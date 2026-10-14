from collections import deque
from dataclasses import dataclass
from time import monotonic

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(frozen=True, slots=True)
class ScheduledSequence:
    seq: Sequence
    is_prefill: bool


@dataclass(slots=True)
class SchedulerOutput:
    scheduled: list[ScheduledSequence]

    @property
    def seqs(self):
        return [item.seq for item in self.scheduled]

    @property
    def is_prefill(self):
        return self.scheduled[0].is_prefill


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.scheduler_policy = config.scheduler_policy
        self.prefill_chunk_size = config.prefill_chunk_size
        self.preemption_wait = config.preemption_wait_ms / 1000
        self.reserve_full_prefill = config.scheduler_reserve_full_isl
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.arrival_time = seq.last_service_time = monotonic()
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        if self.scheduler_policy == "aging":
            scheduled = self._schedule_aging()
        else:
            scheduled = self._schedule_decode_first()
        if not scheduled:
            raise RuntimeError(
                "No request can be scheduled with the current token budget and KV capacity "
                f"(waiting={len(self.waiting)}, running={len(self.running)}, "
                f"free_kv_blocks={len(self.block_manager.free_block_ids)})"
            )
        return SchedulerOutput(scheduled)

    def _schedule_aging(self, force_preemption: bool = False) -> list[ScheduledSequence]:
        candidates = [ScheduledSequence(seq, is_prefill=False) for seq in self.running]
        candidates.extend(ScheduledSequence(seq, is_prefill=True) for seq in self.waiting)
        # Resident partial prefills are active too. Service aging applies within
        # the resident group; non-residents can only enter in original FIFO order.
        candidates.sort(key=lambda item: (
            not bool(item.seq.block_table),
            item.seq.last_service_time if item.seq.block_table else item.seq.arrival_time,
            item.seq.arrival_time, item.seq.seq_id,
        ))
        scheduled = []
        preempted = set()
        preemption_used = False
        now = monotonic()
        remaining = self.max_num_batched_tokens

        for item in candidates:
            if len(scheduled) == self.max_num_seqs or remaining == 0:
                break
            seq = item.seq
            if seq in preempted:
                continue
            # Local WAITING also includes resident partial prefills. Only stop
            # admissions without KV; those residents may keep making progress.
            if preempted and not seq.block_table:
                continue
            if item.is_prefill:
                if not seq.block_table:
                    num_cached_blocks = len(self.block_manager.get_cached_blocks(seq))
                    start = num_cached_blocks * self.block_size
                else:
                    start = seq.num_cached_tokens
                num_tokens = min(seq.num_tokens - start, self.prefill_chunk_size, remaining)
            else:
                start = seq.num_cached_tokens
                num_tokens = 1

            end = start + num_tokens
            if not seq.block_table:
                admission_end = self._reservation_end(seq) if self.reserve_full_prefill else end
                can_schedule = self.block_manager.can_allocate(
                    seq, admission_end, reserved_blocks=self._unallocated_reserved_blocks(),
                ) != -1
                if not can_schedule:
                    # Waiting, including preempted requests, never evicts a
                    # resident. Do not let a later arrival bypass the FIFO head.
                    break
            else:
                admission_end = end
                can_schedule = self.block_manager.can_append(seq, end)
            if not can_schedule:
                if preemption_used:
                    continue
                if not force_preemption and now - seq.last_service_time < self.preemption_wait:
                    continue
                selected = {chosen.seq for chosen in scheduled}
                unselected = [candidate for candidate in candidates if candidate.seq not in selected]
                if not self._preempt_for_kv(seq, admission_end, unselected, preempted):
                    continue
                preemption_used = True

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks, end)
            else:
                self.block_manager.may_append(seq, end)
            if item.is_prefill and end == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            seq.num_scheduled_tokens = num_tokens
            seq.is_prefill = item.is_prefill
            scheduled.append(item)
            remaining -= num_tokens

        # No plan means no allocations or successful preemptions were committed.
        # Retry once without the fairness delay to recover forward progress.
        if not scheduled and not force_preemption and self.max_num_seqs > 0 and remaining > 0:
            return self._schedule_aging(force_preemption=True)
        return scheduled

    def _reservation_end(self, seq: Sequence) -> int:
        # The last emitted token is never forwarded once max_tokens is reached.
        return seq.num_prompt_tokens + max(0, seq.max_tokens - 1)

    def _unallocated_reserved_blocks(self) -> int:
        if not self.reserve_full_prefill:
            return 0
        # Only future growth is reserved. Existing physical blocks (including
        # shared prefixes) are already deducted from the free-block pool once.
        return sum(
            max(0, (self._reservation_end(seq) + self.block_size - 1) // self.block_size
                - len(seq.block_table))
            for seq in (*self.running, *self.waiting) if seq.block_table
        )

    def _preempt_for_kv(
        self, seq: Sequence, end: int,
        unselected: list[ScheduledSequence], preempted: set[Sequence],
    ) -> bool:
        if not seq.block_table:
            return False
        priority = (seq.arrival_time, seq.seq_id)
        eligible = [
            item.seq for item in unselected
            if item.seq not in preempted and item.seq.block_table
            and (item.seq.arrival_time, item.seq.seq_id) > priority
        ]
        # Computed tokens conservatively estimate replay work. Prefix reuse may
        # reduce the actual cost; this does not estimate GPU execution time.
        eligible.sort(key=lambda victim: (
            victim.num_cached_tokens, -victim.arrival_time, -victim.seq_id,
        ))
        victims = []
        for victim in eligible:
            victims.append(victim)
            if not self.block_manager.can_schedule_after_preempt(seq, end, victims):
                continue
            for victim in victims:
                if victim.status is SequenceStatus.RUNNING:
                    self.running.remove(victim)
                else:
                    self.waiting.remove(victim)
                self.preempt(victim)
                preempted.add(victim)
            return True
        return False

    def _schedule_decode_first(self) -> list[ScheduledSequence]:
        scheduled = []
        num_batched_tokens = 0

        # decode
        while (
            self.running
            and len(scheduled) < self.max_num_seqs
            and num_batched_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            end = seq.num_cached_tokens + 1
            while not self.block_manager.can_append(seq, end):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq, end)
                scheduled.append(ScheduledSequence(seq, is_prefill=False))

                num_batched_tokens += seq.num_scheduled_tokens

        # prefill
        for seq in list(self.waiting):
            if len(scheduled) == self.max_num_seqs:
                break
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = len(self.block_manager.get_cached_blocks(seq))
                start = num_cached_blocks * self.block_size
            else:
                start = seq.num_cached_tokens
            num_tokens = seq.num_tokens - start
            if remaining < num_tokens and any(item.is_prefill for item in scheduled):
                break
            num_scheduled_tokens = min(num_tokens, remaining)
            end = start + num_scheduled_tokens
            if not seq.block_table:
                if self.block_manager.can_allocate(seq, end) == -1:
                    continue
                self.block_manager.allocate(seq, num_cached_blocks, end)
            else:
                if not self.block_manager.can_append(seq, end):
                    continue
                self.block_manager.may_append(seq, end)
            seq.num_scheduled_tokens = num_scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            scheduled.append(ScheduledSequence(seq, is_prefill=True))

        self.running.extendleft(reversed([item.seq for item in scheduled if not item.is_prefill]))
        return scheduled

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int | None]):
        assert len(token_ids) == len(output.scheduled)
        completed_at = monotonic()
        for item, token_id in zip(output.scheduled, token_ids):
            seq = item.seq
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.last_service_time = completed_at
            if token_id is None:
                assert item.is_prefill and seq.num_cached_tokens < seq.num_tokens
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
