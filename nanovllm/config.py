import os
from dataclasses import dataclass
from math import isfinite
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduler_policy: str = "aging"
    prefill_chunk_size: int = 256
    preemption_wait_ms: float = 50.0
    scheduler_reserve_full_isl: bool = True

    def __post_init__(self):
        if self.scheduler_policy not in ("aging", "decode_first"):
            raise ValueError("scheduler_policy must be 'aging' or 'decode_first'")
        if not isinstance(self.prefill_chunk_size, int) or self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be a positive integer")
        if not isfinite(self.preemption_wait_ms) or self.preemption_wait_ms < 0:
            raise ValueError("preemption_wait_ms must be finite and non-negative")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
