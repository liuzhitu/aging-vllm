# 里程碑：M7b 动态优先级调度

状态：代码已实现，待统一 GPU 验证；尚未验收。日期：2026-10-12。

## 目标

按照请求最近成功获得计算的时间，统一选择 waiting（等待）与 running（运行）中的请求，替换默认 decode-first（优先解码）顺序。每个 prefill（输入预填充）请求单轮计算量有上限，结合 M7a 的按需 KV Cache（注意力键值缓存）分配实现计算机会轮换。

本轮不实现 M8 的等待阈值和主动 KV 准入抢占；不编写或修改测试代码，不运行 CPU 测试。

## 调查

HEAD 为 `bde300ea63a8f24a92db3b1135ef605cae04a729`；本轮开始时已有 M7a 对 `block_manager.py` 与 `scheduler.py` 的未提交修改，以及学习、方案、工程记录文件。本轮保留这些改动，新增修改仅涉及 `config.py`、`sequence.py`、`scheduler.py`。

实际路径：`LLMEngine.add_request()` → `Scheduler.add()` → `LLMEngine.step()` → `Scheduler.schedule()` → `ModelRunner.run()` → `Scheduler.postprocess()`。

旧路径先处理 running decode，再以剩余预算安排 waiting prefill；已有一个 prefill 后，后续放不下的 prefill 会停止。本地部分 prefill 留在 waiting 且持有 KV，完成 prefill 才转入 running。runner 已按每个计划项的阶段选择 mixed（混合批次）的 varlen（变长序列）执行，不依赖 decode 一定排在前面。

## 设计

```text
age = 当前单调时间 - 最近成功服务时间
排序键 = (last_service_time, seq_id)，从小到大
```

同轮当前时间相同，无需保存 score（优先级分数）或运行动态堆。首次入队初始化服务时间；成功执行后一次取提交时间，同批请求共用。仅计划、分配、缓存命中、被跳过和被抢占均不更新时间。

每轮临时保存请求身份与阶段；候选快照随后不因队列转换改变阶段。每个请求最多入选一次；KV 不足时可以跳过。动态优先级表达未获计算机会时长，不表示 GPU 耗时公平，也不提供所有资源拥塞下的时延上界。

## 实现

### `nanovllm/engine/sequence.py`

`Sequence.__init__()` 增加 `last_service_time = 0.0`。该初值不是正式到达时刻，生产请求在 `Scheduler.add()` 中取得实际值。没有新增到达时间、缓存分数或等待轮数。字段只由 scheduler 所在进程使用，序列化协议不变，worker（执行进程）不参与排序。

### `nanovllm/config.py`

`scheduler_policy` 默认 `aging`，另支持 `decode_first`；`prefill_chunk_size` 默认 256。配置入口检查策略名称和正整数分块大小，避免调度器运行时才发现无效策略或不能推进的分块。256 是起点，不是经测量的最佳配置。

`LLMEngine` 已从 kwargs 读取 Config 字段，因此不增加新的转发接口。例如 `LLM(model, scheduler_policy="aging", prefill_chunk_size=256)`；对照选择 `scheduler_policy="decode_first"`。对照分支不应用单请求 chunk 上限，但仍使用 M7a 按需 KV，不能称为原始 upstream（上游）基线。

### `nanovllm/engine/scheduler.py`

- `__init__()` 保存两项配置；`add()` 用 `monotonic()` 初始化服务时间。
- `schedule()` 选择策略，将各自返回的计划项列表包装为同一种 `SchedulerOutput`（批次执行计划）；共用无进展错误，不能把空计划交给 runner。
- `_schedule_decode_first()` 保留 M7a 的原选择逻辑；整理为列表返回，不改变其请求排序或分块规则。
- `_schedule_aging()` 从 running 与 waiting 建立 `ScheduledSequence` 候选列表，再按服务时间及 seq_id 排序。prefill 每次最多计算 `min(待计算量, prefill_chunk_size, 剩余全局预算)`；decode 每次计算 1 token。沿用 M7a 的目标终点容量接口，多个部分 prefill 可同批执行。
- 未选请求留在原集合；部分 prefill 留在 waiting，完成 prefill 进入 running。原队列顺序不再等于动态执行顺序，计划顺序由候选选择顺序决定。
- decode 扩容失败时，仅从本轮快照中排序更靠后的 decode 候选逆序挑选 victim（被抢占请求）；这保护排序靠前的请求以及所有已选请求。移出 running 后沿用 `preempt()` 释放并入队，局部集合记录被抢占身份，防止后来按旧 decode 快照再次入选。若释放较低优先级运行请求仍不足，当前 decode 自身退回 waiting，本轮不重新入选。
- `preempt()` 不更新时间；没有新增等待阈值或抢占 waiting 部分 prefill 的主动准入逻辑。
- `postprocess()` 在结果长度检查后取一次 `completed_at`，每请求提交 KV 进度后更新时间，位置在部分 prefill 的 None 分支之前。采样资格、结束判断及最终 KV 释放保持原逻辑。

运行流程：

```text
统一候选排序
  → 确定每请求本轮计算量
  → 按需 KV 检查/扩容
  → 按选择顺序执行
  → 提交有效 KV 和同批服务时间
  → 下一轮重新排序
```

例如仅有一个批次槽位，A 刚完成 decode，B 已较久未服务：B 先获得一个 prefill chunk；提交后 B 的时间变新，A 重新获得等待优势。能否成功执行仍取决于 KV，不能仅根据排序承诺一定入选。

## 验证

实际执行：`git diff --check` 无差异格式错误；`git diff -- ...` 审阅三处源文件；`rg` 核对配置、时间字段、Scheduler 构造与调用点；读取 `LLMEngine` 确认新 Config 字段已有传递路径。

这些只是差异检查和源码审阅。按用户要求，未运行 CPU 测试，没有编写或修改测试代码；本轮也未执行 GPU 验证。不声称请求轮换、混合输出、抢占恢复或性能已实测通过。

## GPU 验证缺口

后续统一在 GPU 验证：跨阶段排序、受限批次轮换、多个部分 prefill、成功提交时间、前缀命中预算、纯 decode 的 CUDA Graph（GPU 执行图）路径、抢占保护、重复入选防止以及最终 KV 回收。具体代码等待用户明确要求。

现有套件不能宣称可原样验收新策略：

- 部分断言明确期待 decode-first 顺序，需按策略区分。
- `test/test.py` 的两个手工 `SimpleNamespace` 配置缺少 `scheduler_policy` 与 `prefill_chunk_size`，直接构造 Scheduler 时需要补齐；没有为人工 fixture（构造场景）给生产构造器加入静默默认值。
- M7a 已改变旧人工 KV 抢占场景的有效性，见对应记录。
- 随机采样按计划顺序分配 RNG（随机数）时，新排序可能改变逐请求输出；正确性比较不能直接套用旧随机输出完全一致的要求。

## 工程记录

### 架构决策

统一选择发生在 scheduler，不由 runner 猜阶段或排优先级。保留旧策略为明确实验对照，不引入通用策略注册框架。服务时间保存在请求本身，score 临时推导；被抢占集合只在本轮存在，不形成长期重复状态。

### 不变量

- 计划 token 总数与请求数不超过各自上限；每请求最多入选一次。
- 已选请求持有的 KV 不能被本轮后续抢占释放。
- 有效 KV 仅在成功执行后提交；缓存采用沿用 M7a 契约。
- 每次成功部分计算也算服务，同批提交时间一致。
- 未获服务请求和被抢占请求保留原服务时间。
- decode-first 与 aging 共享同一结果提交、阶段采样与完成释放协议。

### 缺陷或新发现

动态排序可能增大 decode token 间隔，不能把公平机会直接等同于更低时延或更高吞吐。M8 前，waiting 扩容可能继续被 KV 阻挡；所有请求均无法执行时明确报错。

`LLMEngine.step()` 的吞吐显示仍按计划首项阶段分类，不能据其值判断混合批次的实际吞吐。当前保持不变，后续性能实验单独记录真实计算量和时间。

## 后续工作

M7a/M7b 代码已修改，整个 M7 仍待统一 GPU 验证，不标记验收完成。M8 再实现基于等待阈值的 KV 主动准入、共享块释放预检查和重算风险控制。

建议阅读：`Config` → `Scheduler.add()` → `_schedule_aging()` → `postprocess()`。
