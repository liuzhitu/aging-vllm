# 里程碑：M7a 按本轮计算范围分配 KV

状态：代码已修改，待统一 GPU 验证；不是已验收里程碑。日期：2026-10-12。

## 目标

让 KV Cache（注意力键值缓存）容量随本轮计算终点增长，避免分块 prefill（输入预填充）首次进入时占用整个 prompt 所需的块。先保留 decode-first（优先解码）选择规则；动态优先级是后续 M7b。

## 调查

修改前基于 `bde300ea63a8f24a92db3b1135ef605cae04a729`。`BlockManager.can_allocate()` 根据 `Sequence.num_blocks` 检查整请求容量，`allocate()` 分配至整请求末尾；`Scheduler.schedule()` 只有首次 prefill 分配，后续部分 prefill 默认容量足够。decode 扩容则根据 token 长度的模数推测是否需要新块，没有检查真实持有容量。

执行主链：`LLMEngine.step()` → `Scheduler.schedule()` → `ModelRunner.run()` → `Scheduler.postprocess()`。分配属于调度阶段，计算进度属于执行后提交阶段。

## 设计

```text
end = 有效 KV token 数 + 本轮计划计算 token 数
目标块数 = ceil(end / block_size)
新增块数 = max(0, 目标块数 - len(block_table))
```

首次采用缓存前缀时，缓存本身是有效 KV，计入起点；本轮 GPU 计算预算只用于后缀。缓存采用与新块分配的总容量检查先完成，再修改所有权。

不增加持久容量字段，不修改 `Sequence.num_blocks`：全部 token 的逻辑块数和 `len(block_table)` 表示的实际容量必须区分。保留旧 GPU 脚本直接调用的整请求默认分配方式，生产 scheduler 显式传入本轮终点。

## 实现

### `nanovllm/engine/block_manager.py`

- 新增 `get_cached_blocks()`：只查询连续完整块前缀，返回块 ID，不分配或修改引用计数。
- `can_allocate(seq, num_tokens=None)`：按目标终点检查容量；已被其他请求持有的前缀块可共享，不需要取空闲块；缓存但空闲的块仍需从空闲池取回。返回值沿用缓存块数或 -1。
- `allocate(seq, num_cached_blocks, num_tokens=None)`：先采用缓存块，再只为本轮终点分配新块，避免自己的新分配覆盖待采用的缓存块。
- `can_append()` / `may_append()`：根据目标块数与实际 block_table 长度检查、补足缺口，支持续算 prefill 一轮新增多个块和 decode 扩容。默认终点仍是当前完整 token 长度。
- `hash_blocks()` 与 `deallocate()` 沿用原实现。哈希只根据本轮成功计算进度登记完整块；分配空白容量不登记缓存。

### `nanovllm/engine/scheduler.py`

- decode 分支用 `num_cached_tokens + 1` 明确本轮终点，容量足够时不重复分配。
- 首次 prefill 先查询缓存前缀，计算本轮 token 数，再检查并分配目标容量。
- 已持有 KV 的部分 prefill 每轮检查扩容，随后才写入计划。
- waiting 改用当轮列表快照遍历，KV 不足时继续考虑后续候选；完整 prefill 按请求身份移出 waiting，避免误删队首。原有“已有 prefill 后，后续放不下的请求停止”规则保留。
- 整轮没有可执行请求时抛出包含 waiting/running 数与空闲 KV 块数的 RuntimeError，避免空计划进入 runner，也不隐藏扩容无进展。
- `postprocess()` 不变：模型执行成功后才提交 KV 计算进度，部分 prefill 不生成输出 token。

## 验证

按用户要求，没有运行 CPU 测试，没有编写或修改测试代码，也没有在本轮执行 GPU 测试。

实际完成的只读检查：`git diff --check` 无差异格式错误；`git diff --stat` 显示两个源文件共新增 56 行、删除 27 行；`git status --short` 确认测试文件未修改。通过 `rg` 审阅生产调用点，并阅读 `ModelRunner.prepare_prefill()` / `prepare_decode()` 的槽位构造契约。

这些是差异与源码审阅，不是运行正确性证据，不声称 GPU 输出、性能或所有权恢复已通过验证。

## GPU 验证缺口

尚待统一 GPU 验证：首次部分 prefill 实际块数、跨块续算、一次新增多个块、前缀共享/空闲缓存采用、纯 decode 和 CUDA Graph（GPU 执行图）、KV 不足后的状态与结束回收。具体测试代码在用户明确要求后再写。

现有 `test/test.py / run_preemption_control_plane_case()` 给 257-token 请求预分配两个 256-token 块，计算进度设为 256，再期待 decode 缺块而被抢占。新检查正确识别两个块已覆盖第 257 个位置，不会触发该抢占，因此该场景的旧断言不再适用。未修改它，不能将整个旧套件宣称为可原样通过。

## 工程记录

### 架构决策

先建立本轮容量契约，再改动态优先级，避免把调度选择变化和 KV 所有权变化混在一次验收中。块管理器负责容量与引用计数，scheduler 负责本轮计算范围，runner 使用已经保证足够的 block_table。

选择可选终点参数保留已存在的直接分配调用，而非未经用户授权改测试。前缀查询与容量检查在生产首次分配路径各查询一次，这是当前简单接口的开销；未测量收益或回归，不借机新增缓存查询状态。

### 不变量

- 入选计划的请求必须有覆盖本轮 end 的 KV 容量。
- 容量分配不能增加有效计算进度，缓存采用除外。
- 缓存前缀共享维持引用计数，空白块不能成为缓存命中。
- 请求本轮最多入选一次，计划总 token 数和请求数仍受全局预算约束。
- 部分 prefill 留在 waiting 且可持有 KV；完成 prefill 进入 running；结束时释放所有权。

### 缺陷或新发现

按需分配减少提前占有，并不改变预分配 GPU 缓存池的总大小。后续 prefill 扩容仍可能被其他请求占用的 KV 阻挡；首版不主动抢占 waiting 部分 prefill。若整轮无法推进，明确报错，完整拥塞恢复待 M8。

旧 decode 边界判断会在已有足够块时误判缺块；新判断按实际容量纠正，该变化影响旧抢占 fixture（人工构造场景）的预期。

## 后续工作

先统一 GPU 验证 M7a，再按后续授权接入 M7b 的服务时间、跨队列候选排序和单请求计算分块上限。M8 再定义动态准入抢占和共享块释放预检查。
