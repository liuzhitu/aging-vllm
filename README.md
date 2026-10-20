# aging-vllm

**基于 nano-vLLM 的轻量级 LLM 推理引擎，围绕 aging 调度协同设计混合批次、分块预填充与 KV Cache 管理。**

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的精简推理链路，实现了以 aging（按最近成功服务时间轮换）为核心的调度机制：优先推进持有 KV 的驻留请求，在驻留请求之间优先服务更久未获计算机会的请求，新请求按 FIFO（先到先服务）顺序准入。

为配合这一服务顺序，Chunked Prefill（分块预填充）限制单请求每轮的计算量，Mixed Batch（混合批次）让预填充与解码共享执行轮次；按需 KV 分配、未来容量预算和受控抢占共同约束缓存资源使用。项目打通了调度计划、模型执行与请求状态提交的完整链路，执行成功后才更新有效 KV 进度与服务时间。

## 我们做了什么

| 方向 | 实现内容 | 优化目标 |
|---|---|---|
| **Aging 调度** | 优先推进持有 KV 的请求，在驻留请求间按最近成功服务时间排序；新请求按到达顺序准入 | 让驻留请求轮换获得计算机会，并保持新请求准入顺序 |
| **混合批次执行** | 同一批次容纳 decode 与 prefill 请求，通过一次变长序列前向执行；每个请求独立记录执行阶段 | 让正在生成的请求与新输入共享一次执行轮次 |
| **分块 Prefill** | 长输入分轮计算；`aging` 限制单请求的 chunk 大小，支持多个部分 prefill 同批执行 | 分散长输入计算量，增加请求之间的计算交错机会 |
| **按需 KV 分配** | 根据本轮计算终点分配或扩展物理块，支持跨块续算与前缀缓存共享 | 减少尚未计算部分提前占用的 KV 块 |
| **KV 预算准入** | 默认将剩余输入与最大输出增长纳入容量预算；未来预算与当前物理分配分开计算 | 为已准入请求后续增长保留容量预算，减少反复抢占 |
| **受控抢占与恢复** | 保护本轮已选请求；驻留请求仅可抢占较晚到达的持有者，按已计算 token 数挑选；释放前模拟共享块引用变化 | 扩容受阻时释放足够容量，优先选择预计重算量较少的请求 |
| **逐请求结果提交** | 部分 prefill 只提交 KV 进度；decode 与完成 prefill 的请求才采样；执行成功后更新服务时间 | 保持调度计划、有效 KV 进度和输出 token 的一致契约 |

前缀缓存、纯 decode 的 CUDA Graph 路径和原有 `LLM` / `SamplingParams` 使用方式继续沿用上游能力。

## 带来了什么优化

### 长输入到达时，降低正在生成请求的输出间隔尖峰

在 **RTX 4090 / Qwen3-0.6B** 上，先让 16 个短请求进入 decode，再注入一个长输入。短输入为 128 token，批次预算为 2048 token。对比原始基线与本项目早期 decode-first 混合批次实现，记录已有请求的最大 TPOT（相邻输出 token 的实际等待时间）。

| 注入的长输入 | 原始基线 max TPOT | 混合批次 max TPOT | 尖峰降低 |
|---|---:|---:|---:|
| 4K token | 64.33 ms | 40.47 ms | **37.1%** |
| 8K token | 142.10 ms | 41.22 ms | **71.0%** |
| 16K token | 360.80 ms | 66.38 ms | **81.6%** |

混合批次让长输入 prefill 期间的每一轮都继续输出 16 个 decode token。16K 场景的最大等待从 **360.80 ms 降到 66.38 ms**。

实验使用相同的 [TPOT benchmark](bench_tpot_interference.py)，每个输入长度测量一次，计时在每轮执行前后同步 CUDA。原始数据见 [基线结果](tpot-results/baseline/summary.csv) 和 [混合批次结果](tpot-results/chunked/summary.csv)。该组数据对应早期混合批次实现，单独说明这项机制的效果。

### 多长输入同批推进，减少提前占块与 Prefill 轮次

2026-10-14 的 RTX 4090 / Qwen3-0.6B 单次调度实验，对比早期 mixed-batch 提交 `bde300e` 与当前调度实现 `62388bd`，批次预算为 2048 token，当前 `aging` 的 chunk 为 256 token：

| 多长输入场景 | 早期 mixed-batch | 当前 aging |
|---|---:|---:|
| 首轮执行安排 | 1 个请求 × 2048 token | 8 个请求 × 256 token |
| 首轮物理 KV 占用 | 16 块 | **8 块** |
| 含 prefill 的执行轮次 | 23 轮 | **16 轮** |
| 最慢请求首次获得计算的等待 | 576.25 ms | **35.12 ms** |

按需分配使首轮物理占块减少 **50%**；分块调度让多个长输入从第一轮就参与计算，prefill 轮次减少约 **30.4%**。首次获服务时间指开始计算，首 token 延迟另行计量。

### KV 压力下，减少抢占与重复计算

同次实验限制可用 KV 块数为 32、48、64，分别测量 KV 压力与共享前缀场景。当前 `aging` 默认开启未来容量预算，结果如下：

| 场景 / KV 块池 | 早期 mixed-batch 抢占次数 / 重算 token | 当前 aging 抢占次数 / 重算 token |
|---|---:|---:|
| KV 压力 / 32 | 4 / 1461 | **0 / 0** |
| KV 压力 / 48 | 1 / 1275 | **0 / 0** |
| KV 压力 / 64 | 2 / 1528 | **0 / 0** |
| 共享前缀 / 32 | 6 / 1533 | **0 / 0** |
| 共享前缀 / 48 | 0 / 0 | **0 / 0** |
| 共享前缀 / 64 | 0 / 0 | **0 / 0** |

上述组合均完成全部计划请求，当前实现的实际占块与未来预算之和均未超过块池容量。这些数据是指定负载下的单次资源管理观察，性能与输出验证分别进行。

## 核心执行链路

```text
请求进入 waiting 队列
    ↓
Scheduler：aging 驻留请求服务排序 + FIFO 准入
    ↓
确定每个请求的阶段与本轮 token 数
    ↓
BlockManager：容量预算检查 + 按需分配 / 扩容
    ↓
SchedulerOutput：逐请求执行计划
    ↓
ModelRunner：混合批次一次 varlen forward / 纯 decode 快路径
    ↓
按请求采样，提交 KV 进度与服务时间
    ↓
完成请求释放 KV；其余请求进入下一轮
```

资源不足时，调度器先检查允许抢占的请求集合以及实际可释放块数，再执行抢占。请求保留 token 历史，重新进入 prefill 以恢复 KV；每轮最多成功执行一次抢占决策。

## 快速使用

在具备 CUDA 的 Linux 环境安装，使用本地模型目录：

```bash
git clone https://github.com/liuzhitu/aging-vllm.git
cd aging-vllm
pip install -e .
```

发行包名称为 `aging-vllm`，Python 导入包名沿用 `nanovllm`：

```python
from nanovllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        "/path/to/Qwen3-0.6B",
        scheduler_policy="aging",
        prefill_chunk_size=256,
        max_num_batched_tokens=2048,
        max_num_seqs=32,
        scheduler_reserve_full_isl=True,
    )
    params = SamplingParams(temperature=0.6, max_tokens=256)
    outputs = llm.generate(["介绍一下大语言模型的 KV Cache。"], params)
    print(outputs[0]["text"])
```

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `scheduler_policy` | `"aging"` | 驻留请求按服务时间轮换；也可选 `"decode_first"` 做策略对照 |
| `prefill_chunk_size` | `256` | `aging` 中单请求每轮的 prefill token 上限 |
| `max_num_batched_tokens` | `16384` | 全批次计算 token 预算 |
| `max_num_seqs` | `512` | 每轮请求数上限 |
| `scheduler_reserve_full_isl` | `True` | `aging` 准入时计入剩余输入与最大输出增长预算 |
| `preemption_wait_ms` | `50.0` | 正常抢占前的未获服务等待阈值；整轮无进展时可执行一次恢复扫描 |

## 代码导航

| 文件 | 职责 |
|---|---|
| [scheduler.py](nanovllm/engine/scheduler.py) | 批次计划、请求排序、准入、抢占与执行后提交 |
| [block_manager.py](nanovllm/engine/block_manager.py) | KV 块分配、前缀共享、引用计数与释放容量模拟 |
| [model_runner.py](nanovllm/engine/model_runner.py) | 模型输入构造、混合前向与采样 |
| [sequence.py](nanovllm/engine/sequence.py) | 请求状态、token 历史与计算进度 |
| [llm_engine.py](nanovllm/engine/llm_engine.py) | 调度、执行和结果回收的主循环 |
| [config.py](nanovllm/config.py) | 调度与推理配置 |

## 致谢

项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 开发，沿用其轻量推理引擎与模型实现，并在此基础上扩展调度和 KV 管理。遵循 [MIT License](LICENSE)。
