# Ascend V4.1 与 vLLM-Ascend 持平计划

2026-11-17。目标已由需求方定死：**后端与 vLLM-Ascend 持平；先只做 bs8 一档**。本文件把"持平"翻译成可验收的算术，并补上 2026-11-17 账本对比报告里自评出的六个缺口。

判据基线：TS TPOT `267.812ms` / stage `330.0ms`（stage 418 + 415 账本）；vLLM-Ascend `31.887ms` / 每层 `0.80ms`（stage 420，eager）。持平 = **TPOT ≤ 33ms**（1.03x 以内），即 stage ≈ 40ms。

**口径纪律（2026-11-18 实测，务必遵守）**：同一份代码同一天的 bs8 负载，`eager` TPOT `1.1291s`、
`graph` TPOT `0.2742s`，相差 `4.11x`。冻结基线与 vLLM 侧都是 graph 口径，因此**一切比较、
一切"改了会不会更快"的实验都必须在 graph 模式下做**；Phase A 的 op 账本（30,452 次/步）
采自 eager，只能解释 eager 的 host 分发开销，不能直接当 graph 模式的物价表。

## 一、持平要求什么：先看算术能不能到

以 2026-11-18 的同采集器账本（`kernel_details.csv` 重算，两侧同口径）逐项相减：

| 动作 | 剩余 stage | 倍数 | 依据 |
| --- | --- | --- | --- |
| 现状 | 329.8ms | 1.00x | TS 30,452 次/步、8.25ms/层（eager 账本；graph 实测步时 262.6ms） |
| 删只在 TS 出现的算子（微算子 + silu + 镜像往返 28，**不含原估的 121.9**） | ≈180ms | 1.83x | 实测 204.2ms，其中 93.9ms 是误归因 |
| 再融掉「启动次数多一个量级」的族（ReduceSum/Add/Mul/Cast…） | ≈113ms | 2.93x | 实测 67.4ms |
| 把 MoE dispatch 修到 vLLM 水平 | ≈88ms | 3.75x | 实测 24.8ms（次数同为 40/步） |
| vLLM 参照 | 37.8ms | — | 2,286 次/步、0.94ms/层 |

结论：**93% 的残差（204.2 + 67.4 = 271.6 / 292.1）是「同一件事用错粒度」，不需要新算子就能拿掉；真正要在 kernel 层追的只有 MoE dispatch 的 24.8ms。** 这也是此前多轮调参没有收益的原因——被调的参数不是这 93% 的成因。

**2026-11-18 修正**：镜像不能再当 121.9ms 抵扣（它不可删，真实往返 ≈28ms，且融合写已把它拿掉但无实测收益，见
`outputs/v41-debug/l2-fused-mirror-result-20261118.md`）。因此上表剩下三步的最乐观落后仍是
`≈88ms` 对 `37.8ms`，**光靠"融算子"到不了 33ms，必须同时动量化链的表示（BF16-only 或原生量化 MLA）**。

## 二、补的第一次测量：同采集器残差归因（Phase A）

现状两份账本用了不同采集器（vLLM: `msprof --task-time=l0`，TS: `torch_npu.profiler`），插桩代价差 13x，**无法比较同名算子的绝对耗时**——而"101ms 去哪了"恰恰只能这样回答。

### 2.1 采集器选型（2026-11-17 实测，结论已改）

先尝试"把 TS 也塞进 `msprof --application`"，两次都失败，且是两种独立的硬冲突：

| 尝试 | 结果 | 根因 |
| --- | --- | --- |
| `--task-time=l0` | 8 个 `ts-host` 全部 `<defunct>`，`ValueError: The memory capacity is unbalanced`（`distributed_initializer.py:321`） | msprof 每个 worker 的 profiling agent 预留 HBM 不等，超过 TS 的 `min < local*0.9` 守卫 |
| `--task-time=on`（更轻的 agent） | 守卫通过，但 `Scheduler process 0 ... closed its ready pipe` + `signal 10`，引擎称 `received signal 10; terminating scheduler` | msprof 用 SIGUSR1 与子进程通信，TS 引擎同样处理 SIGUSR1，语义冲突 |

结论：**`msprof --application` 不能用于 TokenSpeed**（不是参数问题，是信号与显存两侧的结构性冲突）。l0 那轮还额外写出 57G 中间数据。

因此对称采集改到另一侧：**给 vLLM 挂 `torch_npu.profiler`**，与 TS 冻结账本（stage 415）同一采集器、同一 bs8 负载、同一 eager 模式。实现落在我们自己的 staged adapter（`outputs/v41-debug/deepseek_v41/`，非镜像源码）：

- `debug_capture.py` 新增 `profile_forward_hook()`：按 forward 计数开窗（先跳 `VLLM_V41_PROFILE_SKIP_FORWARDS` 步热机，再采 `VLLM_V41_PROFILE_STEPS` 步），`_ExperimentalConfig(profiler_level=Level1, export_type=[Text])`，`atexit` 兜底关窗；
- `model.py` 的 `forward()` 顶部调用该钩子（每 step 一次）；
- `run_v41_vllm_cluster.py` 新增 `--profile-npu`，注入 `PYTHONPATH=/logs/vllm-debug` 与 `VLLM_V41_PROFILE_DIR=/logs/vllm-profile`，输出留在 stage 内。

产物口径：`ASCEND_PROFILER_OUTPUT/{kernel_details,op_statistic,step_trace_time,api_statistic}.csv`，与 TS 415 的 `rank0-csv.tar` 同名同源，可以直接按 op name 相加后除以 step 数对撞。

### 2.2 已完成（2026-11-18）

同采集器账本已拿到，两侧都从原始 `kernel_details.csv` 重算（步数由每层必现 kernel 反推）：

| | 启动次数/步 | 设备时长/步 | 每层 |
| --- | --- | --- | --- |
| TokenSpeed | 30,452 | 329.8 ms | 8.25 ms |
| vLLM-Ascend | 2,286 | 37.8 ms | 0.94 ms |

残差 292.1ms，**100% 归因**：只在 TS 出现 204.2ms、同名 TS 更贵 +103.8ms（其中 67.4ms 来自「启动次数多一个量级」）、vLLM 独有 −15.7ms。

由此更正两处旧的认知：

1. `pure-decode-summary.json` 的「每步次数」把 10 步窗口当 149 步归一，MoE 一类被放大 15x，不再引用它的 per-op 次数；
2. 真正需要 kernel 层面追的只有 **MoE dispatch 24.8ms**（次数同为 40/步，TS 6.5x/4.5x；vLLM 的 AICPU kernel 是 duration 3.45ms + wait 41.7ms，TS 是 duration 15.39ms + wait 0 —— 更像 token 数/布局或缺少异步 MC2 重叠，不是 kernel 实现本身）。

细节见 `outputs/v41-debug/phase-a-attribution-20261118.md`。

### 2.4 修正：L2 的真实现状与代价（2026-11-18，动代码前核对）

原计划写的是"删 BF16 镜像 = 121.9ms"。核对源码后两处都不成立：

1. **镜像不能删**。`_forward_v41_ascend_decode` 在 `npu` 上无条件读 `swa_bf16(layer_id)` 和 `global_bf16(owner)`，没有回退到量化页的分支；把 `_direct_paged_bf16()` 改 false 只会让 decode 直接报缺字段。Ascend 上 `swa.is_cuda` 为假，所以 SWA 每层都走"写量化页 → gather 回来 → 写镜像"的往返（40 层）+ 全局 8 个 owner，每步 48 次往返。
2. **121.9ms 是误归因**。`GatherV3 69.64ms` 覆盖 185 次/步，其中镜像只有 48 次（约 18ms）；`ViewCopy 31.92ms` 只有 6 次且来自 `index_copy_` 的 prefill workspace，不是镜像；`ScatterNdUpdate 20.33ms` 里镜像写约占 10.6ms。**镜像往返的真实代价约 28ms**，不是 121.9ms。

因此已经落地的是"量化 + 镜像一次写"（`dsv41.cache_scatter_with_mirror`）：pack 一次、解刚 pack 出来的字节、直接写镜像，**按字节等价**（host 单测覆盖 `swa`/`global`、重复 slot 与非法 slot），每层少一次设备 gather。剩下的"单表示"（BF16-only，等于 vLLM 的做法）要连 `_pack_rows` 那条约 45–50ms 的 ATen 量化链一起拿掉，但代价是 4× 页字节和 PD/L2 策略重做，需要单独决策。



对撞口径（同一份脚本既算 TS 也算 vLLM）：

输出三类条目：

1. **vLLM 有、TS 没有**（`AivKernel`、`SparseFlashMla`、`RmsNorm`、`aclnnDynamicQuantV2`…）→ 这是要补的融合算子清单；
2. **同名但 TS 慢 k 倍**（如 `MoeDistributeDispatchV2`、`MatMulV2`、`QuantBatchMatmulV3`）→ 单核效率差，需看 `cube_utilization`/`aic_mac_ratio` 列判断是形状、分块还是调度问题；
3. **TS 有、vLLM 没有**（`Adds`/`Div`/`ReduceSum` 三连、`AsStrided`、`ScatterNdUpdate` 等）→ 已经知道要靠融合或删除消灭。

Phase A 门槛：`101ms` 残差里 **≥90% 必须落到具名算子或具名结构性差异**上，未归因部分 ≤10ms。达不到就不进入 Phase C——说明还需要第三种采集手段。

## 六之二、2026-11-18 口径修正：eager 账本作废于排优先级

三次外推全部失败，方向一致——eager 账本严重高估：

| 候选 | eager 账本预测 | graph 实测 |
| --- | --- | --- |
| L2 镜像往返 | ~110 ms | ~40 ms（合并写法本身 0 收益） |
| 融合 paged scatter | ~70 ms（GatherV3 份额） | ~4 ms |
| 融合 SwiGLU | 12.6 ms | ≤0.34 ms |

SwiGLU 一项有干净的设备证据：同设备、40 层、bs8、**graph 重放**计时，
`npu_clipped_swiglu` 0.2302 ms/step、ATen 拼装 0.5695 ms/step——即使完全不融合也只有
0.57ms 量级，而 eager 账本记的是 12.60ms。TS eager TPOT 1.1291s → graph 0.2742s（4.11x）
说明 eager 账本的 per-op 时间主要是 host 启动开销，graph 重放整段消掉了它。

**纪律（此后强制）**：

1. 排优先级只认 **graph 口径**的设备时间账本；eager 账本仅用于「有哪些算子」的清单，不用于「值多少毫秒」。
2. 候选微基准一律用 **NPU Graph 捕获 + 重放**计时，禁止 eager 循环计时。
3. graph 账本采集走 TS 自带 `POST /start_profile`（`activities=["CPU","GPU"]`、
   `num_steps=10`、`output_dir` 必须落在可写的 `/logs`，不能落在只读 `/ts`）。
   该路径已在 `ascend-v41-w8a8-serving-handoff.md:222` 验证可捕获已捕获的 Graph decode 重放。
4. 只对 graph 口径下真实占比 **≥5%** 的项动手；低于此的融合移植不再单独占一次重启。

已落地的端到端收益（graph/bs8，两次 repeat）：吞吐 27.6516 → 34.4571 tok/s（**+24.6%**），
TPOT 0.267813 → 0.215158 s（−19.7%），correctness ok。距 vLLM 0.031887s 仍 6.7x。
详见 `outputs/v41-debug/p1-fused-op-ports-20261118.md`。

### 2.5 Gate 1：单表示的容量核算（2026-11-18，已通过）

结论：**内存不是单表示的障碍，单表示比现状更省**。

- 镜像的代价是既成事实：40 个 `swa_bf16` + 4 个 `global_bf16` 一共 **44 KiB/token**（`docs/ascend-v41-w8a8-serving-handoff.md`），同一 arena 预算下 token 容量从 4,812,287 掉到 1,180,287。
- 现网一次 resolve 的 parent 是 `4,147,200` 字节 / 64 tokens = **64,800 B/token**。
- BF16-only 去掉的是 packed 副本（`swa` 40×528 = 21,120 与 `global_kv` 4×288 = 1,152，合计 **22,272 B/token**），保留的是镜像本身与 `index_k`。即 **≈42.5 KB/token，比现状少 34%**，容量反而升到 ≈1.8M tokens。
- 需求侧只有 bs8 × 1024 ctx = **8,192 tokens**，即便按最悲观的"页字节翻 4 倍"假设也不过 2.1 GiB。容量判据在任何口径下都通过，且 **不需要** 缩小 `max_model_len`。

真正的门槛不在内存，而在两个表示边界：`index_k` 必须保持量化（indexer 内核按 FP4/uint8 页读），以及 PD transfer / L2 策略按 group 字段搬运；它们才是单表示改造的工作量所在。

## 三、阶段、门槛与顺序（单位改对）| 阶段 | 内容 | 门槛 | 工时 |
| --- | --- | --- | --- |
| A | 同采集器对称测量 + 残差归因表 | 残差归因率 ≥90% | 1.5h（一次 30min 采集 + 解析） |
| B | **L2 先做**（已按实测修正，见 2.4）：先落"量化+镜像一次写"，再做"单表示"决策（BF16-only，容量已核可，见 2.5） | 融合一步：`outDir` 上 launches 减少 48/步、stage 下降即为通过；单表示一步：causal-pair capture **bit-identical** | 融合 0.5d，单表示 2–3d |
| C | 按 A 的表逐项做 kernel：先补 vLLM 有 TS 没有的融合核（RmsNorm/RmsNormCast/AivKernel/GroupedMatmulSwigluQuant），再消灭「千次启动」族，最后查 MoE dispatch 的 token 数/异步重叠 | 每项：实测节省 ≥ 该项在 A 表中的份额；每次提交都要紧 atol 对照 + launch 数下降可复算 | 按表排期 |
| D | 持平验收 | TPOT ≤ 33ms；launches/step ≤ 2.5k；causal pair + 负控无漂移 | 1d |

顺序修正（相对上一版）：**B 在 C 之前**。理由：镜像 121.8ms 比整个微算子块 75.2ms 更大、不需新算子、可 bit-identical 验证，是唯一"确定能拿到"的 1.59x；C 的每一项都要过数值门槛，风险高得多。上一版把 C 排在前是错的。

单点门槛口径修正（相对上一版）：上一版写"每层掉 ≥1ms（≥40ms/步）"让单个三连去承担微算子块的 53%，探针必然失败。改为：**单个三连 `Adds+Div+ReduceSum` 实测 ≥8ms/步**（该三连合计 23.4ms/步），再按"还有多少个同形三连 × 单次收益"外推，不外推就不立项。

## 四、数值与回归条款（上一版缺失）

- 融合会改 `mean/eps` 的结合顺序与累加顺序：每个新 kernel 必须对 ATen 三连实现做紧 `atol` 对照单测（`test/` 下按 `AGENTS.md` 要求用真实类，不 mock）。
- 每次改动都在**同一个 commit** 上重冻基线，不同 commit 之间的数字不进对比表；基线哈希按 `baseline-artifact-hashes-20261117.txt` 的方式留档。
- `L2` 改了缓存表示，`test/runtime/test_slot_state.py` 的 `SlotStateExporter` 覆盖必须仍通过。
- indexer/attention 侧改动按 `docs/design/attention-prologue.md` 的 numerics 契约验收。

## 五、止损（数字化）

- A 阶段归因率 <90% → 不立项 C，先解决测量。
- B 完成后 `data_movement > 120ms` → 镜像不是杠杆，回到 `go-no-go-report-20261117.md` 结论。
- C 做完前两轮（预计 2 周）后 TPOT 仍 > 120ms → 910B3 上 TS 的结构上限就是 2.5–3x，**持平不可达**；此时以数据向需求方重新报目标（而不是继续追加 kernel）。
- 持平不可达时的备选结论要明说：`1.6–2.5x` 的 TS 是否值得部署，取决于需求方对"持平"的依赖强度——这是决策问题，不是工程问题。

## 六、bs8 优先的范围说明

- 本轮全部测量、验收、门槛都锁在 bs8（concurrency 8、每请求 1 token/step），不做任何 batch 扩展优化。
- 只保留一次 flank 观测（bs1 与 bs32 各一次 TPOT），用途是**判断微算子链是否在小 batch 下被放大**，若 bs32 明显更接近 vLLM，则"必须改 kernel"的结论需要打折。这次观测不算优化项，不设门槛。
- 已知会牺牲的东西：`Allow` 分块策略、预填充分块大小这些对 batch 敏感的参数本轮不调。

## 七、已知局限（写进验收报告，不隐藏）

- vLLM 账本只解析到 TP0（device 0）一个 worker，其余 7 个会话未回写 CSV；MoE EP16 的 rank 间差异未覆盖，Phase A 的 TS 采集也按同样单 rank 口径，避免引入不可比项。
- l0 插桩把绝对耗时放大 ~13x；Phase A 的用途是**同名算子的相对比较**，不能用于跨引擎绝对耗时。
- msprof 采集无法进 CI，只能手工重跑；每次重跑都要重冻基线与留哈希，避免"数字漂了没人知道"。

## 复现

```bash
# vLLM 侧（已完成）
wsl.exe bash outputs/v41-debug/start_vllm_msprof_l0.sh
wsl.exe bash outputs/v41-debug/finish_msprof_run.sh
# TS 侧（Phase A）
wsl.exe bash outputs/v41-debug/start_ts_msprof.sh
wsl.exe bash outputs/v41-debug/finish_ts_msprof.sh
# 同名算子对撞
wsl.exe bash outputs/v41-debug/compare_op_ledgers.sh
```
