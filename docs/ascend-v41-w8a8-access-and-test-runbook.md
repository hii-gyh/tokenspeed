# DeepSeek-V4.1-W8A8 Ascend 远端访问与测试交接 Runbook

更新时间：2026-11-17 09:35 CST
适用仓库：`C:\Users\Lenovo\codes\tokenspeed`
目的：让接手 agent 能在不重新摸索环境的情况下，安全地完成只读复核、服务切换、固定负载 benchmark、文本门禁和 NPU profile 采集。

> 本文是操作手册，不代表性能或正确性验收通过。最新详细历史证据见 `docs/ascend-v41-w8a8-serving-handoff.md`，结构化结果见 `outputs/v41-debug/`。

## 1. 当前状态速览

截至 2026-11-17 09:26 CST 的只读复核：

- TokenSpeed 418 双节点容器均 `Up`；rank0 `/health` 返回 200，`/v1/models` 返回模型权重路径和 `deepseek-v41`。
- vLLM-Ascend 420 双节点容器均 `Exited (137)`，五轮历史结果已保留。
- 两套服务均为 `world_size=16`，当前两节点 16 卡 footprint 不能让两套同规格服务同时驻留；有效对照采用**交替驻留**，不是同时在线 A/B。
- TS 418 与 vLLM 420 的严格固定文本门禁均未通过；不能用 token 数完整、HTTP 200 或 Paris keyword smoke 代替文本正确性。
- 当前没有取得 TS 相对 vLLM-Ascend 的可验证性能优势。TS 418 五轮均值 27.6516 tok/s，vLLM 420 为 209.5961 tok/s。
- rank0 root、`/dev/shm` 和 `/mnt` NFS 空间紧张；未经明确授权不要清理 modelscope 缓存、历史 shm profile 或其他远端文件。
- 每 2 小时进展汇报定时任务已暂停，与远端推理服务状态无关。

## 2. 本地工作区与 Git 注意事项

### 2.1 路径

| 用途 | 路径 |
|---|---|
| Windows 主工作区 | `C:\Users\Lenovo\codes\tokenspeed` |
| WSL 主工作区 | `/mnt/c/Users/Lenovo/codes/tokenspeed` |
| 隔离 fresh worktree | `C:\Users\Lenovo\codes\tokenspeed-fresh-gate-5585` |
| WSL fresh worktree | `/mnt/c/Users/Lenovo/codes/tokenspeed-fresh-gate-5585` |
| 分支 | `cursor/ascend-v41-w8a8-serving` |
| 当前已推送 HEAD | `631ce69de2cbb5df575fba3ddad56825cd072104` |
| origin | `https://github.com/hii-gyh/tokenspeed.git` |
| upstream | `https://github.com/lightseekorg/tokenspeed.git` |

### 2.2 重要工作区边界

- 主工作区有大量既有 modified 文件，很多是 CRLF/格式噪声；**禁止 `git add -A`、`git clean`、`reset --hard` 或批量 checkout 恢复**。
- 多个 `test/ci/npu/` V4.1 脚本仍是未跟踪文件，但它们是所有远端启动和测试的关键工具，不能因“清理未跟踪文件”而删除。
- 以下运行时文件含未提交改动，不能视为已验证候选补丁：
  - `python/tokenspeed/runtime/execution/device.py`
  - `python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py`
  - `python/tokenspeed/runtime/models/deepseek_v41.py`
- fresh worktree 的 HEAD 为 `5585f22b1`，但曾补入未提交部署源码；480/481/427/646/418 快照不等于 clean committed-source 门禁。
- 提交前必须遵循 `AGENTS.md`：运行完整的 `pre-commit run --all-files`，不要只跑窄范围 lint，不要跳过 hooks。

## 3. SSH 与远端节点

### 3.1 节点和凭证

| 角色 | 公网地址 | 私网地址 | 主机名 |
|---|---|---|---|
| rank0 | `39.156.164.107` | `192.168.11.2` | `bms-84092057-001` |
| rank1 | `39.156.164.108` | `192.168.11.3` | `bms-84092057-002` |

- SSH 用户必须使用 `root`；默认 `hii` 用户在 rank1 曾返回 publickey/password permission denied。
- WSL 私钥：`/home/hii/.ssh/tokenspeed_v41_ed25519`。
- 已验证 BatchMode key 登录可用；不要重新生成或覆盖既有密钥。
- `.109/.110` 属于 PD 排查节点，不在当前两节点聚合服务 footprint 内；不要触碰其 `pd-pat-*` 工作负载。

WSL 中连接：

```bash
ssh -i /home/hii/.ssh/tokenspeed_v41_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=10 \
  root@39.156.164.107

ssh -i /home/hii/.ssh/tokenspeed_v41_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=10 \
  root@39.156.164.108
```

Windows PowerShell 中可通过 WSL 执行：

```powershell
wsl -e bash -lc 'ssh -i /home/hii/.ssh/tokenspeed_v41_ed25519 -o BatchMode=yes root@39.156.164.107 "hostname"'
```

Python runner 默认读取 `Path.home()/.ssh/tokenspeed_v41_ed25519`，也可用 `--key-path` 显式指定。`TS_SSH_PW` 是脚本支持的备用认证方式，但不要把密码写入命令、文档或 JSON。

## 4. 远端运行环境

| 项目 | 当前值/说明 |
|---|---|
| 容器工具 | `nerdctl`；`docker` 不存在 |
| 镜像 | `quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc` |
| CANN | 9.2.1 |
| vLLM | `0.32.0+empty` |
| 容器架构/Python | aarch64，Python 3.12 |
| 模型权重 | `/mnt/weights/DeepSeek-V4.1-Flash-w8a8`，只读挂载 |
| served model name | `deepseek-v41` |
| API | rank0 `0.0.0.0:8000` |
| 拓扑 | DP2 / attn TP8 / dense TP8 / MoE TP1 / EP16，world_size=16 |
| 模型层数 | 40 |
| 当前最大长度 | `max_model_len=1024`；2048-token prompt 会被配置阻止 |
| TS 服务 Python | `/tmp/ts-native/bin/python` |
| TS CLI | `python -m tokenspeed.cli`；不存在 `/tmp/ts-native/bin/ts` |

TS 容器内执行 Python 必须显式设置：

```bash
export PYTHONPATH=/ts/python:/ts/tokenspeed-kernel/python:/ts/tokenspeed-kernel-npu/python
export PYTHONDONTWRITEBYTECODE=1
```

否则 `tokenspeed` 可能不可导入。`/ts` 是只读挂载，普通 `py_compile` 可能因无法写 `__pycache__` 报 `Read-only file system`；语法检查可用 AST parse，不要把该报错误判为语法错误。

已观察到的容器包版本：torch 2.10.0+cpu、numpy 1.26.4、pytest 8.4.2。普通 packaged pytest 在 CPU 环境可能因 dsv41 包 `__init__` 导入缺失的 `cann_ops_transformer` 而在 collection 阶段失败。

本地 WSL `.venv-wsl` 可用于 runner：Python 3.14.4、paramiko 5.1.0、pytest 9.2.0；当前缺 numpy/torch，不能据此运行模型或部分 pytest。进入方式：

```bash
cd /mnt/c/Users/Lenovo/codes/tokenspeed
source .venv-wsl/bin/activate
```

## 5. 当前服务、stage 和日志位置

### 5.1 TokenSpeed 418（当前驻留）

| 项目 | 值 |
|---|---|
| run_id | `20261116T201549Z-418-cluster2` |
| 容器名 | `tokenspeed-v41-cluster2-20261116t201549z-418` |
| 远端 stage | `/root/gyh_dev/tokenspeed_cluster/20261116T201549Z-418` |
| 容器内源码 | `/ts`，只读 |
| 容器内日志 | `/logs/serve.log` |
| 主机日志 | `/root/gyh_dev/tokenspeed_cluster/20261116T201549Z-418/logs/serve.log` |
| 本地结果 | `outputs/v41-debug/20261116T201549Z-418-cluster2/` |

### 5.2 vLLM-Ascend 420（已停止，证据保留）

| 项目 | 值 |
|---|---|
| run_id | `20261116T191538Z-420-vllm-cluster2` |
| 容器名 | `vllm-v41-cluster2-20261116t191538z-420` |
| 远端 stage | `/root/gyh_dev/vllm_cluster/20261116T191538Z-420` |
| 容器内 entry | `/probe/vllm_entry.sh` |
| 容器内日志 | `/logs/serve.log` |
| 主机日志 | `/root/gyh_dev/vllm_cluster/20261116T191538Z-420/logs/serve.log` |
| 本地结果 | `outputs/v41-debug/20261116T191538Z-420-vllm-cluster2/` |

通用 stage 规则：

- TS：`/root/gyh_dev/tokenspeed_cluster/<run-id>`，挂载为 `/ts:ro`。
- vLLM：`/root/gyh_dev/vllm_cluster/<run-id>`，挂载为 `/probe:ro`。
- 两边 `logs` 均挂载为 `/logs:rw`。
- 模型权重均只读挂载。

## 6. 只读状态复核命令

以下命令不重启服务、不清理资源。

查看容器和资源：

```bash
hostname
nerdctl ps -a --format '{{.Names}}\t{{.Status}}'
df -h / /dev/shm /mnt
npu-smi info
```

rank0 本机 API 检查：

```bash
curl -sS -o /dev/null -w 'health=%{http_code}\n' http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

TS runner 的 readiness 使用 `/readiness`；vLLM 使用 `/health`。vLLM 的 `/healthz`、`/ready`、`readiness` 曾返回 404，不要混用。

查看日志：

```bash
tail -c 240000 /root/gyh_dev/tokenspeed_cluster/20261116T201549Z-418/logs/serve.log
nerdctl logs --tail 2000 tokenspeed-v41-cluster2-20261116t201549z-418
```

如果需要从本地访问 API，可建立 SSH 本地端口转发；目前明确验证的是 rank0 loopback，未把公网 8000 可达性当作既定事实：

```bash
ssh -i /home/hii/.ssh/tokenspeed_v41_ed25519 \
  -L 18000:127.0.0.1:8000 root@39.156.164.107
# 随后访问 http://127.0.0.1:18000
```

## 7. 启动和切换服务

### 7.1 启动前硬闸门

1. 两节点 NPU 必须空闲，rank0 8000 端口可用，镜像和权重存在。
2. 明确当前哪些自定义容器仍驻留；不要假设 runner 会清理它们。
3. 先检查 root、`/dev/shm`、NFS 可用空间。TS entry 每次启动会创建/复用 `/tmp/ts-native` 并执行 pip 安装，rank0 root 接近满时可能阻塞。
4. 两套 world_size=16 服务不能共驻；切换时先停止当前服务并确认 HBM 释放。
5. 启动和权重加载通常需要约 18–29 分钟；不要因短时间无日志就强删容器。

### 7.2 TokenSpeed 双节点 Graph

从 WSL 主工作区、激活 `.venv-wsl` 后：

```bash
python test/ci/npu/run_v41_cluster.py \
  --execution-mode graph \
  --success-action keep-tokenspeed \
  --bench-protocol stream \
  --bench-repeat 5 \
  --skip-benchmark
```

说明：

- `--skip-benchmark` 用于先拉起服务，再单独执行严格文本采集和固定负载测试。
- 若不跳过，runner 会调用 `bench_v41_steady.py`，但 runner 当前没有显式传 16 请求/128 token 预热；正式验收应使用第 8 节的完整参数。
- `--success-action keep-tokenspeed` 保留服务；`restore-vllm` 涉及清理和恢复原始 `deepseek-v41-dp`，使用前先确认状态。
- TS runner 只会自动停止名为 `deepseek-v41-dp` 的原始 vLLM，不会自动停止自定义 420/431/646/418 容器。

### 7.3 vLLM-Ascend 双节点 Graph

释放 TS 后执行：

```bash
python test/ci/npu/run_v41_vllm_cluster.py \
  --execution-mode graph \
  --success-action keep-vllm \
  --bench-protocol stream \
  --bench-repeat 5 \
  --skip-benchmark
```

Graph 由 `--no-enforce-eager` 实现；eager 对照才使用 `--enforce-eager`。不得把 TS Graph 与 vLLM eager 的结果作为竞争力证据。

### 7.4 手动停止原则

- 使用 `nerdctl stop -t 60 <container>`，先保留日志和 launch/summary，再决定是否 `nerdctl rm`。
- 权重加载中禁止强删容器。
- 禁止 `fuser` 和 `npu-smi reset`。
- 启动超时 runner 可能设置 `deferred_while_model_may_be_loading` 并保留容器；接手时先检查真实状态，不要重复启动同名或第二套服务。

## 8. 固定负载性能测试契约

正式对比必须保持：

- concurrency：8
- requests：16
- prompt：约 48–52 tokens
- max_tokens：128
- protocol：stream
- warmup：16 requests × 128 tokens
- measured repeats：5，五轮全部保留，不能挑最好轮次
- `ignore_eos=true`
- Graph/eager 对等、权重/量化/拓扑一致、prefix-cache 状态可解释
- prompt tag 唯一，避免跨服务 prefix cache 污染
- direct 与 SMG/router 分开记录

标准脚本：`test/ci/npu/bench_v41_steady.py`，只依赖 Python stdlib。可在 rank0 执行：

```bash
python3 - --api http://127.0.0.1:8000 \
  --model deepseek-v41 \
  --engine tokenspeed \
  --protocol stream \
  --topology dp2-tp8-ep16 \
  --prompt-tag <unique-run-tag> \
  --concurrency 8 --requests 16 \
  --max-tokens 128 --repeat 5 \
  --warmup-requests 16 --warmup-max-tokens 128 \
  --timeout 1200 < test/ci/npu/bench_v41_steady.py
```

vLLM 对照仅把 `--engine` 改为 `vllm-ascend` 并使用独立 prompt tag。

注意：该脚本的 Paris 检查只是 keyword smoke。`outputs/v41-debug/bench_v41_steady_nogate.py` 是临时 performance-only 副本，会在 smoke 失败时保留测量轮次；它产生的 `RC=0` 或完整 token 数**不代表正确性通过**。正式报告必须同时引用独立严格文本门禁。

Profiler wall time、kernel sum、单步计时只用于归因，不能替代上述端到端五轮 benchmark。

## 9. 文本正确性门禁

固定采集器：`outputs/v41-debug/check_v41_generation.py`。三个 greedy prompt：

1. `The capital of France is`
2. `Count from 1 to 20, separated by spaces.`
3. `请用两句话说明为什么公平性能测试需要充分预热。`

请求参数：`max_tokens=64`、`temperature=0.0`、`ignore_eos=true`，API 为 rank0 `/v1/completions`。

在 WSL 主工作区执行并保存：

```bash
ssh -i /home/hii/.ssh/tokenspeed_v41_ed25519 root@39.156.164.107 \
  'python3 -' < outputs/v41-debug/check_v41_generation.py \
  > outputs/v41-debug/<run-id>-generation.json
```

验收不能只看关键词：

- Paris 要是连贯的首都回答，不能混入乱码、重复 JSON/code 或 off-topic 内容。
- count 要形成正确的 1..20 序列，且尾部不能继续退化。
- Chinese 要是连贯的两句中文，不能出现 mojibake、英文/code 碎片或跑题。

当前 418、427、480、481 和 vLLM 420/431 均有失败记录。九级张量定位方案在 `outputs/v41-debug/v41-correctness-ladder-20261117.json`，但它只是设计，尚无 live NPU hidden/router/logits/token-ID 对比证据。

## 10. NPU profile 与分析工具

### 10.1 采集脚本

| 用途 | 脚本 |
|---|---|
| 纯 decode 驱动 | `outputs/v41-debug/profile_v41_pure_decode.py` |
| 受控纯 prefill 驱动 | `outputs/v41-debug/profile_v41_pure_prefill_long.py` |
| rank0 CSV 流式提取 | `outputs/v41-debug/extract_v41_live_profile_rank0.py` |
| CSV 聚合 | `outputs/v41-debug/summarize_v41_npu_profile.py` |
| 索引复用前后 profile 对比 | `outputs/v41-debug/compare_v41_index_reuse_profile.py` |
| 历史 step 分析 | `outputs/v41-debug/analyze_v41_direct_steps.py` |

采集前必须：

- 确认服务已充分预热，Graph capture/首次编译不进入正式 profile。
- 确认 `PROFILE_OUTPUT_DIR` 是服务容器可写且主机可取回的路径。
- 先检查 `/dev/shm` 和 root 空间；空间不足时不要启动新 profile。
- NPU profiler 要求 positive `num_steps`，activities 使用 `CPU`、`GPU`（映射到 NPU activity）。
- 采集结束调用 `/stop_profile`；失败时也要收尾，不能遗留 profiler 状态。
- vLLM 当前未传 `--profiler-config` 的内置 profiler 端点不可直接 attach；外部 `msprof` attach 尚未验证能产出有效 NPU kernel CSV。

CSV 聚合：

```bash
python outputs/v41-debug/summarize_v41_npu_profile.py \
  --input <ASCEND_PROFILER_OUTPUT-or-parent> \
  --output <summary.json> --top 30
```

有效报告至少应有 `kernel_details.csv`、`op_statistic.csv`、`api_statistic.csv`；缺列或空 CSV 不能作为有效 profile。按 `Step Id` 拆分 prefill/decode，长 prefill 报告的 decode-like step 必须排除。kernel sum 和 stream 时间不能简单当作 wall time 相加。

## 11. 当前有效基线

固定短 mixed/decode-dominated stream 负载、交替驻留：

| 服务 | 吞吐五轮均值 | TPOT mean | TTFT p50 | 严格文本门禁 |
|---|---:|---:|---:|---|
| TS 418 | 27.6516 tok/s | 0.267813 s | 2.625798 s | fail：paris/count/chinese |
| vLLM 420 | 209.5961 tok/s | 0.031887 s | 0.812222 s | fail：paris/count/chinese |

TS/vLLM：吞吐 0.13193x，TPOT 8.39872x，TTFT p50 3.23286x。请求级粗略分解约 94.29% 超额位于 decode/post-first-token，但 TTFT 含排队/prefix-cache/prefill，TPOT 含调度/streaming/host；这不是 paired kernel 归因。

已有有效 TS 优化证据：index-key reuse 在同服务五轮 A/B 中吞吐提升 1.14203x、TPOT 降低 14.80%，但不能据此声称已经赶超 vLLM。

主要基线文件：

- `outputs/v41-debug/20261116T201549Z-418-cluster2/418-benchmark-nogate.json`
- `outputs/v41-debug/20261116T201549Z-418-cluster2/418-generation.json`
- `outputs/v41-debug/20261116T201549Z-418-cluster2/418-gate.json`
- `outputs/v41-debug/20261116T191538Z-420-vllm-cluster2/420-benchmark-nogate.json`
- `outputs/v41-debug/20261116T191538Z-420-vllm-cluster2/420-generation.json`
- `outputs/v41-debug/20261116T191538Z-420-vllm-cluster2/420-gate.json`
- `outputs/v41-debug/20261116T201549Z-418-vs-420-comparison.json`
- `outputs/v41-debug/prefill-decode-bottleneck-analysis-20261116.json`
- `outputs/v41-debug/pure-prefill-ledger-481-20261116.json`
- `outputs/v41-debug/phase0-gate-20261117T0753Z.json`

`outputs/` 被 gitignore 排除；不要期待这些 JSON 出现在提交中。

## 12. 已知故障与未闭合问题

1. **文本退化未定位**：mirror、Graph replay、reload-forward、MoE ctx/router、hidden-state、logits/parser 均只有嫌疑，缺少分层张量证据。
2. **646 启动失败未闭合**：rank1 权重加载出现 `aclrtMemcpy 107017`，dmesg 有 devmm pin-user-pages 失败；418 重试成功不证明根因消失。
3. **配对账本缺失**：纯 decode 和部分纯 prefill 主要是 TS-only rank0 证据，缺同 batch/同长度 vLLM 对照及其他 DP/EP/rank 覆盖。
4. **共驻不可行**：单套服务每卡约 56–60GB，vLLM 420 rank0 每卡约 60.4/65.5GB；第二套 world_size=16 无空间。
5. **空间风险**：最近复核 rank0 root 5.4G 可用、`/dev/shm` 3.9G 可用，NFS 0 可用；rank1 `/dev/shm` 26G 可用。
6. **646 相关日志复核**：

   ```bash
   dmesg -T | grep -iE 'devmm|pin_user|Get user pages|Make host|Fill source|ts-host'
   ```

7. **SMG 当前未运行**：历史 direct vs SMG 约 0.63% 吞吐差异只是历史 artifact，不能写成当前 router 对照已闭合。
8. **PD 暂停**：12 次真实运行全部 `performance_benefit=false`，聚合路径未具备竞争力前不重启 PD 优化。

## 13. 接手后的推荐第一步

先只做只读复核，避免重复历史工作：

1. 阅读本 runbook、`docs/ascend-v41-w8a8-serving-handoff.md` 末尾最新状态和 `phase0-gate-20261117T0753Z.json`。
2. SSH 检查 418/420 容器、rank0 `/health` 与 `/v1/models`、两节点磁盘/shm/NPU。
3. 不重复跑已有 418/420 五轮基线；先解决文本正确性采集链路和远端空间闸门。
4. 优先实现默认关闭、不改变生产 numerics 的分层张量采集，再建立配对纯 decode 账本，之后补配对纯 prefill 账本。
5. 后端优化必须按“算子/子链路回归 → Graph replay 正确性 → 子链路基准 → fresh server 严格文本门禁 → 五轮同口径 A/B”验收。
6. 聚合路径相对 vLLM 取得可验证优势且文本门禁关闭后，再恢复 PD 工作。
