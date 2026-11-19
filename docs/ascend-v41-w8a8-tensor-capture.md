# DeepSeek-V4.1 eager tensor capture

本文记录用于 TokenSpeed / vLLM-Ascend 对齐排查的诊断探针。它不是性能优化路径，也不改变默认服务行为；未显式设置环境变量时不创建 sink，不复制 tensor。

## 边界

- 仅用于单请求、eager forward 的数值诊断。Graph capture/replay 不写 tensor；误在 Graph capture 中启用 writer 会失败。
- 默认只采样 rank 0，最多捕获显式配置的 step 数。每个 tensor 同步复制到 CPU 后用 `torch.save` 写入独立文件，因此不能将捕获实例的时延或吞吐用于性能结论。
- arm 文件不存在时，启动 dummy forward 和普通请求均不捕获。arm 文件内容 `0` 捕获下一次 forward；内容 `1` 跳过下一次 forward（通常为 prefill），捕获再下一次 decode forward。
- 捕获不证明 NPU Graph replay、prefix restore 或端到端文本正确性。严格正确性仍需独立 gate。

## TokenSpeed 环境变量

| 变量 | 含义 |
| --- | --- |
| `TOKENSPEED_V41_DEBUG_CAPTURE_DIR` | 非空输出目录；启用探针的必要条件 |
| `TOKENSPEED_V41_DEBUG_CAPTURE_RANKS` | 逗号分隔的 rank，诊断默认 `0` |
| `TOKENSPEED_V41_DEBUG_CAPTURE_MAX_STEPS` | 实际 capture 会话数上限；未 arm、非目标 rank 或被跳过的 forward 不消耗配额 |
| `TOKENSPEED_V41_DEBUG_CAPTURE_ARM_FILE` | rank 0 entry 上的 arm 文件路径 |

`manifest.json` 以 `layer_id/name/occurrence` 形成稳定 `capture_key`，记录 shape、dtype、文件、forward 顺序和 prefill/decode 模式。公共比较 key 不包括 `input_ids`、`positions`、`final_hidden` 和 `logits` 的 layer 编号；这些点使用 `layer_none`。

## 启动和请求驱动

以下命令均在仓库根目录的 WSL 中执行。启动会创建新 stage 和容器；`--skip-benchmark` 是捕获实例的必需约束。

```bash
.venv-wsl/bin/python test/ci/npu/run_v41_cluster.py \
  --execution-mode eager \
  --success-action keep-tokenspeed \
  --skip-benchmark \
  --debug-capture \
  --key-path /home/hii/.ssh/tokenspeed_v41_ed25519
```

vLLM 使用同一镜像和拓扑交替驻留，不能与 TokenSpeed world-size-16 服务同时运行：

```bash
.venv-wsl/bin/python test/ci/npu/run_v41_vllm_cluster.py \
  --execution-mode eager \
  --success-action keep-tokenspeed \
  --skip-benchmark \
  --debug-capture \
  --key-path /home/hii/.ssh/tokenspeed_v41_ed25519
```

vLLM adapter 通过 stage 的 `logs/vllm-debug` 和 `PYTHONPATH` 加载，不改镜像内安装源码。`sitecustomize.py` 扩展 V4.1 package path 并注册本地 `DeepseekV41ForCausalLM` adapter。

两个服务 readiness 后，分别运行请求驱动。它依次执行固定 greedy prefill（arm=`0`、max tokens=1）和 decode（arm=`1`、max tokens=2），在 finally 中删除 arm 文件，再通过 SFTP 下载 manifest 和 tensor：

```bash
.venv-wsl/bin/python outputs/v41-debug/drive_v41_capture.py tokenspeed \
  --stage /root/gyh_dev/tokenspeed_cluster/<RUN_ID> \
  --output outputs/v41-debug/causal-pair/<RUN_ID>-ts

.venv-wsl/bin/python outputs/v41-debug/drive_v41_capture.py vllm \
  --stage /root/gyh_dev/vllm_cluster/<RUN_ID> \
  --output outputs/v41-debug/causal-pair/<RUN_ID>-vllm
```

## 比较和负控

```bash
python test/ci/npu/compare_v41_captures.py \
  outputs/v41-debug/causal-pair/<TS_STEP> \
  outputs/v41-debug/causal-pair/<VLLM_STEP> \
  --atol 0.02 --rtol 0.02 \
  --report outputs/v41-debug/causal-pair/<RUN_ID>-report.json
```

比较器只比较双方共同 key；共同 key 顺序不同、没有共同 key 或任一 tensor 文件缺失都会失败。额外捕获点只计入 coverage。整数索引始终要求精确相等，不受浮点容差影响；浮点报告最大绝对误差、首个扁平分歧索引、NaN/Inf 和 logits top1。`selection_index_weights_bf16` 与 `selection_index_weights_fp32` 是不同精度合同，不伪装成同一 tensor key。

本地负控位于 `test/ci/npu/test_compare_v41_captures.py`，覆盖 BF16 单元素差异、整数索引在宽松浮点容差下仍失败、logits top1 改变和 missing-key coverage。真实远端负控需要复制同一 capture 后只改一个元素，再确认比较器报告该 key；不能用单元测试代替这一步。下载真实 capture 后执行：

```bash
/mnt/c/Users/Lenovo/AppData/Local/Python/bin/python.exe \
  "C:\Users\Lenovo\codes\tokenspeed\outputs\v41-debug\negative_control_v41_capture.py" \
  "C:\Users\Lenovo\codes\tokenspeed\outputs\v41-debug\causal-pair\<CAPTURE_STEP>" \
  "C:\Users\Lenovo\codes\tokenspeed\outputs\v41-debug\causal-pair\<CAPTURE_STEP>-negative"
```

脚本先做原目录自比较，再复制整个 capture、仅修改一个有限浮点元素，并要求零容差比较在同一 key 报告恰好一个 mismatch；结果写入复制目录的 `negative-control.json`。
