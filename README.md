# trn2-bench — AWS Trainium2 (trn2.48xlarge) 评测套件与报告

对 AWS Trainium2 做的一次完整单机评测（2026-11-07/08，us-east-2b 29h Capacity Block），
覆盖 GEMM 算力、HBM 带宽、STREAM、H2D/D2H、集合通信、以及 vLLM 模型推理
（Qwen3-32B、DeepSeek-R1-Distill-Llama-70B），并与 AWS 官方规格逐项核对。

**报告**：[`report.md`](report.md) ｜ PDF：`Trainium2_trn2.48xlarge_评测报告.pdf`

## 结果速览

| 项目 | 实测 | 官方标称 | 达成率 |
|---|---|---|---|
| GEMM BF16（整机 16 芯片） | 7.59 PFLOPs | 10.67 PF | 71% |
| GEMM BF16（单芯片） | 594 TF | 667 TF | 89% |
| HBM 带宽（整机，DMA） | 41.2 TB/s | 46 TB/s | 90% |
| NeuronLink（all-reduce busbw/芯片） | ≈1.30 TB/s | 1.28 TB/s | ≈100% |
| Qwen3-32B TP32 decode 吞吐 | 1099 tok/s（64 并发） | — | — |
| DS-R1-Distill-70B TP64 decode 吞吐 | 774 tok/s（64 并发） | — | — |
| 单流延迟（两模型） | 13.0 / 13.4 ms/tok | — | — |

环境：Neuron SDK 2.33 · torch-neuronx 2.14（原生 `neuron` device）· vLLM-Neuron 0.29 · LNC=2（64 逻辑核 × 24 GB）

## 目录结构

```
report.md                  # 完整评测报告（含官方规格核对、踩坑记录）
*.pdf                      # 报告 PDF 版
scripts/
  00_setup.sh              # NVMe RAID0 挂载 + 系统信息采集
  common.py                # 多进程每 rank 绑一个逻辑核（NEURON_RT_VISIBLE_CORES）
  01_gemm.py               # GEMM TFLOPs（BF16/FP32，链式方阵乘）
  02_stream.py             # STREAM copy/scale/add/triad（框架 elementwise 路径）
  03_h2d.py                # Host<->Device 带宽
  04_collectives.sh        # nccom-test：all-reduce/all-gather/reduce-scatter/all-to-all
  05_hbm_copy.py           # 设备内 DMA copy_（真实 HBM 带宽）
  06_model_bench.sh        # vLLM throughput + latency
  07_decode_heavy.sh       # decode 为主的吞吐复测（512 输出）
  run_micro.sh / finish_suite.sh   # 微基准编排
  aggregate.py             # 汇总 results/ 生成 markdown 表格
results/                   # 原始数据：jsonl / nccom 输出 / vLLM json（大日志已截断）
  summary_tables.md        # aggregate.py 生成的全量数据表
```

## 复现

```bash
# trn2.48xlarge + Neuron DLAMI (Ubuntu 24.04, SDK 2.33)
bash scripts/00_setup.sh
bash scripts/run_micro.sh && bash scripts/finish_suite.sh     # 微基准
bash scripts/06_model_bench.sh Qwen/Qwen3-32B qwen3-32b-tp32 32 4096
bash scripts/06_model_bench.sh deepseek-ai/DeepSeek-R1-Distill-Llama-70B ds-r1-70b-tp64 64 4096
bash scripts/07_decode_heavy.sh
python3 scripts/aggregate.py results > results/summary_tables.md
```

关键环境变量（详见报告 §7 踩坑记录）：多进程分核 `NEURON_RT_VISIBLE_CORES`、
无 EFA 实例跑 vLLM 需 `NEURON_SKIP_EFA_AFFINITY=1`。
