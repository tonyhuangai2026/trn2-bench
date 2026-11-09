# AWS Trainium2 (trn2.48xlarge) Benchmark Report

**Date**: 2026-11-07 · **Region**: us-east-2b (Capacity Block, 29h, $1037.06)
**Instance**: i-0xxxxxxxxxxxxxxxx · **AMI**: Neuron DLAMI Ubuntu 24.04, SDK 2.33.0
**Software stack**: torch-neuronx 2.14 (native `neuron` device), neuronx-cc 2.28, NRT 2.35, vLLM-Neuron 0.29, nccom-test

## 0. Hardware Topology

- 16× Trainium2 chips, LNC=2 logical-core config → **64 logical NeuronCores**, 24 GB HBM each (1.5 TB HBM per node)
- Each chip connects to 4 neighbors via NeuronLink (2D torus), spanning 2 NUMA domains
- 192 vCPU, 2 TB RAM, 4× 1.9 TB NVMe (assembled as RAID0 = 7 TB for this test)
- No EFA attached on this instance (single-node benchmark; vLLM requires `NEURON_SKIP_EFA_AFFINITY=1`)

> Throughout this report, "per-core" means one LNC=2 logical core = 2 physical cores = 1/4 chip;
> 1 chip = 4 logical cores. Conversion: aggregate(chip) = 4 × per-core.

## 1. GEMM / PFLOPs

Method: chained square matmuls `y = y @ W` (16 matmuls fused into one graph), BF16/FP32,
executed independently on each logical core (data-parallel micro-benchmark, no communication).

| dtype | ranks | N | per-core TFLOPs | aggregate TFLOPs |
|---|---|---|---|---|
| bf16 | 1 | 4096 | 140.4 | 140 |
| bf16 | 1 | 8192 | **148.6** | 149 |
| bf16 | 1 | 16384 | 145.5 | 145 |
| bf16 | 4 (1 chip) | 4096 | 118.7 | **475** |
| bf16 | 4 (1 chip) | 8192 | 106.5 | 426 |
| bf16 | 64 (full node) | 4096 | 118.6 | **7591** |
| bf16 | 64 (full node) | 8192 | 105.2 | 6730 |
| fp32 | 1 | 4096 | 36.3 | 36 |

**Findings**
- Single logical core peaks at **148.6 TF BF16** ≈ 594 TF per chip, i.e. **89%** of the rated dense BF16 (~667 TF/chip).
- Full-node aggregate: **7.59 PFLOPs BF16** (@N=4096), **71%** of the rated ~10.7 PF. With all cores busy,
  per-core drops from 148 → 118 TF, mainly because HBM bandwidth/power is shared among the 4 cores within a chip.
- FP32 ≈ 1/4 of BF16 (36 TF/core), matching the architecture ratio. N=1024 is too small to saturate
  the systolic array (43 TF).

## 2. HBM Bandwidth

Two paths were measured:

**a) DMA `copy_` (true HBM bandwidth, read+write)**

| ranks | array | per-core GB/s | node aggregate |
|---|---|---|---|
| 1 | 1 GB | 640 | — |
| 64 | 1 GB | 643 | **41.2 TB/s** |

Per-core bandwidth is identical with 1 vs 64 concurrent cores (640 ≈ 643 GB/s) → each logical core
has an independent DMA path and can saturate its own HBM slice; no cross-core contention.

Node-level 41.2 TB/s ≈ **89%** of the rated 46.4 TB/s (2.9 TB/s × 16) — measured HBM is very close to spec.

**b) STREAM copy/scale/add/triad (framework elementwise path, fp32, 1 GB arrays)**

| kernel | per-core GB/s | node aggregate GB/s |
|---|---|---|
| copy | 80 | 5,103 |
| scale | 160 | 10,216 |
| add | 147 | 9,437 |
| triad | 91 | 5,847 |

The framework elementwise path reaches only ~25% of the DMA path: the bottleneck is the compute
engine's elementwise throughput, not HBM (evidence: perfectly linear 1→4→64 core scaling, i.e.
no bandwidth contention). On Trainium, bandwidth-bound ops should rely on compiler fusion / DMA;
hand-written elementwise chains are not worthwhile.

## 3. H2D / D2H (PCIe)

| ranks | block size | H2D (per-stream / aggregate) | D2H (per-stream / aggregate) |
|---|---|---|---|
| 1 | 4 GB | 12.9 GB/s | 3.4 GB/s |
| 16 | 1 GB | 2.0 / **32 GB/s** | 1.0 / 15 GB/s |

- Single-stream H2D 12.9 GB/s (~half of PCIe Gen5 x8, single-queue limit); 16 concurrent streams aggregate to 32 GB/s.
- **D2H is markedly asymmetric** (3.4 GB/s single-stream); use multiple concurrent streams when
  pulling large results back (aggregate reaches 15–30 GB/s).
- Small transfers (1 MB) drop to ~3 GB/s effective bandwidth; launch overhead is on the order of ~0.3 ms.

## 4. Collective Communication (nccom-test, bf16, single-node NeuronLink)

For each entry: 32 B small-message latency / peak bus bandwidth:

| op | 4 ranks (1 chip) | 16 ranks (4 chips) | 64 ranks (full node) |
|---|---|---|---|
| all-reduce | 7.4 µs / 195 GB/s | 8.2 µs / 199 GB/s | 9.9 µs / **325 GB/s** |
| all-gather | 5.7 µs / 212 GB/s | 11.5 µs / 177 GB/s | 22.0 µs / **379 GB/s** |
| reduce-scatter | 8.2 µs / 145 GB/s | 11.4 µs / 190 GB/s | 14.4 µs / **275 GB/s** |
| all-to-all | 5.7 µs / 91 GB/s | 13.3 µs / 26 GB/s | 33.8 µs / **22 GB/s** |

**Findings**
- Excellent small-message latency: intra-chip all-reduce at 7.4 µs, and only ~10 µs across all
  64 cores — well suited to the frequent small synchronizations of tensor-parallel inference.
- Large-message all-reduce/all-gather reach 300+ GB/s busbw (saturation requires ≥512 MB; the ramp
  is slow — at 128 MB only ~55% of peak).
- **All-to-all is the clear weak spot**: only ~22 GB/s busbw at 64 ranks, plateauing as early as 8 MiB.
  Expert parallelism (MoE dispatch/combine) on a single trn2 node will be limited by this;
  prefer TP/DP sharding over EP for large MoE models.

## 5. Model Benchmarks (vLLM-Neuron 0.29)

Config: `NEURON_SKIP_EFA_AFFINITY=1`, `--no-enable-prefix-caching --max-num-batched-tokens 4096`,
1024 input / 512 output tokens × 64 prompts (throughput); 128 in / 128 out, batch=1 (latency).

**a) Throughput (64 concurrent requests)**

| model | TP | scenario | req/s | total tok/s | output tok/s |
|---|---|---|---|---|---|
| Qwen3-32B | 32 | 1024 in / 128 out (prefill-heavy) | 5.97 | 6875 | 764 |
| Qwen3-32B | 32 | 1024 in / 512 out (decode-heavy) | 2.15 | 3297 | **1099** |
| DS-R1-Distill-70B | 64 | 1024 in / 128 out | 4.29 | 4938 | 549 |
| DS-R1-Distill-70B | 64 | 1024 in / 512 out | 1.51 | 2322 | **774** |

**b) Single-stream latency (batch=1, 128 in / 128 out)**

| model | TP | avg latency | per token |
|---|---|---|---|
| Qwen3-32B | 32 | 1.66 s | **13.0 ms/tok** (77 tok/s) |
| DS-R1-Distill-70B | 64 | 1.72 s | **13.4 ms/tok** (74 tok/s) |

**Findings**
- The 70B model on the full node (TP=64) matches the 32B model on half the node (TP=32) in
  single-stream latency (13.0 vs 13.4 ms/tok): 2.2× the parameters is almost exactly offset by
  2× the TP cores — confirming NeuronLink's low small-message all-reduce latency and high TP
  scaling efficiency.
- Decode-phase output throughput: ~1.1k tok/s for Qwen3-32B@TP32, ~0.77k tok/s for 70B@TP64
  (64 concurrent, bf16, untuned: chunked prefill limited to batch=1, no continuous-batching
  tuning, no quantization). These are out-of-the-box vLLM-Neuron numbers, not upper bounds.
- Compilation cost is significant: first trace+compile takes 20–40 min per model per config
  (the neuron-cache on NVMe is reusable).

### 5.1 DeepSeek MoE (Original Models) Compatibility Study (2026-11-08)

Conclusion: **DeepSeek V4 / V4.1-Flash currently have no usable inference path on Trainium;
the original V3/R1 can run via NxD Inference but requires a separate engagement.**

| path | version | DeepSeek support | evidence |
|---|---|---|---|
| vLLM-Neuron | 0.29.0.1.2.0 (latest on PyPI) | ❌ architecture allowlist only contains `LlamaForCausalLM` / `Qwen3ForCausalLM` / `GptOssForCausalLM` | Verified: loading V4-Flash-0731 is rejected with `deepseek_v4_fp8 quantization is currently not supported in neuron` |
| NxD Inference | 0.10.18399 (latest) | ⚠️ DeepSeek **V3 family only** (`DeepseekV3Attention` with MLA); can run V3/R1 671B | package ships `models/deepseek/modeling_deepseek.py`; no V4 code anywhere |
| Other (community) | — | ❌ no third-party DeepSeek V4 implementation for Trainium found | — |

- This report uses **R1-Distill-Llama-70B** to represent DeepSeek — the only official DeepSeek
  model on a supported architecture (the R1-Distill-Qwen series is Qwen2-architecture, also not
  on the allowlist).
- Prerequisites for running the original V3/R1 671B: ~700 GB of FP8 weights must be downloaded
  in advance (did not fit in this block's remaining time), and the NxDI path has ~1 h-scale
  compile time. Recommend a ≥12 h capacity block and consider trn2u/UltraServer.

## 6. Verification Against AWS Official Specifications

Official sources: AWS Neuron docs, Trainium2 architecture page; aws.amazon.com/ec2/instance-types/trn2/
(checked 2026-11-08).

| metric | measured | AWS official | ratio |
|---|---|---|---|
| BF16 dense compute / chip | 594 TF (148.6×4) | 667 TFLOPS | **89%** |
| BF16 dense compute / node | 7.59 PF | 10.67 PF (667×16) | **71%** |
| FP32 compute / chip | 145 TF (36.3×4) | 181 TFLOPS | **80%** |
| HBM capacity / node | 1.5 TB (neuron-ls, 16×96 GB) | 1.5 TB HBM3 | ✓ |
| HBM bandwidth / chip | 2.57 TB/s (643×4, DMA r+w) | 2.9 TB/s | **89%** |
| HBM bandwidth / node | 41.2 TB/s | 46 TB/s | **90%** |
| NeuronLink / chip | all-reduce busbw ≈1.30 TB/s (325×4) | 1.28 TB/s (v3) | **≈100%** |
| FP8 compute | not measured (framework bf16 path) | 1299 TFLOPS/chip | — |
| EFA networking | not attached on this instance | 3.2 Tbps EFAv3 | — |

**Verification summary**
- **HBM bandwidth (89–90%) and NeuronLink (≈100%) closely match official specs** — the two
  hardest validation points. Per-chip bus bandwidth of the 64-rank all-reduce lands exactly on
  the NeuronLink-v3 rated value, i.e. the collectives implementation already saturates the link.
- 89% GEMM efficiency per chip is normal compiler/micro-benchmark territory (the official peak is
  the systolic-array theoretical value); the 71% full-node figure is explained by intra-chip
  HBM/power sharing across 4 logical cores (see §1) and does not contradict official data.
- The low single-core STREAM/elementwise numbers (§2b) are a compute-engine path characteristic,
  unrelated to HBM spec; the DMA path verified 89%.
- Officially listed per-chip DMA 3.5 TB/s, SBUF 224 MiB, 8× NeuronCore-v3 + 16 CC-Cores are all
  consistent with neuron-ls output and observed behavior.

## 7. Pitfalls (Reproducibility Notes)

1. **Multi-process runs must pin one core per rank via `NEURON_RT_VISIBLE_CORES`** (set before
   importing torch_neuronx), otherwise rank 0 grabs all 64 logical cores and the remaining ranks
   fail with `cores busy, ret=-16`.
2. On trn2 without EFA, vLLM-Neuron needs `NEURON_SKIP_EFA_AFFINITY=1`, otherwise EngineCore
   fails to start.
3. vLLM-Neuron enables APC by default but requires segmented prefill: pass
   `--no-enable-prefix-caching` or set `--max-num-batched-tokens ∈ {512..8192}`.
4. A very large single-core DMA copy (4 GB × 2) hits the NRT 30 s execution timeout
   (FATAL-RT-UNDEFINED-STATE); chunk the copy instead.
5. nccom-test op names are `allr/allg/redsct/alltoall`; there is no `--check none` option.

## Appendix

- Raw data: `results/*.jsonl`, `results/collectives/*.txt`, `results/summary_tables.md`
- Reproduce: `scripts/00_setup.sh` → `run_micro.sh` / `finish_suite.sh` → `06_model_bench.sh`
