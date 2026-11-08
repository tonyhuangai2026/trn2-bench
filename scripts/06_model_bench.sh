#!/bin/bash
# vLLM-Neuron model benchmarks: throughput + latency
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export HF_HOME=/mnt/nvme/hf
export NEURON_COMPILE_CACHE_URL=/mnt/nvme/neuron-cache
export NEURON_SKIP_EFA_AFFINITY=1
R=~/results/models; mkdir -p $R
MODEL=$1; TAG=$2; TP=${3:-32}; LEN=${4:-4096}
echo "=== $(date -u +%T) throughput $TAG tp=$TP"
timeout 14400 vllm bench throughput --model "$MODEL" --tensor-parallel-size $TP \
  --max-model-len $LEN --no-enable-prefix-caching --max-num-batched-tokens 4096 --input-len 1024 --output-len 512 --num-prompts 64 \
  --output-json $R/${TAG}_throughput.json > $R/${TAG}_throughput.log 2>&1
tail -5 $R/${TAG}_throughput.log
echo "=== $(date -u +%T) latency $TAG"
timeout 7200 vllm bench latency --model "$MODEL" --tensor-parallel-size $TP \
  --max-model-len $LEN --no-enable-prefix-caching --max-num-batched-tokens 4096 --input-len 128 --output-len 128 --batch-size 1 --num-iters 5 --num-iters-warmup 2 \
  --output-json $R/${TAG}_latency.json > $R/${TAG}_latency.log 2>&1
tail -5 $R/${TAG}_latency.log
