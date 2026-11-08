#!/bin/bash
# decode-heavy throughput rerun: 1024 in / 512 out (explicit random-output-len)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export HF_HOME=/mnt/nvme/hf NEURON_COMPILE_CACHE_URL=/mnt/nvme/neuron-cache NEURON_SKIP_EFA_AFFINITY=1
R=~/results/models
bench() { # model tag tp
  echo "=== $(date -u +%T) decode-heavy $2 tp=$3"
  timeout 10800 vllm bench throughput --model "$1" --tensor-parallel-size $3 \
    --max-model-len 4096 --no-enable-prefix-caching --max-num-batched-tokens 4096 \
    --dataset-name random --random-input-len 1024 --random-output-len 512 --num-prompts 64 \
    --output-json $R/$2_tp512.json > $R/$2_tp512.log 2>&1
  tail -4 $R/$2_tp512.log
}
bench Qwen/Qwen3-32B qwen3-32b-tp32 32
bench deepseek-ai/DeepSeek-R1-Distill-Llama-70B ds-r1-70b-tp64 64
echo DECODE_HEAVY_DONE
