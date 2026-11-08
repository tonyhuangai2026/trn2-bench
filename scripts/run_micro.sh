#!/bin/bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
cd ~/trn2-bench/scripts; R=~/results; mkdir -p $R
run() { # name nproc script args...
  local name=$1 np=$2; shift 2
  echo "=== $(date -u +%T) $name (np=$np)"
  rm -f $R/$name.jsonl
  OUT=$R/$name.jsonl timeout 3600 torchrun --nproc_per_node=$np "$@" 2>&1 | tee $R/$name.log | grep -E '^\{' | tail -30
}
for np in 1 4 64; do
  run gemm_bf16_np$np $np 01_gemm.py --dtype bf16 --sizes 1024,2048,4096,8192,16384
done
run gemm_fp32_np1 1 01_gemm.py --dtype fp32 --sizes 2048,4096,8192
for np in 1 4 64; do
  run stream_np$np $np 02_stream.py --mb 1024
done
run h2d_np1 1 03_h2d.py
run h2d_np16 16 03_h2d.py   # 16 并发传输看聚合带宽
echo "=== $(date -u +%T) micro done"
