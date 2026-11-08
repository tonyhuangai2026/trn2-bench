#!/bin/bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
cd ~/trn2-bench/scripts; R=~/results; mkdir -p $R
run() { local name=$1 np=$2; shift 2
  echo "=== $(date -u +%T) $name (np=$np)"; rm -f $R/$name.jsonl
  OUT=$R/$name.jsonl timeout 5400 torchrun --nproc_per_node=$np "$@" > $R/$name.log 2>&1
  grep -hE '^\{' $R/$name.log | tail -8
}
run gemm_bf16_np4  4  01_gemm.py --dtype bf16 --sizes 1024,2048,4096,8192,16384
run gemm_bf16_np64 64 01_gemm.py --dtype bf16 --sizes 1024,2048,4096,8192,16384
run stream_np4  4  02_stream.py --mb 1024
run stream_np64 64 02_stream.py --mb 1024
run h2d_np16 16 03_h2d.py --sizes_mb 16,64,256,1024
echo "=== micro-multi done"
bash ~/trn2-bench/scripts/04_collectives.sh
