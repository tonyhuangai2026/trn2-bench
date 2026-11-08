#!/bin/bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
cd ~/trn2-bench/scripts; R=~/results
run() { local name=$1 np=$2; shift 2
  echo "=== $(date -u +%T) $name (np=$np)"; rm -f $R/$name.jsonl
  OUT=$R/$name.jsonl timeout 5400 torchrun --nproc_per_node=$np "$@" > $R/$name.log 2>&1
  grep -hE '^\{' $R/$name.log | tail -6
}
run gemm_bf16_np64 64 01_gemm.py --dtype bf16 --sizes 1024,2048,4096,8192,16384
run h2d_np16 16 03_h2d.py --sizes_mb 16,64,256,1024
run hbm_copy_np1 1 05_hbm_copy.py --mb 4096
run hbm_copy_np64 64 05_hbm_copy.py --mb 1024
echo "=== micro-multi done"
bash ~/trn2-bench/scripts/04_collectives.sh
echo "=== ALL MICRO DONE"
