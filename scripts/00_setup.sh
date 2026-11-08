#!/bin/bash
# Mount 4x NVMe as RAID0 at /mnt/nvme, collect system info
set -x
if ! mountpoint -q /mnt/nvme; then
  DEVS=$(lsblk -dpno NAME,MODEL | grep -i "Instance Storage" | awk '{print $1}')
  sudo mdadm --create /dev/md0 --level=0 --raid-devices=$(echo $DEVS | wc -w) $DEVS --force --run
  sudo mkfs.xfs -f /dev/md0 && sudo mkdir -p /mnt/nvme && sudo mount /dev/md0 /mnt/nvme && sudo chown ubuntu:ubuntu /mnt/nvme
fi
mkdir -p ~/results
{ neuron-ls; neuronx-cc --version 2>&1|head -3; dpkg -l | grep -i neuron; ls /opt | grep -i venv; nproc; free -g; uname -a; } > ~/results/sysinfo.txt 2>&1
