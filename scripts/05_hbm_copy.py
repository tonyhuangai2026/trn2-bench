# Device-to-device copy_ bandwidth (DMA path) as HBM bandwidth proxy
import argparse
from common import *
p = argparse.ArgumentParser(); p.add_argument("--mb", type=int, default=4096); p.add_argument("--iters", type=int, default=20)
a = p.parse_args()
N = a.mb * 2**20 // 4
A = torch.randn(N, device=dev); B = torch.zeros(N, device=dev)
for _ in range(3): B.copy_(A)
wait(); barrier(); t = time.perf_counter()
for _ in range(a.iters): B.copy_(A)
wait(); el = time.perf_counter() - t; barrier()
gather_print({"bench": "hbm_copy", "mb": a.mb, "GBps_rw": 2 * N * 4 * a.iters / el / 1e9})
