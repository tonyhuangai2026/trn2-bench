# Host<->device transfer bandwidth
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--sizes_mb", default="1,4,16,64,256,1024,4096")
p.add_argument("--iters", type=int, default=10)
a = p.parse_args()
for mb in map(int, a.sizes_mb.split(",")):
    h = torch.randn(mb * 2**18)  # fp32
    d = h.to(dev); sync(); wait(); _ = d.cpu()
    barrier(); t = time.perf_counter()
    for _ in range(a.iters):
        d = h.to(dev); sync(); wait()
    h2d = mb * a.iters / 1024 / (time.perf_counter() - t)
    barrier(); t = time.perf_counter()
    for _ in range(a.iters):
        d2 = (d + 0).cpu() if False else d.cpu()
    d2h = mb * a.iters / 1024 / (time.perf_counter() - t)
    barrier()
    gather_print({"bench": "h2d_d2h", "mb": mb, "h2d_GBps": h2d * 1.073741824, "d2h_GBps": d2h * 1.073741824})
