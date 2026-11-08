# STREAM copy/scale/add/triad + HBM bandwidth on device-resident tensors
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--mb", type=int, default=1024)     # MB per array
p.add_argument("--iters", type=int, default=30)
p.add_argument("--dtype", default="fp32")
a = p.parse_args()
dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
esz = torch.tensor([], dtype=dt).element_size()
N = a.mb * 2**20 // esz
A = torch.ones(N, dtype=dt, device=dev); B = torch.full((N,), 2.0, dtype=dt, device=dev)
C = torch.zeros(N, dtype=dt, device=dev); s = 3.0
sync(); wait()
kernels = {  # name: (fn, bytes moved per call)
  "copy":  (lambda: (A, B, A * 1.0 + 0.0), 2),   # c = a (arith forces real copy)
  "scale": (lambda: (A, B, B * s), 2),
  "add":   (lambda: (A, B, A + B), 3),
  "triad": (lambda: (A, B, A + s * B), 3),
}
for name, (fn, nb) in kernels.items():
    for _ in range(3):
        C = fn()[2]; sync()
    wait(); barrier(); t = time.perf_counter()
    for _ in range(a.iters):
        C = fn()[2]; sync()
    wait(); el = time.perf_counter() - t; barrier()
    gather_print({"bench": "stream", "kernel": name, "dtype": a.dtype, "mb": a.mb,
                  "GBps": nb * N * esz * a.iters / el / 1e9})
