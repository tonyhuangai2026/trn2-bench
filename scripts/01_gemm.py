# GEMM micro-benchmark: chained square matmuls (y = y @ w) inside one graph
import argparse
from common import *
p = argparse.ArgumentParser()
p.add_argument("--sizes", default="1024,2048,4096,8192,16384")
p.add_argument("--dtype", default="bf16")
p.add_argument("--chain", type=int, default=16)   # matmuls per graph
p.add_argument("--iters", type=int, default=20)
a = p.parse_args()
dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[a.dtype]
for n in map(int, a.sizes.split(",")):
    chain = max(2, min(a.chain, int(a.chain * (8192 / n) ** 2)) if n > 8192 else a.chain)
    x = (torch.randn(n, n) * 0.01).to(dt).to(dev)
    w = (torch.eye(n) + torch.randn(n, n) * (0.01 / n ** 0.5)).to(dt).to(dev)
    @torch.compile(backend=os.environ.get("BACKEND","neuron")) if os.environ.get("COMPILE") else (lambda f: f)
    def step(y):
        for _ in range(chain): y = y @ w
        return y
    y = step(x); sync(); wait()          # compile + warmup
    y = step(y); sync(); wait()
    barrier(); t = time.perf_counter()
    for _ in range(a.iters):
        y = step(y); sync()
    wait(); dt_s = time.perf_counter() - t
    barrier()
    flops = 2 * n ** 3 * chain * a.iters
    gather_print({"bench": "gemm", "dtype": a.dtype, "n": n, "chain": chain,
                  "time_s": dt_s, "tflops": flops / dt_s / 1e12})
