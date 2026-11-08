#!/usr/bin/env python3
"""Aggregate all jsonl + nccom results in ~/results into report tables (markdown)."""
import json, re, sys, glob, os, collections
RD = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/results")
out = []
def sec(t): out.append(f"\n## {t}\n")
def row(*c): out.append("| " + " | ".join(str(x) for x in c) + " |")

# GEMM
sec("GEMM (chained square matmul)")
row("dtype", "ranks", "N", "per-core TFLOPs", "aggregate TFLOPs"); row(*["---"]*5)
for f in sorted(glob.glob(f"{RD}/gemm_*.jsonl")):
    m = re.search(r"gemm_(\w+)_np(\d+)", f); dt, np_ = m.group(1), int(m.group(2))
    d = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l); d[r["n"]].append(r["tflops"])
    for n, v in sorted(d.items()):
        row(dt, np_, n, f"{sum(v)/len(v):.1f}", f"{sum(v):.0f}")

sec("STREAM (framework elementwise, fp32)")
row("ranks", "kernel", "per-core GB/s", "aggregate GB/s"); row(*["---"]*4)
for f in sorted(glob.glob(f"{RD}/stream_np*.jsonl"), key=lambda x: int(re.search(r"np(\d+)", x).group(1))):
    np_ = int(re.search(r"np(\d+)", f).group(1))
    d = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l); d[r["kernel"]].append(r["GBps"])
    for k in ("copy", "scale", "add", "triad"):
        v = d[k]; row(np_, k, f"{sum(v)/len(v):.0f}", f"{sum(v):.0f}")

sec("HBM copy_ (DMA, read+write)")
row("ranks", "MB", "per-core GB/s", "aggregate TB/s"); row(*["---"]*4)
for f in sorted(glob.glob(f"{RD}/hbm_copy_np*.jsonl")):
    np_ = int(re.search(r"np(\d+)", f).group(1)); v = []
    for l in open(f):
        r = json.loads(l); v.append(r["GBps_rw"]); mb = r["mb"]
    if v: row(np_, mb, f"{sum(v)/len(v):.0f}", f"{sum(v)/1000:.2f}")

sec("Host<->Device bandwidth (fp32 tensors)")
row("ranks", "MB", "H2D GB/s (per-proc/agg)", "D2H GB/s (per-proc/agg)"); row(*["---"]*4)
for f in sorted(glob.glob(f"{RD}/h2d_np*.jsonl")):
    np_ = int(re.search(r"np(\d+)", f).group(1))
    d = collections.defaultdict(lambda: ([], []))
    for l in open(f):
        r = json.loads(l); d[r["mb"]][0].append(r["h2d_GBps"]); d[r["mb"]][1].append(r["d2h_GBps"])
    for mb, (h, dd) in sorted(d.items()):
        row(np_, mb, f"{sum(h)/len(h):.1f} / {sum(h):.0f}", f"{sum(dd)/len(dd):.1f} / {sum(dd):.0f}")

sec("Collectives (nccom-test, bf16)")
row("op", "ranks", "size", "algbw GB/s", "busbw GB/s"); row(*["---"]*5)
for f in sorted(glob.glob(f"{RD}/collectives/*.txt")):
    m = re.search(r"(\w+)_r(\d+)", os.path.basename(f))
    body = open(f, errors="ignore").read()
    for ln in body.splitlines():
        # nccom-test table rows: size ... time algbw busbw
        mm = re.match(r"\s*(\d[\d.]*\s*[KMG]?i?B?)\s+.*?([\d.]+)\s+([\d.]+)\s*$", ln)
        if mm and not ln.strip().startswith("#"):
            row(m.group(1), m.group(2), mm.group(1).strip(), mm.group(2), mm.group(3))
print("\n".join(out))
