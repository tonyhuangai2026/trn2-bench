import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
LR = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0)); WORLD = int(os.environ.get("WORLD_SIZE", 1))
STRIDE = int(os.environ.get("CORE_STRIDE", 1))
if "NEURON_RT_VISIBLE_CORES" not in os.environ:
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(LR * STRIDE)   # 1 logical core per process
import torch, torch_neuronx
dev = torch.device("neuron:0")
def sync(): pass
def wait(): torch.neuron.synchronize()
_pg = None
def barrier():
    global _pg
    if WORLD == 1: return
    import torch.distributed as dist
    if _pg is None:
        dist.init_process_group("gloo"); _pg = True
    dist.barrier()
def gather_print(rec):
    out = os.environ.get("OUT", "/tmp/bench.jsonl")
    with open(out, "a") as f: f.write(json.dumps({"rank": RANK, "world": WORLD, **rec}) + "\n")
    if RANK == 0: print(json.dumps(rec), flush=True)
