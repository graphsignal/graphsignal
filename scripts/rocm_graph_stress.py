"""Torch HIP-graph replay stress: captures a transformer-ish forward into a
CUDA/HIP graph at several batch sizes and replays them back to back, the way an
inference engine's decode loop does. Exits 0 when every replay completed."""
import sys, time, torch

torch.manual_seed(0)
dev = 'cuda'
d_model, n_layers = 1024, 8
layers = torch.nn.ModuleList([torch.nn.Sequential(
    torch.nn.Linear(d_model, 4 * d_model), torch.nn.GELU(),
    torch.nn.Linear(4 * d_model, d_model), torch.nn.LayerNorm(d_model))
    for _ in range(n_layers)]).to(dev).half()

def fwd(x):
    for l in layers:
        x = x + l(x)
    return x.softmax(-1)

replays = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
import os
eager = os.environ.get('EAGER') == '1'      # launch the forward eagerly instead of replaying graphs
cleanup = os.environ.get('CLEANUP') == '1'  # destroy the graphs and sync before interpreter exit
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
graphs = {}
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for bs in batch_sizes:
        x = torch.randn(bs, d_model, device=dev, dtype=torch.half)
        for _ in range(3):
            fwd(x)  # warm-up
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            y = fwd(x)
        graphs[bs] = (g, x, y)
torch.cuda.synchronize()
print(f'captured {len(graphs)} graphs', flush=True)

t0 = time.time()
for i in range(replays):
    bs = batch_sizes[i % len(batch_sizes)]
    g, x, y = graphs[bs]
    x.copy_(torch.randn_like(x))
    if eager:
        fwd(x)
    else:
        g.replay()
    if i % 100 == 0:
        torch.cuda.synchronize()
        print(f'replay {i}', flush=True)
torch.cuda.synchronize()
print(f'done: {replays} {"eager launches" if eager else "replays"} in {time.time() - t0:.1f}s', flush=True)
if cleanup:
    graphs.clear()
    del g
    torch.cuda.synchronize()
    print('graphs destroyed', flush=True)
