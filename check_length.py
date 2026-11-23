"""Can BishLM read more than it was trained on?

    python check_length.py --ckpt runs/bishlm/ckpt.pt --length 1024

BishLM trains on windows of 256 tokens. Here we give it longer windows and
measure the loss in each part. If the loss stays the same, the model still
understands; if it goes up, the window is too long for it.
"""
import argparse
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from bishlm.config import Config
from bishlm.model import BishLM

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/bishlm/ckpt.pt")
p.add_argument("--length", type=int, default=1024)
p.add_argument("--rows", type=int, default=64)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
cfg = replace(Config(**ck["cfg"]), seq_len=args.length)
model = BishLM(cfg).to(device)
model.load_state_dict(ck["model"], strict=False)  # strict=False: RoPE tables are rebuilt for the new length
model.eval()

data = np.memmap("data/valid.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(0)
starts = rng.integers(0, len(data) - args.length - 1, args.rows)
x = torch.from_numpy(np.stack([data[s:s + args.length] for s in starts]).astype(np.int64)).to(device)
y = torch.from_numpy(np.stack([data[s + 1:s + args.length + 1] for s in starts]).astype(np.int64)).to(device)
losses = []
with torch.no_grad(), model.eager():
    for i in range(0, args.rows, 8):
        logits, _, _ = model(x[i:i + 8])
        losses.append(F.cross_entropy(logits.flatten(0, 1), y[i:i + 8].flatten(), reduction="none").view(-1, args.length).cpu())
loss = torch.cat(losses)
print(f"trained on windows of {Config(**ck['cfg']).seq_len} tokens, tested on {args.length}\n")
for a in range(0, args.length, 256):
    print(f"tokens {a:4d} to {a + 256:4d}:  loss {loss[:, a:a + 256].mean():.3f}")
