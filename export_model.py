"""Turn a training checkpoint into a small file for sharing: the weights only.

A training checkpoint also holds the optimizer state (to resume training), which
is twice the size of the model itself. This keeps just the weights, stored as
16-bit floats, so the file is about a sixth of the size. sample.py, chat.py and
`train.py --init` load it like any checkpoint.

    python export_model.py --ckpt runs/bishlm/ckpt.pt --out models/bishlm.pt
    python export_model.py --ckpt runs/instruct/ckpt.pt --out models/instruct.pt
"""
import argparse
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="a checkpoint saved by train.py")
p.add_argument("--out", required=True, help="where to write the weights-only file")
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
weights = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
small = {"cfg": ck["cfg"], "preset": ck.get("preset"), "model": weights,
         "step": ck["step"], "tokens": ck["tokens"]}

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save(small, args.out)
n = sum(v.numel() for v in weights.values())
print(f"{args.out}: {n / 1e6:.1f}M numbers, {os.path.getsize(args.out) / 1e6:.0f} MB "
      f"(step {ck['step']}, trained on {ck['tokens']:,} tokens)")
