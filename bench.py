"""Measure training speed of every preset on this machine.

    python bench.py                # all presets
    python bench.py bishlm plain   # some presets
"""
import sys
import time

import torch

from bishlm.config import PRESETS
from bishlm.model import BishLM

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
B, T = 16, 256


def sync():
    if device == "mps":
        torch.mps.synchronize()


for name in [a for a in sys.argv[1:] if not a.startswith("--")] or PRESETS:
    cfg = PRESETS[name]
    model = BishLM(cfg).to(device)
    if "--compile" in sys.argv:
        model.compile_parts()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def step():
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.balance_experts()

    for _ in range(3):
        step()
    sync()
    t = time.time()
    for _ in range(5):
        step()
    sync()
    dt = (time.time() - t) / 5
    p = model.count_params()
    print(f"{name:16s} total {p['total'] / 1e6:5.1f}M  active {p['active'] / 1e6:4.1f}M  "
          f"{dt * 1000:5.0f} ms/step  {B * T / dt:7,.0f} tok/s  {B * T / dt * 28800 / 1e6:5.0f}M tok/8h")
