"""Step 2: train the model.

    python train.py --preset bishlm --steps 45000 --out runs/bishlm
    python train.py --preset small --minutes 10 --out runs/live

Every few hundred steps it prints the loss and a sample story, and saves
runs/<name>/ckpt.pt and runs/<name>/log.jsonl (used for the charts).
"""
import argparse
import json
import math
import os
import time

from dataclasses import replace

import numpy as np
import torch

from bishlm.config import PRESETS
from bishlm.generate import generate
from bishlm.instruct import prompt_for
from bishlm.model import BishLM
from bishlm.tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--preset", default="bishlm", choices=list(PRESETS))
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = no limit)")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--out", default="runs/bishlm")
p.add_argument("--no-compile", action="store_true")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt (model, optimizer, step)")
p.add_argument("--init", default="", help="fine-tuning: start from the weights in this checkpoint")
p.add_argument("--data", default="", help="fine-tuning data: data/<name>-train.bin and its -mask.bin")
p.add_argument("--seq-len", type=int, default=0, help="tokens per window (0 = the preset's 256)")
p.add_argument("--mix", type=float, default=0.2, help="with --data: share of rows taken from the stories, so old skills stay")
args = p.parse_args()

torch.manual_seed(args.seed)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
cfg = PRESETS[args.preset]
if args.seq_len:  # a longer window: works without retraining thanks to KDA and NoPE (see check_length.py)
    cfg = replace(cfg, seq_len=args.seq_len)
os.makedirs(args.out, exist_ok=True)

# --- data: one long list of token ids, we cut random windows out of it -------
train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
valid_data = np.memmap("data/valid.bin", dtype=np.uint16, mode="r")
tok = Tokenizer.load("data/tokenizer.json")
T = cfg.seq_len


def rows(data, n, rng, mask=None):
    starts = rng.integers(0, len(data) - T - 1, n)
    x = np.stack([data[s:s + T] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64)
    if mask is not None:  # only grade the marked tokens (the stories, not the requests)
        m = np.stack([mask[s + 1:s + T + 1] for s in starts])
        y[m == 0] = -100
    return x, y


def batch(data, B, rng, mask=None, mix=0.0):
    if mask is None:
        x, y = rows(data, B, rng)
    else:  # fine-tuning: most rows from the new data, some from the stories
        k = int(round(B * mix))
        x, y = rows(data, B - k, rng, mask)
        if k:
            x2, y2 = rows(train_data, k, rng)
            x, y = np.concatenate([x, x2]), np.concatenate([y, y2])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


rng = np.random.default_rng(args.seed)
if args.data:
    ft_data = np.memmap(f"data/{args.data}-train.bin", dtype=np.uint16, mode="r")
    ft_mask = np.memmap(f"data/{args.data}-train-mask.bin", dtype=np.uint8, mode="r")
    fv_data = np.memmap(f"data/{args.data}-valid.bin", dtype=np.uint16, mode="r")
    fv_mask = np.memmap(f"data/{args.data}-valid-mask.bin", dtype=np.uint8, mode="r")
    valid_batches = [batch(fv_data, args.batch, np.random.default_rng(i), fv_mask) for i in range(20)]
else:
    valid_batches = [batch(valid_data, args.batch, np.random.default_rng(i)) for i in range(20)]

# --- model and optimizer -------------------------------------------------------
model = BishLM(cfg).to(device)
if args.init:  # fine-tuning starts from a trained model
    model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
    print(f"weights from {args.init}")
if not args.no_compile and device == "mps":
    model.compile_parts()
counts = model.count_params()
print(f"{args.preset}: {counts['total'] / 1e6:.1f}M parameters, {counts['active'] / 1e6:.1f}M active per token")

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95))


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))   # cosine down to 10%


@torch.no_grad()
def evaluate():
    model.eval()
    losses = [model(x, y)[2]["loss_main"].item() for x, y in valid_batches]
    model.train()
    return sum(losses) / len(losses)


def save():
    # write to a temp file first, so a crash mid-save never breaks the last good checkpoint
    path = os.path.join(args.out, "ckpt.pt")
    torch.save({"cfg": cfg.__dict__, "preset": args.preset, "model": model.state_dict(),
                "opt": opt.state_dict(), "step": step, "tokens": tokens,
                "rng": rng.bit_generator.state, "elapsed": time.time() - start}, path + ".tmp")
    os.replace(path + ".tmp", path)


first = 0
tokens = 0
start = time.time()
ckpt_path = os.path.join(args.out, "ckpt.pt")
if args.resume and os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
    first, tokens = ck["step"] + 1, ck["tokens"]
    start -= ck.get("elapsed", 0)
    print(f"resumed from step {ck['step']} ({tokens:,} tokens)")
log = open(os.path.join(args.out, "log.jsonl"), "a" if args.resume else "w")
step = first
for step in range(first, args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(ft_data, args.batch, rng, ft_mask, args.mix) if args.data else batch(train_data, args.batch, rng)
    _, loss, stats = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    model.balance_experts()
    tokens += x.numel()

    last = step == args.steps - 1
    out_of_time = args.minutes and time.time() - start > args.minutes * 60
    if step % 50 == 0 or last or out_of_time:
        elapsed = time.time() - start
        row = {"step": step, "tokens": tokens, "time": round(elapsed, 1), "lr": lr_at(step),
               "loss": loss.item(), "grad_norm": grad_norm.item(),
               **{k: v.item() for k, v in stats.items()}}
        moes = model.moe_layers()
        if moes:
            row["expert_load"] = [m.load.tolist() for m in moes]
            row["dropped"] = sum(m.dropped.item() for m in moes) / len(moes)
        if step % args.eval_every == 0 or last or out_of_time:
            row["val_loss"] = evaluate()
            opening = prompt_for("Write a story that uses the words: dragon, ice cream, shop.") if args.data else "Once upon a time"
            prompt = torch.tensor([tok.encode(opening)], device=device)
            story = tok.decode(generate(model, prompt, 60, temperature=0.8)[0].tolist())
            row["sample"] = story
            print(f"\n--- step {step}  val loss {row['val_loss']:.3f} ---\n{story}\n", flush=True)
            save()
        print(f"step {step:6d}  loss {row['loss']:.3f}  {tokens / elapsed:,.0f} tok/s  "
              f"{elapsed / 60:.1f} min", flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    if out_of_time:
        break
save()
print(f"done: {tokens:,} tokens in {(time.time() - start) / 60:.1f} min")
