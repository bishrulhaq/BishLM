"""Step 3: let the trained model write.

    python sample.py --ckpt runs/bishlm/ckpt.pt --prompt "Once upon a time"
    python sample.py --ckpt runs/bishlm/ckpt.pt --compare     # normal vs speculative speed
"""
import argparse
import time

import torch

from bishlm.config import Config
from bishlm.generate import generate, generate_speculative
from bishlm.instruct import prompt_for
from bishlm.model import BishLM
from bishlm.tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/bishlm/ckpt.pt")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--instruct", default="", help="a request for an instruction-tuned model, e.g. \"Write a story about a dragon.\"")
p.add_argument("--tokens", type=int, default=200)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--compare", action="store_true", help="time normal vs speculative decoding")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=device)
cfg = Config(**ck["cfg"])
model = BishLM(cfg).to(device)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.load("data/tokenizer.json")
start = prompt_for(args.instruct) if args.instruct else args.prompt
prompt = torch.tensor([tok.encode(start)], device=device)
print(f"step {ck['step']}, trained on {ck['tokens']:,} tokens\n")


def timed(fn):
    torch.manual_seed(args.seed)
    t = time.time()
    out = fn()
    return out, time.time() - t


if not args.compare or model.mtp is None:
    ids, _ = timed(lambda: generate(model, prompt, args.tokens, args.temperature))
    print(tok.decode(ids[0].tolist()))
else:
    # warm-up: the Apple GPU prepares kernels for every new length the first time it
    # sees it, so we run once untimed. Otherwise the first method looks slower.
    generate(model, prompt, args.tokens + 2, args.temperature, stop_at_eot=False)
    ids, t1 = timed(lambda: generate(model, prompt, args.tokens, args.temperature, stop_at_eot=False))
    n1 = ids.shape[1] - prompt.shape[1]
    (ids2, st), t2 = timed(lambda: generate_speculative(model, prompt, args.tokens, args.temperature,
                                                        stop_at_eot=False))
    print(tok.decode(ids2[0].tolist()), "\n")
    print(f"normal:      {n1} tokens in {t1:.2f}s = {n1 / t1:.1f} tokens/s (1 token per pass)")
    print(f"speculative: {st['tokens']} tokens in {t2:.2f}s = {st['tokens'] / t2:.1f} tokens/s "
          f"({st['tokens_per_pass']:.2f} tokens per pass, {st['accepted']} drafts accepted)")
