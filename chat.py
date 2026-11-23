"""Step 5: chat with BishLM about stories (after instruction tuning).

    python chat.py --ckpt runs/instruct/ckpt.pt

Type a request, like:  Write a story about a cat who finds a ball. Include dialogue.
Type "continue" to hear more of the same story, or "quit" to stop.
"""
import argparse
import sys

import torch

from bishlm.chat import QUIT, context_for, cut_at_sentence
from bishlm.config import Config
from bishlm.generate import generate
from bishlm.model import BishLM
from bishlm.tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/instruct/ckpt.pt")
p.add_argument("--tokens", type=int, default=70, help="longest reply, in tokens")
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--repeat-penalty", type=float, default=1.1, help="1.0 = off; higher = fewer repeated words")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location=device)
model = BishLM(Config(**ck["cfg"])).to(device)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.load("data/tokenizer.json")
torch.manual_seed(args.seed)
typed = sys.stdin.isatty()  # when the messages come from a file, print them too

print('BishLM story chat. Ask for a story, type "continue" for more, or "quit".\n')
story = []  # the ids of the story we are writing
while True:
    try:
        message = input("You: ") if typed else sys.stdin.readline()
    except EOFError:
        break
    if not message or message.strip().lower() in QUIT:
        break
    if not typed:
        print("You:", message.strip())
    ids, new = context_for(message, story, tok)
    if ids is None:
        print("BishLM: That story has ended. Ask me for a new one!\n")
        continue
    out = generate(model, torch.tensor([ids], device=device), args.tokens, args.temperature,
                   repeat_penalty=args.repeat_penalty)[0].tolist()
    reply = out[len(ids):]
    ended = bool(reply) and reply[-1] == tok.eot_id
    reply = reply[:-1] if ended else cut_at_sentence(tok, reply)
    story = ids + reply
    print("BishLM:", tok.decode(reply).strip().replace("\n", " "))
    if ended:
        print("        (The end.)")
        story = []
    print()
