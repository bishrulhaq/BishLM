"""Step 4 (optional): data for instruction tuning.

    python prepare_instruct.py              # all of TinyStories-Instruct (2.6 GB)
    python prepare_instruct.py --max-mb 300 # only the first 300 MB (enough for a short fine-tune)

Makes data/instruct-train.bin, data/instruct-train-mask.bin and the same for valid.
The mask marks the tokens we grade: 1 for the story, 0 for the request.
TinyStories-Instruct: Eldan and Li, 2023. Licence: CDLA-Sharing-1.0.
"""
import argparse
import os
import random
import time
import urllib.request

import numpy as np

from bishlm.instruct import encode_example, parse
from bishlm.tokenizer import EOT, Tokenizer

URL = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/"
FILES = {"train": "TinyStories-Instruct-train.txt", "valid": "TinyStories-Instruct-valid.txt"}

p = argparse.ArgumentParser()
p.add_argument("--max-mb", type=int, default=0, help="read only this much of the train file (0 = all)")
args = p.parse_args()

tok = Tokenizer.load("data/tokenizer.json")
rng = random.Random(1234)
for split, name in FILES.items():
    path = os.path.join("data", name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(URL + name, path)
    t = time.time()
    with open(path, encoding="utf-8") as f:
        text = f.read(args.max_mb * 1_000_000) if split == "train" and args.max_mb else f.read()
    entries = text.split(EOT)[:-1] if split == "train" and args.max_mb else text.split(EOT)
    ids, mask, kept = [], [], 0
    for e in entries:
        fields = parse(e)
        if not fields["story"]:
            continue
        a, b = encode_example(tok, fields, rng)
        ids.extend(a)
        mask.extend(b)
        kept += 1
    np.array(ids, dtype=np.uint16).tofile(f"data/instruct-{split}.bin")
    np.array(mask, dtype=np.uint8).tofile(f"data/instruct-{split}-mask.bin")
    print(f"instruct-{split}: {kept:,} stories -> {len(ids):,} tokens ({time.time() - t:.0f}s)")
