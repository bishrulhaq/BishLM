"""Step 1: download TinyStories, train the tokenizer, turn all stories into token ids.

    python prepare_data.py

Makes data/tokenizer.json, data/train.bin and data/valid.bin.
TinyStories (Eldan and Li, 2023) is about 2.7 million short stories written
in simple English. Licence: CDLA-Sharing-1.0.
"""
import os
import sys
import time
import urllib.request
from multiprocessing import Pool

import numpy as np

from bishlm.tokenizer import EOT, Tokenizer

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
DATA = "data"
TOKENIZER_SAMPLE = 40_000_000   # characters used to learn the merges


def download():
    os.makedirs(DATA, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(URL + name, path)


_tok = None


def _init(merges):
    global _tok
    _tok = Tokenizer(merges)


def _encode_chunk(stories):
    out = []
    for s in stories:
        s = s.strip()
        if s:
            out.extend(_tok.encode(s))
            out.append(_tok.eot_id)
    return np.array(out, dtype=np.uint16)


def encode_file(tok, src, dst, workers=8):
    with open(src, encoding="utf-8") as f:
        stories = f.read().split(EOT)
    chunks = [stories[i:i + 2000] for i in range(0, len(stories), 2000)]
    t = time.time()
    with Pool(workers, initializer=_init, initargs=(tok.merges,)) as pool:
        parts = pool.map(_encode_chunk, chunks, chunksize=4)
    ids = np.concatenate(parts)
    ids.tofile(dst)
    print(f"{dst}: {len(stories):,} stories -> {len(ids):,} tokens ({time.time() - t:.0f}s)")


def main():
    download()
    tok_path = os.path.join(DATA, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.load(tok_path)
    else:
        with open(os.path.join(DATA, FILES["train"]), encoding="utf-8") as f:
            sample = f.read(TOKENIZER_SAMPLE).replace(EOT, " ")
        t = time.time()
        tok = Tokenizer.train(sample, vocab_size=4096)
        tok.save(tok_path)
        print(f"tokenizer trained in {time.time() - t:.0f}s, vocab {tok.vocab_size}")
    for split in sys.argv[1:] or ["valid", "train"]:
        encode_file(tok, os.path.join(DATA, FILES[split]), os.path.join(DATA, f"{split}.bin"))


if __name__ == "__main__":
    main()
