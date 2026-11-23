"""A byte-level BPE tokenizer, written from scratch.

Idea: start with the 256 possible bytes as tokens. Then, again and again,
find the pair of neighbouring tokens that appears most often and glue it
into one new token. After 3839 merges we have a vocabulary of 4096 tokens
(256 bytes + 3839 merges + 1 special end-of-story token).
"""
import json
from collections import Counter, defaultdict

import regex

# Split text into words first, so merges never cross a word boundary.
# This is the same pattern GPT-2 uses.
SPLIT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
EOT = "<|endoftext|>"


class Tokenizer:
    def __init__(self, merges=None):
        self.merges = merges or []                      # list of (a, b) pairs, in order
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.cache = {}

    @property
    def vocab_size(self):
        return self.eot_id + 1

    # --- training -----------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size=4096, verbose=True):
        n_merges = vocab_size - 256 - 1
        words = Counter(SPLIT.findall(text))
        # every unique word as a list of byte ids, with how often it appears
        seqs = [list(w.encode("utf-8")) for w in words]
        freq = list(words.values())

        pair_count = Counter()
        where = defaultdict(set)                         # pair -> words that contain it
        for wi, s in enumerate(seqs):
            for p in zip(s, s[1:]):
                pair_count[p] += freq[wi]
                where[p].add(wi)

        merges = []
        for m in range(n_merges):
            if not pair_count:
                break
            best = max(pair_count, key=pair_count.get)
            new_id = 256 + m
            merges.append(best)
            for wi in list(where[best]):
                s = seqs[wi]
                # remove this word's old pairs
                for p in zip(s, s[1:]):
                    pair_count[p] -= freq[wi]
                    if pair_count[p] <= 0:
                        del pair_count[p]
                    where[p].discard(wi)
                # glue every (a, b) into new_id
                out, i = [], 0
                while i < len(s):
                    if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[wi] = out
                # add its new pairs
                for p in zip(out, out[1:]):
                    pair_count[p] += freq[wi]
                    where[p].add(wi)
            if verbose and (m + 1) % 500 == 0:
                tok = cls(merges)
                print(f"merge {m + 1}: {tok.vocab[new_id]!r}")
        return cls(merges)

    # --- encode / decode ------------------------------------------------------
    def _encode_word(self, word):
        ids = self.cache.get(word)
        if ids is not None:
            return ids
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            # glue the pair that was learned earliest
            pairs = [(self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            ids[i:i + 2] = [256 + rank]
        self.cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for w in SPLIT.findall(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids if i != self.eot_id)
        return data.decode("utf-8", errors="replace")

    # --- save / load ------------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls([tuple(p) for p in json.load(f)["merges"]])
