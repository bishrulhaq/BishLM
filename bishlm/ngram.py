"""N-gram embeddings, the Qwen3.8-Flash-Next way.

A normal embedding gives each token one vector. But "once upon a" means
more than its three tokens one by one. So we also give each 2-gram
(last 2 tokens) and 3-gram (last 3 tokens) its own vector.

There are far too many possible n-grams to store (4096^3 = 68 billion
3-grams), so we hash each one into a smaller table, like Qwen does:

    id = (tok_t * m0) XOR (tok_t-1 * m1) [XOR (tok_t-2 * m2)]   mod prime

Two different n-grams can land on the same row (a collision). Using
several hash heads, each with its own multipliers and prime, makes it very
unlikely that two n-grams collide in every head.

A lookup is just reading a row, so this adds parameters but almost no compute.
Qwen injects the result at layer 2, into every residual stream, through a
gate that lets each stream decide how much to take:

    gate_i = sigmoid( RMSNorm(key) . RMSNorm(stream_i) / sqrt(d) )
    stream_i = stream_i + gate_i * value
"""
import random

import torch
import torch.nn as nn

from .layers import RMSNorm


def is_prime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def primes_from(start, count):
    out, n = [], start
    while len(out) < count:
        if is_prime(n):
            out.append(n)
        n += 1
    return out


class NGramEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.orders = cfg.ngram_orders
        n_heads = len(self.orders) * cfg.ngram_heads
        self.primes = primes_from(cfg.ngram_table, n_heads)
        rng = random.Random(1234)                                    # fixed seed: same hashes every run
        self.mults = [[rng.randrange(1, 1 << 20) * 2 + 1 for _ in range(max(self.orders))]
                      for _ in range(n_heads)]                       # odd multipliers
        self.head_order = [n for n in self.orders for _ in range(cfg.ngram_heads)]
        self.tables = nn.ModuleList(nn.Embedding(p, cfg.ngram_dim) for p in self.primes)
        width = n_heads * cfg.ngram_dim
        self.value = nn.Linear(width, cfg.d_model, bias=False)
        self.key = nn.Linear(width, cfg.d_model, bias=False)
        self.key_norm, self.stream_norm = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.pad = cfg.vocab_size                                    # stands for "before the text"
        nn.init.zeros_(self.value.weight)                            # start as "no effect"

    def hash_ids(self, idx):
        """idx: (B, T) token ids -> list of (B, T) row ids, one per hash head."""
        prev = [idx]
        for s in range(1, max(self.orders)):
            prev.append(torch.cat([torch.full_like(idx[:, :s], self.pad), idx[:, :-s]], dim=1))
        ids = []
        for head, n in enumerate(self.head_order):
            h = prev[0] * self.mults[head][0]
            for s in range(1, n):
                h = torch.bitwise_xor(h, prev[s] * self.mults[head][s])
            ids.append(torch.remainder(h, self.primes[head]))
        return ids

    def lookup(self, idx):
        rows = [table(i) for table, i in zip(self.tables, self.hash_ids(idx))]
        return torch.cat(rows, dim=-1)                               # (B, T, heads * ngram_dim)

    def inject(self, streams, e):
        """streams: (B, T, n, d), e: (B, T, width) -> streams with n-gram info added."""
        d = streams.shape[-1]
        key = self.key_norm(self.key(e)).unsqueeze(2)                # (B, T, 1, d)
        gate = torch.sigmoid((key * self.stream_norm(streams)).sum(-1, keepdim=True) / d ** 0.5)
        return streams + gate * self.value(e).unsqueeze(2)
