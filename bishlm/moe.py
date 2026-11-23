"""Mixture of experts, the Kimi K3 way.

Instead of one big feed-forward layer, we keep many small ones ("experts").
A router scores every expert for every token and only the top-k run.
So the model can hold many parameters while each token only pays for a few.

Kimi K3 details we keep (from its tech report):
  - sigmoid scores, not softmax:           s = sigmoid(W_r x)
  - choose top-k of (s + b), where b is a balancing bias
  - weights from the scores alone:         p = s / sum(s over the chosen experts)
  - shared experts that every token uses
  - no auxiliary loss: only the bias b balances the load.
    After each step, experts that got too many tokens get a lower b,
    experts that got too few get a higher b (DeepSeek-V3 rule).

To run fast on a laptop GPU, every expert gets a fixed number of seats
("capacity"). Tokens queue up at each expert they picked; the first `cap`
in each queue get a seat. All experts then run together as one batched
matrix multiply. With good balancing almost no token misses its seat; we
count the ones that do.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLU


class MoE(nn.Module):
    def __init__(self, cfg, capacity_factor=2.0):
        super().__init__()
        self.n, self.k, d, h = cfg.n_experts, cfg.top_k, cfg.d_model, cfg.expert_hidden
        self.router = nn.Linear(d, self.n, bias=False)
        self.register_buffer("bias", torch.zeros(self.n))           # balancing bias (no gradients)
        self.register_buffer("load", torch.zeros(self.n))           # tokens per expert, last batch
        self.register_buffer("dropped", torch.zeros(()))            # share of tokens that lost a seat
        # all experts' SwiGLU weights stacked: expert e uses w_gate[e], w_up[e], w_down[e]
        self.w_gate = nn.Parameter(torch.randn(self.n, d, h) * 0.02)
        self.w_up = nn.Parameter(torch.randn(self.n, d, h) * 0.02)
        self.w_down = nn.Parameter(torch.randn(self.n, h, d) * 0.02)
        self.shared = nn.ModuleList(SwiGLU(d, h) for _ in range(cfg.n_shared))
        self.rate, self.capacity_factor = cfg.balance_rate, capacity_factor

    def route(self, x):
        """x: (N, d) -> chosen experts (N, k) and their weights (N, k)."""
        s = torch.sigmoid(self.router(x))                            # (N, n)
        pick = (s + self.bias).detach()                              # bias only picks...
        top = []
        for _ in range(self.k):                                      # top-k as k argmax passes
            i = pick.argmax(-1, keepdim=True)                        # (fast on Apple GPUs)
            top.append(i)
            pick = pick.scatter(-1, i, float("-inf"))
        top = torch.cat(top, dim=-1)                                 # (N, k)
        w = s.gather(-1, top)                                        # ...weights use s alone
        return top, w / w.sum(-1, keepdim=True)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                                 # (N, d) all tokens
        N, n = x.shape[0], self.n
        top, w = self.route(x)

        # weight of every expert for every token, 0 when not chosen: (N, n)
        gates = torch.zeros(N, n, device=x.device, dtype=x.dtype).scatter(1, top, w)
        chosen = gates > 0
        # seats per expert while training; when writing text, room for every token
        # (so the answer never depends on how many tokens are in the batch)
        cap = min(N, int(self.capacity_factor * N * self.k / n)) if self.training else N
        # queue number of each token at each expert it picked: 0, 1, 2, ...
        queue = chosen.long().cumsum(0) - 1
        seated = chosen & (queue < cap)
        # seat id = expert * cap + queue number; everything else goes to a spare seat
        seat = torch.where(seated, torch.arange(n, device=x.device) * cap + queue,
                           torch.full_like(queue, n * cap)).flatten()
        tokens = torch.arange(N, device=x.device).repeat_interleave(n)
        seat_tok = torch.zeros(n * cap + 1, dtype=torch.long, device=x.device).scatter(0, seat, tokens)
        seat_w = torch.zeros(n * cap + 1, device=x.device, dtype=x.dtype).scatter(0, seat, gates.flatten())
        seat_tok, seat_w = seat_tok[:-1].view(n, cap), seat_w[:-1].view(n, cap)   # drop the spare seat

        xe = x[seat_tok]                                             # (n, cap, d)
        he = F.silu(xe @ self.w_gate) * (xe @ self.w_up)             # (n, cap, h)
        ye = (he @ self.w_down) * seat_w.unsqueeze(-1)               # (n, cap, d)
        out = torch.zeros_like(x).index_add(0, seat_tok.flatten(), ye.flatten(0, 1))

        for expert in self.shared:
            out = out + expert(x)
        if self.training:
            with torch.no_grad():
                self.load.copy_(chosen.sum(0).float())
                self.dropped.copy_(1 - seated.sum() / chosen.sum())
        return out.reshape(shape)

    def forward_reference(self, x):
        """Slow and simple: loop over experts, no seats, nothing dropped. For checking."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        top, w = self.route(x)
        out = torch.zeros_like(x)
        for e in range(self.n):
            tok, slot = (top == e).nonzero(as_tuple=True)
            h = F.silu(x[tok] @ self.w_gate[e]) * (x[tok] @ self.w_up[e])
            out.index_add_(0, tok, (h @ self.w_down[e]) * w[tok, slot, None])
        for expert in self.shared:
            out = out + expert(x)
        return out.reshape(shape)

    @torch.no_grad()
    def balance(self):
        """Call after each optimizer step: nudge the bias toward an even load."""
        self.bias += self.rate * torch.sign(self.load.mean() - self.load)
