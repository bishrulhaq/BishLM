"""The basic parts every modern LLM uses: RMSNorm, SwiGLU, RoPE and attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Scale a vector so its average square is 1, then apply a learned gain."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Feed-forward layer: up-project, gate with SiLU, down-project."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def rope_tables(seq_len, head_dim, base=10000.0):
    """cos and sin tables for rotary position embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair of channels by an angle that grows with the position."""
    T = x.shape[-2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:T], sin[:T]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    """Full causal attention: every token looks at every earlier token.

    Grouped-query attention (several query heads share one key/value head).
    Optional RoPE (the plain model) or NoPE (the hybrid, like Kimi K3 and GLM).
    Optional sigmoid output gate (Kimi K3's "Gated" attention).
    """

    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.out = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, self.h * self.hd, bias=False) if cfg.attn_gate else None
        self.use_rope = cfg.use_rope
        if self.use_rope:
            cos, sin = rope_tables(cfg.seq_len * 2, self.hd)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)    # (B, h, T, hd)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.h * self.hd)
        if self.gate is not None:
            o = o * torch.sigmoid(self.gate(x))
        return self.out(o)
