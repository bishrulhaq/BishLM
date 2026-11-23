"""Kimi Delta Attention (KDA): linear attention with a fixed-size memory.

Full attention keeps every past key and value, so its memory grows with the
text. KDA keeps one small matrix S per head instead (d_k x d_v numbers) and
updates it at every token:

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

In words, for every token:
  1. forget a little:  S = Diag(alpha) S        (alpha in [0,1], one per channel)
  2. read what S already stores for this key:  old = S^T k
  3. write the correction:  S = S + beta k (v - old)^T
  4. read the output for the query:  o = S^T q

Source: Kimi Linear paper (arXiv 2510.26692) and the Kimi K3 report.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


def kda_recurrent(q, k, v, g, beta, S=None):
    """Token-by-token KDA. Slow, but it is the definition. Used for checking.

    q, k: (B, H, T, dk)   v: (B, H, T, dv)   g: (B, H, T, dk) log forget gate (<= 0)
    beta: (B, H, T)       S: (B, H, dk, dv) starting memory (zeros if None)
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    if S is None:
        S = q.new_zeros(B, H, dk, dv)
    out = []
    for t in range(T):
        S = S * g[:, :, t].exp().unsqueeze(-1)                          # 1. forget
        old = torch.einsum("bhkv,bhk->bhv", S, k[:, :, t])              # 2. read old value
        S = S + beta[:, :, t, None, None] * torch.einsum(               # 3. write correction
            "bhk,bhv->bhkv", k[:, :, t], v[:, :, t] - old)
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, :, t]))        # 4. read output
    return torch.stack(out, dim=2), S


def unit_lower_inverse(M):
    """(I + M)^-1 for a strictly lower-triangular M of size C x C (C a power of 2).

    M is nilpotent (M^C = 0), so (I + M)^-1 = I - M + M^2 - ... - M^(C-1)
    = (I - M)(I + M^2)(I + M^4)...  : only log2(C) matrix products.
    """
    C = M.shape[-1]
    eye = torch.eye(C, device=M.device, dtype=M.dtype)
    P = -M
    out = eye + P
    step = 2
    while step < C:
        P = P @ P
        out = out @ (eye + P)
        step *= 2
    return out


def kda_chunked(q, k, v, g, beta, chunk=32, S=None):
    """The same maths as kda_recurrent, but chunk by chunk with matrix multiplies.

    Inside a chunk of C tokens every step can be written down at once:
      G      = running sum of g inside the chunk (how much was forgotten so far)
      k~ = k * exp(G),  k^ = k * exp(-G),  q~ = q * exp(G)
      M[j,i] = beta_i (k~_j . k^_i)  for i < j        (how write i changes token j)
      U      = (I + M)^-1 (V - K~ S0)                  (all corrections at once)
      O      = Q~ S0 + tril(beta_j q~_t . k^_j) U
      S_end  = exp(G_last) S0 + sum_j (k_j exp(G_last - G_j) beta_j) u_j^T
    Only the hand-off of S from one chunk to the next is a loop.

    Products like k~_j . k^_i only need exp(G_j - G_i), so inside them we
    measure G from the middle of the chunk. Then no exp is bigger than
    exp(C/2 * |g_min|) = exp(80) for C = 32, g_min = -5, which still fits
    in float32 (limit about exp(88)).
    """
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    C = chunk
    pad = (-T) % C
    if pad:  # zeros: no forgetting, no writing, no reading
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    N = (T + pad) // C
    q, k, v, g = (x.reshape(B, H, N, C, -1) for x in (q, k, v, g))
    beta = beta.reshape(B, H, N, C)

    G = g.cumsum(dim=-2)                               # (B, H, N, C, dk)
    Gm = G - G[:, :, :, C // 2 - 1:C // 2, :]           # measured from the chunk middle
    k_t, q_t = k * G.exp(), q * G.exp()                # for S0 terms (always <= 1)
    k_tm, k_hm, q_tm = k * Gm.exp(), k * (-Gm).exp(), q * Gm.exp()

    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(0)

    # mask BEFORE anything else touches the product: the masked (future) entries
    # can be exp(160) = inf, and inf * 0 would turn into NaN in the backward pass
    M = (k_tm @ k_hm.transpose(-1, -2)).masked_fill(~strict, 0.0) * beta.unsqueeze(-2)   # M[j, i]
    Minv = unit_lower_inverse(M)                                     # (B, H, N, C, C)

    W = Minv @ k_t                                     # U = U0 - W S0
    U0 = Minv @ v
    P = (q_tm @ k_hm.transpose(-1, -2)).masked_fill(~causal, 0.0) * beta.unsqueeze(-2)
    decay_last = G[:, :, :, -1:, :]                    # (B, H, N, 1, dk)
    k_end = k * (decay_last - G).exp() * beta.unsqueeze(-1)

    if S is None:
        S = q.new_zeros(B, H, dk, dv)
    outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ S
        outs.append(q_t[:, :, n] @ S + P[:, :, n] @ U)
        S = decay_last[:, :, n].transpose(-1, -2).exp() * S + k_end[:, :, n].transpose(-1, -2) @ U
    o = torch.stack(outs, dim=2).reshape(B, H, N * C, dv)
    return o[:, :, :T], S


class ShortConv(nn.Module):
    """Causal depthwise convolution: each channel mixes the last few tokens."""

    def __init__(self, dim, kernel):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=False)

    def forward(self, x):                              # (B, T, dim)
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)


class KDA(nn.Module):
    """One Kimi Delta Attention layer."""

    def __init__(self, cfg):
        super().__init__()
        d, self.h, self.hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        inner = self.h * self.hd
        self.q, self.k, self.v = (nn.Linear(d, inner, bias=False) for _ in range(3))
        self.conv_q, self.conv_k, self.conv_v = (ShortConv(inner, cfg.conv_kernel) for _ in range(3))
        self.beta = nn.Linear(d, self.h, bias=True)                  # write strength per head
        self.g_down = nn.Linear(d, self.hd, bias=False)              # low-rank forget gate
        self.g_up = nn.Linear(self.hd, inner, bias=True)
        self.A = nn.Parameter(torch.zeros(self.h))                   # gate scale per head
        self.out_gate = nn.Linear(d, inner, bias=False)
        self.norm = RMSNorm(self.hd)
        self.out = nn.Linear(inner, d, bias=False)
        self.g_min, self.chunk = cfg.g_min, cfg.kda_chunk
        assert self.chunk / 2 * abs(self.g_min) < 88, "exp(-G) would overflow float32"
        assert self.chunk & (self.chunk - 1) == 0, "chunk must be a power of 2"

    def heads(self, x):                                # (B, T, h*hd) -> (B, h, T, hd)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.hd).transpose(1, 2)

    def gates(self, x):
        z = self.g_up(self.g_down(x))                                # (B, T, h*hd)
        z = z.view(*x.shape[:2], self.h, self.hd) * self.A.exp()[:, None]
        g = self.g_min * torch.sigmoid(z)                            # in [g_min, 0]
        beta = torch.sigmoid(self.beta(x))                           # (B, T, h)
        return g.transpose(1, 2), beta.transpose(1, 2)

    def forward(self, x, recurrent=False):
        B, T, _ = x.shape
        q = self.heads(F.silu(self.conv_q(self.q(x))))
        k = self.heads(F.silu(self.conv_k(self.k(x))))
        v = self.heads(F.silu(self.conv_v(self.v(x))))
        q = F.normalize(q, dim=-1) * self.hd ** -0.5
        k = F.normalize(k, dim=-1)
        g, beta = self.gates(x)
        if recurrent:
            o, _ = kda_recurrent(q, k, v, g, beta)
        else:
            o, _ = kda_chunked(q, k, v, g, beta, self.chunk)
        o = self.norm(o).transpose(1, 2).reshape(B, T, self.h * self.hd)
        return self.out(o * torch.sigmoid(self.out_gate(x)))
