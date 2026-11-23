"""mHC: Manifold-Constrained Hyper-Connections, the GLM-5.3-Flash way.

A normal residual connection is one lane:      x = x + F(norm(x))

Hyper-connections widen it to n lanes (streams). X has shape (n, d).
For every layer F:
    input  = H_pre  X           mix the n lanes into one input      (1 x n)
    y      = F(input)
    X      = H_res  X + H_post^T y
                                 mix the lanes with each other       (n x n)
                                 and spread the output back          (1 x n)

The "manifold constraint": H_res must be doubly stochastic (every row and
every column sums to 1). Sinkhorn-Knopp gets there by normalising rows,
then columns, again and again (20 times). Mixing with such a matrix can
never blow the signal up or shrink it to nothing, even after many layers.

The numbers in H_pre, H_post and H_res are partly fixed (learned biases b)
and partly depend on the current token (alpha * x' phi), as in the paper:
    x' = RMSNorm(vec(X))
    H_pre  = sigmoid(alpha_pre  * x' phi_pre  + b_pre)
    H_post = 2 sigmoid(alpha_post * x' phi_post + b_post)
    H_res  = Sinkhorn(exp(alpha_res * x' phi_res + b_res))
"""
import torch
import torch.nn as nn

from .layers import RMSNorm


def sinkhorn(logits, iters=20):
    """Turn any square matrix of logits into a doubly stochastic matrix."""
    M = (logits - logits.amax(dim=(-1, -2), keepdim=True)).exp()
    for _ in range(iters):
        M = M / M.sum(-1, keepdim=True)        # rows sum to 1
        M = M / M.sum(-2, keepdim=True)        # columns sum to 1
    return M


class HyperConnection(nn.Module):
    """Wraps one sublayer F (attention or feed-forward) with mHC."""

    def __init__(self, cfg, fn, layer_index):
        super().__init__()
        n, d = cfg.hc_streams, cfg.d_model
        self.n, self.iters, self.fn = n, cfg.sinkhorn_iters, fn
        self.norm_all = RMSNorm(n * d)
        self.phi = nn.Linear(n * d, 2 * n + n * n, bias=False)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))            # paper: alpha starts at 0.01
        # Static start, like the original Hyper-Connections paper (our choice):
        #   H_pre reads lane (layer mod n), H_post writes 1 to every lane, H_res ~ identity.
        b_pre = torch.full((n,), -4.0)
        b_pre[layer_index % n] = 4.0
        self.b_pre = nn.Parameter(b_pre)
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.full((n, n), -4.0).fill_diagonal_(0.0))
        self.norm = RMSNorm(d)                                       # pre-norm for F

    def coefficients(self, X):
        n = self.n
        h = self.phi(self.norm_all(X.flatten(-2)))                   # (B, T, 2n + n*n)
        a_pre, a_post, a_res = self.alpha
        H_pre = torch.sigmoid(a_pre * h[..., :n] + self.b_pre)                       # (B, T, n)
        H_post = 2 * torch.sigmoid(a_post * h[..., n:2 * n] + self.b_post)           # (B, T, n)
        H_res = sinkhorn(a_res * h[..., 2 * n:].unflatten(-1, (n, n)) + self.b_res, self.iters)
        return H_pre, H_post, H_res

    def read(self, X):
        """Coefficients, and the one input F gets: H_pre X (then RMSNorm)."""
        H_pre, H_post, H_res = self.coefficients(X)
        inp = (H_pre.unsqueeze(-1) * X).sum(-2)                      # (B, T, d)
        return self.norm(inp), H_post, H_res

    @staticmethod
    def write(X, y, H_post, H_res):
        """Mix the lanes, then spread F's output back: H_res X + H_post^T y."""
        return H_res @ X + H_post.unsqueeze(-1) * y.unsqueeze(-2)

    def forward(self, X):                                            # X: (B, T, n, d)
        inp, H_post, H_res = self.read(X)
        return self.write(X, self.fn(inp), H_post, H_res)


class Residual(nn.Module):
    """The normal one-lane residual, for the plain model. X: (B, T, 1, d)."""

    def __init__(self, cfg, fn, layer_index=0):
        super().__init__()
        self.fn, self.norm = fn, RMSNorm(cfg.d_model)

    def forward(self, X):
        return X + self.fn(self.norm(X.squeeze(2))).unsqueeze(2)
