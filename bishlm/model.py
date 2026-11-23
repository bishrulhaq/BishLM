"""The whole model, built from the parts in this folder."""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kda import KDA
from .layers import Attention, RMSNorm, SwiGLU
from .mhc import HyperConnection, Residual
from .moe import MoE
from .ngram import NGramEmbedding


def is_full_attention(cfg, i):
    """Layer i (0-based) uses full attention? With linear attention on: every 4th layer."""
    if not cfg.use_linear_attn:
        return True
    return (i + 1) % (cfg.linear_per_full + 1) == 0


class Block(nn.Module):
    """One layer: a token mixer (attention or KDA), then a feed-forward (dense or MoE)."""

    def __init__(self, cfg, i):
        super().__init__()
        wrap = HyperConnection if cfg.use_mhc else Residual
        mixer = Attention(cfg) if is_full_attention(cfg, i) else KDA(cfg)
        dense = (not cfg.use_moe) or i < cfg.first_dense
        ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden) if dense else MoE(cfg)
        self.mixer = wrap(cfg, mixer, 2 * i)
        self.ffn = wrap(cfg, ffn, 2 * i + 1)

    def forward(self, X):
        return self.ffn(self.mixer(X))


class MTPHead(nn.Module):
    """Multi-token prediction: guess the token AFTER the next one.

    At position t the main model has a hidden state h_t and predicts token t+1.
    MTP joins h_t with the embedding of token t+1, runs one extra layer, and
    predicts token t+2 with the same output head (DeepSeek-V3 style, used by Qwen).
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm_h, self.norm_e = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.attn = Residual(cfg, Attention(cfg))
        self.ffn = Residual(cfg, SwiGLU(cfg.d_model, cfg.ffn_hidden))
        self.norm_out = RMSNorm(cfg.d_model)

    def forward(self, h, next_emb):
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        X = self.ffn(self.attn(x.unsqueeze(2))).squeeze(2)
        return self.norm_out(X)


class BishLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_streams = cfg.hc_streams if cfg.use_mhc else 1
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ngram = NGramEmbedding(cfg) if cfg.use_ngram else None
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                         # weight tying
        self.mtp = MTPHead(cfg) if cfg.use_mtp else None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight is not None and m.weight.dim() == 2:
            if m.weight.abs().sum() > 0:                             # keep zero-initialised weights at zero
                nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, idx):
        """Token ids (B, T) -> last hidden states (B, T, d), before the final norm."""
        X = self.embed(idx).unsqueeze(2).expand(-1, -1, self.n_streams, -1)   # copy into every lane
        e = self.ngram.lookup(idx) if self.ngram is not None else None
        for i, block in enumerate(self.blocks):
            if e is not None and i == self.cfg.ngram_layer - 1:     # "at layer 2"
                X = self.ngram.inject(X, e)
            X = block(X)
        return X.mean(dim=2)                                         # collapse lanes: plain mean (GLM)

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        logits = self.head(self.norm(h))
        if targets is None:
            return logits, None, {}
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        stats = {"loss_main": loss.detach()}
        if self.mtp is not None:
            # at position t: hidden h_t + embedding of token t+1 (= targets[t]) -> predict targets[t+1]
            m = self.mtp(h, self.embed(targets.clamp(min=0)))     # -100 marks ungraded tokens
            logits2 = self.head(m)[:, :-1]
            loss2 = F.cross_entropy(logits2.flatten(0, 1), targets[:, 1:].flatten())
            stats["loss_mtp"] = loss2.detach()
            loss = loss + self.cfg.mtp_weight * loss2
        return logits, loss, stats

    def compile_parts(self):
        """Speed-up for Apple GPUs: let torch.compile fuse KDA and the mHC maths
        into fewer GPU kernels. (Compiling the MoE made it slower, so we skip it.)"""
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256              # one compiled copy per layer
        torch._dynamo.config.accumulated_cache_size_limit = 4096
        self._compiled = []
        for m in self.modules():
            names = ["forward"] if isinstance(m, KDA) else ["read", "write"] if isinstance(m, HyperConnection) else []
            for name in names:
                fast = torch.compile(getattr(m, name), dynamic=False)
                self._compiled.append((m, name, fast))
                setattr(m, name, fast)
        return self

    @contextlib.contextmanager
    def eager(self):
        """Run without the compiled parts (generation uses a new length every step)."""
        compiled = getattr(self, "_compiled", [])
        for m, name, _ in compiled:
            delattr(m, name)                                         # falls back to the class method
        try:
            yield self
        finally:
            for m, name, fast in compiled:
                setattr(m, name, fast)

    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, MoE)]

    def balance_experts(self):
        for m in self.moe_layers():
            m.balance()

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        ngram = sum(p.numel() for p in self.ngram.tables.parameters()) if self.ngram else 0
        mtp = sum(p.numel() for p in self.mtp.parameters()) if self.mtp else 0
        inactive = 0
        for m in self.moe_layers():
            per_expert = (m.w_gate[0].numel() + m.w_up[0].numel() + m.w_down[0].numel())
            inactive += per_expert * (m.n - m.k)
        return {"total": total, "active": total - inactive - ngram - mtp,
                "ngram_table": ngram, "mtp": mtp}
