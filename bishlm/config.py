"""All BishLM settings in one place.

Every idea we borrow has an on/off switch, so we can train the model
with and without it and see what it really changes.
"""
from dataclasses import dataclass, replace


@dataclass
class Config:
    # --- size -------------------------------------------------------------
    vocab_size: int = 4096      # our own BPE tokenizer (bishlm/tokenizer.py)
    seq_len: int = 256          # tokens the model sees at once
    d_model: int = 256          # width of every token vector
    n_layers: int = 8
    n_heads: int = 4            # heads in every attention layer
    n_kv_heads: int = 2         # grouped-query attention: 2 key/value heads shared by 4 query heads
    head_dim: int = 64
    ffn_hidden: int = 704       # hidden size of a dense SwiGLU feed-forward layer

    # --- idea 1 (Kimi K3): mixture of experts -------------------------------
    use_moe: bool = True
    n_experts: int = 8          # routed experts (Kimi K3: 896)
    top_k: int = 2              # experts per token (Kimi K3: 16)
    n_shared: int = 1           # experts every token uses (Kimi K3: 2)
    expert_hidden: int = 256    # each expert is a small SwiGLU
    first_dense: int = 1        # the first layer keeps a dense FFN (Kimi K3 does the same)
    balance_rate: float = 0.01  # how fast the router bias moves (aux-loss-free balancing)

    # --- idea 2 (Kimi K3): linear attention mixed 3:1 -----------------------
    use_linear_attn: bool = True
    linear_per_full: int = 3    # 3 KDA layers, then 1 full attention layer
    kda_chunk: int = 32         # tokens per chunk in the fast (chunked) KDA
    g_min: float = -5.0         # lowest log forget gate per token (Kimi K3 uses -5)
    conv_kernel: int = 4        # short convolution on q, k, v

    # --- idea 3 (Qwen3.8-Flash-Next): n-gram embeddings ---------------------
    use_ngram: bool = True
    ngram_orders: tuple = (2, 3)
    ngram_heads: int = 2        # hash heads per order (Qwen: 8)
    ngram_dim: int = 64         # width of each hash head
    ngram_table: int = 30011    # rows per hash head (a prime; each head gets its own prime near this)
    ngram_layer: int = 2        # injected at layer 2, like Qwen

    # --- idea 4 (Qwen3.8-Flash-Next): multi-token prediction ----------------
    use_mtp: bool = True
    mtp_weight: float = 0.3     # weight of the "token after next" loss (our choice)

    # --- idea 5 (GLM-5.3-Flash): mHC hyper-connections -----------------------
    use_mhc: bool = True
    hc_streams: int = 4         # residual stream widened to 4 copies (GLM: hc_mult 4)
    sinkhorn_iters: int = 20    # GLM: hc_sinkhorn_iters 20

    @property
    def use_rope(self):
        # Hybrid models (Kimi K3, GLM-5.3-Flash) use no position encoding (NoPE)
        # in full attention: the linear layers already know the order of tokens.
        # A plain all-attention model needs RoPE.
        return not self.use_linear_attn

    @property
    def attn_gate(self):
        # Kimi K3's full attention has a sigmoid output gate. The plain model does not.
        return self.use_linear_attn or self.use_moe or self.use_mhc


# --- presets ------------------------------------------------------------------

BISHLM = Config()
PLAIN = replace(BISHLM, use_moe=False, use_linear_attn=False, use_ngram=False,
                use_mtp=False, use_mhc=False)

# Smaller twins for the live run and the one-idea-off experiments.
SMALL = replace(BISHLM, d_model=192, n_layers=8, n_heads=3, n_kv_heads=1,
                ffn_hidden=512, expert_hidden=192)

PRESETS = {
    "bishlm": BISHLM,
    "plain": PLAIN,
    "small": SMALL,
    "small-plain": replace(SMALL, use_moe=False, use_linear_attn=False,
                           use_ngram=False, use_mtp=False, use_mhc=False),
    "small-no-moe": replace(SMALL, use_moe=False),
    "small-no-linear": replace(SMALL, use_linear_attn=False),
    "small-no-ngram": replace(SMALL, use_ngram=False),
    "small-no-mtp": replace(SMALL, use_mtp=False),
    "small-no-mhc": replace(SMALL, use_mhc=False),
}
