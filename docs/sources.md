# Sources: what the real models do

Checked on 18 November 2026, against primary sources only: official tech reports, arXiv papers, Hugging Face `config.json` files and `transformers` modeling code. When a source is a blog or has a conflict, this file says so.

BishLM borrows **ideas**, not code or weights. Each section below ends with "BishLM uses", which says what our tiny version keeps and what it simplifies.

---

## 1. Kimi K3 (Moonshot AI)

- Tech report: https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf
- Config: https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
- Kimi Linear paper (KDA): https://arxiv.org/html/2510.26692

### 1a. Mixture of experts

- Kimi K3 has 896 routed experts. Each token uses 16 of them, plus 2 shared experts that every token uses. The model has 2.78T parameters in total and 104.2B active per token, across 93 layers. The first layer is dense (`first_k_dense_replace: 1`).
- The router uses sigmoid scores, not softmax: `s_i = Sigmoid(W_r x_i)`. It picks the top-k of `s_i + b`, then normalises the weights with `p = s / Σ s` over the chosen experts. The bias `b` is *not* used in the weights.
- Load balancing uses no auxiliary loss ("Kimi K3 adopts auxiliary-loss-free routing"). Only the bias `b` steers tokens toward idle experts. K3's new rule for updating that bias is called Quantile Balancing.
- "Stable LatentMoE" means three things: an RMSNorm before the up-projection, a SiTU-GLU activation, and Quantile Balancing. The routed experts also run in a narrower latent width.

**BishLM uses:** sigmoid router, top-k, `s / Σ s` weights, shared expert, a dense first layer, and auxiliary-loss-free bias balancing. We use the simpler DeepSeek-V3 bias rule: raise the bias of under-used experts and lower it for over-used ones. We skip Quantile Balancing, LatentMoE and SiTU-GLU; our experts are plain SwiGLU.

### 1b. Linear attention mixed 3:1 with full attention

- Kimi K3 has 69 KDA layers and 24 Gated MLA layers. The report says: "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3:1 mixing ratio". There is also one extra MLA layer at the end.
- All MLA layers use **NoPE**, meaning no position encoding at all: "applies No Position Encoding (NoPE) to all MLA layers." The KDA layers give the model its sense of order.
- KDA state update (Kimi Linear paper): `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`, and `o_t = S_tᵀ q_t`.
  - `α_t ∈ [0,1]^{d_k}` is a per-channel forget gate. Gated DeltaNet uses one number per head instead.
  - `β_t = Sigmoid(W_β x_t)` is one write-strength number per head.
  - q and k: `L2Norm(Swish(ShortConv(W x)))`. v: `Swish(ShortConv(W_v x))`. The conv kernel size is 4.
  - Output: `W_o(Sigmoid(W_g x) ⊙ RMSNorm(KDA(...)))`.
  - K3 bounds the decay: `g = g_min · Sigmoid(e^A z)`, with `α = exp(g)` and `g_min = −5`.

**BishLM uses:** the KDA update exactly as above, with per-channel α bounded by `g_min = −5`, plus ShortConv, L2Norm, a sigmoid output gate and RMSNorm. Layers follow a 3:1 pattern, and the full attention layers are NoPE. Our full attention is plain gated multi-head attention, not MLA.

### 1c. Not used

AttnRes (Attention Residuals) and MTP. On MTP, the K3 report lists 1 layer but the HF config has `num_nextn_predict_layers: 0`.

---

## 2. Qwen3.8-Flash-Next (Alibaba)

- Paper: https://arxiv.org/html/2608.30320
- `transformers` code: `src/transformers/models/qwen4_exp/modular_qwen4_exp.py`
- The blog (https://qwen.ai/blog?id=qwen3.8-flash-next) is JavaScript-rendered, so its text could not be read.

### 2a. N-gram embeddings

- It uses 2-grams and 3-grams. README: "N-gram Embedding: 20,000,000 (bigrams/trigrams at layer 2)".
- Lookup uses multi-head hashing: "We use multi-head hashing for embedding lookup." There are 8 heads per n-gram order.
  - Each head computes `id = (tok_t·m0) XOR (tok_{t−1}·m1) [XOR (tok_{t−2}·m2)]`, then takes it modulo a prime.
  - Every head has its own prime table size and its own odd multipliers.
- It enters **at layer 2**, not at the token embedding: "We place it at Layer 2". It goes into every residual stream through a per-stream sigmoid gate: `sigmoid(⟨RMSNorm(key), RMSNorm(stream)⟩ / √d)` times a shared value projection.
- The 51B parameter figure counts the whole n-gram table. Lookups do not add matrix-multiply cost per token.

**BishLM uses:** 2-grams and 3-grams, multi-head XOR hashing with prime table sizes, injection at layer 2 into every mHC stream, and the per-stream sigmoid gate. We skip Qwen's dilated conv and host-RAM offload.

### 2b. Multi-token prediction (MTP)

- It has 1 MTP layer: "MTP: 1 layer, trained with multi-steps", about 4B parameters.
- It is used for speculative decoding: the MTP layer drafts tokens, and the main model checks them.
- The loss weight is NOT stated in primary sources.
- Conflict: Table 4 says "four-step" speculative decoding, while `next_n=4` corresponds to three MTP steps.

**BishLM uses:** one MTP block that predicts the token after next (t+2), in the DeepSeek-V3 style: join the main hidden state with the embedding of token t+1, then run one block and the shared output head. MTP loss weight `0.3` is **our choice**. Speculative decoding drafts 1 token and verifies it.

---

## 3. GLM-5.3-Flash (Z.ai)

- Config: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json
- Card: https://huggingface.co/zai-org/GLM-5.3-Flash
- Docs: https://docs.z.ai/guides/vlm/glm-5.3-flash
- mHC paper (DeepSeek): https://arxiv.org/html/2512.24880
- Original Hyper-Connections paper: https://arxiv.org/abs/2409.19606

### Facts

- Size: 320B total and 18B active per the card text. Conflict: the safetensors metadata says 321B.
- 45 layers: 34 KDA and 11 DeepSeek Sparse Attention, all NoPE.
- MoE: 288 routed experts, top-8, 1 shared expert, sigmoid router.
- `mhc: true`, `hc_mult: 4`, `hc_sinkhorn_iters: 20`. Context is 1M tokens (`max_position_embeddings 1048576`).

### mHC: Manifold-Constrained Hyper-Connections

- The residual stream is widened to n copies: `x_l ∈ R^{n×C}`.
- Update rule: `x_{l+1} = H_res x_l + H_postᵀ F(H_pre x_l)`.
  - `H_pre` (1×n) mixes the n streams into one layer input.
  - `H_post` (1×n) spreads the layer output back over the streams.
  - `H_res` (n×n) mixes the streams with each other.
- The coefficients are static plus dynamic: `x′ = RMSNorm(vec(x_l))`, then `H̃ = α·(x′ φ) + b`, with α initialised to 0.01.
- Activations: `H_pre = σ(H̃_pre)`, `H_post = 2σ(H̃_post)`, and `H_res = Sinkhorn-Knopp(exp(H̃_res))` with 20 iterations.
  - Sinkhorn-Knopp makes H_res doubly stochastic: every row and every column sums to 1. That constraint (the "manifold") keeps the signal from blowing up or dying across many layers.
- GLM copies the embedding into all 4 streams at the start. At the end it collapses them with an unweighted mean: "Unlike DeepSeek-V4, this is an unweighted mean."

**BishLM uses:** mHC as above, with n = 4 streams, 20 Sinkhorn iterations, and both static and dynamic coefficients. The embedding is copied into all streams and the mean is taken at the end. The static biases start like the original Hyper-Connections paper, so that `H_res ≈ I`, layer `l` reads stream `l mod n`, and `H_post = 1`. **This init is our choice**, because the mHC paper does not state its bias init.

---

## Hugging Face `transformers` support (for reference only; BishLM does not use these)

- GLM-5.3-Flash: `models/glm5_next`
- Qwen3.8-Flash-Next: `models/qwen4_exp`
- Kimi K3: no native class; it needs `trust_remote_code`. `models/kimi_linear` exists but has no AttnRes or LatentMoE.
