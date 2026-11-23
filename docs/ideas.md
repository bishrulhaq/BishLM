# The five ideas, in plain words

Each section covers the problem, the idea, how BishLM does it, and the switch that turns it off. The facts about the real models, with links, are in [sources.md](sources.md).

---

## 1. Mixture of experts (from Kimi K3)

**Problem.** A bigger feed-forward layer knows more, but every token has to pay for all of it.

**Idea.** Keep many small feed-forward layers ("experts"). A small router gives every expert a score for each token, and only the best few run. The model holds many parameters, but each token only uses a few of them. Kimi K3 has 896 experts and uses 16 per token.

**Balancing.** If one expert becomes everyone's favourite, the others never learn. Kimi K3 uses no extra loss for this. Each expert has a bias that only affects the *choice*. After every step, busy experts get a lower bias and idle ones a higher bias. The weights that mix the experts' outputs never include the bias.

**BishLM:** 8 experts, 2 per token, plus 1 shared expert that every token uses. The first layer stays dense. Code: [bishlm/moe.py](../bishlm/moe.py). Switch: `use_moe`.

---

## 2. Linear attention mixed 3:1 (from Kimi K3)

**Problem.** Full attention compares each new token with *every* earlier token. Its memory (the KV cache) grows with the text, and so does its compute.

**Idea.** Kimi Delta Attention keeps one fixed-size matrix S per head, like a small notebook. For every token it:
1. forgets a little (each channel has its own forget gate),
2. reads what the notebook already says for this key,
3. writes only the *correction* (new value minus old value),
4. reads the answer for the query.

The memory never grows. To keep exact recall, every 4th layer is still full attention (3 KDA : 1 full). Those full layers use no position encoding at all, because the KDA layers already know the order.

**Making it fast.** The loop over tokens is slow on a GPU. The chunked version does 32 tokens at once with matrix multiplies and gives the same numbers (a test checks this).

**BishLM:** layers 1–3, 5–7 use KDA; layers 4 and 8 use full attention. Code: [bishlm/kda.py](../bishlm/kda.py). Switch: `use_linear_attn`.

---

## 3. N-gram embeddings (from Qwen3.8-Flash-Next)

**Problem.** A token embedding only knows one token. Common phrases like "once upon a" have to be rebuilt from scratch in the first layers every time.

**Idea.** Also give each 2-gram and 3-gram its own vector. There are far too many to store, so each n-gram is hashed into a table, like lockers that several n-grams share. Several hash heads, each with its own hash, make it very unlikely that two n-grams share a locker in every head. A lookup only reads rows, so it adds parameters but almost no compute.

**BishLM:** 2 heads for 2-grams and 2 heads for 3-grams, with tables of about 30,000 rows. The result is added at layer 2 into every residual lane, through a gate. Code: [bishlm/ngram.py](../bishlm/ngram.py). Switch: `use_ngram`.

---

## 4. Multi-token prediction (from Qwen3.8-Flash-Next)

**Problem.** A language model writes one token per pass through the whole network.

**Idea.** Add a small extra head that guesses the token *after* the next one. During training this is an extra lesson: plan one step further. When writing, the guess becomes a draft. The next pass checks the draft for free, and if the model agrees we get two tokens from one pass. The text is exactly the same as normal decoding (a test checks this).

**BishLM:** one MTP block (attention + feed-forward), with loss weight 0.3. Code: `MTPHead` in [bishlm/model.py](../bishlm/model.py), decoding in [bishlm/generate.py](../bishlm/generate.py). Switch: `use_mtp`.

---

## 5. mHC hyper-connections (from GLM-5.3-Flash)

**Problem.** In a deep model, everything flows through one residual "lane". Each layer adds its output to that single lane.

**Idea.** Widen the lane into 4 lanes. Each layer reads a mix of the lanes (H_pre), writes its output back to them (H_post), and the lanes mix with each other (H_res). Without a rule, H_res could slowly blow the signal up or shrink it over many layers. The rule ("manifold constraint") is that every row and every column of H_res sums to 1. Sinkhorn-Knopp gets there by normalising rows, then columns, 20 times.

**BishLM:** 4 lanes and 20 Sinkhorn steps. At the start H_res is close to the identity, so the model begins like a normal residual model. At the end, the lanes are averaged. Code: [bishlm/mhc.py](../bishlm/mhc.py). Switch: `use_mhc`.
