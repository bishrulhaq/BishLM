"""Checks for every part of the model.   Run:  python -m pytest -q

Each test proves one claim we make in the video.
"""
import os

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)


# --- tokenizer -------------------------------------------------------------------

from bishlm.tokenizer import Tokenizer


def test_tokenizer_round_trip():
    text = "Once upon a time, a little cat named Tom found a red ball. 123!"
    tok = Tokenizer.train(text * 50, vocab_size=300, verbose=False)
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(text)) < len(text.encode("utf-8"))      # merges make it shorter


@pytest.mark.skipif(not os.path.exists("data/tokenizer.json"), reason="run prepare_data.py first")
def test_trained_tokenizer():
    tok = Tokenizer.load("data/tokenizer.json")
    assert tok.vocab_size == 4096
    s = "The dog was happy. He ran to the park with his friend."
    assert tok.decode(tok.encode(s)) == s



# --- MoE ---------------------------------------------------------------------------------

from bishlm.config import SMALL
from bishlm.moe import MoE


def test_moe_matches_simple_loop():
    m = MoE(SMALL, capacity_factor=8.0)                              # room for every token
    x = torch.randn(4, 64, SMALL.d_model)
    assert torch.allclose(m(x), m.forward_reference(x), atol=1e-6)


def test_router_weights_sum_to_one():
    m = MoE(SMALL)
    top, w = m.route(torch.randn(100, SMALL.d_model))
    assert top.shape == (100, SMALL.top_k)
    assert torch.allclose(w.sum(-1), torch.ones(100))
    assert (top[:, 0] != top[:, 1]).all()                            # two different experts


def test_bias_balances_the_load():
    m = MoE(SMALL)
    m.bias[0] = 0.5                                                  # start unfair: expert 0 wins almost always
    x = torch.randn(2048, SMALL.d_model)
    m.train()
    m(x)
    before = m.load[0].item()
    for _ in range(200):
        m(x)
        m.balance()
    assert m.load[0].item() < before * 0.5



# --- KDA (linear attention) ------------------------------------------------------------

from bishlm.kda import kda_chunked, kda_recurrent, unit_lower_inverse


def kda_inputs(T=77, dtype=torch.float64):
    B, H, dk, dv = 2, 3, 8, 5
    q = torch.randn(B, H, T, dk, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, dk, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, dv, dtype=dtype)
    g = -5 * torch.rand(B, H, T, dk, dtype=dtype)
    beta = torch.rand(B, H, T, dtype=dtype)
    return q, k, v, g, beta


def test_unit_lower_inverse():
    M = torch.randn(4, 32, 32).tril(-1) * 0.3
    I = torch.eye(32)
    assert torch.allclose(unit_lower_inverse(M) @ (I + M), I.expand(4, 32, 32), atol=1e-4)


def test_kda_chunked_equals_recurrent():
    q, k, v, g, beta = kda_inputs()
    o1, S1 = kda_recurrent(q, k, v, g, beta)
    o2, S2 = kda_chunked(q, k, v, g, beta, chunk=32)
    assert torch.allclose(o1, o2, atol=1e-10) and torch.allclose(S1, S2, atol=1e-10)


def test_kda_float32_worst_case_gradients():
    # strongest forgetting everywhere: the case that could overflow
    q, k, v, g, beta = kda_inputs(T=64, dtype=torch.float32)
    g = torch.full_like(g, -5.0)
    xs = [t.requires_grad_() for t in (q, k, v, g, beta)]
    o, _ = kda_chunked(*xs, chunk=32)
    o.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in xs)


def test_kda_memory_is_fixed_size():
    q, k, v, g, beta = kda_inputs(T=200)
    _, S = kda_recurrent(q, k, v, g, beta)
    assert S.shape == (2, 3, 8, 5)                                   # does not grow with T



# --- n-gram embeddings --------------------------------------------------------------------

from bishlm.ngram import NGramEmbedding, is_prime


def test_ngram_hashing():
    ng = NGramEmbedding(SMALL)
    assert all(is_prime(p) for p in ng.primes)
    a = torch.tensor([[5, 9, 7, 5, 9, 7]])
    ids = ng.hash_ids(a)
    # same 3 tokens in a row -> same row in every head (positions 2 and 5)
    assert all(int(h[0, 2]) == int(h[0, 5]) for h in ids)
    assert all((h >= 0).all() and (h < p).all() for h, p in zip(ids, ng.primes))


def test_ngram_starts_with_no_effect():
    ng = NGramEmbedding(SMALL)
    X = torch.randn(1, 6, 4, SMALL.d_model)
    e = ng.lookup(torch.randint(0, 4096, (1, 6)))
    assert torch.equal(ng.inject(X, e), X)                           # value projection starts at zero



# --- mHC -------------------------------------------------------------------------------------

from bishlm.mhc import HyperConnection, sinkhorn


def test_sinkhorn_doubly_stochastic():
    M = sinkhorn(torch.randn(10, 4, 4), iters=20)
    assert torch.allclose(M.sum(-1), torch.ones(10, 4), atol=1e-3)
    assert torch.allclose(M.sum(-2), torch.ones(10, 4), atol=1e-6)
    assert (M >= 0).all()


def test_mhc_starts_close_to_identity():
    hc = HyperConnection(SMALL, torch.nn.Identity(), layer_index=0)
    X = torch.randn(2, 5, 4, SMALL.d_model)
    _, _, H_res = hc.coefficients(X)
    assert torch.allclose(H_res, torch.eye(4).expand_as(H_res), atol=0.1)



# --- the whole model -------------------------------------------------------------------------

from bishlm.config import PRESETS
from bishlm.generate import generate, generate_speculative
from bishlm.model import BishLM


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs(name):
    cfg = PRESETS[name]
    m = BishLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    logits, loss, _ = m(x, x)
    assert logits.shape == (2, 40, cfg.vocab_size) and torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("name", ["small", "small-plain"])
def test_model_cannot_see_the_future(name):
    """Changing token 30 must not change any prediction before position 30."""
    m = BishLM(PRESETS[name]).eval()
    with torch.no_grad():
        for p in m.modules():                                        # make the n-gram path active
            if isinstance(p, NGramEmbedding):
                p.value.weight.normal_(0, 0.02)
        x = torch.randint(0, 4096, (1, 50))
        y = x.clone()
        y[0, 30] = (y[0, 30] + 1) % 4096
        a, b = m(x)[0], m(y)[0]
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5)
    assert not torch.allclose(a[:, 30:], b[:, 30:], atol=1e-5)


def test_speculative_decoding_gives_the_same_text():
    m = BishLM(SMALL).eval()
    prompt = torch.randint(0, 4095, (1, 8))
    normal = generate(m, prompt, 30, temperature=0, stop_at_eot=False)
    fast, stats = generate_speculative(m, prompt, 30, temperature=0, stop_at_eot=False)
    assert torch.equal(normal[0, :38], fast[0, :38])
    assert stats["passes"] <= 30
