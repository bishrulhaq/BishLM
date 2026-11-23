"""Writing text with the trained model, with and without speculative decoding.

Normal decoding: one forward pass of the model -> one new token.

Speculative decoding with the MTP head: the main model gives token t+1,
and in the same pass the MTP head guesses token t+2 (a "draft"). In the
next pass we feed the draft in too. If the main model agrees with the
draft, we get two tokens from one pass. If not, we throw the draft away.
Accepting only when the model's own choice equals the draft keeps the
output exactly the same as normal decoding.
"""
import torch


def pick(logits, temperature):
    """Choose the next token from logits (V,)."""
    if temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


def penalize(logits, recent, penalty):
    """Make tokens we used a moment ago less likely, so a small model does not get
    stuck saying "free free free". 1.0 = off."""
    if penalty == 1.0 or not recent:
        return logits
    logits = logits.clone()
    ids = torch.tensor(sorted(set(recent)), device=logits.device)
    seen = logits[ids]
    logits[ids] = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits


@torch.no_grad()
def generate(model, idx, n_new, temperature=0.8, stop_at_eot=True, repeat_penalty=1.0):
    """Normal decoding. idx: (1, T) prompt ids -> (1, T + up to n_new) ids."""
    with model.eager():
        return _generate(model, idx, n_new, temperature, stop_at_eot, repeat_penalty)


def _generate(model, idx, n_new, temperature, stop_at_eot, repeat_penalty=1.0):
    was_training = model.training
    model.eval()
    ids = idx[0].tolist()
    eot = model.cfg.vocab_size - 1
    for _ in range(n_new):
        x = torch.tensor([ids[-model.cfg.seq_len:]], device=idx.device)
        logits, _, _ = model(x)
        nxt = pick(penalize(logits[0, -1], ids[-64:], repeat_penalty), temperature)
        ids.append(nxt)
        if stop_at_eot and nxt == eot:
            break
    model.train(was_training)
    return torch.tensor([ids], device=idx.device)


@torch.no_grad()
def generate_speculative(model, idx, n_new, temperature=0.8, stop_at_eot=True):
    """Speculative decoding with the MTP head. Returns (ids, stats)."""
    with model.eager():
        return _generate_speculative(model, idx, n_new, temperature, stop_at_eot)


def _generate_speculative(model, idx, n_new, temperature, stop_at_eot):
    assert model.mtp is not None, "this model has no MTP head"
    was_training = model.training
    model.eval()
    ids = idx[0].tolist()
    start, L = len(ids), model.cfg.seq_len
    eot = model.cfg.vocab_size - 1
    draft, passes, accepted = None, 0, 0
    while len(ids) - start < n_new:
        inp = ids + ([draft] if draft is not None else [])
        cut = max(0, len(inp) - L)
        x = torch.tensor([inp[cut:]], device=idx.device)
        h = model.hidden(x)                                  # (1, T, d)
        logits = model.head(model.norm(h))[0]
        passes += 1
        pos = len(ids) - 1 - cut                             # position that predicts the next token
        new = [pick(logits[pos], temperature)]
        if draft is not None and new[0] == draft:            # the model agrees with the draft
            accepted += 1
            new.append(pick(logits[pos + 1], temperature))   # and we get one more for free
        ids.extend(new)
        if stop_at_eot and eot in new:
            ids = ids[:len(ids) - len(new) + new.index(eot) + 1]
            break
        # new draft: MTP at the last position joins h (up to the newest token's
        # predecessor) with the embeddings of the tokens that came after
        n_h = len(ids) - 1 - cut                             # hidden states we can use
        hs = h[:, :n_h][:, -L:]
        nxt = torch.tensor([ids[cut + 1:cut + 1 + n_h]], device=idx.device)[:, -L:]
        m = model.mtp(hs, model.embed(nxt))
        draft = pick(model.head(m)[0, -1], temperature)
    model.train(was_training)
    made = len(ids) - start
    return torch.tensor([ids], device=idx.device), {
        "tokens": made, "passes": passes, "accepted": accepted,
        "tokens_per_pass": made / max(1, passes)}
