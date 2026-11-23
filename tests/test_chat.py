"""Checks for the story chat.   Run:  python -m pytest -q"""
from bishlm.chat import context_for, cut_at_sentence
from bishlm.instruct import prompt_for
from bishlm.tokenizer import Tokenizer

TOK = Tokenizer.train((prompt_for("Write a story.") + "Sam was happy. He ran to the park and") * 30,
                      vocab_size=300, verbose=False)


def test_a_request_starts_a_new_story():
    ids, new = context_for("Write a story.", [1, 2, 3], TOK)
    assert new
    assert TOK.decode(ids) == prompt_for("Write a story.")


def test_continue_keeps_the_story():
    ids, new = context_for("continue", [1, 2, 3], TOK)
    assert not new and ids == [1, 2, 3]


def test_continue_after_the_end_asks_for_a_new_story():
    ids, new = context_for("continue", [], TOK)
    assert ids is None and not new


def test_replies_end_at_a_full_sentence():
    ids = TOK.encode("Sam was happy. He ran to the park and")
    assert TOK.decode(cut_at_sentence(TOK, ids)) == "Sam was happy."


def test_the_repeat_penalty_lowers_used_tokens_only():
    import torch
    from bishlm.generate import penalize
    logits = torch.tensor([2.0, -1.0, 3.0])
    out = penalize(logits, [0, 1], 2.0)
    assert out.tolist() == [1.0, -2.0, 3.0]
    assert penalize(logits, [0], 1.0) is logits
