"""Checks for instruction tuning.   Run:  python -m pytest -q"""
import random

from bishlm.instruct import encode_example, make_request, parse, prompt_for
from bishlm.tokenizer import Tokenizer

ENTRY = """Features: Dialogue
Words: plane, excited, holiday
Summary: Tom and Anna fly on a big plane to a sunny place.
Story:
Tom and Anna are brother and sister.
"Look, a plane!" Tom says."""


def test_parse_reads_every_field():
    f = parse(ENTRY)
    assert f["features"] == "Dialogue"
    assert f["words"] == "plane, excited, holiday"
    assert f["summary"].startswith("Tom and Anna fly")
    assert f["story"] == 'Tom and Anna are brother and sister.\n"Look, a plane!" Tom says.'


def test_story_can_come_before_the_summary():
    f = parse("Story: Sam had a tower of books.\nIt fell down.\nSummary: A tower falls.")
    assert f["story"] == "Sam had a tower of books.\nIt fell down."
    assert f["summary"] == "A tower falls."


def test_request_uses_the_notes():
    f = parse(ENTRY)
    reqs = {make_request(f, random.Random(i)) for i in range(40)}
    assert any("uses the words: plane, excited, holiday" in r for r in reqs)
    assert any(r.startswith("Write a story about this: Tom and Anna") for r in reqs)
    assert any("Include dialogue." in r for r in reqs)


def test_only_the_story_is_graded():
    text = prompt_for("Write a story.") + "Sam was happy."
    tok = Tokenizer.train(text * 30, vocab_size=300, verbose=False)
    ids, mask = encode_example(tok, {"story": "Sam was happy."}, random.Random(0))
    n_prompt = mask.index(1)
    assert len(ids) == len(mask)
    assert tok.decode(ids[:n_prompt]).startswith("### Instruction:")
    assert tok.decode(ids[n_prompt:]) == "Sam was happy."      # decode drops the end token
    assert all(m == 1 for m in mask[n_prompt:])
