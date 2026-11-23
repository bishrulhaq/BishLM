"""Instruction tuning: teach the story model to follow a request.

TinyStories-Instruct (by the TinyStories authors) puts notes in front of
every story:

    Features: Dialogue
    Words: plane, excited, holiday
    Summary: Tom and Anna fly on a big plane to a sunny place.
    Story:
    Tom and Anna are brother and sister. ...

We turn those notes into a plain request, the way chat models are trained:

    ### Instruction:
    Write a story that uses the words: plane, excited, holiday. Include dialogue.
    ### Story:
    Tom and Anna are brother and sister. ...

During training, only the story is graded (mask = 1); the request is not
(mask = 0). The model learns to answer requests, not to write them.
"""
import random

FIELDS = ("Features", "Words", "Summary", "Random sentence", "Story")
FEATURES = {
    "dialogue": "dialogue", "twist": "a twist", "foreshadowing": "foreshadowing",
    "badending": "a sad ending", "moralvalue": "a lesson", "conflict": "a problem to solve",
}


def parse(entry):
    """One TinyStories-Instruct entry -> {'features', 'words', 'summary', 'story'}."""
    out = {}
    current = None
    story = []
    for line in entry.strip().split("\n"):
        key = next((f for f in FIELDS if line.startswith(f + ":")), None)
        if key:
            current = key
            value = line[len(key) + 1:].strip()
            if key == "Story":
                if value:
                    story.append(value)
            else:
                out[key.lower().replace(" ", "_")] = value
        elif current == "Story":
            story.append(line)
    out["story"] = "\n".join(story).strip()
    return out


def prompt_for(request):
    """The text we give the model before it writes."""
    return f"### Instruction:\n{request}\n### Story:\n"


def make_request(f, rng=random):
    """Build a request from the notes; mixes short and long kinds."""
    words = f.get("words", "")
    summary = f.get("summary", "")
    feats = [FEATURES.get(x.strip().lower().replace(" ", ""), "") for x in f.get("features", "").split(",")]
    feats = [x for x in feats if x]
    kind = rng.random()
    if words and (kind < 0.5 or not summary):
        req = f"Write a story that uses the words: {words}."
    elif summary:
        req = f"Write a story about this: {summary}"
    else:
        req = "Write a story."
    if feats and rng.random() < 0.6:
        req += " Include " + " and ".join(feats) + "."
    return req


def encode_example(tok, f, rng=random):
    """-> (ids, mask): the request (mask 0), then the story and end token (mask 1)."""
    p = tok.encode(prompt_for(make_request(f, rng)))
    s = tok.encode(f["story"]) + [tok.eot_id]
    return p + s, [0] * len(p) + [1] * len(s)
