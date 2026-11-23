"""Chatting with the instruction-tuned BishLM about stories.

The model was never trained on chats, only on "request -> story". So the
chat is a small loop around that:

    a request   -> a fresh start: the request in the instruction format
    "continue"  -> the same story so far, and the model writes on
    "quit"      -> stop

Each reply is a few sentences, cut at the end of a sentence, so it feels
like a chat. The model only sees its last seq_len tokens (512 after
instruction tuning), so in a long story it forgets the beginning.
"""
from .instruct import prompt_for

CONTINUE = {"continue", "go on", "more", "and then?", "what happened next?"}
QUIT = {"quit", "exit", "bye"}
ENDS = (".", "!", "?", '."', '!"', '?"')


def context_for(message, story_ids, tok):
    """-> (ids to give the model, is_new_story), or (None, False) for "continue" with no story."""
    if message.strip().lower() in CONTINUE:
        return (story_ids, False) if story_ids else (None, False)
    return tok.encode(prompt_for(message.strip())), True


def cut_at_sentence(tok, new_ids):
    """Keep the new tokens up to the last full sentence (all of them if there is none)."""
    for i in range(len(new_ids), 0, -1):
        if tok.decode(new_ids[:i]).rstrip().endswith(ENDS):
            return new_ids[:i]
    return new_ids
