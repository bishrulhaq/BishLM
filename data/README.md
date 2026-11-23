# data/

This folder is empty in the repo. It fills up in one of two ways.

**Try the trained model** (no training needed):

```bash
python download_model.py
```

This downloads the trained BishLM from
[bishrulhaq.com/downloads/bishlm-v1.zip](https://bishrulhaq.com/downloads/bishlm-v1.zip)
(about 99 MB) and puts `tokenizer.json` here and the two models in `models/`.

**Train your own:**

```bash
python prepare_data.py
```

This downloads TinyStories (about 2.2 GB of text), trains the tokenizer and writes
`tokenizer.json`, `train.bin` and `valid.bin` here (about 3.4 GB in all). See the main README
for the steps after that.
