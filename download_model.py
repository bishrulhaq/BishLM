"""Download the trained BishLM, so you can try it without training it yourself.

    python download_model.py
    python chat.py --ckpt models/instruct.pt

It downloads bishlm-v1.zip (about 99 MB) from bishrulhaq.com, checks its SHA-256,
and unpacks it here:

    models/bishlm.pt      the base model (writes stories)
    models/instruct.pt    the instruction-tuned model (follows requests, story chat)
    data/tokenizer.json   the tokenizer both models need

Only the Python standard library is used.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://bishrulhaq.com/downloads/bishlm-v1.zip"
SHA256 = "7a440456965c188b975e2d0351cd095b62db163c38d479aee41a47d3c6126245"
FILES = ["models/bishlm.pt", "models/instruct.pt", "data/tokenizer.json"]

p = argparse.ArgumentParser()
p.add_argument("--url", default=URL, help="where to download the zip from")
p.add_argument("--keep-zip", action="store_true", help="keep the zip after unpacking")
args = p.parse_args()

root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)
zip_path = "bishlm-v1.zip"


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "BishLM-download/1.0"})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:5.1f} of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(path + ".part", path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


if not (os.path.exists(zip_path) and sha256(zip_path) == SHA256):
    print(f"downloading {args.url}")
    download(args.url, zip_path)

if sha256(zip_path) != SHA256:
    os.remove(zip_path)
    sys.exit("the download is damaged (SHA-256 does not match). Please run this again.")
print("checksum ok")

with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    # unpack only the files we expect, so a bad zip cannot write anywhere else
    for name in FILES:
        if name not in names:
            sys.exit(f"{name} is missing from the zip")
    old_tok = "data/tokenizer.json"
    if os.path.exists(old_tok) and open(old_tok, "rb").read() != z.read(old_tok):
        os.replace(old_tok, "data/tokenizer.mine.json")
        print("your own data/tokenizer.json differs: kept it as data/tokenizer.mine.json")
    for name in FILES:
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with z.open(name) as src, open(name, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        print(f"  {name}")

if not args.keep_zip:
    os.remove(zip_path)
print("done. Try:  python chat.py --ckpt models/instruct.pt")
