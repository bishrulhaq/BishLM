#!/bin/sh
# Runs after overnight.sh: the live 10-minute run, the five one-idea-off runs
# (plus the full small model, all with the same steps), and the speed test.
cd "$(dirname "$0")"
while pgrep -f "overnight.sh" >/dev/null; do sleep 60; done
run() {  # run <preset> <out> <extra args...>
  p="$1"; o="$2"; shift 2
  for try in 1 2 3 4 5; do
    python3 -u train.py --preset "$p" --out "$o" --resume "$@" >> "$o.log" 2>&1 && return 0
    echo "crash $try at $(date), restarting in 60 s" >> "$o.log"; sleep 60
  done
}
run small runs/live --minutes 10 --steps 2000 --eval-every 300
for p in small small-no-moe small-no-linear small-no-ngram small-no-mtp small-no-mhc; do
  run "$p" "runs/$p" --steps 2000 --eval-every 500
done
python3 sample.py --ckpt runs/bishlm/ckpt.pt --prompt "Once upon a time" > runs/story-bishlm.txt 2>&1
python3 sample.py --ckpt runs/plain/ckpt.pt --prompt "Once upon a time" > runs/story-plain.txt 2>&1
python3 sample.py --ckpt runs/bishlm/ckpt.pt --compare --tokens 200 > runs/speed-bishlm.txt 2>&1
echo "all done $(date)" > runs/followup.done
