#!/bin/sh
# Train BishLM, then the plain model, with the same steps and data.
# A crash (a GPU error, the drive unplugged) restarts from the last checkpoint.
cd "$(dirname "$0")"
run() {  # run <preset> <out>
  for try in 1 2 3 4 5 6 7 8 9 10; do
    python3 -u train.py --preset "$1" --steps 45000 --eval-every 1000 --out "$2" --resume >> "$2.log" 2>&1 && return 0
    echo "crash $try at $(date), restarting in 60 s" >> "$2.log"
    sleep 60
  done
}
mkdir -p runs
run bishlm runs/bishlm
run plain runs/plain
