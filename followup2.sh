#!/bin/sh
# After followup.sh: instruction tuning of the finished BishLM (with a 512-token window),
# then before/after samples, the story chat and the length check.
cd "$(dirname "$0")"
while [ ! -f runs/followup.done ]; do sleep 60; done
for try in 1 2 3 4 5; do
  # --no-compile: at 512 tokens torch.compile makes a kernel too big for Apple GPUs
  python3 -u train.py --preset bishlm --init runs/bishlm/ckpt.pt --data instruct --mix 0.2 --seq-len 512 --batch 8 \
    --no-compile --steps 2000 --lr 3e-4 --warmup 50 --eval-every 500 --out runs/instruct --resume >> runs/instruct.log 2>&1 && break
  echo "crash $try at $(date), restarting in 60 s" >> runs/instruct.log; sleep 60
done
rm -f runs/instruct-before.txt runs/instruct-after.txt
for req in "Write a story that uses the words: dragon, ice cream, shop." "Write a story about a girl who finds a lost puppy. Include dialogue."; do
  echo "### $req" >> runs/instruct-before.txt
  python3 sample.py --ckpt runs/bishlm/ckpt.pt --instruct "$req" --tokens 160 >> runs/instruct-before.txt 2>&1
  echo "### $req" >> runs/instruct-after.txt
  python3 sample.py --ckpt runs/instruct/ckpt.pt --instruct "$req" --tokens 200 >> runs/instruct-after.txt 2>&1
done
python3 sample.py --ckpt runs/instruct/ckpt.pt --prompt "Once upon a time" > runs/instruct-still-stories.txt 2>&1
for seed in 0 1 2; do
  python3 chat.py --ckpt runs/instruct/ckpt.pt --seed $seed < chat-demo.txt > runs/chat-demo-$seed.txt 2>&1
done
python3 check_length.py --ckpt runs/bishlm/ckpt.pt --length 1024 > runs/length-bishlm.txt 2>&1
python3 check_length.py --ckpt runs/plain/ckpt.pt --length 1024 > runs/length-plain.txt 2>&1
echo "all done $(date)" > runs/followup2.done
