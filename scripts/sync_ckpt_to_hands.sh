#!/usr/bin/env bash
# Copy one checkpoint's pretrained_model to HANDS: ~/policies/<EXP_ID>/<step>/
#   scripts/sync_ckpt_to_hands.sh E001 runs/act_orange_2cam_b8 [last|070000] [--go]
# Default is a dry run (rsync -n). Pass --go to actually transfer.
set -euo pipefail
EXP=${1:?exp id, e.g. E001}; RUN=${2:?run dir}; STEP=${3:-last}; GO=${4:-}
SRC="$RUN/checkpoints/$STEP/pretrained_model"
[ -d "$SRC" ] || { echo "no such checkpoint: $SRC" >&2; exit 1; }
REAL_STEP=$(basename "$(cd "$RUN/checkpoints/$STEP" && pwd -P)")
ssh -o BatchMode=yes -o ConnectTimeout=5 hands true || { echo "ssh hands failed — ask user for MacBook relay path" >&2; exit 2; }
DEST="hands:~/policies/$EXP/$REAL_STEP/"
FLAGS=(-avh --progress)
[ "$GO" = "--go" ] || { FLAGS+=(-n); echo "[dry-run] add --go to transfer"; }
ssh hands "mkdir -p ~/policies/$EXP/$REAL_STEP"
rsync "${FLAGS[@]}" "$SRC/" "$DEST"
[ "$GO" = "--go" ] && ssh hands "cd ~/policies/$EXP/$REAL_STEP && ls -la && du -sh ."
