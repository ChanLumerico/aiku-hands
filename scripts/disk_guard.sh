#!/usr/bin/env bash
# Report disk headroom and intermediate checkpoints. Never deletes anything — only suggests.
#   scripts/disk_guard.sh [min_free_gb=5]
set -uo pipefail
MIN=${1:-5}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FREE_GB=$(df -g ~ | awk 'NR==2{print $4}')
df -h ~ | sed -n 1,2p
echo "free: ${FREE_GB} GB (threshold ${MIN} GB)"
for run in "$ROOT"/runs/*/; do
  [ -d "$run/checkpoints" ] || continue
  echo "== $(basename "$run")"
  last=$(readlink "$run/checkpoints/last" 2>/dev/null || true)
  for ck in "$run"/checkpoints/[0-9]*; do
    sz=$(du -sh "$ck" | cut -f1)
    tag=""; [ "$(basename "$ck")" = "$(basename "${last:-x}")" ] && tag=" (last — keep)"
    echo "  $(basename "$ck")  $sz$tag"
  done
done
if [ "${FREE_GB:-0}" -lt "$MIN" ]; then
  echo "⚠️  free < ${MIN} GB: STOP and ask the user. Candidates = non-last intermediate checkpoints above (confirm before rm)."
  exit 1
fi
