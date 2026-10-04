#!/bin/bash
# Clone every upstream repo at the pinned commit into $TP. Idempotent.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$TP"
get() {  # dir repo commit
  [ -d "$TP/$1/.git" ] || git clone -q "$2" "$TP/$1"
  git -C "$TP/$1" fetch -q origin && git -C "$TP/$1" checkout -q "$3"
  echo "$1 @ $(git -C "$TP/$1" rev-parse --short HEAD)"
}
get jaxued "$JAXUED_REPO" "$JAXUED_COMMIT"
get dcd "$DCD_REPO" "$DCD_COMMIT"
get sfl "$SFL_REPO" "$SFL_COMMIT"
get baselines "$BASELINES_REPO" "$BASELINES_COMMIT"
# Local changes to jaxued: wandb.Video(format="gif") so logging works headless, and a --minimax
# flag on maze_paired.py (adversary reward = -student return; JaxUED ships no minimax baseline).
for p in gif minimax; do
  git -C "$TP/jaxued" apply --check "$ATLAS/cluster/patches/jaxued_$p.patch" 2>/dev/null \
    && git -C "$TP/jaxued" apply "$ATLAS/cluster/patches/jaxued_$p.patch" && echo "jaxued: $p patch applied" \
    || echo "jaxued: $p patch already applied"
done
