#!/bin/bash
# Clones the upstream repos at the pinned commits and applies the patches. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$TP"
get() {  # dir repo commit
  [ -d "$TP/$1/.git" ] || git clone -q "$2" "$TP/$1"
  git -C "$TP/$1" fetch -q origin && git -C "$TP/$1" checkout -q "$3"
  echo "$1 @ $(git -C "$TP/$1" rev-parse --short HEAD)"
}
get jaxued "$JAXUED_REPO" "$JAXUED_COMMIT"
get kinetix "$KINETIX_REPO" "$KINETIX_COMMIT"
get sfl "$SFL_REPO" "$SFL_COMMIT"
# jaxued: headless GIF logging, and a --minimax flag (JaxUED has no minimax baseline)
for p in gif minimax; do
  git -C "$TP/jaxued" apply --check "$ATLAS/cluster/patches/jaxued_$p.patch" 2>/dev/null \
    && git -C "$TP/jaxued" apply "$ATLAS/cluster/patches/jaxued_$p.patch" && echo "jaxued: $p patch applied" \
    || echo "jaxued: $p patch already applied"
done
# kinetix: fixes for upstream bugs on jax 0.9.  sfl: a minimax baseline for JaxNav.
apply_once() {  # repo patch
  if git -C "$TP/$1" apply --check "$ATLAS/cluster/patches/$2" 2>/dev/null; then
    git -C "$TP/$1" apply "$ATLAS/cluster/patches/$2" && echo "$1: $2 applied"
  elif git -C "$TP/$1" apply --reverse --check "$ATLAS/cluster/patches/$2" 2>/dev/null; then
    echo "$1: $2 already applied"
  else
    echo "ERROR: $2 neither applies nor is already applied to $TP/$1"; exit 1
  fi
}
apply_once kinetix kinetix_acl27.patch
apply_once sfl sfl_minimax.patch
