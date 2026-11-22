#!/bin/zsh
# Overnight live-room search. Runs from a FROZEN copy of the current commit, so editing code
# tonight can't change what the matches are testing. Stops at 07:00. Keeps the Mac awake.
cd "$(dirname "$0")"
REPO="$PWD"
STAMP=$(date +%Y%m%d-%H%M)
OUT="$REPO/runs/overnight-$STAMP"
WT="/tmp/mech-overnight-$STAMP"
mkdir -p "$OUT"
{
  echo "=== overnight start $(date) commit $(git rev-parse --short HEAD) ==="
  if ! git diff --quiet HEAD -- bot main.py; then
    echo "WARNING: uncommitted changes in bot/ or main.py are NOT included; commit first if you want them tested."
  fi
  git worktree add --detach "$WT" HEAD >/dev/null 2>&1 || { echo "worktree failed"; exit 1; }
  source "$REPO/.venv/bin/activate"
  pip install -q websockets==13.1 certifi
  export SSL_CERT_FILE="$(python -m certifi)"
  cd "$WT"
  caffeinate -dimsu python tools/overnight.py --until 07:00 --out "$OUT"
  echo "=== overnight end $(date) ==="
  cp "$OUT/REPORT.md" "$REPO/runs/REPORT.md" 2>/dev/null
} 2>&1 | tee "$OUT/overnight.log"
