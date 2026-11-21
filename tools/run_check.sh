#!/usr/bin/env bash
# Local conformance check: plain python, then the real sandboxed docker run (what the scorer does).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 check/check.py -- python3 -u main.py
docker build --platform linux/amd64 -t mech-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --platform linux/amd64 --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 \
  -v "$PWD/arena:/arena:ro" mech-bot
