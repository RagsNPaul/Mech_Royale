#!/bin/zsh
# Runs one live-room match with our bot, then analyzes it. Output also saved to runs/live_log.txt
cd "$(dirname "$0")"
mkdir -p runs
{
  echo "=== $(date) ==="
  source .venv/bin/activate
  pip install -q websockets==13.1 certifi
  export SSL_CERT_FILE="$(python -m certifi)"
  echo "using CA bundle $SSL_CERT_FILE"
  python -c "from websockets.asyncio.client import connect; print('websockets ok')"
  python tools/live_relay.py --matches 1
  LATEST="runs/$(ls -t runs | grep -v -e live_log -e REPORT -e results -e overnight -e "\." | head -1)"
  echo "=== analyzing $LATEST ==="
  python tools/analyze_runs.py "$LATEST"
  echo "=== done $(date) ==="
} 2>&1 | tee -a runs/live_log.txt
