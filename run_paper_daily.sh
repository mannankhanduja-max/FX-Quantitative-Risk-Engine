#!/bin/bash
#
# Daily paper run, for launchd.
#
# Credentials live in ~/.alpaca_env, OUTSIDE this repository, mode
# 600. That file is the only place they exist on disk. It is not in
# the repo, not in git, and not in your shell history.
#
#   printf 'export ALPACA_API_KEY_ID=%s\nexport ALPACA_API_SECRET_KEY=%s\n' \
#     'YOUR_KEY' 'YOUR_SECRET' > ~/.alpaca_env
#   chmod 600 ~/.alpaca_env
#
# Install the schedule:
#   cp com.mannan.paperfx.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.mannan.paperfx.plist
#
# Stop it:
#   launchctl unload ~/Library/LaunchAgents/com.mannan.paperfx.plist

set -uo pipefail

REPO="$HOME/fx-risk-engine"
ENV_FILE="$HOME/.alpaca_env"
LOG_DIR="$REPO/results"
LOG="$LOG_DIR/paper_daily.log"

mkdir -p "$LOG_DIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

{
  echo "===== $(stamp) ====="

  if [ ! -f "$ENV_FILE" ]; then
    echo "no $ENV_FILE - create it first (see the header of this script)"
    exit 1
  fi

  # Refuse to run if the credential file is readable by anyone else.
  perms=$(stat -f "%Lp" "$ENV_FILE")
  if [ "$perms" != "600" ]; then
    echo "$ENV_FILE has mode $perms, expected 600. Refusing."
    echo "  chmod 600 $ENV_FILE"
    exit 1
  fi

  # shellcheck source=/dev/null
  . "$ENV_FILE"

  cd "$REPO" || { echo "no repo at $REPO"; exit 1; }

  # The signal comes from the local consolidated cache, so refresh it
  # before deciding anything. If the fetch fails, do NOT trade on a
  # stale cache - a signal computed from last week's bars is worse
  # than no signal.
  if ! python3 fetch_data.py >/dev/null 2>&1; then
    echo "fetch_data.py failed - not trading on a stale cache"
    exit 1
  fi

  python3 paper_trade.py --submit
  echo "--- reconcile ---"
  python3 paper_trade.py --reconcile

  echo "exit=$?"
} >> "$LOG" 2>&1
