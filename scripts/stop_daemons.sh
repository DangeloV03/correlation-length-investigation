#!/usr/bin/env bash
# Stop coex and correlation tmux sessions.
#
# Usage:
#   ./scripts/stop_daemons.sh          # kill tmux sessions only
#   ./scripts/stop_daemons.sh --slurm  # also scancel your flex_sim jobs

set -euo pipefail

for SESSION in coex-campaign correlation-campaign flex-investigation; do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Killed tmux session '$SESSION'"
  fi
done

if pkill -f "python.*coex.run_all" 2>/dev/null; then
  echo "Killed stray coex.run_all"
fi
if pkill -f "python.*coex.analyzer" 2>/dev/null; then
  echo "Killed stray coex.analyzer"
fi
if pkill -f "python.*correlation.run_all" 2>/dev/null; then
  echo "Killed stray correlation.run_all"
fi

if [[ "${1:-}" == "--slurm" ]]; then
  scancel -u "$(whoami)" -n flex_sim 2>/dev/null || scancel -u "$(whoami)"
  echo "Cancelled Slurm flex_sim jobs for $(whoami)"
fi
