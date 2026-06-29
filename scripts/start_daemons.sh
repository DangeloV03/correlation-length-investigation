#!/usr/bin/env bash
# Start coex run_all and analyzer in a detached tmux session.
#
# Usage (on Della login node):
#   ./scripts/start_daemons.sh
#   tmux attach -t coex-campaign
#   Ctrl-b d                            # detach without stopping

set -euo pipefail

SESSION="coex-campaign"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DAEMON_SETUP='module load anaconda3/2024.10 2>/dev/null; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1; export PYTHONPATH="'"${PROJECT_DIR}"':${PYTHONPATH:-}"'

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  stop:    ./scripts/stop_daemons.sh"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n run_all -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:run_all" \
  "${DAEMON_SETUP}; python -u -m coex.run_all" C-m

tmux new-window -t "$SESSION" -n analyzer -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:analyzer" \
  "${DAEMON_SETUP}; python -u -m coex.analyzer --depth-first" C-m

echo "Started tmux session '$SESSION' with windows: run_all, analyzer"
echo "  attach:  tmux attach -t $SESSION"
echo "  list:    tmux list-windows -t $SESSION"
echo "  detach:  Ctrl-b then d"
