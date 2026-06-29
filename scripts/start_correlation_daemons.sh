#!/usr/bin/env bash
# Start correlation production dispatcher in tmux.
#
# Usage:
#   python -m correlation.generate_jobs   # run first
#   ./scripts/start_correlation_daemons.sh

set -euo pipefail

SESSION="correlation-campaign"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DAEMON_SETUP='module load anaconda3/2024.10 2>/dev/null; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate lattice; export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; export PYTHONUNBUFFERED=1; export PYTHONPATH="'"${PROJECT_DIR}"':${PYTHONPATH:-}"'

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n run_all -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:run_all" \
  "${DAEMON_SETUP}; python -u -m correlation.run_all" C-m

echo "Started tmux session '$SESSION'"
echo "  attach:  tmux attach -t $SESSION"
