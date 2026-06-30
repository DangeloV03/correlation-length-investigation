#!/bin/bash
# Submit L=256 epsilon sweep: eps in [-1.700, -1.800] step 0.005
# 21 eps values x 16 replicas = 336 Slurm array tasks (IDs 0-335)
#
# Layout: eps_idx = task_id // 16, replica_id = task_id % 16
# eps values: -1.700, -1.705, ..., -1.800
#
# Usage:
#   bash scripts/submit_eps_sweep.sh [output_base_dir]
#   Default: /scratch/gpfs/WJACOBS/vd7294/correlation-length-investigation/eps_sweep_L256
#
# To throttle concurrency add %N to the array spec, e.g. --array=0-335%50

set -euo pipefail

SCRATCH_BASE="/scratch/gpfs/WJACOBS/vd7294/correlation-length-investigation"
OUTPUT_BASE="${1:-${SCRATCH_BASE}/eps_sweep_L256}"
REPO_ROOT="$(realpath "$(dirname "$0")/..")"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=eps_sweep_L256
#SBATCH --array=0-335
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=${LOG_DIR}/job_%A_%a.out
#SBATCH --error=${LOG_DIR}/job_%A_%a.err

module load anaconda3/2024.10
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

EPS_IDX=\$((SLURM_ARRAY_TASK_ID / 16))
REPLICA_ID=\$((SLURM_ARRAY_TASK_ID % 16))
EPS=\$(python3 -c "print(f'{-1.700 + \${EPS_IDX} * 0.005:.3f}')")

echo "[eps_sweep] node: \$(hostname)"
echo "[eps_sweep] array_id=\${SLURM_ARRAY_TASK_ID}  eps_idx=\${EPS_IDX}  eps=\${EPS}  replica=\${REPLICA_ID}"

python "${REPO_ROOT}/tests/eps_sweep_runner.py" \\
    --epsilon "\${EPS}" \\
    --replica-id "\${REPLICA_ID}" \\
    --output-dir "${OUTPUT_BASE}"
EOF

echo "Submitted array job (336 tasks, IDs 0-335)."
echo "Output base: ${OUTPUT_BASE}"
echo "Logs:        ${LOG_DIR}/job_<array_id>_<task_id>.out"
echo ""
echo "Monitor one task:  tail -f ${LOG_DIR}/job_*_0.out"
echo "Plot when done:    python ${REPO_ROOT}/tests/plot_eps_sweep.py ${OUTPUT_BASE}"
