#!/bin/bash
# Submit a single G(r) analysis + plot job to Della's GPU partition.
#
# Usage:
#   bash scripts/submit_gr_plot.sh /path/to/snapshot_metadata.csv [output_dir]
#
# The script resolves paths to absolute so the job finds files regardless of
# which directory Slurm starts in.

set -euo pipefail

METADATA_PATH="$(realpath "${1:?Usage: $0 <snapshot_metadata.csv> [output_dir]}")"
OUTPUT_DIR="${2:-$(dirname "$METADATA_PATH")}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
REPO_ROOT="$(realpath "$(dirname "$0")/..")"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gr_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=${OUTPUT_DIR}/gr_plot_%j.out
#SBATCH --error=${OUTPUT_DIR}/gr_plot_%j.err

module load anaconda3/2024.10
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

echo "[gr_plot] metadata: ${METADATA_PATH}"
echo "[gr_plot] output:   ${OUTPUT_DIR}"
echo "[gr_plot] node:     \$(hostname)"

python "${REPO_ROOT}/tests/plot_gr_della.py" \
    "${METADATA_PATH}" \
    --output-dir "${OUTPUT_DIR}"
EOF
