#!/bin/bash
# Submit the L=256 / 1000-chunk G(r) test to Della.
#
# Usage:
#   bash scripts/submit_gr_test_L256.sh [output_dir]
#   (default: ~/gr_test_L256)

set -euo pipefail

OUTPUT_DIR="${1:-$HOME/gr_test_L256}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
REPO_ROOT="$(realpath "$(dirname "$0")/..")"

mkdir -p "$OUTPUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gr_L256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=${OUTPUT_DIR}/gr_L256_%j.out
#SBATCH --error=${OUTPUT_DIR}/gr_L256_%j.err

module load anaconda3/2024.10
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

echo "[gr_L256] node: \$(hostname)"
echo "[gr_L256] output: ${OUTPUT_DIR}"

python "${REPO_ROOT}/tests/gr_test_L256.py" --output-dir "${OUTPUT_DIR}"
EOF

echo "Submitted. Output: ${OUTPUT_DIR}"
echo "Watch: tail -f ${OUTPUT_DIR}/gr_L256_*.out"
