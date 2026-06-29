#!/bin/bash
# Submit the quick G(r) end-to-end test to a Della GPU node.
#
# Usage:
#   bash scripts/submit_quick_gr_test.sh [output_dir]
#   (default output_dir: ~/quick_gr_test)

set -euo pipefail

OUTPUT_DIR="${1:-$HOME/quick_gr_test}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
REPO_ROOT="$(realpath "$(dirname "$0")/..")"

mkdir -p "$OUTPUT_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=quick_gr_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=${OUTPUT_DIR}/quick_gr_%j.out
#SBATCH --error=${OUTPUT_DIR}/quick_gr_%j.err

module load anaconda3/2024.10
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"

echo "[quick_gr_test] node: \$(hostname)"
echo "[quick_gr_test] output: ${OUTPUT_DIR}"

python "${REPO_ROOT}/tests/quick_gr_test.py" --output-dir "${OUTPUT_DIR}"
EOF

echo "Submitted. Output will appear in: ${OUTPUT_DIR}"
echo "Watch progress: tail -f ${OUTPUT_DIR}/quick_gr_*.out"
