# Quickstart

Run from the repository root (`correlation-length-investigation/`).

## Setup

```bash
cd /scratch/gpfs/WJACOBS/$USER
git clone <repo-url> correlation-length-investigation
cd correlation-length-investigation

module load anaconda3/2024.10
conda create -n lattice python=3.11 -y
conda activate lattice
pip install maturin
pip install -r requirements.txt

cd ~/software/lattice-gas
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
./build-rust-lib.sh

cd /scratch/gpfs/WJACOBS/$USER/correlation-length-investigation
source scripts/env.sh
```

Check `config/slurm_config.yml` before submitting Slurm jobs.

## Stage 1: Fit `mu_coex_FITTED`

```bash
python -m coex.generate_samples --ly 16
python -m coex.run_all
python -m coex.analyzer --depth-first
```

Local smoke test:

```bash
python -m coex.generate_samples --ly 8
python -m coex.run_all --local --once --max-concurrent 1
python -m coex.analyzer --once
```

Or use tmux:

```bash
./scripts/start_daemons.sh
tmux attach -t coex-campaign
```

## Stage 2: Save correlation snapshots

```bash
python -m correlation.generate_jobs --L 32 64 128
python -m correlation.run_all
```

Local smoke test:

```bash
python -m correlation.generate_jobs \
  --L 16 \
  --eq-time 10 \
  --prod-time 20 \
  --prod-chunks 4 \
  --num-parallel-runs 1
python -m correlation.run_all --local --once --max-concurrent 1
```

Or use tmux:

```bash
./scripts/start_correlation_daemons.sh
```

Each result directory contains `snapshot_metadata.csv` and `snapshots/replica_*/chunk_*.npy`.

## Stage 3: Analyze correlation length

```bash
python -m correlation.analyzer correlation_results/<result_dir>
```

Outputs: `fourier_modes.csv`, `G_r.csv`, `correlation_length.csv`.

Default spin mapping: `BONDING -> +1`, `INERT/EMPTY -> -1`. Use `--mapping active_vs_empty` for `INERT -> 0`.

## Tests

```bash
python -m pytest tests/test_pipeline.py -v
```
