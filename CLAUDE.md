# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup (must be run before any Python commands)
source scripts/env.sh          # Sets PYTHONPATH

# Run all pipeline tests
python -m pytest tests/test_pipeline.py -v

# Run estimator validation ladder
python -m pytest tests/test_estimator_validation.py -v --tb=short

# Run a single test
python -m pytest tests/test_pipeline.py::test_queue_manifest_deduplicates_pending -v

# Skip slow tests (Metropolis MC, large finite-size)
python -m pytest tests/ -v -m "not slow"

# Run only slow tests
python -m pytest tests/ -v -m slow

# Full validation report (add --mc for Metropolis MC cross-checks)
python tests/run_validation_report.py [--mc]
```

**Stage-by-stage pipeline execution (on Della HPC):**
```bash
# Stage 1: Generate coexistence sweep jobs and run them
python coex/generate_samples.py
python coex/run_all.py --once          # or omit --once to run as long-lived daemon
python coex/analyzer.py --once         # fits mu_coex_FITTED into manage.csv

# Stage 2: Generate and run correlation snapshot jobs
python correlation/generate_jobs.py
python correlation/run_all.py --once

# Stage 3: Analyze snapshots
python correlation/analyzer.py --once
```

## Architecture

This is a **3-stage HPC pipeline** for measuring spatial correlation lengths (ξ) in 2D lattice-gas (Ising-like) systems. The pipeline depends on a private Rust extension called `lattice_gas` (a `HeteroChain` FLEX sampler), which must be compiled separately.

### Stage 1 — Coexistence Fitting (`coex/`)
Sweeps chemical potential μ across a parameter grid (ε, Δμ, Ly), runs short lattice-gas simulations at each μ value, then fits the coexistence μ as the zero-crossing of the φ/ψ order-parameter curves.

- `generate_samples.py` — builds the parameter grid and writes job JSONs; uses `flex_coex.py` for an analytical first guess (`mu_coex_FLEX`) before sweeping
- `json_runner.py` — executes a single coex job via `lattice_gas`, writes `output.csv`
- `analyzer.py` — long-running watcher that fits φ(μ)/ψ(μ) and writes `mu_coex_FITTED` to `manage.csv`
- `flex_coex.py` — computes analytical equilibrium μ via closed-form expression or `fsolve`

### Stage 2 — Snapshot Collection (`correlation/`)
Reads `manage.csv` rows with a fitted `mu_coex_FITTED`, then runs production simulations at coexistence for varying system sizes L ∈ {16, 32, 64, 128, 256}. Saves chunk-level lattice snapshots as `.npy` arrays.

- `generate_jobs.py` — reads `manage.csv`, writes one job JSON per (ε, Δμ, L) combination
- `runner.py` — runs a single production job: equilibrate → chunked production → save `snapshot.npy` per chunk + `snapshot_metadata.csv`

### Stage 3 — Correlation Analysis (`correlation/analyzer.py`)
Loads snapshots, converts lattice sites (0=EMPTY, 2=BONDING) to spins (±1), computes the connected real-space correlation function G(r), and fits ξ.

1. FFT autocorrelation → `C2D = ifft2(|fft2(spins)|²) / N`
2. Subtract m² → connected correlation `G2D`
3. Radial average → `G(r)` (averaged over all snapshots)
4. Fit `G(r) = A·exp(-r/ξ)` for r ≥ 1

Outputs: `G_r.csv`, `correlation_length.csv` per result directory.

**Planned:** PyTorch GPU parallelization for running multiple replicas concurrently. Currently CPU-only (numpy FFT); validate correctness first, then add GPU path.

### Queue & Dispatch (`common/queue_manifest.py`)
All job dispatching uses file-locked JSON manifests (`run_all_queue.json`, `correlation_queue.json`). Schema: `{"pending": [...], "in_flight": {...}}`. Key operations: `merge_pending` (deduplicates), `pop_next_pending` (FIFO), `mark_in_flight`/`remove_in_flight`. The `run_all.py` dispatchers in both `coex/` and `correlation/` are long-running daemons that submit via `simple_slurm` and re-enqueue failures.

### Data Flow
```
lattice_gas (Rust)
    ↓
coex/json_runner.py → output.csv → coex/analyzer.py → manage.csv
                                                            ↓
                                              correlation/generate_jobs.py
                                                            ↓
                                              correlation/runner.py → snapshots/*.npy
                                                            ↓
                                              correlation/analyzer.py → correlation_length.csv
```

### Testing Strategy
- `test_pipeline.py` — unit tests for queue deduplication, coex CSV schema, path stability, spin mapping, and metadata roundtrip
- `test_estimator_validation.py` — validation ladder: (1) exact exponential recovery from synthetic data, (2) Fourier-synthesized Ornstein–Zernike snapshots, (3) finite-size threshold checks, (4) Metropolis MC cross-checks (marked `@pytest.mark.slow`)
- `ising_reference.py` — helper module providing analytical high-T ξ, critical β, and a lightweight Metropolis MC sampler used by validation tests
- Slow tests require `--mc` flag or `-m slow` to run; default `pytest` skips them

### Key Configuration
- `config/slurm_config.yml` — Slurm directives (partition, account, CPUs, memory, time)
- `pyproject.toml` — pytest markers (`slow`) and testpaths
- Generated outputs (`samples/`, `results/`, `correlation_results/`, `manage.csv`, `*.json` queues) are git-ignored
