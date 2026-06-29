# correlation-length investigation

Measure correlation length from lattice-gas simulations by first fitting coexistence chemical potentials, then collecting chunk-level lattice snapshots and analyzing spatial correlations.

See [QUICKSTART.md](QUICKSTART.md) for setup and run commands.

## Workflow

1. **Coexistence (`coex/`)** — fit `mu_coex_FITTED` away from equilibrium.
   - `python -m coex.generate_samples`
   - `python -m coex.run_all`
   - `python -m coex.analyzer`

2. **Correlation production (`correlation/`)** — save lattice snapshots at fitted coexistence.
   - `python -m correlation.generate_jobs`
   - `python -m correlation.run_all`
   - `python -m correlation.runner` (single job)

3. **Correlation analysis (`correlation/`)** — estimate ξ from snapshots.
   - `python -m correlation.analyzer <result_dir>`

## Project layout

```text
correlation-length-investigation/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── coex/                 # Stage 1: mu_coex fitting
│   ├── generate_samples.py
│   ├── json_runner.py
│   ├── run_all.py
│   ├── analyzer.py
│   ├── paths.py
│   └── flex_coex.py
├── correlation/          # Stage 2–3: snapshots and analysis
│   ├── generate_jobs.py
│   ├── runner.py
│   ├── run_all.py
│   ├── analyzer.py
│   └── paths.py
├── common/
│   └── queue_manifest.py
├── config/
│   └── slurm_config.yml
├── scripts/
│   ├── env.sh
│   ├── start_daemons.sh
│   ├── start_correlation_daemons.sh
│   └── stop_daemons.sh
├── tests/
│   └── test_pipeline.py
└── docs/
    └── Correlation Function in Ising Models.pdf
```

## Generated outputs (gitignored)

| Path | Contents |
| --- | --- |
| `samples/` | Coex μ-sweep job JSONs |
| `results/` | Coex density outputs and `phi_psi.csv` |
| `manage.csv` | Coex ledger with `mu_coex_FITTED` |
| `correlation_samples/` | Correlation production job JSONs |
| `correlation_results/` | Snapshots, metadata, and analysis CSVs |
| `run_all_queue.json` | Coex dispatch queue |
| `correlation_queue.json` | Correlation dispatch queue |

## Requirements

- Python 3.11+
- `numpy`, `scipy`, `pandas`, `pyyaml`, `simple-slurm`, `pytest`
- Private `lattice_gas` package (Rust extension)

Run all commands from the repository root after `source scripts/env.sh` (sets `PYTHONPATH`).
