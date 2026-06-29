"""
Create square-lattice correlation production jobs from fitted coex rows.

Run the coex phase first (`generate_samples.py`, `run_all.py`, `analyzer.py`).
This script reads numeric mu_coex_FITTED values from manage.csv, then creates
one production JSON per requested epsilon and square lattice size.
"""

from __future__ import annotations

import argparse
import json
import os

from correlation.paths import (
    MANIFEST,
    RESULTS_DIR,
    SAMPLES_DIR,
    SQUARE_L_VALUES,
    correlation_job_filename,
    read_completed_coex_rows,
)
from coex.generate_samples import (
    DELTA_F,
    DMU_MAX,
    DMU_MIN,
    DMU_STEP,
    EPS_MAX,
    EPS_MIN,
    EPS_STEP,
    K,
    MANAGE_CSV,
    SCHEME,
    frange,
)
from common.queue_manifest import merge_pending

DEFAULT_RUN_SETTINGS = {
    "beta": 1.0,
    "initial_fraction": 0.8,
    "num_parallel_runs": 8,
    "eq_time": 100000.0,
    "prod_time": 100000.0,
    "prod_chunks": 1000,
    "seed_base": 5000,
}


def _lookup(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in ["epsilon", "delta_f", "delta_mu", "k", "scheme"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate correlation production JSON jobs")
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--samples-dir", default=SAMPLES_DIR)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--eps-min", type=float, default=EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=EPS_STEP)
    parser.add_argument("--dmu-min", type=float, default=DMU_MIN)
    parser.add_argument("--dmu-max", type=float, default=DMU_MAX)
    parser.add_argument("--dmu-step", type=float, default=DMU_STEP)
    parser.add_argument(
        "--L",
        type=int,
        nargs="+",
        default=SQUARE_L_VALUES,
        help="Square lattice sizes (Lx = Ly = L)",
    )
    parser.add_argument("--eq-time", type=float, default=None)
    parser.add_argument("--prod-time", type=float, default=None)
    parser.add_argument("--prod-chunks", type=int, default=None)
    parser.add_argument("--num-parallel-runs", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    args = parser.parse_args()

    coex_rows = read_completed_coex_rows(args.manage)
    if not coex_rows:
        print(f"No numeric mu_coex_FITTED rows found in '{args.manage}'. Run the coex phase first.")
        return

    os.makedirs(args.samples_dir, exist_ok=True)
    eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
    dmu_values = frange(args.dmu_min, args.dmu_max, args.dmu_step)
    l_values = sorted(set(args.L))

    pending_paths: list[str] = []
    n_files = 0
    n_existing = 0
    n_skipped = 0

    for epsilon in eps_values:
        for delta_mu in dmu_values:
            lookup = _lookup({
                "epsilon": epsilon,
                "delta_f": DELTA_F,
                "delta_mu": delta_mu,
                "k": K,
                "scheme": SCHEME,
            })
            row = coex_rows.get(lookup)
            if row is None:
                print(f"[skip] no mu_coex_FITTED for eps={epsilon} dmu={delta_mu}")
                n_skipped += 1
                continue

            mu_coex_fitted = float(row["mu_coex_FITTED"])
            for l_val in l_values:
                run_settings = dict(DEFAULT_RUN_SETTINGS)
                if args.eq_time is not None:
                    run_settings["eq_time"] = args.eq_time
                if args.prod_time is not None:
                    run_settings["prod_time"] = args.prod_time
                if args.prod_chunks is not None:
                    run_settings["prod_chunks"] = args.prod_chunks
                if args.num_parallel_runs is not None:
                    run_settings["num_parallel_runs"] = args.num_parallel_runs
                if args.num_batches is not None:
                    run_settings["num_batches"] = args.num_batches

                job = {
                    "epsilon": epsilon,
                    "delta_f": DELTA_F,
                    "delta_mu": delta_mu,
                    "k": K,
                    "scheme": SCHEME,
                    "Lx": l_val,
                    "Ly": l_val,
                    "mu": mu_coex_fitted,
                    "mu_coex_FITTED": mu_coex_fitted,
                    "run_settings": run_settings,
                    "results_base": args.results_dir,
                }

                filename = correlation_job_filename(SCHEME, epsilon, delta_mu, l_val)
                filepath = os.path.join(args.samples_dir, filename)
                if os.path.isfile(filepath):
                    n_existing += 1
                else:
                    with open(filepath, "w") as f:
                        json.dump(job, f, indent=2)
                        f.write("\n")
                    n_files += 1
                pending_paths.append(filepath)

    merge_pending(pending_paths, path=args.manifest)
    print(f"Wrote {n_files} new JSON files to '{args.samples_dir}/' ({n_existing} existed)")
    print(f"Queued {len(pending_paths)} path(s) into '{args.manifest}'")
    print(f"Skipped {n_skipped} coex lookup(s) without fitted mu")


if __name__ == "__main__":
    main()
