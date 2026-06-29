"""
Create square-lattice correlation production jobs from coexistence chemical potential.

Default (--mu-source fitted): numeric mu_coex_FITTED in manage.csv after coex.analyzer.
Alternatives:
  flex  — mu_coex_FLEX from manage.csv (coex.generate_samples only; no analyzer)
  exact — analytical equilibrium mu from flex_coex.coex_chemical_potential (no manage.csv)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from correlation.paths import (
    MANIFEST,
    RESULTS_DIR,
    SAMPLES_DIR,
    SQUARE_L_VALUES,
    correlation_job_filename,
    read_completed_coex_rows,
)
from coex.flex_coex import coex_chemical_potential
from coex.generate_samples import (
    DELTA_F,
    DMU_MAX,
    DMU_MIN,
    DMU_STEP,
    EPS_MAX,
    EPS_MIN,
    EPS_STEP,
    FLEX_INDEX,
    K,
    MANAGE_CSV,
    SCHEME,
    frange,
)
from common.queue_manifest import merge_pending

MU_SOURCE_COLUMNS = {
    "fitted": "mu_coex_FITTED",
    "flex": "mu_coex_FLEX",
}

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


def _exact_mu_coex(epsilon: float, delta_mu: float) -> float:
    mu = coex_chemical_potential(
        epsilon,
        DELTA_F,
        delta_mu,
        K,
        scheme=FLEX_INDEX,
    )
    return float(np.asarray(mu).ravel()[0])


def _exact_mu_coex(epsilon: float, delta_mu: float) -> float:
    mu = coex_chemical_potential(
        epsilon,
        DELTA_F,
        delta_mu,
        K,
        scheme=FLEX_INDEX,
    )
    return float(np.asarray(mu).ravel()[0])


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
    parser.add_argument(
        "--mu-source",
        choices=["fitted", "flex", "exact"],
        default="fitted",
        help="Coexistence mu: fitted (analyzer), flex (generate_samples), or exact (analytic)",
    )
    args = parser.parse_args()

    if args.mu_source != "exact":
        mu_column = MU_SOURCE_COLUMNS[args.mu_source]
        coex_rows = read_completed_coex_rows(args.manage, mu_column=mu_column)
        if not coex_rows:
            print(
                f"No numeric {mu_column} rows found in '{args.manage}'. "
                f"Run coex.generate_samples (flex) or coex.analyzer (fitted) first, "
                f"or use --mu-source exact."
            )
            return
    else:
        coex_rows = {}
        mu_column = ""
        print("[generate_jobs] using analytical mu_coex (exact equilibrium; no manage.csv)")

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
            if args.mu_source == "exact":
                mu_coex = _exact_mu_coex(epsilon, delta_mu)
            else:
                row = coex_rows.get(_lookup({
                    "epsilon": epsilon,
                    "delta_f": DELTA_F,
                    "delta_mu": delta_mu,
                    "k": K,
                    "scheme": SCHEME,
                }))
                if row is None:
                    mu_coex = None
                else:
                    mu_coex = float(row[mu_column])
            if mu_coex is None:
                print(f"[skip] no mu for eps={epsilon} dmu={delta_mu} (source={args.mu_source})")
                n_skipped += 1
                continue
            if mu_coex > 0 and args.mu_source != "exact":
                print(f"[skip] mu_coex={mu_coex:.6f} > 0 for eps={epsilon} dmu={delta_mu}")
                n_skipped += 1
                continue
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
                    "mu": mu_coex,
                    "mu_coex_FITTED": mu_coex,
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
    print(f"Skipped {n_skipped} grid point(s) (source={args.mu_source})")


if __name__ == "__main__":
    main()
