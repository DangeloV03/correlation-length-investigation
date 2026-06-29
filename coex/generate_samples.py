"""
generate_samples.py

Full grid campaign over (epsilon, delta_mu) for homo scheme 1 / K=1.
Grid: epsilon in [-2.5, -1.5] step 0.1, delta_mu in [-1, 5] step 0.25.
Default chain sizes: Ly in {8, 16} with Lx = 10 * Ly (override with --ly).

  1. Enumerate every (epsilon, delta_mu, Ly) on the configured grid.
  2. Skip combos already finished (analyzed, ran, or with results/) or in the queue.
  3. Compute mu_coex_FLEX per combo; skip if mu_coex_FLEX > 0.
  4. Sweep mu over mu_coex_FLEX +/- 0.1 (10 points), write JSON files.
  5. Append new rows to manage.csv and merge jobs into run_all_queue.json.

Existing JSON files and manage.csv rows are never overwritten or removed; re-runs
only append missing combos/files.

Usage:
    python generate_samples.py
    python generate_samples.py --ly 8 16
    python generate_samples.py --ly 32
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd

from coex.flex_coex import coex_chemical_potential
from coex.paths import COMBO_KEY_FIELDS, combo_dir, combo_has_results, combo_key_from_dict, iter_output_csvs
from common.queue_manifest import merge_pending, read_manifest

# ---------------------------------------------------------------------------
# Campaign constants
# ---------------------------------------------------------------------------

SCHEME = "homo"
FLEX_INDEX = 1
DEFAULT_LY_VALUES = [8, 16]
LX_MULTIPLIER = 10  # Lx = LX_MULTIPLIER * Ly
DELTA_F = 0.0
K = 1.0

EPS_MIN = -2.5
EPS_MAX = -1.5
EPS_STEP = 0.1
DMU_MIN = -1.0
DMU_MAX = 5.0
DMU_STEP = 0.25

MU_WINDOW = 0.1
N_MU_POINTS = 10

RUN_SETTINGS = {
    "beta": 1.0,
    "k": K,
    "initial_condition": "slab_half_active_half_empty",
    "num_parallel_runs": 4,
    "eq_time": 10000.0,
    "prod_time": 10000.0,
    "seed_base": 1000,
}

OUTPUT_DIR = "samples"
MANAGE_CSV = "manage.csv"
MANIFEST_PATH = "run_all_queue.json"
RESULTS_DIR = "results"

COMBO_KEY_FIELDS = COMBO_KEY_FIELDS  # re-export
MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_FITTED",
    "mu_coex_FITTED_error",
    "RequestForAdditionalData",
    "combo_path",
]


def frange(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive float range [lo, hi] at fixed step (rounded to 6 decimals)."""
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 6) for i in range(n)]


def grid_outer_combos() -> list[tuple[float, float]]:
    """Full Cartesian grid over (epsilon, delta_mu)."""
    pairs: list[tuple[float, float]] = []
    for epsilon in frange(EPS_MIN, EPS_MAX, EPS_STEP):
        for delta_mu in frange(DMU_MIN, DMU_MAX, DMU_STEP):
            pairs.append((epsilon, delta_mu))
    return pairs


def combo_dict(epsilon: float, delta_mu: float, ly: int, lx: int) -> dict:
    return {
        "epsilon": epsilon,
        "delta_f": DELTA_F,
        "delta_mu": delta_mu,
        "k": K,
        "scheme": SCHEME,
        "Lx": lx,
        "Ly": ly,
    }


def eps_filename_tag(epsilon: float) -> str:
    return "eps" + str(abs(epsilon)).replace(".", "p")


def dmu_filename_tag(delta_mu: float) -> str:
    body = str(abs(delta_mu)).replace(".", "p")
    if delta_mu < 0:
        return f"dm-{body}"
    return f"dm{body}"


def mu_sweep(mu_coex_flex: float) -> list[float]:
    values = np.linspace(mu_coex_flex - MU_WINDOW, mu_coex_flex + MU_WINDOW, N_MU_POINTS)
    return [round(float(v), 6) for v in values]


def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict] = []
    for line_no, row in enumerate(raw_rows, start=2):
        row.setdefault("mu_coex_FITTED", row.get("mu_coex_SIM", ""))
        row.setdefault("mu_coex_FITTED_error", row.get("mu_coex_SIM_error", ""))
        row.setdefault("combo_path", "")
        missing = [f for f in COMBO_KEY_FIELDS if f not in row or row[f] is None]
        if missing:
            print(
                f"WARNING: skipping malformed manage.csv line {line_no} "
                f"(missing {missing})"
            )
            continue
        rows.append(row)
    return rows


def write_manage(manage_path: str, rows: list[dict]) -> None:
    """Rewrite manage.csv from the full row list (caller must preserve existing rows)."""
    if os.path.isfile(manage_path):
        prior_count = len(read_manage(manage_path))
        if len(rows) < prior_count:
            raise RuntimeError(
                f"Refusing to write {manage_path}: {len(rows)} rows would replace "
                f"{prior_count} existing rows. manage.csv rows are never deleted."
            )

    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def append_manage_rows(manage_path: str, new_rows: list[dict]) -> int:
    """Append only new combo rows; never modify or drop existing manage.csv rows."""
    existing_rows = read_manage(manage_path)
    existing_keys = {combo_key_from_dict(row) for row in existing_rows}
    to_add = [
        row for row in new_rows
        if combo_key_from_dict(row) not in existing_keys
    ]
    write_manage(manage_path, existing_rows + to_add)
    return len(to_add)


def combo_from_json_path(json_path: str) -> tuple | None:
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            job = json.load(f)
        return combo_key_from_dict(job)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def manage_row_is_complete(row: dict, results_dir: str) -> bool:
    """True when this ledger row should not be regenerated."""
    if str(row.get("isAnalyzed", "")).strip():
        return True
    if str(row.get("isRan", "")).strip():
        return True
    combo = {f: row[f] for f in COMBO_KEY_FIELDS}
    return combo_has_results(combo, results_dir)


def collect_active_combo_keys(
    manage_path: str,
    manifest_path: str,
    samples_dir: str,
    results_dir: str,
) -> dict[tuple, str]:
    """Return combo_key -> reason for combos that should not be regenerated.

    manage.csv is append-only (rows are never deleted), but incomplete ledger rows
    without results may still receive missing sample JSONs and queue entries.
    """
    active: dict[tuple, str] = {}

    for row in read_manage(manage_path):
        if not manage_row_is_complete(row, results_dir):
            continue
        key = combo_key_from_dict(row)
        if str(row.get("isAnalyzed", "")).strip():
            active.setdefault(key, "manage:analyzed")
        elif str(row.get("isRan", "")).strip():
            active.setdefault(key, "manage:ran")
        else:
            active.setdefault(key, "manage:results")

    manifest = read_manifest(manifest_path)
    for json_path in manifest.get("pending", []) + list(manifest.get("in_flight", {}).values()):
        key = combo_from_json_path(json_path)
        if key is not None:
            active.setdefault(key, "queue")

    for csv_path in iter_output_csvs(results_dir):
        try:
            df = pd.read_csv(csv_path, nrows=1)
            if not all(col in df.columns for col in COMBO_KEY_FIELDS):
                continue
            job = {
                f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                else df[f].iloc[0]
                for f in COMBO_KEY_FIELDS
            }
            key = combo_key_from_dict(job)
            active.setdefault(key, "results")
        except Exception:
            continue

    return active


def main():
    parser = argparse.ArgumentParser(description="Generate grid campaign sample JSONs")
    parser.add_argument(
        "--ly",
        type=int,
        nargs="+",
        default=DEFAULT_LY_VALUES,
        help="Chain lengths Ly (Lx = 10 * Ly). Default: 8 16",
    )
    args = parser.parse_args()
    ly_values = sorted(set(args.ly))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outer_pairs = grid_outer_combos()
    n_eps = len(frange(EPS_MIN, EPS_MAX, EPS_STEP))
    n_dmu = len(frange(DMU_MIN, DMU_MAX, DMU_STEP))
    print(
        f"Grid: epsilon [{EPS_MIN}, {EPS_MAX}] step {EPS_STEP} ({n_eps} pts) x "
        f"delta_mu [{DMU_MIN}, {DMU_MAX}] step {DMU_STEP} ({n_dmu} pts) "
        f"-> {len(outer_pairs)} (epsilon, delta_mu) pairs per Ly"
    )
    print(f"Ly values: {ly_values}  (Lx = {LX_MULTIPLIER} * Ly)")

    active_combos = collect_active_combo_keys(
        MANAGE_CSV, MANIFEST_PATH, OUTPUT_DIR, RESULTS_DIR,
    )
    existing_rows = read_manage(MANAGE_CSV)
    existing_keys = {combo_key_from_dict(row) for row in existing_rows}

    pending_paths: list[str] = []
    new_manage_rows: list[dict] = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_files = 0
    n_existing_json = 0
    skipped_flex = 0
    skipped_dedup = 0

    for ly in ly_values:
        lx = LX_MULTIPLIER * ly
        print(f"\n--- Ly={ly}, Lx={lx} ---")

        for epsilon, delta_mu in outer_pairs:
            combo = combo_dict(epsilon, delta_mu, ly, lx)
            key = combo_key_from_dict(combo)

            if key in active_combos:
                print(f"[skip dedup:{active_combos[key]}] Ly={ly} eps={epsilon}, dmu={delta_mu}")
                skipped_dedup += 1
                continue

            try:
                mu_coex_flex = coex_chemical_potential(
                    epsilon=epsilon,
                    df=DELTA_F,
                    dmu=delta_mu,
                    chem_rec_baserate=K,
                    DRIVEN=True,
                    scheme=FLEX_INDEX,
                )
                mu_coex_flex = float(np.asarray(mu_coex_flex).ravel()[0])
            except Exception as exc:
                print(f"[skip flex] Ly={ly} eps={epsilon}, dmu={delta_mu}: {exc}")
                skipped_flex += 1
                continue

            if mu_coex_flex > 0:
                print(
                    f"[skip flex] Ly={ly} eps={epsilon}, dmu={delta_mu}: "
                    f"mu_coex_FLEX={mu_coex_flex:.6f} > 0"
                )
                skipped_flex += 1
                continue

            print(f"Ly={ly} eps={epsilon}, dmu={delta_mu}: mu_coex_FLEX={mu_coex_flex:.6f}")
            mu_values = mu_sweep(mu_coex_flex)

            if key not in existing_keys:
                new_manage_rows.append({
                    "epsilon": epsilon,
                    "delta_f": DELTA_F,
                    "delta_mu": delta_mu,
                    "k": K,
                    "scheme": SCHEME,
                    "Lx": lx,
                    "Ly": ly,
                    "mu_coex_FLEX": mu_coex_flex,
                    "isSubmitted": timestamp,
                    "isRan": "",
                    "isAnalyzed": "",
                    "mu_coex_FITTED": "",
                    "mu_coex_FITTED_error": "",
                    "RequestForAdditionalData": 0,
                    "combo_path": combo_dir(combo),
                })
                existing_keys.add(key)

            outer_tag = f"{eps_filename_tag(epsilon)}_{dmu_filename_tag(delta_mu)}"
            for idx, mu in enumerate(mu_values):
                job = {
                    **combo,
                    "mu": mu,
                    "mu_coex_FLEX": mu_coex_flex,
                    "run_settings": RUN_SETTINGS,
                }
                filename = f"{SCHEME}_{outer_tag}_Ly{ly}_mu{idx:02d}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    n_existing_json += 1
                else:
                    with open(filepath, "w") as f:
                        json.dump(job, f, indent=2)
                    n_files += 1
                pending_paths.append(filepath)

    merge_pending(pending_paths, path=MANIFEST_PATH)
    n_added = append_manage_rows(MANAGE_CSV, new_manage_rows)

    print(f"\nWrote {n_files} new JSON files to '{OUTPUT_DIR}/' "
          f"({n_existing_json} already existed, left unchanged)")
    print(f"Queued {len(pending_paths)} job path(s) into '{MANIFEST_PATH}' "
          f"(merge_pending skips duplicates already pending/in-flight)")
    print(f"Added {n_added} new rows to '{MANAGE_CSV}' "
          f"({len(existing_rows)} existing, unchanged)")
    print(f"Skipped {skipped_dedup} complete dedup, {skipped_flex} FLEX filter")


if __name__ == "__main__":
    main()
