"""
Path helpers and schemas for correlation-length production runs.

The coex phase still uses results/ + manage.csv. Correlation production reads
completed manage.csv rows and writes chunk snapshots under correlation_results/.
"""

from __future__ import annotations

import csv
import os

from coex.paths import combo_dir_name, param_tag

SAMPLES_DIR = "correlation_samples"
RESULTS_DIR = "correlation_results"
MANIFEST = "correlation_queue.json"
METADATA_CSV = "snapshot_metadata.csv"
ANALYSIS_CSV = "correlation_length.csv"
SNAPSHOTS_DIR = "snapshots"

COEX_LOOKUP_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme"]
SQUARE_L_VALUES = [16, 32, 48, 64, 96, 128, 256]


METADATA_FIELDS = [
    "run_id",
    "replica_id",
    "chunk",
    "snapshot_path",
    "time",
    "chunk_time",
    "epsilon",
    "delta_f",
    "delta_mu",
    "k",
    "scheme",
    "Lx",
    "Ly",
    "mu",
    "mu_coex_FITTED",
    "beta",
    "eq_time",
    "prod_time",
    "prod_chunks",
    "seed",
]


def eps_filename_tag(epsilon: float) -> str:
    return "eps" + str(abs(float(epsilon))).replace(".", "p")


def dmu_filename_tag(delta_mu: float) -> str:
    dmu = float(delta_mu)
    body = str(abs(dmu)).replace(".", "p")
    if dmu < 0:
        return f"dm-{body}"
    return f"dm{body}"


def correlation_dir_name(params: dict) -> str:
    return f"correlation_{combo_dir_name(params)}"


def correlation_dir(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(base, correlation_dir_name(params))


def snapshots_dir(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(correlation_dir(params, base), SNAPSHOTS_DIR)


def replica_snapshot_dir(params: dict, run_id: int, base: str = RESULTS_DIR) -> str:
    return os.path.join(snapshots_dir(params, base), f"replica_{int(run_id):05d}")


def snapshot_path(params: dict, run_id: int, chunk_idx: int, base: str = RESULTS_DIR) -> str:
    filename = f"chunk_{int(chunk_idx):05d}.npy"
    return os.path.join(replica_snapshot_dir(params, run_id, base), filename)


def metadata_csv_path(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(correlation_dir(params, base), METADATA_CSV)


def analysis_csv_path(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(correlation_dir(params, base), ANALYSIS_CSV)


def correlation_job_filename(scheme: str, epsilon: float, delta_mu: float, l_val: int) -> str:
    return (
        f"{scheme}_{eps_filename_tag(epsilon)}_"
        f"{dmu_filename_tag(delta_mu)}_L{int(l_val)}.json"
    )


def lookup_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in COEX_LOOKUP_FIELDS)


def read_completed_coex_rows(manage_path: str) -> dict[tuple[str, ...], dict]:
    if not os.path.isfile(manage_path):
        return {}

    by_key: dict[tuple[str, ...], dict] = {}
    with open(manage_path, newline="") as f:
        for row in csv.DictReader(f):
            fitted = str(row.get("mu_coex_FITTED", "")).strip()
            if not fitted or fitted.lower() == "nan":
                continue
            try:
                float(fitted)
            except ValueError:
                continue
            by_key[lookup_key(row)] = row
    return by_key
