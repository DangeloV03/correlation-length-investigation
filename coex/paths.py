"""
Path helpers for coexistence campaign results and combo directories.

Layout example::

    results/{size}_{scheme}_deltaF{df}_dmu{dmu}_epsilon{eps}/
        phi_psi.csv
        mu_sweeps/
            mu{tag}/output.csv
            mu{tag}/final_lattice_*.npy
"""

from __future__ import annotations

import csv
import os
import pathlib
import re
import shutil
from typing import Iterator

import pandas as pd

RESULTS_DIR = "results"
COMBO_KEY_FIELDS = ["epsilon", "delta_f", "delta_mu", "k", "scheme", "Lx", "Ly"]
PHI_PSI_CSV = "phi_psi.csv"
MU_SWEEPS_DIR = "mu_sweeps"
MU_RUN_DIR_PATTERN = re.compile(r"^mu\d")


def param_tag(value: float) -> str:
    """Filesystem-safe tag for a float: -2.4 -> m2p4, 0.5 -> 0p5."""
    v = float(value)
    body = str(abs(v)).replace(".", "p")
    if v < 0:
        return f"m{body}"
    return body


def size_tag(lx: int | float, ly: int | float) -> str:
    return f"{int(lx)}x{int(ly)}"


def combo_dir_name(params: dict) -> str:
    """Folder name for one outer parameter combo."""
    return (
        f"{size_tag(params['Lx'], params['Ly'])}_"
        f"{params['scheme']}_"
        f"deltaF{param_tag(params['delta_f'])}_"
        f"dmu{param_tag(params['delta_mu'])}_"
        f"epsilon{param_tag(params['epsilon'])}"
    )


def combo_dir(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(base, combo_dir_name(params))


def mu_dir_name(mu: float) -> str:
    return f"mu{round(abs(float(mu)) * 1_000_000):07d}"


def mu_sweeps_dir(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(combo_dir(params, base), MU_SWEEPS_DIR)


def mu_dir(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(mu_sweeps_dir(params, base), mu_dir_name(params["mu"]))


def phi_psi_csv_path(params: dict, base: str = RESULTS_DIR) -> str:
    return os.path.join(combo_dir(params, base), PHI_PSI_CSV)


def combo_key_from_dict(combo: dict) -> tuple[str, ...]:
    return tuple(str(combo[f]) for f in COMBO_KEY_FIELDS)


def combo_dict_from_csv_row(row: dict) -> dict:
    return {f: row[f] for f in COMBO_KEY_FIELDS}


def legacy_combo_dir_names(params: dict) -> list[str]:
    """Previous result dirname patterns (newest first)."""
    scheme = params["scheme"]
    ly = params["Ly"]
    eps_tag = str(abs(float(params["epsilon"]))).replace(".", "")
    dmu = float(params["delta_mu"])
    dmu_body = str(abs(dmu)).replace(".", "p")
    dmu_tag = f"dm-{dmu_body}" if dmu < 0 else f"dm{dmu_body}"
    return [
        f"{scheme}_eps{eps_tag}_{dmu_tag}_Ly{ly}",
        f"{scheme}_eps{eps_tag}_Ly{ly}",
    ]


def _move_mu_dir(src: pathlib.Path, dst: pathlib.Path, *, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if dry_run:
            return
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
        if src.is_dir() and not any(src.iterdir()):
            src.rmdir()
        return
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def nest_flat_mu_dirs(results_dir: str, *, dry_run: bool = False) -> int:
    """Move combo/mu* run dirs into combo/mu_sweeps/mu* (one-time fix)."""
    root = pathlib.Path(results_dir)
    if not root.is_dir():
        return 0
    moved = 0
    for combo_path in sorted(root.iterdir()):
        if not combo_path.is_dir():
            continue
        sweeps = combo_path / MU_SWEEPS_DIR
        for child in sorted(combo_path.iterdir()):
            if not child.is_dir() or not MU_RUN_DIR_PATTERN.match(child.name):
                continue
            dst = sweeps / child.name
            if child.resolve() == dst.resolve():
                continue
            print(f"nest {combo_path.name}/{child.name} -> {MU_SWEEPS_DIR}/")
            _move_mu_dir(child, dst, dry_run=dry_run)
            moved += 1
    return moved


def iter_output_csvs(results_dir: str) -> Iterator[pathlib.Path]:
    """Yield every mu-level output.csv (current and legacy layouts)."""
    root = pathlib.Path(results_dir)
    if not root.is_dir():
        return

    seen: set[pathlib.Path] = set()
    patterns = (
        f"*/{MU_SWEEPS_DIR}/*/output.csv",  # combo/mu_sweeps/mu.../output.csv
        "*/*/output.csv",                     # legacy flat or pre-migration
    )
    for pattern in patterns:
        for csv_path in sorted(root.glob(pattern)):
            if csv_path.parent.name == MU_SWEEPS_DIR:
                continue
            resolved = csv_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield csv_path


def _combo_has_mu_csv(combo_path: pathlib.Path) -> bool:
    sweeps = combo_path / MU_SWEEPS_DIR
    if sweeps.is_dir() and any(sweeps.glob("*/output.csv")):
        return True
    for child in combo_path.iterdir():
        if child.is_dir() and MU_RUN_DIR_PATTERN.match(child.name):
            if (child / "output.csv").is_file():
                return True
    return False


def read_combo_from_output_csv(csv_path: os.PathLike | str) -> dict | None:
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if df.empty or not all(col in df.columns for col in COMBO_KEY_FIELDS + ["mu"]):
            return None
        job = {
            f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
            else df[f].iloc[0]
            for f in COMBO_KEY_FIELDS
        }
        job["mu"] = float(df["mu"].iloc[0])
        return job
    except Exception:
        return None


def discover_combo_results(results_dir: str) -> dict:
    """Group mu-level output.csv files by outer combo key."""
    grouped: dict[tuple[str, ...], dict] = {}

    for csv_path in iter_output_csvs(results_dir):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "mu" not in df.columns:
                continue
            required = COMBO_KEY_FIELDS + ["mu"]
            if not all(col in df.columns for col in required):
                continue

            mu = float(df["mu"].iloc[0])
            if df["mu"].nunique() != 1:
                continue

            job = {
                f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                else df[f].iloc[0]
                for f in COMBO_KEY_FIELDS
            }
            job["mu"] = mu

            run_settings_fields = [
                "beta", "num_parallel_runs", "eq_time", "prod_time", "seed_base",
            ]
            if all(col in df.columns for col in run_settings_fields):
                job["run_settings"] = {
                    f: df[f].iloc[0].item() if hasattr(df[f].iloc[0], "item")
                    else df[f].iloc[0]
                    for f in run_settings_fields
                }
                job["run_settings"]["initial_condition"] = "slab_half_active_half_empty"
            else:
                job["run_settings"] = None

            combo_key = combo_key_from_dict(job)
        except Exception:
            continue

        if combo_key not in grouped:
            grouped[combo_key] = {"job": job, "points": []}
        grouped[combo_key]["points"].append((mu, df))

    return grouped


def combo_has_results(params: dict, base: str = RESULTS_DIR) -> bool:
    """True if any mu output.csv exists for this combo (new or legacy layout)."""
    new_dir = pathlib.Path(combo_dir(params, base))
    if new_dir.is_dir() and _combo_has_mu_csv(new_dir):
        return True
    for legacy in legacy_combo_dir_names(params):
        legacy_path = pathlib.Path(base) / legacy
        if legacy_path.is_dir() and _combo_has_mu_csv(legacy_path):
            return True
    return False


def write_phi_psi_csv(
    params: dict,
    mu_vals,
    phi_vals,
    phi_errs,
    psi_vals,
    psi_errs,
    base: str = RESULTS_DIR,
) -> str:
    path = phi_psi_csv_path(params, base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "phi", "phi_err", "psi", "psi_err"])
        for row in zip(mu_vals, phi_vals, phi_errs, psi_vals, psi_errs):
            writer.writerow(row)
    return path
