"""
analyzer.py

Long-running watcher that:
  1. Polls results/ for completed output.csv files.
  2. Groups results by outer combo (scheme, Lx, Ly, epsilon, delta_f, delta_mu, k).
  3. For each combo with enough data, computes phi(mu) and psi(mu).
  4. Waits for all N_INITIAL_MU_POINTS (10) initial mu jobs before analyzing.
  5. Checks for sign change in phi:
       - Sign change found  -> refine mu in the bracket until dense or budget exhausted.
       - Neighbors of min(psi) near zero -> finalize (resolved) if min(psi) <= PSI_COEX_MAX.
       - Interior min(psi) with sign change -> finalize with argmin(psi) once bracket
         is dense or no new mu values remain, and min(psi) <= PSI_COEX_MAX.
       - No sign change     -> extend mu window; NaN only if budget exhausted.
     Max MAX_ADDITIONAL_REQUESTS additional data requests per combo.
  6. Finds min(psi) -> mu_coex_FITTED.
  7. Saves phi/psi CSV inside each combo folder under results/.
  8. Updates manage.csv with mu_coex_FITTED, isAnalyzed, combo_path, RequestForAdditionalData.

Usage:
    python analyzer.py [--results results] [--manage manage.csv] [--interval 10]
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from coex.paths import (
    COMBO_KEY_FIELDS,
    RESULTS_DIR,
    combo_dir,
    combo_dir_name,
    discover_combo_results,
    write_phi_psi_csv,
)
from common.queue_manifest import prepend_pending, read_manifest

MANAGE_CSV = "manage.csv"
SAMPLES_DIR = "samples"
POLL_INTERVAL = 10.0  # seconds
MAX_ADDITIONAL_REQUESTS = 10
N_INITIAL_MU_POINTS = 10  # must match generate_samples.N_MU_POINTS
N_REFINEMENT_POINTS = 10
PHI_NEIGHBOR_SIGMA_K = 2.0  # |phi| <= k * max(phi_err, PHI_ABS_TOL) counts as "close"
PHI_ABS_TOL = 0.05
PSI_COEX_MAX = 0.025  # min(psi) must be at or below this to accept mu_coex_FITTED
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


# ---------------------------------------------------------------------------
# manage.csv helpers
# ---------------------------------------------------------------------------

def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict] = []
    for line_no, row in enumerate(raw_rows, start=2):
        row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
        row.setdefault("mu_coex_FITTED_error", "")
        row.setdefault("combo_path", "")
        missing = [f for f in COMBO_KEY_FIELDS if f not in row or row[f] is None]
        if missing:
            print(
                f"[analyzer] WARNING: skipping malformed manage.csv line {line_no} "
                f"(missing {missing})",
                file=sys.stderr,
            )
            continue
        rows.append(row)
    return rows


def write_manage(manage_path: str, rows: list[dict]):
    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def _combo_field_match(row_val, combo_val) -> bool:
    """Match manage.csv strings to job dict values (exact or numeric)."""
    if row_val == combo_val:
        return True
    rs = str(row_val).strip()
    cs = str(combo_val).strip()
    if rs == cs:
        return True
    try:
        return float(rs) == float(cs)
    except (TypeError, ValueError):
        return False


def find_manage_row(rows: list[dict], combo: dict) -> int | None:
    """Return index of the manage row matching this combo, or None."""
    for i, row in enumerate(rows):
        if all(_combo_field_match(row.get(f, ""), combo.get(f, "")) for f in COMBO_KEY_FIELDS):
            return i
    return None


def update_manage_field(manage_path: str, combo: dict, updates: dict):
    """Update specific fields on the matching combo row, only if currently empty
    (for timestamp fields) or always (for numeric fields like RequestForAdditionalData)."""
    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(f"[analyzer] WARNING: no manage row found for {combo}", file=sys.stderr)
        return
    for field, value in updates.items():
        # Timestamp fields: only write if empty (first writer wins)
        if field in ("isRan", "isAnalyzed") and rows[idx].get(field, ""):
            continue
        rows[idx][field] = value
    write_manage(manage_path, rows)


# ---------------------------------------------------------------------------
# Physics calculations
# ---------------------------------------------------------------------------

def calculate_phi_psi(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute phi = <rho_active - rho_inert - rho_empty> and psi = |phi|
    with error propagation."""
    rho_a = df["rho_active"].mean()
    rho_i = df["rho_inert"].mean()
    rho_e = df["rho_empty"].mean()

    std_a = df["rho_active"].std()
    std_i = df["rho_inert"].std()
    std_e = df["rho_empty"].std()

    phi = rho_a - rho_i - rho_e
    psi = abs(phi)

    # Error propagation for psi = |phi|
    # d(psi)/d(rho_a) = sign(phi), etc.
    if psi > 0:
        prefactor = -phi / psi  # = -sign(phi)
        psi_error = np.sqrt(
            (std_a * phi / psi) ** 2
            + (std_i * prefactor) ** 2
            + (std_e * prefactor) ** 2
        )
        phi_error = np.sqrt(std_a**2 + std_i**2 + std_e**2)
    else:
        psi_error = 0.0
        phi_error = 0.0

    return phi, phi_error, psi, psi_error


def build_curves(points: list[tuple]) -> tuple:
    """Given [(mu, df), ...], return sorted arrays of mu, phi, phi_err, psi, psi_err."""
    points_sorted = sorted(points, key=lambda x: x[0])
    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = [], [], [], [], []
    for mu, df in points_sorted:
        phi, phi_err, psi, psi_err = calculate_phi_psi(df)
        mu_vals.append(mu)
        phi_vals.append(phi)
        phi_errs.append(phi_err)
        psi_vals.append(psi)
        psi_errs.append(psi_err)
    return (
        np.array(mu_vals),
        np.array(phi_vals),
        np.array(phi_errs),
        np.array(psi_vals),
        np.array(psi_errs),
    )


# ---------------------------------------------------------------------------
# Analysis CSV output
# ---------------------------------------------------------------------------

def write_combo_curves(combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
                       results_dir=RESULTS_DIR):
    job = dict(zip(COMBO_KEY_FIELDS, combo_key))
    csv_path = write_phi_psi_csv(
        job, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs, base=results_dir,
    )
    print(f"[analyzer] Curves saved: {csv_path}")


# ---------------------------------------------------------------------------
# Job dispatching
# ---------------------------------------------------------------------------

def make_job_json(
    job_template: dict,
    mu: float,
    samples_dir: str,
    *,
    results_base: str | None = None,
    manage_csv: str | None = None,
) -> str:
    """Write a new JSON job file for a given mu and return its path."""
    scheme = str(job_template["scheme"])
    Ly = int(job_template["Ly"])
    mu_tag = f"mu{round(abs(mu) * 1_000_000):07d}"
    filename = f"{scheme}_Ly{Ly}_{mu_tag}.json"
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, filename)

    # Coerce all values to native Python types for JSON serialization
    def to_native(v):
        if hasattr(v, "item"):  # catches all numpy scalars
            return v.item()
        return v

    job = {k: to_native(v) for k, v in job_template.items()}
    job["mu"] = round(float(mu), 6)
    if results_base is not None:
        job["results_base"] = results_base
    if manage_csv is not None:
        job["manage_csv"] = manage_csv

    with open(filepath, "w") as f:
        json.dump(job, f, indent=2)
    return filepath


def enqueue_jobs(
    mu_values: list[float],
    job_template: dict,
    samples_dir: str,
    manifest_path: str = "run_all_queue.json",
    *,
    results_dir: str | None = None,
    manage_path: str | None = None,
) -> tuple[int, list[str]]:
    """Write JSON files and prepend them to the run_all queue (priority stack).

    Returns (number of paths newly added to pending, all json paths).
    """
    paths = []
    for mu in mu_values:
        json_path = make_job_json(
            job_template,
            mu,
            samples_dir,
            results_base=results_dir,
            manage_csv=manage_path,
        )
        done_json = os.path.join(samples_dir, "done", os.path.basename(json_path))
        if os.path.isfile(done_json):
            os.remove(done_json)
        paths.append(json_path)

    manifest = read_manifest(manifest_path)
    known = set(manifest.get("pending", [])) | set(manifest.get("in_flight", {}).values())

    n_added = prepend_pending(paths, path=manifest_path)

    for json_path, mu in zip(paths, mu_values):
        if json_path not in known and n_added > 0:
            print(f"[analyzer] Enqueued {json_path} (mu={mu:.6f})")
    if n_added == 0 and paths and not all(p in known for p in paths):
        print(
            f"[analyzer] WARNING: wrote {len(paths)} JSON(s) but added 0 to "
            f"'{manifest_path}' (paths may be duplicates in that manifest)",
            file=sys.stderr,
        )
    return n_added, paths


def _jobs_already_in_manifest(paths: list[str], manifest_path: str) -> bool:
    """True when every path is already pending or in flight."""
    if not paths:
        return False
    manifest = read_manifest(manifest_path)
    known = set(manifest.get("pending", [])) | set(manifest.get("in_flight", {}).values())
    return all(p in known for p in paths)


# ---------------------------------------------------------------------------
# Coexistence resolution check
# ---------------------------------------------------------------------------

def phi_is_close_to_zero(phi: float, phi_err: float) -> bool:
    """True if |phi| is within k sigma of zero (with an absolute floor)."""
    threshold = PHI_NEIGHBOR_SIGMA_K * max(phi_err, PHI_ABS_TOL)
    return abs(phi) <= threshold


def is_coex_resolved(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> bool:
    """True when min(psi) has neg/pos phi neighbors that are both near zero."""
    if len(mu_vals) < 3:
        return False

    min_idx = int(np.argmin(psi_vals))
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return False

    phi_left = phi_vals[min_idx - 1]
    phi_right = phi_vals[min_idx + 1]
    if phi_left >= 0 or phi_right <= 0:
        return False

    return (
        phi_is_close_to_zero(phi_left, phi_errs[min_idx - 1])
        and phi_is_close_to_zero(phi_right, phi_errs[min_idx + 1])
    )


def psi_min_index(psi_vals: np.ndarray) -> int:
    return int(np.argmin(psi_vals))


def min_psi_value(psi_vals: np.ndarray) -> float:
    return float(np.min(psi_vals))


def is_psi_minimum_acceptable(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is small enough to trust mu_coex_FITTED."""
    return min_psi_value(psi_vals) <= PSI_COEX_MAX


def interior_psi_minimum(psi_vals: np.ndarray) -> bool:
    """True when argmin(psi) is not at the edge of the sampled mu range."""
    if len(psi_vals) < 3:
        return False
    min_idx = psi_min_index(psi_vals)
    return 0 < min_idx < len(psi_vals) - 1


def has_phi_sign_change(phi_vals: np.ndarray) -> bool:
    signs = np.sign(phi_vals)
    return not np.all(signs == signs[0])


def sign_change_bracket(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
) -> tuple[float, float] | None:
    """Return (mu_lo, mu_hi) bracketing phi=0, or None if no sign change."""
    if not has_phi_sign_change(phi_vals):
        return None
    pos_mask = phi_vals > 0
    neg_mask = phi_vals < 0
    mu_pos = float(mu_vals[pos_mask][np.argmin(np.abs(phi_vals[pos_mask]))])
    mu_neg = float(mu_vals[neg_mask][np.argmin(np.abs(phi_vals[neg_mask]))])
    return min(mu_pos, mu_neg), max(mu_pos, mu_neg)


def unsampled_mus(candidate_mus: list[float], mu_vals: np.ndarray) -> list[float]:
    return [
        m for m in candidate_mus
        if not any(abs(m - float(existing)) < 1e-6 for existing in mu_vals)
    ]


def refinement_mus(mu_lo: float, mu_hi: float, mu_vals: np.ndarray) -> list[float]:
    """Return new mu points inside [mu_lo, mu_hi], subdividing when coarse grid fills bracket."""
    for n in (N_REFINEMENT_POINTS, 2 * N_REFINEMENT_POINTS, 4 * N_REFINEMENT_POINTS):
        candidates = [round(float(m), 6) for m in np.linspace(mu_lo, mu_hi, n)]
        new_mus = unsampled_mus(candidates, mu_vals)
        if new_mus:
            return new_mus

    in_bracket = sorted(
        float(m) for m in mu_vals if mu_lo - 1e-9 <= float(m) <= mu_hi + 1e-9
    )
    midpoints = [
        round(0.5 * (in_bracket[i] + in_bracket[i + 1]), 6)
        for i in range(len(in_bracket) - 1)
    ]
    return unsampled_mus(midpoints, mu_vals)


def count_in_bracket(mu_vals: np.ndarray, mu_lo: float, mu_hi: float) -> int:
    return int(np.sum((mu_vals >= mu_lo) & (mu_vals <= mu_hi)))


def extension_window(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    toward_edge: int | None = None,
) -> tuple[float, float]:
    """Return (mu_lo, mu_hi) for a window extension.

    If toward_edge is 0, extend below the current range; if len(mu)-1, extend above.
    Otherwise infer direction from phi sign (all positive -> lower mu, else higher).
    """
    window = float(mu_vals[-1] - mu_vals[0])
    if toward_edge == 0:
        return float(mu_vals[0] - window), float(mu_vals[0])
    if toward_edge is not None and toward_edge == len(mu_vals) - 1:
        return float(mu_vals[-1]), float(mu_vals[-1] + window)
    if np.all(phi_vals > 0):
        return float(mu_vals[0] - window), float(mu_vals[0])
    return float(mu_vals[-1]), float(mu_vals[-1] + window)


def compute_mu_coex_sim_error(
    mu_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
) -> float:
    """Replica-scatter scale at mu neighbors bracketing argmin(psi).

    This is a phi uncertainty proxy used for resolution checks, not a Delta-mu
    error bar on mu_coex_FITTED.
    """
    min_idx = psi_min_index(psi_vals)
    if min_idx == 0 or min_idx == len(mu_vals) - 1:
        return float(phi_errs[min_idx])
    return float(max(phi_errs[min_idx - 1], phi_errs[min_idx + 1]))


def fit_zero_crossing_with_error(
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
) -> tuple[float, float] | tuple[None, None]:
    """Weighted linear fit to phi(mu); return (mu_zero, mu_zero_err).

    Uses points inside the sign-change bracket plus one neighbour on each side.
    Weights are 1/phi_err^2. Error on the zero crossing is propagated from the
    fit covariance matrix. Returns (None, None) when no sign change exists or
    the fit is degenerate.
    """
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return None, None

    mu_lo, mu_hi = bracket
    sorted_mus = np.sort(mu_vals)
    lo_idx = int(np.searchsorted(sorted_mus, mu_lo))
    hi_idx = int(np.searchsorted(sorted_mus, mu_hi))
    expanded_lo = float(sorted_mus[max(0, lo_idx - 1)])
    expanded_hi = float(sorted_mus[min(len(sorted_mus) - 1, hi_idx + 1)])

    mask = (mu_vals >= expanded_lo - 1e-9) & (mu_vals <= expanded_hi + 1e-9)
    mu_fit = mu_vals[mask]
    phi_fit = phi_vals[mask]
    err_fit = phi_errs[mask]

    if len(mu_fit) < 2:
        return None, None

    w = np.where(err_fit > 0, 1.0 / (err_fit ** 2), 1.0)
    try:
        coeffs, cov = np.polyfit(mu_fit, phi_fit, deg=1, w=np.sqrt(w), cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    slope, intercept = coeffs
    if abs(slope) < 1e-12:
        return None, None

    mu_zero = float(-intercept / slope)

    # Error propagation: var(mu_zero) via delta method
    var_s, var_i = float(cov[0, 0]), float(cov[1, 1])
    cov_si = float(cov[0, 1])
    var_mu = (
        (intercept / slope ** 2) ** 2 * var_s
        + (1.0 / slope) ** 2 * var_i
        - 2.0 * (intercept / slope ** 3) * cov_si
    )
    mu_zero_err = float(np.sqrt(max(0.0, var_mu)))
    return mu_zero, mu_zero_err


def finalize_combo(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Set mu_coex_FITTED via linear fit zero crossing, save curves, mark combo analyzed."""
    mu_coex_fitted, fitted_error = fit_zero_crossing_with_error(mu_vals, phi_vals, phi_errs)
    if mu_coex_fitted is None:
        # fallback to argmin(psi) if fit fails (e.g. only 1 bracket point)
        min_idx = int(np.argmin(psi_vals))
        mu_coex_fitted = float(mu_vals[min_idx])
        fitted_error = compute_mu_coex_sim_error(mu_vals, phi_errs, psi_vals)
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: mu_coex_FITTED = {mu_coex_fitted:.6f}, "
          f"error = {fitted_error:.6f}{suffix}")

    write_combo_curves(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_FITTED": mu_coex_fitted,
        "mu_coex_FITTED_error": fitted_error,
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })


def finalize_unstable(
    combo_key: tuple,
    combo: dict,
    tag: str,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    n_requests: int,
    reason: str = "",
):
    """Mark unstable combo analyzed with mu_coex_FITTED=NaN after max refinement requests."""
    suffix = f" ({reason})" if reason else ""
    print(f"[analyzer] {tag}: unstable, mu_coex_FITTED=NaN{suffix}")
    write_combo_curves(
        combo_key, mu_vals, phi_vals, phi_errs, psi_vals, psi_errs,
        results_dir=results_dir,
    )
    update_manage_field(manage_path, combo, {
        "mu_coex_FITTED": "NaN",
        "mu_coex_FITTED_error": "NaN",
        "isAnalyzed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RequestForAdditionalData": n_requests,
        "combo_path": combo_dir(combo, results_dir),
    })


# ---------------------------------------------------------------------------
# Core analysis logic per combo
# ---------------------------------------------------------------------------

def _request_more_data(
    *,
    tag: str,
    combo_key: tuple,
    combo: dict,
    new_mus: list[float],
    job: dict,
    manage_path: str,
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
    samples_dir: str,
    manifest_path: str,
    results_dir: str,
    pending_points: dict[tuple, int],
    action: str,
) -> bool:
    """Enqueue refinement/extension jobs if needed. Returns True if jobs were queued."""
    if not new_mus:
        return False

    n_added, paths = enqueue_jobs(
        new_mus,
        job,
        samples_dir,
        manifest_path,
        results_dir=results_dir,
        manage_path=manage_path,
    )
    if n_added == 0:
        if _jobs_already_in_manifest(paths, manifest_path):
            if n_requests == 0:
                rows[idx]["RequestForAdditionalData"] = 1
                write_manage(manage_path, rows)
                pending_points[combo_key] = n_points
                print(
                    f"[analyzer] {tag}: {action} jobs already in queue; "
                    f"synced RequestForAdditionalData=1",
                )
            else:
                pending_points.setdefault(combo_key, n_points)
                print(f"[analyzer] {tag}: {action} jobs already queued, waiting")
            return True
        print(f"[analyzer] {tag}: failed to queue {action} jobs")
        return False

    n_requests += 1
    rows[idx]["RequestForAdditionalData"] = n_requests
    write_manage(manage_path, rows)
    pending_points[combo_key] = n_points
    print(f"[analyzer] {tag}: {action}, queued {n_added} jobs "
          f"(request {n_requests}/{MAX_ADDITIONAL_REQUESTS})")
    return True


def _request_psi_improvement(
    *,
    tag: str,
    combo_key: tuple,
    combo: dict,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    psi_vals: np.ndarray,
    manage_path: str,
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
    samples_dir: str,
    manifest_path: str,
    results_dir: str,
    pending_points: dict[tuple, int],
) -> bool:
    """Queue refinement/extension when min(psi) is still above PSI_COEX_MAX."""
    if n_requests >= MAX_ADDITIONAL_REQUESTS:
        return False

    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is not None and interior_psi_minimum(psi_vals):
        mu_lo, mu_hi = bracket
        new_mus = refinement_mus(mu_lo, mu_hi, mu_vals)
        if new_mus:
            return _request_more_data(
                tag=tag,
                combo_key=combo_key,
                combo=combo,
                new_mus=new_mus,
                job=job,
                manage_path=manage_path,
                rows=rows,
                idx=idx,
                n_requests=n_requests,
                n_points=n_points,
                samples_dir=samples_dir,
                manifest_path=manifest_path,
                results_dir=results_dir,
                pending_points=pending_points,
                action="refining (min psi above threshold)",
            )

    min_idx = psi_min_index(psi_vals)
    toward_edge = min_idx if not interior_psi_minimum(psi_vals) else None
    new_lo, new_hi = extension_window(mu_vals, phi_vals, toward_edge=toward_edge)
    new_mus = refinement_mus(new_lo, new_hi, mu_vals)
    if new_mus:
        return _request_more_data(
            tag=tag,
            combo_key=combo_key,
            combo=combo,
            new_mus=new_mus,
            job=job,
            manage_path=manage_path,
            rows=rows,
            idx=idx,
            n_requests=n_requests,
            n_points=n_points,
            samples_dir=samples_dir,
            manifest_path=manifest_path,
            results_dir=results_dir,
            pending_points=pending_points,
            action="extending (min psi above threshold)",
        )
    return False


def _finalize_if_psi_acceptable(
    *,
    finalize_reason: str,
    unstable_reason: str,
    combo_key: tuple,
    combo: dict,
    tag: str,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    samples_dir: str,
    manifest_path: str,
    pending_points: dict[tuple, int],
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
) -> bool:
    """Finalize when min(psi) is acceptable; otherwise refine or mark unstable."""
    if is_psi_minimum_acceptable(psi_vals):
        finalize_combo(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason=finalize_reason,
        )
        return True

    psi_min = min_psi_value(psi_vals)
    print(
        f"[analyzer] {tag}: min(psi)={psi_min:.4f} > {PSI_COEX_MAX}, "
        f"rejecting {finalize_reason!r}",
    )
    if _request_psi_improvement(
        tag=tag,
        combo_key=combo_key,
        combo=combo,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        psi_vals=psi_vals,
        manage_path=manage_path,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        results_dir=results_dir,
        pending_points=pending_points,
    ):
        return True

    reason = unstable_reason or f"min(psi)={psi_min:.4f} > {PSI_COEX_MAX}"
    finalize_unstable(
        combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
        psi_vals, psi_errs, manage_path, results_dir, n_requests,
        reason=reason,
    )
    return True


def _analyze_sign_change(
    *,
    combo_key: tuple,
    combo: dict,
    tag: str,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    samples_dir: str,
    manifest_path: str,
    pending_points: dict[tuple, int],
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
) -> None:
    """Phi changes sign: refine bracket, then finalize with argmin(psi)."""
    bracket = sign_change_bracket(mu_vals, phi_vals)
    if bracket is None:
        return
    mu_lo, mu_hi = bracket

    finalize_kwargs = dict(
        combo_key=combo_key,
        combo=combo,
        tag=tag,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        phi_errs=phi_errs,
        psi_vals=psi_vals,
        psi_errs=psi_errs,
        manage_path=manage_path,
        results_dir=results_dir,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        pending_points=pending_points,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
    )

    if is_coex_resolved(mu_vals, phi_vals, phi_errs, psi_vals):
        _finalize_if_psi_acceptable(
            **finalize_kwargs,
            finalize_reason="neighbors resolved",
            unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
        )
        return

    in_bracket = count_in_bracket(mu_vals, mu_lo, mu_hi)
    bracket_dense = in_bracket >= N_REFINEMENT_POINTS
    interior_min = interior_psi_minimum(psi_vals)

    if bracket_dense and interior_min:
        _finalize_if_psi_acceptable(
            **finalize_kwargs,
            finalize_reason="dense bracket",
            unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
        )
        return

    if not interior_min:
        if n_requests >= MAX_ADDITIONAL_REQUESTS:
            finalize_unstable(
                combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
                psi_vals, psi_errs, manage_path, results_dir, n_requests,
                reason="min(psi) at edge of mu window",
            )
            return

        min_idx = psi_min_index(psi_vals)
        new_lo, new_hi = extension_window(mu_vals, phi_vals, toward_edge=min_idx)
        new_mus = refinement_mus(new_lo, new_hi, mu_vals)
        if not new_mus:
            _finalize_if_psi_acceptable(
                **finalize_kwargs,
                finalize_reason="argmin(psi)",
                unstable_reason="min(psi) at edge of mu window",
            )
            return

        direction = "lower" if min_idx == 0 else "higher"
        print(f"[analyzer] {tag}: min(psi) at {direction} edge, extending "
              f"[{new_lo:.4f}, {new_hi:.4f}]")
        _request_more_data(
            tag=tag,
            combo_key=combo_key,
            combo=combo,
            new_mus=new_mus,
            job=job,
            manage_path=manage_path,
            rows=rows,
            idx=idx,
            n_requests=n_requests,
            n_points=n_points,
            samples_dir=samples_dir,
            manifest_path=manifest_path,
            results_dir=results_dir,
            pending_points=pending_points,
            action="extending toward edge",
        )
        return

    # Interior minimum: try refinement while bracket is sparse and budget remains.
    if not bracket_dense and n_requests < MAX_ADDITIONAL_REQUESTS:
        new_mus = refinement_mus(mu_lo, mu_hi, mu_vals)
        if new_mus:
            print(f"[analyzer] {tag}: sign change, refining "
                  f"[{mu_lo:.4f}, {mu_hi:.4f}] with {len(new_mus)} points "
                  f"({in_bracket}/{N_REFINEMENT_POINTS} in bracket)")
            queued = _request_more_data(
                tag=tag,
                combo_key=combo_key,
                combo=combo,
                new_mus=new_mus,
                job=job,
                manage_path=manage_path,
                rows=rows,
                idx=idx,
                n_requests=n_requests,
                n_points=n_points,
                samples_dir=samples_dir,
                manifest_path=manifest_path,
                results_dir=results_dir,
                pending_points=pending_points,
                action="refining",
            )
            if queued:
                return
            if not bracket_dense:
                print(
                    f"[analyzer] {tag}: refinement already queued; "
                    f"waiting for new data before finalizing",
                )
                return

    _finalize_if_psi_acceptable(
        **finalize_kwargs,
        finalize_reason="argmin(psi)",
        unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
    )


def _analyze_no_sign_change(
    *,
    combo_key: tuple,
    combo: dict,
    tag: str,
    job: dict,
    mu_vals: np.ndarray,
    phi_vals: np.ndarray,
    phi_errs: np.ndarray,
    psi_vals: np.ndarray,
    psi_errs: np.ndarray,
    manage_path: str,
    results_dir: str,
    samples_dir: str,
    manifest_path: str,
    pending_points: dict[tuple, int],
    rows: list[dict],
    idx: int,
    n_requests: int,
    n_points: int,
) -> None:
    """Phi same sign everywhere: extend mu window until sign change or budget exhausted."""
    if n_requests >= MAX_ADDITIONAL_REQUESTS:
        finalize_unstable(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason="max requests, no sign change",
        )
        return

    new_lo, new_hi = extension_window(mu_vals, phi_vals)
    if np.all(phi_vals > 0):
        print(f"[analyzer] {tag}: all phi>0, extending window lower "
              f"[{new_lo:.4f}, {new_hi:.4f}]")
    else:
        print(f"[analyzer] {tag}: all phi<0, extending window higher "
              f"[{new_lo:.4f}, {new_hi:.4f}]")

    new_mus = refinement_mus(new_lo, new_hi, mu_vals)
    if not new_mus:
        finalize_unstable(
            combo_key, combo, tag, mu_vals, phi_vals, phi_errs,
            psi_vals, psi_errs, manage_path, results_dir, n_requests,
            reason="no unsampled mu in extension window",
        )
        return

    _request_more_data(
        tag=tag,
        combo_key=combo_key,
        combo=combo,
        new_mus=new_mus,
        job=job,
        manage_path=manage_path,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        results_dir=results_dir,
        pending_points=pending_points,
        action="extending window",
    )


def _enrich_job_template(job: dict, results_dir: str, manage_path: str) -> dict:
    """Ensure refinement JSONs carry the same paths as initial coex jobs."""
    enriched = dict(job)
    enriched["results_base"] = results_dir
    enriched["manage_csv"] = manage_path
    return enriched


def _manifest_has_activity(manifest_path: str) -> bool:
    manifest = read_manifest(manifest_path)
    return bool(manifest.get("pending")) or bool(manifest.get("in_flight"))


def analyze_combo(combo_key: tuple, data: dict, manage_path: str,
                  results_dir: str, samples_dir: str, manifest_path: str,
                  pending_points: dict[tuple, int]):
    job = _enrich_job_template(data["job"], results_dir, manage_path)
    points = data["points"]

    combo = {f: job[f] for f in COMBO_KEY_FIELDS}
    tag = combo_dir_name(job)

    mu_vals, phi_vals, phi_errs, psi_vals, psi_errs = build_curves(points)
    n_points = len(mu_vals)

    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(
            f"[analyzer] WARNING: {tag}: no manage.csv row for combo "
            f"{ {f: combo[f] for f in COMBO_KEY_FIELDS} }",
            file=sys.stderr,
        )
        return
    n_requests = int(rows[idx].get("RequestForAdditionalData", 0))

    if rows[idx].get("isAnalyzed", ""):
        return

    if n_requests == 0 and n_points < N_INITIAL_MU_POINTS:
        print(f"[analyzer] {tag}: waiting for initial batch "
              f"({n_points}/{N_INITIAL_MU_POINTS})")
        return

    if n_requests > 0:
        points_at_request = pending_points.get(combo_key)
        if points_at_request is not None and n_points <= points_at_request:
            if not _manifest_has_activity(manifest_path):
                print(
                    f"[analyzer] {tag}: refinement stalled (queue empty, "
                    f"still {n_points} mu points); re-analyzing",
                )
                pending_points.pop(combo_key, None)
            elif (
                n_points >= N_INITIAL_MU_POINTS
                and has_phi_sign_change(phi_vals)
                and interior_psi_minimum(psi_vals)
            ):
                _finalize_if_psi_acceptable(
                    combo_key=combo_key,
                    combo=combo,
                    tag=tag,
                    job=job,
                    mu_vals=mu_vals,
                    phi_vals=phi_vals,
                    phi_errs=phi_errs,
                    psi_vals=psi_vals,
                    psi_errs=psi_errs,
                    manage_path=manage_path,
                    results_dir=results_dir,
                    samples_dir=samples_dir,
                    manifest_path=manifest_path,
                    pending_points=pending_points,
                    rows=rows,
                    idx=idx,
                    n_requests=n_requests,
                    n_points=n_points,
                    finalize_reason="initial batch complete (no new refinement data)",
                    unstable_reason=f"min(psi) > {PSI_COEX_MAX} after max requests",
                )
                pending_points.pop(combo_key, None)
                return
            else:
                return

    common = dict(
        combo_key=combo_key,
        combo=combo,
        tag=tag,
        job=job,
        mu_vals=mu_vals,
        phi_vals=phi_vals,
        phi_errs=phi_errs,
        psi_vals=psi_vals,
        psi_errs=psi_errs,
        manage_path=manage_path,
        results_dir=results_dir,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        pending_points=pending_points,
        rows=rows,
        idx=idx,
        n_requests=n_requests,
        n_points=n_points,
    )

    if has_phi_sign_change(phi_vals):
        _analyze_sign_change(**common)
    else:
        _analyze_no_sign_change(**common)


def select_depth_first_combo(
    grouped: dict,
    manage_path: str,
) -> tuple[str, ...] | None:
    """Pick one unfinished combo: finish in-flight refinements first, else lowest epsilon."""
    rows = read_manage(manage_path)
    open_combos: list[tuple[float, tuple[str, ...], bool]] = []

    for combo_key, data in grouped.items():
        combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
        idx = find_manage_row(rows, combo)
        if idx is None or rows[idx].get("isAnalyzed", ""):
            continue
        n_requests = int(rows[idx].get("RequestForAdditionalData", 0) or 0)
        open_combos.append((float(combo["epsilon"]), combo_key, n_requests > 0))

    if not open_combos:
        return None

    in_refinement = [(eps, key) for eps, key, active in open_combos if active]
    if in_refinement:
        return min(in_refinement, key=lambda item: item[0])[1]

    return min(open_combos, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Watch results/ and compute mu_coex_FITTED per combo."
    )
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--manage", default=MANAGE_CSV)
    parser.add_argument("--samples", default=SAMPLES_DIR)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--manifest", default="run_all_queue.json",
                        help="Queue manifest for run_all.py")
    parser.add_argument(
        "--depth-first",
        action="store_true",
        help="Finish mu refinement for one epsilon before starting the next",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one poll cycle then exit (debugging)",
    )
    args = parser.parse_args()

    mode = "depth-first" if args.depth_first else "breadth-first"
    print(f"[analyzer] Watching '{args.results}' every {args.interval}s "
          f"({mode}, Ctrl-C to stop)")

    pending_points: dict[tuple, int] = {}  # point count when last follow-up was queued
    last_focus_key: tuple[str, ...] | None = None

    while True:
        grouped = discover_combo_results(args.results)

        active_key = None
        if args.depth_first:
            active_key = select_depth_first_combo(grouped, args.manage)
            if active_key is None:
                time.sleep(args.interval)
                continue
            if active_key != last_focus_key:
                print(
                    f"[analyzer] depth-first: focusing "
                    f"epsilon={grouped[active_key]['job']['epsilon']}",
                )
                last_focus_key = active_key

        for combo_key, data in grouped.items():
            if args.depth_first and combo_key != active_key:
                continue

            rows = read_manage(args.manage)
            combo = {f: data["job"][f] for f in COMBO_KEY_FIELDS}
            idx = find_manage_row(rows, combo)
            if idx is not None and rows[idx].get("isAnalyzed", ""):
                pending_points.pop(combo_key, None)
                continue

            analyze_combo(
                combo_key, data, args.manage, args.results, args.samples, args.manifest,
                pending_points,
            )

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()