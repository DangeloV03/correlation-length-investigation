"""
Analyze saved lattice snapshots to estimate correlation length.

This tool brute-forces correlation length estimation over spatial correlation data 
from saved lattice snapshots, rather than relying on analytic estimators from Ising models.
Both Fourier-space and real-space correlation length fitting are performed, and a radial 
G(r) CSV is written using FFT-based autocorrelation with periodic boundaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from correlation.paths import ANALYSIS_CSV, METADATA_CSV

EMPTY, INERT, BONDING = 0, 1, 2


@dataclass
class FitResult:
    xi: float
    chi: float
    intercept: float
    slope: float
    n_points: int
    max_k_sq: float
    warning: str


@dataclass
class RealSpaceFitResult:
    xi: float
    slope: float
    n_points: int
    r_min: float
    r_max: float
    warning: str


def lattice_to_spins(lattice: np.ndarray, mapping: str = "active_vs_other") -> np.ndarray:
    if mapping == "active_vs_other":
        return np.where(lattice == BONDING, 1.0, -1.0)
    if mapping == "active_vs_empty":
        return np.where(lattice == BONDING, 1.0, np.where(lattice == EMPTY, -1.0, 0.0))
    raise ValueError(f"unknown spin mapping: {mapping}")


def read_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path, newline="") as f:
        return list(csv.DictReader(f))


def snapshot_paths(metadata_path: str) -> list[str]:
    rows = read_metadata(metadata_path)
    base = os.path.dirname(metadata_path)
    paths = []
    for row in rows:
        raw = row.get("snapshot_path", "")
        if not raw:
            continue
        path = raw if os.path.isabs(raw) else os.path.join(base, raw)
        if os.path.isfile(path):
            paths.append(path)
    return paths


def load_spin_snapshots(paths: list[str], mapping: str) -> list[np.ndarray]:
    snapshots = []
    shape = None
    for path in paths:
        lattice = np.load(path)
        spins = lattice_to_spins(lattice, mapping)
        if shape is None:
            shape = spins.shape
        elif spins.shape != shape:
            raise ValueError(f"snapshot shape mismatch: {path} has {spins.shape}, expected {shape}")
        snapshots.append(spins)
    if not snapshots:
        raise ValueError("no readable snapshots found")
    return snapshots


def average_power_spectrum(snapshots: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    n_sites = snapshots[0].size
    power_accum = np.zeros(snapshots[0].shape, dtype=float)
    magnetizations = []
    for spins in snapshots:
        fft = np.fft.fftn(spins)
        power_accum += np.abs(fft) ** 2 / n_sites
        magnetizations.append(float(np.mean(spins)))
    g_hat = power_accum / len(snapshots)
    return g_hat, float(np.mean(magnetizations)), float(np.mean(np.square(magnetizations)))


def radial_mode_series(g_hat: np.ndarray, max_modes: int | None = None) -> list[dict]:
    if g_hat.ndim != 2 or g_hat.shape[0] != g_hat.shape[1]:
        raise ValueError("radial Fourier fit currently expects square 2D snapshots")

    L = g_hat.shape[0]
    n_max = L // 2
    if max_modes is not None:
        n_max = min(n_max, max_modes)

    rows = []
    for n in range(1, n_max + 1):
        k0 = 2.0 * math.pi * n / L
        g_val = float(g_hat[n % L, n % L])
        k_sq = 2.0 * k0 * k0
        rows.append({
            "mode": n,
            "k0": k0,
            "k_sq": k_sq,
            "sin_sq": 2.0 * (math.sin(0.5 * k0) ** 2),
            "g_hat": g_val,
            "inv_g_hat": 1.0 / g_val if g_val > 0 else math.nan,
        })
    return rows


def fit_correlation_length(
    mode_rows: list[dict],
    *,
    L: int,
    min_points: int = 2,
    initial_points: int = 6,
) -> FitResult:
    candidates = [row for row in mode_rows if row["g_hat"] > 0 and np.isfinite(row["inv_g_hat"])]
    if len(candidates) < min_points:
        raise ValueError("not enough positive Fourier modes to fit xi")

    def fit_rows(rows: list[dict]) -> tuple[float, float] | None:
        x_vals = np.asarray([row["sin_sq"] for row in rows], dtype=float)
        y_vals = np.asarray([row["inv_g_hat"] for row in rows], dtype=float)
        slope_val, intercept_val = np.polyfit(x_vals, y_vals, deg=1)
        if intercept_val <= 0 or slope_val <= 0:
            return None
        return float(slope_val), float(intercept_val)

    selected = candidates[:max(min_points, min(initial_points, len(candidates)))]
    initial_fit = fit_rows(selected)
    if initial_fit is None:
        for n_points in range(min_points, min(initial_points, len(candidates)) + 1):
            trial = candidates[:n_points]
            if fit_rows(trial) is not None:
                selected = trial
                initial_fit = fit_rows(trial)
                break
    if initial_fit is None:
        raise ValueError("small-k fit produced nonpositive intercept or slope")

    for _ in range(3):
        fit = fit_rows(selected)
        if fit is None:
            break
        slope, intercept = fit
        xi = float(math.sqrt(slope / intercept))
        filtered = [row for row in candidates if xi * xi * row["sin_sq"] < 1.0]
        if len(filtered) < min_points or fit_rows(filtered) is None:
            break
        if [row["mode"] for row in filtered] == [row["mode"] for row in selected]:
            break
        selected = filtered

    final_fit = fit_rows(selected)
    if final_fit is None:
        raise ValueError("small-k fit produced nonpositive intercept or slope")
    slope, intercept = final_fit
    x = np.asarray([row["sin_sq"] for row in selected], dtype=float)

    xi = float(math.sqrt(slope / intercept))
    chi = float(1.0 / intercept)
    warning = ""
    if xi >= L / 6:
        warning = "xi is not much smaller than L; finite-size effects may be important"

    return FitResult(
        xi=xi,
        chi=chi,
        intercept=float(intercept),
        slope=float(slope),
        n_points=len(selected),
        max_k_sq=float(np.max(x)),
        warning=warning,
    )


def radial_real_space_correlation(
    snapshots: list[np.ndarray],
    *,
    subtract_magnetization: bool = True,
) -> list[dict]:
    shape = snapshots[0].shape
    if len(shape) != 2:
        raise ValueError("real-space radial G(r) currently expects 2D snapshots")

    accum = np.zeros(shape, dtype=float)
    m2_vals = []
    n_sites = snapshots[0].size
    for spins in snapshots:
        fft = np.fft.fftn(spins)
        corr = np.fft.ifftn(np.abs(fft) ** 2).real / n_sites
        accum += corr
        m2_vals.append(float(np.mean(spins)) ** 2)
    corr = accum / len(snapshots)
    if subtract_magnetization:
        corr = corr - float(np.mean(m2_vals))

    Lx, Ly = shape
    bins: dict[int, list[float]] = {}
    for dx in range(Lx):
        mx = min(dx, Lx - dx)
        for dy in range(Ly):
            my = min(dy, Ly - dy)
            r = int(round(math.sqrt(mx * mx + my * my)))
            bins.setdefault(r, []).append(float(corr[dx, dy]))

    return [
        {"r": r, "G_r": float(np.mean(values)), "n_offsets": len(values)}
        for r, values in sorted(bins.items())
    ]


def fit_real_space_xi(
    gr: list[dict],
    *,
    L: int,
    r_min: int = 2,
) -> RealSpaceFitResult:
    """Fit connected G(r) ~ exp(-r/xi) via log(G) vs r in an intermediate distance window."""
    r_max_limit = L / 2
    points = [
        row for row in gr
        if row["r"] > 0
        and row["r"] >= r_min
        and row["r"] <= r_max_limit
        and row["G_r"] > 0
    ]
    if len(points) < 2:
        raise ValueError("not enough positive G(r) points for real-space xi fit")

    r_vals = np.asarray([row["r"] for row in points], dtype=float)
    log_g = np.log(np.asarray([row["G_r"] for row in points], dtype=float))
    slope, _intercept = np.polyfit(r_vals, log_g, deg=1)

    warning = ""
    if len(points) < 4:
        warning = "real-space fit used fewer than 4 points"
    if slope >= 0:
        unphysical = "real-space fit slope is non-negative (unphysical decay)"
        warning = f"{warning}; {unphysical}".strip("; ") if warning else unphysical
        xi = float("nan")
    else:
        xi = float(-1.0 / slope)

    return RealSpaceFitResult(
        xi=xi,
        slope=float(slope),
        n_points=len(points),
        r_min=float(np.min(r_vals)),
        r_max=float(np.max(r_vals)),
        warning=warning,
    )


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def analyze_metadata(
    metadata_path: str,
    *,
    mapping: str = "active_vs_other",
    max_modes: int | None = None,
    output_dir: str | None = None,
) -> tuple[FitResult, RealSpaceFitResult]:
    paths = snapshot_paths(metadata_path)
    snapshots = load_spin_snapshots(paths, mapping)
    Lx, Ly = snapshots[0].shape
    if Lx != Ly:
        raise ValueError(f"correlation analysis expects square snapshots, got {Lx}x{Ly}")

    outdir = output_dir or os.path.dirname(metadata_path)
    g_hat, m_mean, m2_mean = average_power_spectrum(snapshots)
    modes = radial_mode_series(g_hat, max_modes=max_modes)
    fit = fit_correlation_length(modes, L=Lx)
    gr = radial_real_space_correlation(snapshots, subtract_magnetization=True)
    rs_fit = fit_real_space_xi(gr, L=Lx)

    write_csv(
        os.path.join(outdir, "fourier_modes.csv"),
        modes,
        ["mode", "k0", "k_sq", "sin_sq", "g_hat", "inv_g_hat"],
    )
    write_csv(
        os.path.join(outdir, "G_r.csv"),
        gr,
        ["r", "G_r", "n_offsets"],
    )
    write_csv(
        os.path.join(outdir, ANALYSIS_CSV),
        [{
            "n_snapshots": len(snapshots),
            "L": Lx,
            "mapping": mapping,
            "m_mean": m_mean,
            "m2_mean": m2_mean,
            "xi": fit.xi,
            "xi_realspace": rs_fit.xi,
            "chi": fit.chi,
            "fit_intercept": fit.intercept,
            "fit_slope": fit.slope,
            "fit_points": fit.n_points,
            "fit_max_k_sq": fit.max_k_sq,
            "realspace_slope": rs_fit.slope,
            "realspace_fit_points": rs_fit.n_points,
            "realspace_r_min": rs_fit.r_min,
            "realspace_r_max": rs_fit.r_max,
            "warning": fit.warning,
            "realspace_warning": rs_fit.warning,
        }],
        [
            "n_snapshots",
            "L",
            "mapping",
            "m_mean",
            "m2_mean",
            "xi",
            "xi_realspace",
            "chi",
            "fit_intercept",
            "fit_slope",
            "fit_points",
            "fit_max_k_sq",
            "realspace_slope",
            "realspace_fit_points",
            "realspace_r_min",
            "realspace_r_max",
            "warning",
            "realspace_warning",
        ],
    )
    return fit, rs_fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate correlation length from chunk snapshots")
    parser.add_argument(
        "path",
        help="Path to snapshot_metadata.csv or a correlation result directory containing it",
    )
    parser.add_argument(
        "--mapping",
        choices=["active_vs_other", "active_vs_empty"],
        default="active_vs_other",
    )
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metadata_path = args.path
    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, METADATA_CSV)
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(metadata_path)

    fit, rs_fit = analyze_metadata(
        metadata_path,
        mapping=args.mapping,
        max_modes=args.max_modes,
        output_dir=args.output_dir,
    )
    print(
        f"[correlation_analyzer] xi_fourier={fit.xi:.6g} xi_realspace={rs_fit.xi:.6g} "
        f"chi={fit.chi:.6g} fit_points={fit.n_points} rs_points={rs_fit.n_points}"
    )
    if fit.warning:
        print(f"[correlation_analyzer] Fourier WARNING: {fit.warning}")
    if rs_fit.warning:
        print(f"[correlation_analyzer] Real-space WARNING: {rs_fit.warning}")


if __name__ == "__main__":
    main()
