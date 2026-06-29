"""
Compute G(r) and fit correlation length from saved lattice snapshots.

Algorithm (see ising_correlation_length.md):
  1. Load each snapshot (.npy), convert lattice values → ±1 spins.
  2. FFT autocorrelation → connected 2-point correlation G2D = C2D - m².
  3. Radially average G2D → G(r), averaged over all snapshots.
  4. Fit G(r) = A·exp(-r/ξ) to get the correlation length ξ.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from correlation.paths import ANALYSIS_CSV, METADATA_CSV

EMPTY, INERT, BONDING = 0, 1, 2


def lattice_to_spins(lattice: np.ndarray) -> np.ndarray:
    """BONDING sites → +1, everything else → -1."""
    return np.where(lattice == BONDING, 1.0, -1.0)


def read_snapshot_paths(metadata_path: str) -> list[str]:
    base = os.path.dirname(metadata_path)
    paths = []
    with open(metadata_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("snapshot_path", "")
            if not raw:
                continue
            path = raw if os.path.isabs(raw) else os.path.join(base, raw)
            if os.path.isfile(path):
                paths.append(path)
    return paths


def compute_G_r(snapshot_paths: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the radially-averaged connected correlation G(r), averaged over snapshots.

    Returns
    -------
    r_centers : 1-D array of integer distances 0, 1, ..., L//2 - 1
    G_r       : mean connected correlation at each distance
    L         : lattice side length
    """
    accum: np.ndarray | None = None
    m2_sum = 0.0
    n_snaps = 0
    L: int = 0

    for path in snapshot_paths:
        spins = lattice_to_spins(np.load(path))
        if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
            raise ValueError(f"expected square 2D snapshot, got {spins.shape}: {path}")
        L = int(spins.shape[0])
        N = L * L

        F = np.fft.fft2(spins)
        C2D = np.fft.ifft2(F * np.conj(F)).real / N
        C2D = np.fft.fftshift(C2D)

        m = float(np.mean(spins))
        G2D = C2D - m * m

        if accum is None:
            accum = np.zeros_like(G2D)
        accum += G2D
        m2_sum += m * m
        n_snaps += 1

    if accum is None or n_snaps == 0:
        raise ValueError("no snapshots loaded")

    G2D_avg = accum / n_snaps

    # Distance from the shifted origin at (L//2, L//2)
    cy, cx = L // 2, L // 2
    dx, dy = np.meshgrid(np.arange(L) - cx, np.arange(L) - cy)
    r = np.sqrt(dx**2 + dy**2)

    r_max = L // 2
    bin_edges = np.arange(0, r_max + 1, dtype=float) - 0.5
    bin_edges[0] = -0.5

    G_sum, _ = np.histogram(r.ravel(), bins=bin_edges, weights=G2D_avg.ravel())
    counts, _ = np.histogram(r.ravel(), bins=bin_edges)
    r_centers = np.arange(r_max, dtype=float)
    G_r = G_sum / counts

    return r_centers, G_r, L


def fit_xi(
    r_centers: np.ndarray,
    G_r: np.ndarray,
    L: int,
) -> dict:
    """
    Fit G(r) = A·exp(-r/ξ) for r ≥ 1 where G(r) > 0 and r < L/2.

    Returns a dict with xi, xi_err, A, A_err, n_points, r_min, r_max, warning.
    """
    # Skip r=1 contact term where possible; fall back to r=1 for small systems.
    mask = (r_centers >= 2) & (G_r > 0) & (r_centers < L / 2)
    if np.sum(mask) < 3:
        mask = (r_centers >= 1) & (G_r > 0) & (r_centers < L / 2)
    r_fit = r_centers[mask]
    G_fit = G_r[mask]

    warning = ""
    if len(r_fit) < 2:
        raise ValueError("not enough positive G(r) points to fit xi")

    def exp_decay(r, A, xi):
        return A * np.exp(-r / xi)

    # Estimate xi from the log-slope of the first two points; much better than L/4.
    if len(G_fit) >= 2 and G_fit[1] > 0:
        xi0 = float(-(r_fit[1] - r_fit[0]) / np.log(G_fit[1] / G_fit[0]))
        xi0 = max(xi0, 0.5)
    else:
        xi0 = float(L / 4)
    p0 = [float(G_fit[0]), xi0]
    popt, pcov = curve_fit(
        exp_decay, r_fit, G_fit, p0=p0,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=10_000,
    )
    A_fit, xi_fit = float(popt[0]), float(popt[1])
    A_err, xi_err = float(np.sqrt(pcov[0, 0])), float(np.sqrt(pcov[1, 1]))

    if xi_fit >= L / 6:
        warning = "xi >= L/6; finite-size effects may bias the estimate"

    return {
        "xi": xi_fit,
        "xi_err": xi_err,
        "A": A_fit,
        "A_err": A_err,
        "n_points": int(np.sum(mask)),
        "r_min": float(r_fit[0]),
        "r_max": float(r_fit[-1]),
        "warning": warning,
    }


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def analyze(metadata_path: str, *, output_dir: str | None = None) -> dict:
    """
    Full analysis pipeline: load snapshots → compute G(r) → fit ξ → write CSVs.

    Returns the fit result dict.
    """
    paths = read_snapshot_paths(metadata_path)
    if not paths:
        raise FileNotFoundError(f"no readable snapshots listed in {metadata_path}")

    r_centers, G_r, L = compute_G_r(paths)
    fit = fit_xi(r_centers, G_r, L)

    outdir = output_dir or os.path.dirname(metadata_path)

    _write_csv(
        os.path.join(outdir, "G_r.csv"),
        [{"r": int(r), "G_r": float(g)} for r, g in zip(r_centers, G_r)],
        ["r", "G_r"],
    )

    result = {"n_snapshots": len(paths), "L": L, **fit}
    _write_csv(
        os.path.join(outdir, ANALYSIS_CSV),
        [result],
        ["n_snapshots", "L", "xi", "xi_err", "A", "A_err",
         "n_points", "r_min", "r_max", "warning"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute G(r) and fit correlation length from lattice snapshots"
    )
    parser.add_argument(
        "path",
        help="snapshot_metadata.csv or a directory containing it",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metadata_path = args.path
    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, METADATA_CSV)
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(metadata_path)

    r = analyze(metadata_path, output_dir=args.output_dir)
    print(
        f"[analyzer] L={r['L']}  n={r['n_snapshots']}  "
        f"xi={r['xi']:.4f} ± {r['xi_err']:.4f}  "
        f"A={r['A']:.4f}  fit_points={r['n_points']}"
    )
    if r["warning"]:
        print(f"[analyzer] WARNING: {r['warning']}")


if __name__ == "__main__":
    main()
    