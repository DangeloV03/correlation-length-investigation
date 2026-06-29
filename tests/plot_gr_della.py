"""
One-shot Della test: compute G(r) from snapshots and save a plot.

Usage (GPU node on Della):
    python tests/plot_gr_della.py /path/to/snapshot_metadata.csv

Writes:
    G_r.csv          -- radially-averaged connected correlation
    correlation_length.csv  -- xi fit result
    G_r_plot.png     -- G(r) vs r on linear and log scales
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed on Della
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correlation.analyzer import compute_G_r, fit_xi, read_snapshot_paths, _write_csv
from correlation.paths import ANALYSIS_CSV, METADATA_CSV


def plot_G_r(
    r_centers: np.ndarray,
    G_r: np.ndarray,
    fit: dict,
    output_path: str,
) -> None:
    r_plot = np.linspace(1, fit["r_max"], 300)
    G_plot = fit["A"] * np.exp(-r_plot / fit["xi"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    ax = axes[0]
    ax.plot(r_centers[1:], G_r[1:], "o", ms=4, label="G(r) data")
    ax.plot(r_plot, G_plot, "-", lw=1.5,
            label=rf"fit: $\xi$ = {fit['xi']:.3f} ± {fit['xi_err']:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("r (lattice units)")
    ax.set_ylabel("G(r)")
    ax.set_title(f"Connected correlation  (L={fit.get('L', '?')}, n={fit.get('n_snapshots', '?')})")
    ax.legend()

    # Log scale
    ax = axes[1]
    positive = G_r[1:] > 0
    r_pos = r_centers[1:][positive]
    ax.semilogy(r_pos, G_r[1:][positive], "o", ms=4, label="G(r) data")
    ax.semilogy(r_plot, G_plot, "-", lw=1.5,
                label=rf"fit: $\xi$ = {fit['xi']:.3f} ± {fit['xi_err']:.3f}")
    ax.set_xlabel("r (lattice units)")
    ax.set_ylabel("G(r)  [log scale]")
    ax.set_title("Exponential decay check")
    ax.legend()

    if fit.get("warning"):
        fig.text(0.5, 0.01, f"WARNING: {fit['warning']}",
                 ha="center", color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[plot_gr] saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute G(r) and plot correlation length from lattice snapshots"
    )
    parser.add_argument("path", help="snapshot_metadata.csv or directory containing it")
    parser.add_argument("--output-dir", default=None,
                        help="where to write CSVs and plot (default: same dir as metadata)")
    args = parser.parse_args()

    metadata_path = args.path
    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, METADATA_CSV)
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(metadata_path)

    outdir = args.output_dir or os.path.dirname(metadata_path)
    os.makedirs(outdir, exist_ok=True)

    print(f"[plot_gr] reading snapshots from {metadata_path}")
    paths = read_snapshot_paths(metadata_path)
    if not paths:
        raise FileNotFoundError(f"no readable snapshots listed in {metadata_path}")
    print(f"[plot_gr] found {len(paths)} snapshot(s)")

    r_centers, G_r, L = compute_G_r(paths)
    fit = fit_xi(r_centers, G_r, L)
    fit["L"] = L
    fit["n_snapshots"] = len(paths)

    print(f"[plot_gr] xi = {fit['xi']:.4f} ± {fit['xi_err']:.4f}  "
          f"A = {fit['A']:.4f}  fit_points = {fit['n_points']}")
    if fit["warning"]:
        print(f"[plot_gr] WARNING: {fit['warning']}")

    _write_csv(
        os.path.join(outdir, "G_r.csv"),
        [{"r": int(r), "G_r": float(g)} for r, g in zip(r_centers, G_r)],
        ["r", "G_r"],
    )
    _write_csv(
        os.path.join(outdir, ANALYSIS_CSV),
        [fit],
        ["n_snapshots", "L", "xi", "xi_err", "A", "A_err",
         "n_points", "r_min", "r_max", "warning"],
    )

    plot_G_r(r_centers, G_r, fit, os.path.join(outdir, "G_r_plot.png"))


if __name__ == "__main__":
    main()
