"""
Post-processing plot for the L=256 epsilon sweep.

Reads all per-replica G_r.csv and correlation_length.csv files produced by
eps_sweep_runner.py, then generates two figures:

  eps_sweep_Gr.png      -- G(r) vs r for each epsilon (replica-averaged),
                           colored from blue (-1.700) to red (-1.800)
  eps_sweep_xi_vs_eps.png -- xi vs epsilon with error bars

Usage:
    python tests/plot_eps_sweep.py /path/to/eps_sweep_L256
    python tests/plot_eps_sweep.py /path/to/eps_sweep_L256 --output-dir /path/to/plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correlation.paths import ANALYSIS_CSV


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_all_results(base_dir: str) -> tuple[dict, dict]:
    """
    Walk base_dir looking for eps_*/replica_*/correlation_length.csv and G_r.csv.

    Returns
    -------
    xi_data : dict mapping epsilon (float) → list of dicts with xi, xi_err, etc.
    gr_data : dict mapping epsilon (float) → list of (r_centers, G_r) arrays
    """
    xi_data: dict[float, list[dict]] = {}
    gr_data: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}

    for eps_dir in sorted(Path(base_dir).glob("eps_*")):
        if not eps_dir.is_dir():
            continue
        try:
            eps = float(eps_dir.name.replace("eps_", ""))
        except ValueError:
            continue

        for rep_dir in sorted(eps_dir.glob("replica_*")):
            xi_path = rep_dir / ANALYSIS_CSV
            gr_path = rep_dir / "G_r.csv"

            if xi_path.is_file():
                rows = _read_csv(str(xi_path))
                if rows:
                    row = rows[0]
                    try:
                        xi_data.setdefault(eps, []).append({
                            "xi":      float(row["xi"]),
                            "xi_err":  float(row["xi_err"]),
                            "A":       float(row["A"]),
                            "warning": row.get("warning", ""),
                        })
                    except (KeyError, ValueError):
                        pass

            if gr_path.is_file():
                rows = _read_csv(str(gr_path))
                if rows:
                    try:
                        r = np.array([float(r["r"]) for r in rows])
                        g = np.array([float(r["G_r"]) for r in rows])
                        gr_data.setdefault(eps, []).append((r, g))
                    except (KeyError, ValueError):
                        pass

    return xi_data, gr_data


# ── Statistics ────────────────────────────────────────────────────────────────

def weighted_mean_xi(entries: list[dict]) -> tuple[float, float, int]:
    """Inverse-variance weighted mean and combined error, plus replica count."""
    xis    = np.array([e["xi"]     for e in entries])
    errs   = np.array([e["xi_err"] for e in entries])
    errs   = np.where(errs > 0, errs, np.median(errs[errs > 0]) if np.any(errs > 0) else 1.0)
    w      = 1.0 / errs**2
    xi_avg = float(np.sum(w * xis) / np.sum(w))
    xi_err = float(1.0 / np.sqrt(np.sum(w)))
    return xi_avg, xi_err, len(entries)


def average_Gr(gr_list: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average G(r) curves over replicas (all must share the same r grid)."""
    g_stack = np.stack([g for _, g in gr_list])
    return gr_list[0][0], np.mean(g_stack, axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_Gr(gr_data: dict[float, list], outpath: str) -> None:
    eps_vals = sorted(gr_data.keys())
    if not eps_vals:
        print("[plot_eps_sweep] no G(r) data found, skipping G(r) plot")
        return

    cmap = plt.colormaps.get_cmap("coolwarm_r")
    colors = cmap(np.linspace(0, 1, len(eps_vals)))

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for eps, color in zip(eps_vals, colors):
        r_avg, G_avg = average_Gr(gr_data[eps])
        label = rf"$\epsilon$={eps:.3f}"

        ax = axes[0]
        ax.plot(r_avg[1:], G_avg[1:], "-", color=color, lw=1.2, label=label, alpha=0.85)

        ax = axes[1]
        pos = G_avg[1:] > 0
        if np.any(pos):
            ax.semilogy(r_avg[1:][pos], G_avg[1:][pos], "-", color=color,
                        lw=1.2, label=label, alpha=0.85)

    for ax, ylabel, title in zip(
        axes,
        ["G(r)", "G(r)  [log scale]"],
        [f"Connected G(r) — L=256, {len(eps_vals)} ε values",
         "Exponential decay check"],
    ):
        ax.set_xlabel("r  (lattice units)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", lw=0.6, ls="--")

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm_r",
        norm=mcolors.Normalize(vmin=min(eps_vals), vmax=max(eps_vals)),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label=r"$\epsilon$", shrink=0.8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"[plot_eps_sweep] G(r) plot → {outpath}")


def plot_xi_vs_eps(xi_data: dict[float, list], outpath: str) -> None:
    eps_vals = sorted(xi_data.keys())
    if not eps_vals:
        print("[plot_eps_sweep] no xi data found, skipping xi plot")
        return

    xi_avgs, xi_errs, n_reps = [], [], []
    for eps in eps_vals:
        xi_avg, xi_err, n = weighted_mean_xi(xi_data[eps])
        xi_avgs.append(xi_avg)
        xi_errs.append(xi_err)
        n_reps.append(n)

    xi_avgs = np.array(xi_avgs)
    xi_errs = np.array(xi_errs)
    eps_arr  = np.array(eps_vals)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    ax.errorbar(eps_arr, xi_avgs, yerr=xi_errs, fmt="o-", capsize=4,
                color="steelblue", lw=1.5, ms=5, label="weighted mean ± 1σ")
    for eps, entries in xi_data.items():
        ax.scatter([eps] * len(entries),
                   [e["xi"] for e in entries],
                   s=8, color="gray", alpha=0.4, zorder=2)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\xi$  (lattice units)")
    ax.set_title(r"Correlation length vs $\epsilon$ — L=256")
    ax.legend()

    # Log scale (to check power-law divergence)
    ax = axes[1]
    ax.errorbar(eps_arr, xi_avgs, yerr=xi_errs, fmt="o-", capsize=4,
                color="steelblue", lw=1.5, ms=5, label="weighted mean ± 1σ")
    for eps, entries in xi_data.items():
        ax.scatter([eps] * len(entries),
                   [e["xi"] for e in entries],
                   s=8, color="gray", alpha=0.4, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\xi$  [log scale]")
    ax.set_title(r"Log scale (check for divergence)")
    ax.legend()

    # Annotate replica count
    for eps, n in zip(eps_vals, n_reps):
        if n < 16:
            axes[0].annotate(f"n={n}", (eps, xi_data[eps][0]["xi"]),
                             fontsize=6, color="red", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"[plot_eps_sweep] xi vs eps plot → {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot G(r) and xi vs eps from an eps_sweep_L256 output directory"
    )
    parser.add_argument("sweep_dir", help="Base output directory from eps_sweep_runner.py")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write plots (default: sweep_dir)")
    args = parser.parse_args()

    sweep_dir = os.path.abspath(args.sweep_dir)
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")

    outdir = args.output_dir or sweep_dir
    os.makedirs(outdir, exist_ok=True)

    print(f"[plot_eps_sweep] scanning {sweep_dir} …")
    xi_data, gr_data = load_all_results(sweep_dir)

    eps_done = sorted(xi_data.keys())
    total_reps = sum(len(v) for v in xi_data.values())
    print(f"[plot_eps_sweep] found {len(eps_done)} eps values, {total_reps} replicas total")
    if not eps_done:
        print("[plot_eps_sweep] nothing to plot — no completed results found")
        sys.exit(0)

    plot_Gr(gr_data, os.path.join(outdir, "eps_sweep_Gr.png"))
    plot_xi_vs_eps(xi_data, os.path.join(outdir, "eps_sweep_xi_vs_eps.png"))
    print("[plot_eps_sweep] done")


if __name__ == "__main__":
    main()
