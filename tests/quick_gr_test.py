"""
Quick end-to-end test: run one lattice-gas replica on Della, compute G(r), plot.

Uses the analytical equilibrium mu_coex — no Stage 1 / manage.csv needed.
Run on a GPU node (or any node with lattice_gas installed):

    python tests/quick_gr_test.py [--output-dir /path/to/outdir]

Writes to --output-dir (default: /tmp/quick_gr_test/):
    snapshots/       -- .npy chunk snapshots
    snapshot_metadata.csv
    G_r.csv
    correlation_length.csv
    G_r_plot.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coex.flex_coex import coex_chemical_potential
from correlation.analyzer import compute_G_r, fit_xi, _write_csv
from correlation.paths import ANALYSIS_CSV, METADATA_CSV, METADATA_FIELDS

# ── Physics parameters ────────────────────────────────────────────────────────
EPSILON   = 1.5     # binding energy
DELTA_F   = 0.0     # free-energy offset (INERT vs BONDING)
DELTA_MU  = 0.0     # driven asymmetry (0 = equilibrium)
K         = 1.0     # chemical recombination base rate
SCHEME    = 1
BETA      = 1.0

# ── Simulation size / timing ─────────────────────────────────────────────────
L         = 64      # square lattice side
EQ_TIME   = 2_000   # equilibration sweeps
PROD_TIME = 5_000   # total production sweeps
N_CHUNKS  = 10      # snapshots to collect
SEED      = 42

EMPTY, INERT, BONDING = 0, 1, 2


def run_simulation(outdir: str) -> list[str]:
    from lattice_gas import load
    from lattice_gas.boundary_condition import Periodic
    from lattice_gas.ending_criterion import Time
    from lattice_gas.markov_chain import HeteroChain
    from lattice_gas.simulate import simulate

    mu = coex_chemical_potential(
        epsilon=EPSILON, df=DELTA_F, dmu=DELTA_MU,
        chem_rec_baserate=K, DRIVEN=False, scheme=SCHEME,
    )
    print(f"[quick_gr_test] mu_coex (analytical) = {mu:.6f}", flush=True)

    inert_fugacity   = np.exp(BETA * (mu + DELTA_F))
    bonding_fugacity = np.exp(BETA * mu)
    chain = HeteroChain(BETA, EPSILON, DELTA_MU, inert_fugacity, bonding_fugacity, K, SCHEME)
    boundary = Periodic()

    rng = np.random.default_rng(SEED)
    state = np.zeros((L, L), dtype=np.uint32)
    state[rng.random((L, L)) < 0.5] = BONDING

    scratch = os.path.join(outdir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    print(f"[quick_gr_test] equilibrating for {EQ_TIME} sweeps …", flush=True)
    simulate(state, boundary, chain, [], [Time(EQ_TIME)], SEED, scratch)
    state = load.final_state(scratch)
    print("[quick_gr_test] equilibration done", flush=True)

    snap_dir = os.path.join(outdir, "snapshots", "replica_00000")
    os.makedirs(snap_dir, exist_ok=True)

    chunk_time = PROD_TIME / N_CHUNKS
    snap_paths: list[str] = []
    rows: list[dict] = []

    for i in range(N_CHUNKS):
        simulate(state, boundary, chain, [], [Time(chunk_time)], SEED + 1 + i, scratch)
        state = load.final_state(scratch)
        t = load.final_time(scratch)

        path = os.path.join(snap_dir, f"chunk_{i:05d}.npy")
        np.save(path, state)
        snap_paths.append(path)

        rows.append({
            "run_id": 0,
            "replica_id": 0,
            "chunk": i,
            "snapshot_path": os.path.relpath(path, outdir),
            "time": t,
            "chunk_time": chunk_time,
            "epsilon": EPSILON,
            "delta_f": DELTA_F,
            "delta_mu": DELTA_MU,
            "k": K,
            "scheme": SCHEME,
            "Lx": L,
            "Ly": L,
            "mu": mu,
            "mu_coex_FITTED": mu,
            "beta": BETA,
            "eq_time": EQ_TIME,
            "prod_time": PROD_TIME,
            "prod_chunks": N_CHUNKS,
            "seed": SEED,
        })
        print(f"[quick_gr_test] chunk {i + 1}/{N_CHUNKS} saved", flush=True)

    shutil.rmtree(scratch, ignore_errors=True)

    meta_path = os.path.join(outdir, METADATA_CSV)
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})

    print(f"[quick_gr_test] wrote {meta_path}", flush=True)
    return snap_paths


def plot_G_r(
    r_centers: np.ndarray,
    G_r: np.ndarray,
    fit: dict,
    output_path: str,
) -> None:
    r_plot = np.linspace(1, fit["r_max"], 300)
    G_plot = fit["A"] * np.exp(-r_plot / fit["xi"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(r_centers[1:], G_r[1:], "o", ms=4, label="G(r) data")
    ax.plot(r_plot, G_plot, "-", lw=1.5,
            label=rf"fit: $\xi$ = {fit['xi']:.3f} ± {fit['xi_err']:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("r  (lattice units)")
    ax.set_ylabel("G(r)")
    ax.set_title(
        rf"Connected G(r) — L={L}, $\epsilon$={EPSILON}, "
        rf"n_snap={fit['n_snapshots']}"
    )
    ax.legend()

    ax = axes[1]
    pos = G_r[1:] > 0
    ax.semilogy(r_centers[1:][pos], G_r[1:][pos], "o", ms=4, label="G(r) data")
    ax.semilogy(r_plot, G_plot, "-", lw=1.5,
                label=rf"$\xi$ = {fit['xi']:.3f} ± {fit['xi_err']:.3f}")
    ax.set_xlabel("r  (lattice units)")
    ax.set_ylabel("G(r)  [log]")
    ax.set_title("Exponential decay check")
    ax.legend()

    if fit.get("warning"):
        fig.text(0.5, 0.01, f"WARNING: {fit['warning']}",
                 ha="center", color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[quick_gr_test] plot saved → {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/quick_gr_test")
    args = parser.parse_args()

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    print(f"[quick_gr_test] output dir: {outdir}", flush=True)

    snap_paths = run_simulation(outdir)

    print("[quick_gr_test] computing G(r) …", flush=True)
    r_centers, G_r, L_out = compute_G_r(snap_paths)

    fit = fit_xi(r_centers, G_r, L_out)
    fit["n_snapshots"] = len(snap_paths)
    fit["L"] = L_out

    print(
        f"[quick_gr_test] xi = {fit['xi']:.4f} ± {fit['xi_err']:.4f}  "
        f"A = {fit['A']:.4f}  fit_points = {fit['n_points']}",
        flush=True,
    )
    if fit["warning"]:
        print(f"[quick_gr_test] WARNING: {fit['warning']}", flush=True)

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
    print("[quick_gr_test] done", flush=True)


if __name__ == "__main__":
    main()
