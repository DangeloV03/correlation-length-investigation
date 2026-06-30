"""
Single-replica worker for the L=256 epsilon sweep.

Runs one production replica at a fixed epsilon value, saves snapshots,
computes G(r), and fits ξ.  Intended to be called by submit_eps_sweep.sh
as one task per Slurm array element.

Directory layout written:
    {output_dir}/eps_{epsilon:.3f}/replica_{id:05d}/
        snapshots/chunk_*.npy
        snapshot_metadata.csv
        G_r.csv
        correlation_length.csv

Usage:
    python tests/eps_sweep_runner.py \\
        --epsilon -1.750 --replica-id 3 \\
        --output-dir /scratch/gpfs/WJACOBS/vd7294/correlation-length-investigation/eps_sweep_L256
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coex.flex_coex import coex_chemical_potential
from coex.generate_samples import DELTA_F, FLEX_INDEX
from correlation.analyzer import compute_G_r, fit_xi, _write_csv
from correlation.paths import ANALYSIS_CSV, METADATA_CSV, METADATA_FIELDS

DELTA_MU  = 0.0
K         = 1.0
SCHEME    = "homo"
BETA      = 1.0
L         = 256
EQ_TIME   = 50_000
PROD_TIME = 100_000
N_CHUNKS  = 1000

EMPTY, INERT, BONDING = 0, 1, 2


def run_simulation(outdir: str, epsilon: float, replica_id: int) -> list[str]:
    from lattice_gas import load
    from lattice_gas.boundary_condition import Periodic
    from lattice_gas.ending_criterion import Time
    from lattice_gas.markov_chain import HeteroChain
    from lattice_gas.simulate import simulate

    mu = coex_chemical_potential(
        epsilon=epsilon, df=DELTA_F, dmu=DELTA_MU,
        chem_rec_baserate=K, DRIVEN=False, scheme=FLEX_INDEX,
    )
    tag = f"[eps={epsilon:.3f} rep={replica_id}]"
    print(f"{tag} mu_coex = {mu:.6f}", flush=True)
    print(f"{tag} L={L}  eq={EQ_TIME}  prod={PROD_TIME}  chunks={N_CHUNKS}", flush=True)

    inert_fugacity   = np.exp(BETA * (mu + DELTA_F))
    bonding_fugacity = np.exp(BETA * mu)
    chain = HeteroChain(BETA, epsilon, DELTA_MU, inert_fugacity, bonding_fugacity, K, SCHEME)
    boundary = Periodic()

    seed = 42 + replica_id * 1000
    rng = np.random.default_rng(seed)
    state = np.zeros((L, L), dtype=np.uint32)
    state[rng.random((L, L)) < 0.5] = BONDING

    scratch = os.path.join(outdir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    print(f"{tag} equilibrating …", flush=True)
    simulate(state, boundary, chain, [], [Time(EQ_TIME)], seed, scratch)
    state = load.final_state(scratch)
    print(f"{tag} equilibration done", flush=True)

    snap_dir = os.path.join(outdir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    chunk_time = PROD_TIME / N_CHUNKS
    snap_paths: list[str] = []
    rows: list[dict] = []

    for i in range(N_CHUNKS):
        simulate(state, boundary, chain, [], [Time(chunk_time)], seed + 1 + i, scratch)
        state = load.final_state(scratch)
        t = load.final_time(scratch)

        path = os.path.join(snap_dir, f"chunk_{i:05d}.npy")
        np.save(path, state)
        snap_paths.append(path)

        rows.append({
            "run_id": 0,
            "replica_id": replica_id,
            "chunk": i,
            "snapshot_path": os.path.relpath(path, outdir),
            "time": t,
            "chunk_time": chunk_time,
            "epsilon": epsilon,
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
            "seed": seed,
        })

        if (i + 1) % 100 == 0:
            print(f"{tag} {i + 1}/{N_CHUNKS} chunks done", flush=True)

    shutil.rmtree(scratch, ignore_errors=True)

    meta_path = os.path.join(outdir, METADATA_CSV)
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})
    print(f"{tag} metadata → {meta_path}", flush=True)
    return snap_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--replica-id", type=int, required=True)
    parser.add_argument(
        "--output-dir", required=True,
        help="Base dir; results go in eps_{val}/replica_{id:05d}/",
    )
    args = parser.parse_args()

    eps_tag = f"{args.epsilon:.3f}"
    outdir = os.path.join(
        args.output_dir, f"eps_{eps_tag}", f"replica_{args.replica_id:05d}"
    )
    os.makedirs(outdir, exist_ok=True)
    print(f"[eps_sweep_runner] output: {outdir}", flush=True)

    snap_paths = run_simulation(outdir, args.epsilon, args.replica_id)

    print(f"[eps_sweep_runner] computing G(r) …", flush=True)
    r_centers, G_r, L_out = compute_G_r(snap_paths)
    fit = fit_xi(r_centers, G_r, L_out)
    fit.update({
        "n_snapshots": len(snap_paths),
        "L": L_out,
        "epsilon": args.epsilon,
        "replica_id": args.replica_id,
    })

    tag = f"[eps={args.epsilon:.3f} rep={args.replica_id}]"
    print(
        f"{tag} xi = {fit['xi']:.4f} ± {fit['xi_err']:.4f}  "
        f"A = {fit['A']:.4f}  fit_pts = {fit['n_points']}",
        flush=True,
    )
    if fit.get("warning"):
        print(f"{tag} WARNING: {fit['warning']}", flush=True)

    _write_csv(
        os.path.join(outdir, "G_r.csv"),
        [{"r": int(r), "G_r": float(g)} for r, g in zip(r_centers, G_r)],
        ["r", "G_r"],
    )
    _write_csv(
        os.path.join(outdir, ANALYSIS_CSV),
        [fit],
        ["epsilon", "replica_id", "n_snapshots", "L",
         "xi", "xi_err", "A", "A_err", "n_points", "r_min", "r_max", "warning"],
    )
    print("[eps_sweep_runner] done", flush=True)


if __name__ == "__main__":
    main()
