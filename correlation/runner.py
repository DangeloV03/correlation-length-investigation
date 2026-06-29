"""
Run square-lattice production jobs and save chunk lattice snapshots.

Each replica equilibrates once, then runs production in prod_chunks. After each
chunk, the final lattice state is saved as .npy for correlation_analyzer.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil

import numpy as np

from correlation.paths import (
    METADATA_FIELDS,
    METADATA_CSV,
    RESULTS_DIR,
    SNAPSHOTS_DIR,
    correlation_dir,
)
from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate

EMPTY, INERT, BONDING = 0, 1, 2


def build_initial_state(
    Lx: int,
    Ly: int,
    *,
    initial_fraction: float = 0.8,
    seed: int | None = None,
) -> np.ndarray:
    """Random square-lattice initial condition with active sites and vacancies."""
    rng = np.random.default_rng(seed)
    state = np.zeros((Lx, Ly), dtype=np.uint32)
    state[rng.random((Lx, Ly)) < initial_fraction] = BONDING
    return state


def _results_base(params: dict) -> str:
    return params.get("results_base", RESULTS_DIR)


def get_next_id(csv_path: str) -> int:
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        ids = [int(row["run_id"]) for row in reader if row.get("run_id", "").strip()]
    return max(ids) + 1 if ids else 0


def _relative_snapshot_path(full_path: str, outdir: str) -> str:
    return os.path.relpath(full_path, outdir)


def _replica_snapshot_dir(outdir: str, run_id: int) -> str:
    return os.path.join(outdir, SNAPSHOTS_DIR, f"replica_{int(run_id):05d}")


def _snapshot_path(outdir: str, run_id: int, chunk_idx: int) -> str:
    return os.path.join(
        _replica_snapshot_dir(outdir, run_id),
        f"chunk_{int(chunk_idx):05d}.npy",
    )


def run_replica(args):
    replica_id, run_id, seed, params, run_settings, outdir = args

    beta = run_settings["beta"]
    epsilon = params["epsilon"]
    delta_mu = params["delta_mu"]
    delta_f = params["delta_f"]
    k = params["k"]
    scheme = params["scheme"]
    mu = params["mu"]
    Lx = int(params["Lx"])
    Ly = int(params["Ly"])

    eq_time = float(run_settings["eq_time"])
    prod_time = float(run_settings["prod_time"])
    n_chunks = int(run_settings.get("prod_chunks", 100))
    initial_fraction = float(run_settings.get("initial_fraction", 0.8))

    inert_fugacity = np.exp(beta * (mu + delta_f))
    bonding_fugacity = np.exp(beta * mu)
    chain = HeteroChain(
        beta,
        epsilon,
        delta_mu,
        inert_fugacity,
        bonding_fugacity,
        k,
        scheme,
    )

    boundary = Periodic()
    state = build_initial_state(Lx, Ly, initial_fraction=initial_fraction, seed=seed)
    scratch_dir = os.path.join(outdir, f"_scratch_{replica_id}")

    simulate(state, boundary, chain, [], [Time(eq_time)], seed, scratch_dir)
    state = load.final_state(scratch_dir)
    print(f"[correlation_runner] replica={replica_id} equilibration done", flush=True)

    os.makedirs(_replica_snapshot_dir(outdir, run_id), exist_ok=True)
    chunk_time = prod_time / n_chunks
    cumulative_time = 0.0
    rows: list[dict] = []

    for chunk_idx in range(n_chunks):
        chunk_seed = seed + 1 + chunk_idx
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seed, scratch_dir)
        state = load.final_state(scratch_dir)
        cumulative_time += load.final_time(scratch_dir)

        full_snapshot_path = _snapshot_path(outdir, run_id, chunk_idx)
        os.makedirs(os.path.dirname(full_snapshot_path), exist_ok=True)
        np.save(full_snapshot_path, state)

        rows.append({
            "run_id": run_id,
            "replica_id": replica_id,
            "chunk": chunk_idx,
            "snapshot_path": _relative_snapshot_path(full_snapshot_path, outdir),
            "time": cumulative_time,
            "chunk_time": chunk_time,
            "epsilon": epsilon,
            "delta_f": delta_f,
            "delta_mu": delta_mu,
            "k": k,
            "scheme": scheme,
            "Lx": Lx,
            "Ly": Ly,
            "mu": mu,
            "mu_coex_FITTED": params.get("mu_coex_FITTED", mu),
            "beta": beta,
            "eq_time": eq_time,
            "prod_time": prod_time,
            "prod_chunks": n_chunks,
            "seed": seed,
        })
        print(
            f"[correlation_runner] replica={replica_id} chunk {chunk_idx + 1}/{n_chunks} "
            f"t={cumulative_time:.1f} saved={full_snapshot_path}",
            flush=True,
        )

    np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), state)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    return rows


def append_metadata(csv_path: str, rows: list[dict]) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunk-snapshot correlation production")
    parser.add_argument("json_path", help="Path to correlation production job JSON")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: correlation_results/correlation_<combo>/)",
    )
    args = parser.parse_args()

    with open(args.json_path) as f:
        params = json.load(f)

    run_settings = params["run_settings"]
    num_parallel_runs = int(run_settings["num_parallel_runs"])
    num_batches = int(run_settings.get("num_batches", 1))
    seed_base = int(run_settings["seed_base"])
    results_base = _results_base(params)
    outdir = args.outdir or correlation_dir(params, base=results_base)
    os.makedirs(outdir, exist_ok=True)

    print(
        f"[correlation_runner] START {args.json_path} "
        f"eps={params['epsilon']} L={params['Lx']} mu={params['mu']} "
        f"replicas={num_parallel_runs} batches={num_batches} outdir={outdir}",
        flush=True,
    )

    csv_path = os.path.join(outdir, METADATA_CSV)
    for batch_idx in range(num_batches):
        next_id = get_next_id(csv_path)
        tasks = []
        for replica_id in range(num_parallel_runs):
            run_id = next_id + replica_id
            seed = seed_base + run_id * 2
            tasks.append((replica_id, run_id, seed, params, run_settings, outdir))

        with mp.Pool(processes=num_parallel_runs) as pool:
            batch_results = pool.map(run_replica, tasks)

        rows = [row for replica_rows in batch_results for row in replica_rows]
        rows.sort(key=lambda row: (int(row["run_id"]), int(row["chunk"])))
        append_metadata(csv_path, rows)
        print(
            f"[correlation_runner] batch {batch_idx + 1}/{num_batches}: "
            f"wrote {len(rows)} snapshot rows to {csv_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
