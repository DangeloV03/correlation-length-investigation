"""
json_runner.py

Takes a single self-contained JSON job (one outer combo + one mu value) and
runs `num_parallel_runs` independent replicas in parallel (multiprocessing).

For each replica:
  - Build the slab initial condition (x < Lx/2 -> active, rest empty).
  - Construct HeteroChain with fugacities derived from mu, delta_f.
  - Run equilibration (eq_time), discard.
  - Run production (prod_time), record time-averaged densities.
  - Save final lattice as final_lattice_{i}.npy.

Outputs (in a per-run output directory):
  - final_lattice_0.npy ... final_lattice_{n-1}.npy
  - output.csv: id, rho_active, rho_inert, rho_empty, time

Usage:
    python json_runner.py samples/homo_Ly16_mu00.json [--outdir results/.../mu...]
"""

import argparse
import csv
import json
import os
import shutil
import time
import multiprocessing as mp

import numpy as np

from coex.paths import COMBO_KEY_FIELDS, RESULTS_DIR, combo_dir, mu_dir, mu_sweeps_dir
from lattice_gas import load
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.markov_chain import HeteroChain
from lattice_gas.simulate import simulate


def _results_base(params: dict) -> str:
    return params.get("results_base", RESULTS_DIR)


def _manage_csv_path(params: dict) -> str:
    return params.get("manage_csv", "manage.csv")

EMPTY, INERT, BONDING = 0, 1, 2


def build_initial_state(Lx: int, Ly: int) -> np.ndarray:
    """Slab initial condition: x < Lx/2 -> active (BONDING), rest -> empty."""
    state = np.zeros((Lx, Ly), dtype=np.uint32)
    state[: Lx // 2, :] = BONDING
    return state


def compute_densities(state: np.ndarray) -> tuple[float, float, float]:
    total = state.size
    rho_active = float(np.count_nonzero(state == BONDING)) / total
    rho_inert = float(np.count_nonzero(state == INERT)) / total
    rho_empty = float(np.count_nonzero(state == EMPTY)) / total
    return rho_active, rho_inert, rho_empty


def run_replica(args):
    (
        replica_id,
        run_id,
        seed,
        params,
        run_settings,
        outdir,
    ) = args

    beta = run_settings["beta"]
    epsilon = params["epsilon"]
    delta_mu = params["delta_mu"]
    delta_f = params["delta_f"]
    k = params["k"]
    scheme = params["scheme"]
    mu = params["mu"]
    Lx = params["Lx"]
    Ly = params["Ly"]

    eq_time = run_settings["eq_time"]
    prod_time = run_settings["prod_time"]
    n_chunks = run_settings.get("prod_chunks", 10)

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
    state = build_initial_state(Lx, Ly)

    scratch_dir = os.path.join(outdir, f"_scratch_{replica_id}")

    simulate(state, boundary, chain, [], [Time(eq_time)], seed, scratch_dir)
    state = load.final_state(scratch_dir)
    print(f"[json_runner] replica={replica_id} equilibration done", flush=True)

    chunk_time = prod_time / n_chunks
    rho_active_samples = []
    rho_inert_samples = []
    rho_empty_samples = []
    cumulative_time = 0.0

    for chunk_idx in range(n_chunks):
        chunk_seed = seed + 1 + chunk_idx
        simulate(state, boundary, chain, [], [Time(chunk_time)], chunk_seed, scratch_dir)
        state = load.final_state(scratch_dir)
        cumulative_time += load.final_time(scratch_dir)

        rho_active, rho_inert, rho_empty = compute_densities(state)
        rho_active_samples.append(rho_active)
        rho_inert_samples.append(rho_inert)
        rho_empty_samples.append(rho_empty)
        print(
            f"[json_runner] replica={replica_id} chunk {chunk_idx + 1}/{n_chunks} "
            f"t={cumulative_time:.1f}",
            flush=True,
        )

    rho_active = float(np.mean(rho_active_samples))
    rho_inert = float(np.mean(rho_inert_samples))
    rho_empty = float(np.mean(rho_empty_samples))

    np.save(os.path.join(outdir, f"final_lattice_{run_id}.npy"), state)

    shutil.rmtree(scratch_dir, ignore_errors=True)

    return {
        "id": run_id,
        "rho_active": rho_active,
        "rho_inert": rho_inert,
        "rho_empty": rho_empty,
        "time": cumulative_time,
    }


def get_next_id(csv_path: str) -> int:
    """Return the next sequential id to use, continuing from any existing rows."""
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_ids = [int(r["id"]) for r in reader]
    return max(existing_ids) + 1 if existing_ids else 0


def append_to_csv(csv_path: str, rows: list[dict], params: dict):
    """Append rows to output.csv. Rows must already have their final 'id' set."""
    file_exists = os.path.isfile(csv_path)

    run_settings = params["run_settings"]
    fieldnames = [
        "id", "epsilon", "delta_f", "delta_mu", "k",
        "scheme", "Lx", "Ly", "mu",
        "rho_active", "rho_inert", "rho_empty", "time",
        "beta", "num_parallel_runs", "eq_time", "prod_time", "seed_base",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["epsilon"] = params["epsilon"]
            row_out["delta_f"] = params["delta_f"]
            row_out["delta_mu"] = params["delta_mu"]
            row_out["k"] = params["k"]
            row_out["scheme"] = params["scheme"]
            row_out["Lx"] = params["Lx"]
            row_out["Ly"] = params["Ly"]
            row_out["mu"] = params["mu"]
            row_out["beta"] = run_settings["beta"]
            row_out["num_parallel_runs"] = run_settings["num_parallel_runs"]
            row_out["eq_time"] = run_settings["eq_time"]
            row_out["prod_time"] = run_settings["prod_time"]
            row_out["seed_base"] = run_settings["seed_base"]
            writer.writerow(row_out)


def update_manage_csv(manage_path: str, params: dict) -> None:
    """Set isRan and combo_path on the matching manage.csv row when still empty."""
    if not os.path.isfile(manage_path):
        return

    with open(manage_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return

    if "combo_path" not in fieldnames:
        fieldnames.append("combo_path")

    combo_path = combo_dir(params, base=_results_base(params))
    updated = False
    for row in rows:
        if row.get("isRan", "") and row.get("combo_path", ""):
            continue
        if all(str(row[field]) == str(params[field]) for field in COMBO_KEY_FIELDS):
            if not row.get("isRan", ""):
                row["isRan"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if not row.get("combo_path", ""):
                row["combo_path"] = combo_path
            updated = True
            break

    if not updated:
        return

    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the job JSON file")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: results/<combo>/mu...)",
    )
    args = parser.parse_args()

    with open(args.json_path) as f:
        params = json.load(f)

    run_settings = params["run_settings"]
    num_parallel_runs = run_settings["num_parallel_runs"]
    seed_base = run_settings["seed_base"]

    results_base = _results_base(params)
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = mu_dir(params, base=results_base)
    os.makedirs(combo_dir(params, base=results_base), exist_ok=True)
    os.makedirs(mu_sweeps_dir(params, base=results_base), exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    eps = params["epsilon"]
    dmu = params["delta_mu"]
    mu_val = params["mu"]
    ly = params["Ly"]
    print(
        f"[json_runner] START {args.json_path} "
        f"eps={eps} dmu={dmu} mu={mu_val} "
        f"Ly={ly} replicas={num_parallel_runs} outdir={outdir}",
        flush=True,
    )

    csv_path = os.path.join(outdir, "output.csv")
    next_id = get_next_id(csv_path)

    tasks = []
    for replica_id in range(num_parallel_runs):
        run_id = next_id + replica_id
        seed = seed_base + run_id * 2
        tasks.append((replica_id, run_id, seed, params, run_settings, outdir))

    with mp.Pool(processes=num_parallel_runs) as pool:
        results = pool.map(run_replica, tasks)

    results.sort(key=lambda r: r["id"])

    append_to_csv(csv_path, results, params)

    n_rows = len(results)
    print(f"Wrote {n_rows} rows to {csv_path}", flush=True)
    for r in results:
        rid = r["id"]
        ra = r["rho_active"]
        ri = r["rho_inert"]
        re_ = r["rho_empty"]
        t = r["time"]
        print(
            f"  replica {rid}: rho_active={ra:.4f} rho_inert={ri:.4f} "
            f"rho_empty={re_:.4f} time={t:.2f}",
            flush=True,
        )

    update_manage_csv(_manage_csv_path(params), params)


if __name__ == "__main__":
    main()
