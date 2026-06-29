"""
run_all.py

Long-running dispatcher that submits json_runner.py jobs to Slurm via simple_slurm.

Reads run_all_queue.json, keeps up to MAX_CONCURRENT jobs active, and submits
more as slots free up until the pending queue is empty. Re-enqueues failed jobs
at the front; archives completed job JSON files to samples/done/.

Usage (Della login node):
    python run_all.py
    python run_all.py --config slurm_config.yml --interval 30

Local testing (no Slurm):
    python run_all.py --local --samples test_samples
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import common.queue_manifest as qm
from common.queue_manifest import (
    archive_json,
    cleanup_staged_json,
    ensure_job_json,
    mark_in_flight,
    pop_next_pending,
    read_manifest,
    requeue_front,
    remove_in_flight,
    stage_job_json,
)

MAX_CONCURRENT = 100
POLL_INTERVAL = 30.0
SLURM_CONFIG = "config/slurm_config.yml"
RUNNER_MODULE = "coex.json_runner"
SUCCESS_STATES = {"COMPLETED", "COMPLETING"}
FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
}
# Ly -> Slurm wall time (current campaign is Ly=32-heavy)
LY_WALLTIME = {
    8: "02:00:00",
    16: "04:00:00",
    32: "08:00:00",
}
DEFAULT_WALLTIME = "08:00:00"


def load_slurm_config(path: str) -> tuple[dict, list[str], str | None]:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    setup_cmds = raw.pop("setup_cmds", [])
    report_dir = raw.pop("report_dir", None)
    if report_dir:
        report_dir = os.path.expandvars(os.path.expanduser(report_dir))
    for key in ("output", "error"):
        if key in raw:
            raw[key] = os.path.expandvars(os.path.expanduser(raw[key]))
    return raw, setup_cmds, report_dir


def build_slurm(config_path: str = SLURM_CONFIG, *, time: str | None = None, cpus_per_task: int | None = None):
    from simple_slurm import Slurm

    slurm_kwargs, setup_cmds, report_dir = load_slurm_config(config_path)
    if time is not None:
        slurm_kwargs["time"] = time
    if cpus_per_task is not None:
        slurm_kwargs["cpus_per_task"] = cpus_per_task
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    elif slurm_kwargs.get("output"):
        os.makedirs(os.path.dirname(slurm_kwargs["output"]), exist_ok=True)
    slurm = Slurm(**slurm_kwargs)
    for cmd in setup_cmds:
        slurm.add_cmd(cmd)
    return slurm


def slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def normalize_sacct_state(state: str | None) -> str | None:
    if not state:
        return None
    # sacct may return "TIMEOUT+" or "FAILED|0"
    return state.split()[0].split("|")[0].rstrip("+")


def sacct_state(job_id: str) -> str | None:
    try:
        result = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "--format=State"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return normalize_sacct_state(lines[0] if lines else None)


def active_slurm_ids(slurm) -> set[str]:
    slurm.squeue.update_squeue()
    return {str(job_id) for job_id in slurm.squeue.jobs}


def walltime_for_json(json_path: str, config_path: str = SLURM_CONFIG) -> str:
    """Pick Slurm wall time from Ly in the job JSON, else config default."""
    try:
        with open(json_path) as f:
            params = json.load(f)
        ly = int(params["Ly"])
        return LY_WALLTIME.get(ly, DEFAULT_WALLTIME)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        slurm_kwargs, _, _ = load_slurm_config(config_path)
        return slurm_kwargs.get("time", DEFAULT_WALLTIME)


def submit_slurm_job(
    json_path: str,
    python: str,
    config_path: str = SLURM_CONFIG,
) -> str:
    walltime = walltime_for_json(json_path, config_path)
    slurm = build_slurm(config_path, time=walltime)
    abs_json = os.path.abspath(json_path)
    job_id = slurm.sbatch(f"{python} -u -m {RUNNER_MODULE} {abs_json}")
    return str(job_id)


def run_local_job(json_path: str, python: str) -> bool:
    result = subprocess.run([python, "-u", "-m", RUNNER_MODULE, json_path])
    return result.returncode == 0


def cleanup_job_staging(canonical_path: str) -> None:
    staged = os.path.join(qm.STAGING_DIR, os.path.basename(canonical_path))
    cleanup_staged_json(staged)


def finish_job(json_path: str, success: bool) -> None:
    cleanup_job_staging(json_path)
    if success:
        archive_json(json_path)
        print(f"[run_all] Completed: {json_path}")
    else:
        ensure_job_json(json_path)
        requeue_front(json_path)
        print(f"[run_all] Failed, re-queued: {json_path}")


def reconcile_in_flight(slurm=None, local_jobs: dict[str, subprocess.Popen] | None = None) -> None:
    """Detect finished jobs and archive or re-queue their JSON files."""
    manifest = read_manifest()
    in_flight = dict(manifest.get("in_flight", {}))

    if local_jobs is not None:
        finished = []
        for job_id, proc in list(local_jobs.items()):
            ret = proc.poll()
            if ret is None:
                continue
            json_path = in_flight.get(job_id)
            if json_path:
                finish_job(json_path, ret == 0)
                remove_in_flight(job_id)
            finished.append(job_id)
        for job_id in finished:
            local_jobs.pop(job_id, None)
        return

    active_ids = active_slurm_ids(slurm)
    for job_id, json_path in list(in_flight.items()):
        if job_id in active_ids:
            continue

        state = sacct_state(job_id)
        if state is None:
            print(
                f"[run_all] sacct unknown for job {job_id}, "
                f"leaving in_flight (will retry): {json_path}"
            )
            continue
        if state in SUCCESS_STATES:
            finish_job(json_path, success=True)
        elif state in FAILURE_STATES:
            finish_job(json_path, success=False)
        else:
            print(f"[run_all] job {job_id} state={state!r}, re-queuing {json_path}")
            finish_job(json_path, success=False)

        remove_in_flight(job_id)


def submit_up_to_cap(
    *,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
    python: str,
    max_concurrent: int,
    config_path: str = SLURM_CONFIG,
) -> int:
    """Submit jobs until in_flight reaches max_concurrent. Returns count submitted."""
    submitted = 0

    while True:
        manifest = read_manifest()
        in_flight_count = len(manifest.get("in_flight", {}))
        if local_jobs is not None:
            in_flight_count = len(local_jobs)
        if in_flight_count >= max_concurrent:
            break

        json_path = pop_next_pending()
        if json_path is None:
            break

        if not ensure_job_json(json_path):
            print(
                f"[run_all] WARNING: missing JSON (not in samples/ or done/), "
                f"re-queuing for later: {json_path}"
            )
            requeue_front(json_path)
            continue

        staged_path = stage_job_json(json_path)

        if local_jobs is not None:
            proc = subprocess.Popen([python, "-u", "-m", RUNNER_MODULE, staged_path])
            job_id = f"local-{proc.pid}"
            mark_in_flight(job_id, json_path)
            local_jobs[job_id] = proc
        else:
            job_id = submit_slurm_job(staged_path, python, config_path)
            mark_in_flight(job_id, json_path)

        print(f"[run_all] Submitted job {job_id}: {json_path} (staged: {staged_path})")
        submitted += 1

    return submitted


def dispatch_once(
    *,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
    python: str,
    max_concurrent: int,
    config_path: str = SLURM_CONFIG,
) -> tuple[int, int, int]:
    reconcile_in_flight(slurm=slurm, local_jobs=local_jobs)
    n_submitted = submit_up_to_cap(
        slurm=slurm,
        local_jobs=local_jobs,
        python=python,
        max_concurrent=max_concurrent,
        config_path=config_path,
    )
    manifest = read_manifest()
    pending = len(manifest.get("pending", []))
    in_flight = len(manifest.get("in_flight", {}))
    return n_submitted, pending, in_flight


def main():
    parser = argparse.ArgumentParser(description="Slurm dispatcher for json_runner jobs")
    parser.add_argument("--manifest", default=qm.MANIFEST_PATH)
    parser.add_argument("--config", default=SLURM_CONFIG)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run jobs locally via subprocess instead of Slurm",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one dispatch cycle and exit",
    )
    args = parser.parse_args()

    qm.MANIFEST_PATH = args.manifest

    use_local = args.local or not slurm_available()
    if use_local and not args.local:
        print("[run_all] sbatch not found — running in local mode")

    python = sys.executable
    slurm = None
    local_jobs: dict[str, subprocess.Popen] | None = {} if use_local else None

    if not use_local:
        slurm = build_slurm(args.config)

    print(
        f"[run_all] Watching {args.manifest} "
        f"(max {args.max_concurrent} concurrent, interval {args.interval}s)"
    )

    while True:
        n_submitted, pending, in_flight = dispatch_once(
            slurm=slurm,
            local_jobs=local_jobs,
            python=python,
            max_concurrent=args.max_concurrent,
            config_path=args.config,
        )
        print(
            f"[run_all] cycle: submitted={n_submitted}, "
            f"pending={pending}, in_flight={in_flight}"
        )

        if args.once:
            break
        if pending == 0 and in_flight == 0:
            print("[run_all] Queue empty — exiting")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
