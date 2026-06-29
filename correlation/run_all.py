"""
Slurm/local dispatcher for correlation production jobs.

Reads correlation_queue.json, stages job JSONs, and runs correlation.runner.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import common.queue_manifest as qm
from correlation.paths import MANIFEST, SAMPLES_DIR
from common.queue_manifest import (
    archive_json,
    cleanup_staged_json,
    ensure_job_json,
    mark_in_flight,
    pop_next_pending,
    read_manifest,
    remove_in_flight,
    requeue_front,
    stage_job_json,
)
from coex.run_all import (
    FAILURE_STATES,
    MAX_CONCURRENT,
    POLL_INTERVAL,
    SLURM_CONFIG,
    SUCCESS_STATES,
    active_slurm_ids,
    build_slurm,
    sacct_state,
    slurm_available,
    walltime_for_json,
)

RUNNER_MODULE = "correlation.runner"


def submit_slurm_job(json_path: str, python: str, config_path: str = SLURM_CONFIG) -> str:
    walltime = walltime_for_json(json_path, config_path)
    slurm = build_slurm(config_path, time=walltime)
    job_id = slurm.sbatch(f"{python} -u -m {RUNNER_MODULE} {os.path.abspath(json_path)}")
    return str(job_id)


def finish_job(
    json_path: str,
    success: bool,
    *,
    manifest_path: str,
    done_dir: str,
    staging_dir: str,
) -> None:
    cleanup_staged_json(os.path.join(staging_dir, os.path.basename(json_path)))
    if success:
        archive_json(json_path, done_dir=done_dir)
        print(f"[run_correlation_all] Completed: {json_path}")
    else:
        ensure_job_json(json_path, done_dir=done_dir)
        requeue_front(json_path, path=manifest_path)
        print(f"[run_correlation_all] Failed, re-queued: {json_path}")


def reconcile_in_flight(
    *,
    manifest_path: str,
    done_dir: str,
    staging_dir: str,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
) -> None:
    manifest = read_manifest(manifest_path)
    in_flight = dict(manifest.get("in_flight", {}))

    if local_jobs is not None:
        for job_id, proc in list(local_jobs.items()):
            ret = proc.poll()
            if ret is None:
                continue
            json_path = in_flight.get(job_id)
            if json_path:
                finish_job(
                    json_path,
                    ret == 0,
                    manifest_path=manifest_path,
                    done_dir=done_dir,
                    staging_dir=staging_dir,
                )
                remove_in_flight(job_id, path=manifest_path)
            local_jobs.pop(job_id, None)
        return

    active_ids = active_slurm_ids(slurm)
    for job_id, json_path in list(in_flight.items()):
        if job_id in active_ids:
            continue
        state = sacct_state(job_id)
        if state is None:
            print(
                f"[run_correlation_all] sacct unknown for job {job_id}, leaving in_flight: {json_path}",
                file=sys.stderr,
            )
            continue
        success = state in SUCCESS_STATES
        if state in FAILURE_STATES:
            success = False
        finish_job(
            json_path,
            success,
            manifest_path=manifest_path,
            done_dir=done_dir,
            staging_dir=staging_dir,
        )
        remove_in_flight(job_id, path=manifest_path)


def submit_up_to_cap(
    *,
    manifest_path: str,
    done_dir: str,
    staging_dir: str,
    python: str,
    max_concurrent: int,
    config_path: str,
    slurm=None,
    local_jobs: dict[str, subprocess.Popen] | None = None,
) -> int:
    submitted = 0
    while True:
        manifest = read_manifest(manifest_path)
        in_flight_count = len(manifest.get("in_flight", {}))
        if local_jobs is not None:
            in_flight_count = len(local_jobs)
        if in_flight_count >= max_concurrent:
            break

        json_path = pop_next_pending(manifest_path)
        if json_path is None:
            break
        if not ensure_job_json(json_path, done_dir=done_dir):
            print(f"[run_correlation_all] missing JSON, re-queuing: {json_path}")
            requeue_front(json_path, path=manifest_path)
            continue

        staged_path = stage_job_json(json_path, staging_dir=staging_dir)
        if local_jobs is not None:
            proc = subprocess.Popen([python, "-u", "-m", RUNNER_MODULE, staged_path])
            job_id = f"local-{proc.pid}"
            local_jobs[job_id] = proc
        else:
            job_id = submit_slurm_job(staged_path, python, config_path)

        mark_in_flight(job_id, json_path, path=manifest_path)
        print(f"[run_correlation_all] submitted {job_id}: {json_path}")
        submitted += 1

    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatcher for correlation production jobs")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--samples", default=SAMPLES_DIR)
    parser.add_argument("--config", default=SLURM_CONFIG)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    done_dir = os.path.join(args.samples, "done")
    staging_dir = os.path.join(args.samples, "staging")
    qm.MANIFEST_PATH = args.manifest
    qm.DONE_DIR = done_dir
    qm.STAGING_DIR = staging_dir

    use_local = args.local or not slurm_available()
    if use_local and not args.local:
        print("[run_correlation_all] sbatch not found - running in local mode")
    python = sys.executable
    slurm = None
    local_jobs: dict[str, subprocess.Popen] | None = {} if use_local else None
    if not use_local:
        slurm = build_slurm(args.config)

    print(
        f"[run_correlation_all] Watching {args.manifest} "
        f"(max {args.max_concurrent} concurrent, mode={'local' if use_local else 'slurm'})"
    )

    while True:
        reconcile_in_flight(
            manifest_path=args.manifest,
            done_dir=done_dir,
            staging_dir=staging_dir,
            slurm=slurm,
            local_jobs=local_jobs,
        )
        n_submitted = submit_up_to_cap(
            manifest_path=args.manifest,
            done_dir=done_dir,
            staging_dir=staging_dir,
            python=python,
            max_concurrent=args.max_concurrent,
            config_path=args.config,
            slurm=slurm,
            local_jobs=local_jobs,
        )
        manifest = read_manifest(args.manifest)
        pending = len(manifest.get("pending", []))
        in_flight = len(manifest.get("in_flight", {}))
        print(
            f"[run_correlation_all] cycle: submitted={n_submitted}, "
            f"pending={pending}, in_flight={in_flight}"
        )
        if args.once:
            break
        if pending == 0 and in_flight == 0:
            print("[run_correlation_all] Queue empty - exiting")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
