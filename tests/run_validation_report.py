#!/usr/bin/env python3
"""
Print correlation-length validation ladder results (run on Della or locally).

Usage:
  python tests/run_validation_report.py
  python tests/run_validation_report.py --mc   # include slow Metropolis MC steps
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from correlation.analyzer import (
    fit_correlation_length,
    fit_real_space_xi,
    radial_mode_series,
    radial_real_space_correlation,
)
from tests.ising_reference import (
    analyze_spin_snapshots,
    generate_fourier_model_snapshots,
    gr_log_decay_diagnostics,
    ising_critical_beta,
    ising_exact_xi_high_temp,
    reduced_temperature,
    relative_error,
    run_ising_metropolis,
)


def _print_gr_summary(gr, L: int) -> None:
    diag = gr_log_decay_diagnostics(gr, L=L)
    positive = [(row["r"], row["G_r"]) for row in gr if row["r"] > 0 and row["G_r"] > 0]
    print(f"    G(r): {diag['n_positive']} positive points, r_max={diag['r_max_positive']:.0f}")
    if diag["upward_tail"]:
        print("    G(r) WARNING: upward curvature at large r (possible finite-size corruption)")
    if diag["negatives_before_decay"]:
        print("    G(r) WARNING: non-positive values before r ~ L/4")
    if diag["insufficient_dynamic_range"]:
        print("    G(r) WARNING: fewer than 4 positive points for a reliable log fit")
    if positive:
        head = positive[:6]
        tail = positive[-2:] if len(positive) > 8 else []
        parts = [f"({r:.0f},{g:.2e})" for r, g in head]
        if tail and len(positive) > 8:
            parts.append("...")
            parts.extend(f"({r:.0f},{g:.2e})" for r, g in tail)
        print(f"    G(r) preview: {', '.join(parts)}")


def step_1_high_temp_realspace() -> None:
    print("\n=== Step 1: high-T real-space vs exact xi = -1/ln(tanh(beta)) ===")
    beta = 0.43
    xi_exact = ising_exact_xi_high_temp(beta)
    print(f"  beta={beta}, xi_exact(high-T)={xi_exact:.3f}  [max ~0.9-1.1 in paramagnetic 2D Ising]")
    snapshots = run_ising_metropolis(
        128, beta, n_thermal=6_000, n_measure=40, measure_interval=120, seed=101
    )
    fourier, realspace, gr = analyze_spin_snapshots(snapshots)
    print(f"  Fourier xi={fourier.xi:.3f}  real-space xi={realspace.xi:.3f}")
    print(f"  real-space vs exact: {relative_error(realspace.xi, xi_exact)*100:.1f}% error")
    _print_gr_summary(gr, L=128)


def step_2_estimator_agreement() -> None:
    print("\n=== Step 2: Fourier vs real-space agreement (Fourier-model synthesis) ===")
    xi_target = 4.0
    L = 64
    snapshots = generate_fourier_model_snapshots(L, xi_target, n_snapshots=48, seed=202)
    fourier, realspace, gr = analyze_spin_snapshots(snapshots)
    print(f"  target xi={xi_target}, L={L}")
    print(f"  Fourier xi={fourier.xi:.3f}  real-space xi={realspace.xi:.3f}")
    print(f"  cross-estimator error: {relative_error(fourier.xi, realspace.xi)*100:.1f}%")
    _print_gr_summary(gr, L=L)


def step_3_finite_size() -> None:
    print("\n=== Step 3: xi < L/4 finite-size check ===")
    xi_target = 8.0
    L = 128
    snapshots = generate_fourier_model_snapshots(L, xi_target, n_snapshots=48, seed=303)
    fourier, realspace, gr = analyze_spin_snapshots(snapshots)
    print(f"  target xi={xi_target}, L={L}, threshold L/4={L/4}")
    print(f"  Fourier xi={fourier.xi:.3f}  real-space xi={realspace.xi:.3f}")
    print(f"  Fourier OK: {fourier.xi < L/4}  real-space OK: {realspace.xi < L/4}")
    _print_gr_summary(gr, L=L)


def step_4_flex_note() -> None:
    print("\n=== Step 4: FLEX production (not bare Ising) ===")
    beta_c = ising_critical_beta()
    print(f"  2D Ising beta_c = {beta_c:.4f}")
    print(f"  correlation/runner.py uses lattice-gas HeteroChain at fixed (epsilon, mu, beta).")
    print(f"  Near-critical FLEX runs need mu on the coexistence line (mu_coex_FITTED), not a T sweep.")
    t = reduced_temperature(0.434)
    print(f"  Example: beta=0.434 -> t=(T-Tc)/Tc = {t:.4f} (paramagnetic side)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation-length validation ladder report")
    parser.add_argument("--mc", action="store_true", help="Include Metropolis MC step (slow)")
    args = parser.parse_args()

    print("Correlation-length validation ladder")
    if args.mc:
        step_1_high_temp_realspace()
    else:
        print("\n=== Step 1: real-space fit on known exponential G(r) ===")
        xi_true = 2.5
        gr = [{"r": r, "G_r": math.exp(-r / xi_true), "n_offsets": 1} for r in range(1, 25)]
        fit = fit_real_space_xi(gr, L=64, r_min=2)
        print(f"  injected xi={xi_true:.3f}, recovered xi={fit.xi:.3f}")
        _print_gr_summary(gr, L=64)

    step_2_estimator_agreement()
    step_3_finite_size()
    step_4_flex_note()
    print("\nDone. Use --mc for Metropolis cross-check against high-T Ising formula.")


if __name__ == "__main__":
    main()
