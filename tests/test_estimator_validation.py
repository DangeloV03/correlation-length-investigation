"""
Ladder of correlation-length estimator validation tests.

Fast pytest checks use known inputs (exponential G(r), Fourier-model synthesis).
Full Metropolis MC cross-checks are in tests/run_validation_report.py --mc.

Run: pytest tests/test_estimator_validation.py -v
Report: python tests/run_validation_report.py [--mc]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from correlation.analyzer import fit_real_space_xi
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


# --- Test 1: known exponential G(r), real-space fit ---


def test_1_realspace_recovers_exact_exponential():
    """Step 1: validate real-space estimator against a known xi (no MC noise)."""
    xi_true = 2.5
    gr = [
        {"r": r, "G_r": float(math.exp(-r / xi_true)), "n_offsets": 1}
        for r in range(1, 25)
    ]
    fit = fit_real_space_xi(gr, L=64, r_min=2)
    diag = gr_log_decay_diagnostics(gr, L=64)

    assert not diag["insufficient_dynamic_range"]
    assert not diag["upward_tail"]
    assert relative_error(fit.xi, xi_true) < 0.10
    assert fit.xi < 64 / 4


# --- Test 2: Fourier-model synthesis, cross-estimator check ---


def test_2_fourier_model_fourier_vs_realspace():
    """Step 2: moderate xi on L=64; Fourier should match target, estimators same order."""
    xi_target = 4.0
    L = 64
    snapshots = generate_fourier_model_snapshots(L, xi_target, n_snapshots=48, seed=202)

    fourier, realspace, gr = analyze_spin_snapshots(snapshots)
    diag = gr_log_decay_diagnostics(gr, L=L)

    assert diag["n_positive"] >= 4
    assert fourier.xi > 0 and realspace.xi > 0
    assert relative_error(fourier.xi, xi_target) < 0.45
    assert fourier.xi < L / 4
    assert realspace.xi < L / 4


# --- Test 3: larger xi, finite-size threshold ---


@pytest.mark.slow
def test_3_large_xi_below_finite_size_threshold():
    """Step 3: L=128, target xi=8; Fourier fit below L/4."""
    xi_target = 8.0
    L = 128
    snapshots = generate_fourier_model_snapshots(L, xi_target, n_snapshots=48, seed=303)

    fourier, realspace, gr = analyze_spin_snapshots(snapshots)
    diag = gr_log_decay_diagnostics(gr, L=L)

    assert fourier.xi < L / 4
    assert realspace.xi < L / 4
    assert diag["n_positive"] >= 6
    assert relative_error(fourier.xi, xi_target) < 0.50


# --- MC smoke (optional, loose) ---


@pytest.mark.slow
def test_mc_ising_smoke_runs():
    """Metropolis MC snapshots produce finite positive xi estimates."""
    snapshots = run_ising_metropolis(
        64, 0.35, n_thermal=2_000, n_measure=24, measure_interval=80, seed=101
    )
    fourier, realspace, _ = analyze_spin_snapshots(snapshots)
    assert fourier.xi > 0
    assert realspace.xi > 0
    assert fourier.xi < 64


# --- G(r) diagnostic flags ---


def test_gr_flags_curved_upward_tail():
    gr = []
    for r in range(1, 30):
        base = math.exp(-r / 8.0)
        bump = 0.15 * math.exp((r - 18) / 3.0) if r > 15 else 0.0
        gr.append({"r": r, "G_r": base + bump})
    assert gr_log_decay_diagnostics(gr, L=64)["upward_tail"]


def test_gr_flags_insufficient_points():
    gr = [{"r": 1, "G_r": 1.0}, {"r": 2, "G_r": 0.5}]
    assert gr_log_decay_diagnostics(gr, L=32)["insufficient_dynamic_range"]


# --- FLEX vs Ising ---


def test_flex_vs_ising_documentation():
    beta_c = ising_critical_beta()
    t = reduced_temperature(0.434, J=1.0)
    assert beta_c == pytest.approx(0.440686, rel=1e-4)
    assert t == pytest.approx(0.0154, rel=0.05)
    assert math.isnan(reduced_temperature(0.50, J=1.0))


def test_high_temp_formula_max_xi_in_paramagnetic_phase():
    """High-T formula gives xi < 1.2 throughout the paramagnetic phase (2D, J=1)."""
    xi_at_tc = ising_exact_xi_high_temp(ising_critical_beta() - 1e-4)
    xi_high_t = ising_exact_xi_high_temp(0.20)
    assert xi_at_tc > xi_high_t
    assert xi_at_tc < 1.2
