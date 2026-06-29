"""
Reference 2D Ising helpers for correlation-length estimator validation.

The lattice-gas production runner (correlation/runner.py) is *not* a plain Ising
sampler at arbitrary temperature — it uses HeteroChain on three-state FLEX
lattices at fixed (epsilon, mu, beta). For estimator validation we therefore
use:

  1. Analytic high-T correlation length (exact in the paramagnetic phase):
         xi = -1 / ln(tanh(beta * J))
  2. A lightweight Metropolis MC for standard +/-1 Ising (Test 1).
  3. Ornstein–Zernike Fourier synthesis for known xi at moderate/critical
     reduced temperatures without critical slowing down (Tests 2–3).
"""

from __future__ import annotations

import math

import numpy as np

BONDING = 2
EMPTY = 0


def ising_critical_beta(J: float = 1.0) -> float:
    """Inverse temperature at the 2D square-lattice Ising critical point (J > 0)."""
    return math.log(1.0 + math.sqrt(2.0)) / (2.0 * J)


def ising_exact_xi_high_temp(beta: float, J: float = 1.0) -> float:
    """
    Exact exponential correlation length in the high-temperature paramagnetic phase.

    Valid when beta * J is well below beta_c * J (short correlation length).
    """
    t = math.tanh(beta * J)
    if t <= 0.0 or t >= 1.0:
        raise ValueError(f"tanh(beta*J)={t} outside (0, 1)")
    return -1.0 / math.log(t)


def reduced_temperature(beta: float, J: float = 1.0) -> float:
    """t = (T - T_c) / T_c in the paramagnetic phase (beta < beta_c)."""
    beta_c = ising_critical_beta(J)
    if beta >= beta_c:
        return math.nan
    return beta_c / beta - 1.0


def spins_to_lattice(spins: np.ndarray) -> np.ndarray:
    """Map +/-1 spins to FLEX lattice codes for active_vs_other mapping."""
    return np.where(spins > 0, BONDING, EMPTY).astype(np.uint32)


def lattice_to_spins_check(lattice: np.ndarray) -> np.ndarray:
    """Inverse of spins_to_lattice for BONDING/EMPTY-only configs."""
    return np.where(lattice == BONDING, 1.0, -1.0)


def _metropolis_sweep_checkerboard(
    spins: np.ndarray, beta: float, J: float, rng: np.random.Generator
) -> None:
    """Vectorized checkerboard Metropolis update (two sublattice passes per sweep)."""
    L = spins.shape[0]
    parity_grid = (np.arange(L)[:, None] + np.arange(L)) % 2
    for parity in (0, 1):
        mask = parity_grid == parity
        nn = (
            np.roll(spins, 1, axis=0)
            + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1)
            + np.roll(spins, -1, axis=1)
        )
        dE = 2.0 * J * spins * nn
        accept = (dE <= 0.0) | (rng.random(spins.shape) < np.exp(-beta * dE))
        spins[mask & accept] *= -1.0


def run_ising_metropolis(
    L: int,
    beta: float,
    *,
    J: float = 1.0,
    n_thermal: int = 2_000,
    n_measure: int = 32,
    measure_interval: int = 100,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return +/-1 spin snapshots from a 2D periodic Ising Metropolis chain."""
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(L, L))

    for _ in range(n_thermal):
        _metropolis_sweep_checkerboard(spins, beta, J, rng)

    snapshots: list[np.ndarray] = []
    for _ in range(n_measure):
        for _ in range(measure_interval):
            _metropolis_sweep_checkerboard(spins, beta, J, rng)
        snapshots.append(spins.copy())
    return snapshots


def generate_fourier_model_snapshots(
    L: int,
    xi: float,
    *,
    chi: float | None = None,
    n_snapshots: int = 32,
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Synthesize +/-1 spins whose mean power spectrum follows the same small-k law
    the Fourier fitter assumes: 1/g_hat(k) = intercept + slope * sin_sq(k).
    """
    if xi <= 0:
        raise ValueError("xi must be positive")
    if chi is None:
        chi = float(L * L)
    intercept = 1.0 / chi
    slope = xi * xi * intercept

    rng = np.random.default_rng(seed)
    snapshots: list[np.ndarray] = []
    k1d = 2.0 * math.pi * np.arange(L) / L
    sin_x = np.sin(0.5 * k1d) ** 2
    sin_y = np.sin(0.5 * k1d) ** 2
    sin_sq = sin_x[:, None] + sin_y[None, :]
    amplitude = 1.0 / (intercept + slope * sin_sq)
    amplitude[0, 0] = chi

    for _ in range(n_snapshots):
        noise = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
        field_fft = np.sqrt(np.maximum(amplitude, 0.0)) * noise
        field = np.fft.ifftn(field_fft).real
        spins = np.sign(field)
        spins[spins == 0] = 1.0
        snapshots.append(spins.astype(float))
    return snapshots


def relative_error(estimate: float, exact: float) -> float:
    return abs(estimate - exact) / exact


def analyze_spin_snapshots(snapshots: list[np.ndarray]):
    """Run Fourier + real-space correlation-length fits on +/-1 spin configs."""
    from correlation.analyzer import (
        average_power_spectrum,
        fit_correlation_length,
        fit_real_space_xi,
        radial_mode_series,
        radial_real_space_correlation,
    )

    L = snapshots[0].shape[0]
    g_hat, _, _ = average_power_spectrum(snapshots)
    modes = radial_mode_series(g_hat)
    fourier = fit_correlation_length(modes, L=L)
    gr = radial_real_space_correlation(snapshots, subtract_magnetization=True)
    realspace = fit_real_space_xi(gr, L=L)
    return fourier, realspace, gr


def gr_log_decay_diagnostics(gr: list[dict], *, L: int, r_min_fit: int = 2) -> dict:
    """
    Summarize G(r) quality on a log scale (see validation checklist).

    Returns counts and flags for common failure modes.
    """
    positive = [row for row in gr if row["r"] > 0 and row["G_r"] > 0]
    r_vals = np.asarray([row["r"] for row in positive], dtype=float)
    g_vals = np.asarray([row["G_r"] for row in positive], dtype=float)

    upward_tail = False
    if len(positive) >= 4:
        mid = len(positive) // 2
        early_slope = np.polyfit(r_vals[:mid], np.log(g_vals[:mid]), 1)[0]
        late_slope = np.polyfit(r_vals[mid:], np.log(g_vals[mid:]), 1)[0]
        upward_tail = late_slope > 0 and late_slope > early_slope + 0.05

    negatives_before_decay = any(
        row["G_r"] <= 0 for row in gr if r_min_fit <= row["r"] <= L / 4
    )

    return {
        "n_positive": len(positive),
        "upward_tail": upward_tail,
        "negatives_before_decay": negatives_before_decay,
        "insufficient_dynamic_range": len(positive) < 4,
        "r_max_positive": float(r_vals[-1]) if len(positive) else 0.0,
    }
