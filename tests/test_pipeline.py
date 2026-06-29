from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import common.queue_manifest as qm
from coex.analyzer import (
    build_curves,
    calculate_phi_psi,
    fit_zero_crossing_with_error,
    write_combo_curves,
)
from coex.paths import COMBO_KEY_FIELDS, combo_dir, phi_psi_csv_path
from coex.generate_samples import MANAGE_FIELDS
from correlation.analyzer import (
    analyze_metadata,
    average_power_spectrum,
    fit_correlation_length,
    fit_real_space_xi,
    lattice_to_spins,
    radial_mode_series,
)
from correlation.paths import (
    METADATA_FIELDS,
    analysis_csv_path,
    correlation_dir,
    metadata_csv_path,
    snapshot_path,
)


def test_queue_manifest_deduplicates_pending(tmp_path):
    manifest = str(tmp_path / "queue.json")
    qm.merge_pending(["a.json", "b.json"], path=manifest)
    qm.merge_pending(["a.json", "c.json"], path=manifest)
    assert qm.read_manifest(manifest)["pending"] == ["a.json", "b.json", "c.json"]


def test_generate_samples_uses_fitted_mu_schema():
    assert "mu_coex_FITTED" in MANAGE_FIELDS
    assert "mu_coex_FITTED_error" in MANAGE_FIELDS
    assert "mu_coex_SIM" not in MANAGE_FIELDS


def test_calculate_phi_psi_and_zero_crossing():
    def frame(mu, phi):
        rho_active = (phi + 1.0) / 2.0
        return pd.DataFrame({
            "mu": [mu] * 4,
            "rho_active": [rho_active] * 4,
            "rho_inert": [0.25] * 4,
            "rho_empty": [0.25] * 4,
        })

    points = [(-1.0, frame(-1.0, 0.4)), (0.0, frame(0.0, 0.0)), (1.0, frame(1.0, -0.4))]
    mu_vals, phi_vals, phi_errs, psi_vals, _ = build_curves(points)

    assert calculate_phi_psi(points[0][1])[0] > 0
    assert np.allclose(psi_vals, np.abs(phi_vals))
    mu_zero, mu_err = fit_zero_crossing_with_error(mu_vals, phi_vals, phi_errs)
    assert abs(mu_zero) < 1e-8
    assert mu_err >= 0


def test_write_combo_curves_writes_csv_only(tmp_path):
    combo = {
        "epsilon": -2.0,
        "delta_f": 0.0,
        "delta_mu": 1.0,
        "k": 1.0,
        "scheme": "homo",
        "Lx": 160,
        "Ly": 16,
    }
    combo_key = tuple(str(combo[field]) for field in COMBO_KEY_FIELDS)
    write_combo_curves(
        combo_key,
        np.asarray([-1.0, 0.0]),
        np.asarray([0.2, -0.1]),
        np.asarray([0.01, 0.01]),
        np.asarray([0.2, 0.1]),
        np.asarray([0.01, 0.01]),
        results_dir=str(tmp_path),
    )

    assert os.path.isfile(phi_psi_csv_path(combo, base=str(tmp_path)))
    assert not os.path.exists(os.path.join(combo_dir(combo, base=str(tmp_path)), "phi_psi.png"))


def test_correlation_paths_are_stable():
    params = {
        "epsilon": -2.0,
        "delta_f": 0.0,
        "delta_mu": 1.0,
        "k": 1.0,
        "scheme": "homo",
        "Lx": 32,
        "Ly": 32,
    }
    assert correlation_dir(params).startswith("correlation_results/correlation_")
    assert metadata_csv_path(params).endswith("snapshot_metadata.csv")
    assert analysis_csv_path(params).endswith("correlation_length.csv")
    assert snapshot_path(params, 7, 3).endswith("snapshots/replica_00007/chunk_00003.npy")


def test_spin_mapping_defaults_to_active_vs_other():
    lattice = np.asarray([[2, 1], [0, 2]], dtype=np.uint32)
    spins = lattice_to_spins(lattice)
    assert spins.tolist() == [[1.0, -1.0], [-1.0, 1.0]]


def _write_snapshot_metadata(tmp_path, snapshots):
    rows = []
    for idx, arr in enumerate(snapshots):
        path = tmp_path / "snapshots" / "replica_00000" / f"chunk_{idx:05d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        rows.append({
            "run_id": 0,
            "replica_id": 0,
            "chunk": idx,
            "snapshot_path": os.path.relpath(path, tmp_path),
            "time": idx + 1,
            "chunk_time": 1.0,
            "epsilon": -2.0,
            "delta_f": 0.0,
            "delta_mu": 1.0,
            "k": 1.0,
            "scheme": "homo",
            "Lx": arr.shape[0],
            "Ly": arr.shape[1],
            "mu": -4.0,
            "mu_coex_FITTED": -4.0,
            "beta": 1.0,
            "eq_time": 1.0,
            "prod_time": 4.0,
            "prod_chunks": len(snapshots),
            "seed": 123,
        })

    metadata = tmp_path / "snapshot_metadata.csv"
    with open(metadata, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def test_fourier_fit_rejects_nonpositive_slope():
    snapshots = [np.ones((8, 8)), -np.ones((8, 8))]
    g_hat, _, _ = average_power_spectrum(snapshots)
    modes = radial_mode_series(g_hat)
    try:
        fit_correlation_length(modes, L=8)
    except ValueError as exc:
        assert "not enough positive" in str(exc) or "nonpositive" in str(exc)
    else:
        raise AssertionError("expected invalid Fourier fit to fail")


def test_fit_real_space_xi_recovers_exponential_decay():
    xi_true = 3.5
    gr = [
        {"r": r, "G_r": float(np.exp(-r / xi_true)), "n_offsets": 1}
        for r in range(1, 20)
    ]
    fit = fit_real_space_xi(gr, L=32, r_min=2)
    assert fit.n_points >= 4
    assert abs(fit.xi - xi_true) < 0.05
    assert fit.slope < 0


def test_fit_real_space_xi_skips_r_zero_and_nonpositive():
    gr = [
        {"r": 0, "G_r": 1.0, "n_offsets": 1},
        {"r": 1, "G_r": -0.1, "n_offsets": 1},
        {"r": 2, "G_r": 0.5, "n_offsets": 1},
        {"r": 3, "G_r": 0.25, "n_offsets": 1},
        {"r": 4, "G_r": 0.125, "n_offsets": 1},
    ]
    fit = fit_real_space_xi(gr, L=16, r_min=2)
    assert fit.r_min >= 2
    assert fit.n_points == 3


def test_fourier_fit_uses_sin_sq_axis():
    rows = [
        {"mode": 1, "sin_sq": 0.1, "inv_g_hat": 1.0, "g_hat": 1.0},
        {"mode": 2, "sin_sq": 0.2, "inv_g_hat": 1.2, "g_hat": 1.0},
        {"mode": 3, "sin_sq": 0.3, "inv_g_hat": 1.5, "g_hat": 1.0},
    ]
    fit = fit_correlation_length(rows, L=64, min_points=2, initial_points=3)
    assert fit.n_points == 3
    assert fit.max_k_sq == 0.3


def test_analyze_metadata_writes_outputs(tmp_path):
    L = 16
    x, y = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    snapshots = []
    for phase in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        field = (
            np.cos(2.0 * np.pi * (x + y) / L + phase)
            + 0.35 * np.cos(4.0 * np.pi * (x + y) / L + phase)
            + 0.15 * np.cos(6.0 * np.pi * (x + y) / L + phase)
        )
        snapshots.append(np.where(field >= 0.0, 2, 0).astype(np.uint32))
    metadata = _write_snapshot_metadata(tmp_path, snapshots)
    fit, rs_fit = analyze_metadata(str(metadata), max_modes=4)

    assert fit.xi > 0
    assert rs_fit.n_points >= 2
    assert os.path.isfile(tmp_path / "fourier_modes.csv")
    assert os.path.isfile(tmp_path / "G_r.csv")
    analysis_path = tmp_path / "correlation_length.csv"
    assert os.path.isfile(analysis_path)
    with open(analysis_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert "xi_realspace" in row
    assert row["xi_realspace"]
