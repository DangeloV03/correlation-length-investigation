"""
Analyze saved lattice snapshots to estimate correlation length.

This tool brute-forces correlation length estimation over spatial correlation data 
from saved lattice snapshots, rather than relying on analytic estimators from Ising models.
Both Fourier-space and real-space correlation length fitting are performed, and a radial 
G(r) CSV is written using FFT-based autocorrelation with periodic boundaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from correlation.paths import ANALYSIS_CSV, METADATA_CSV

EMPTY, INERT, BONDING = 0, 1, 2


@dataclass
class FitResult:
    xi: float
    chi: float
    intercept: float
    slope: float
    n_points: int
    max_k_sq: float
    warning: str


@dataclass
class RealSpaceFitResult:
    xi: float
    slope: float
    n_points: int
    r_min: float
    r_max: float
    warning: str


@dataclass
class RealSpaceGpuFitResult:
    xi: float
    slope: float
    n_pairs_used: int
    warning: str


def lattice_to_spins(lattice: np.ndarray, mapping: str = "active_vs_other") -> np.ndarray:
    if mapping == "active_vs_other":
        return np.where(lattice == BONDING, 1.0, -1.0)
    if mapping == "active_vs_empty":
        return np.where(lattice == BONDING, 1.0, np.where(lattice == EMPTY, -1.0, 0.0))
    raise ValueError(f"unknown spin mapping: {mapping}")


def read_metadata(metadata_path: str) -> list[dict]:
    with open(metadata_path, newline="") as f:
        return list(csv.DictReader(f))


def snapshot_paths(metadata_path: str) -> list[str]:
    rows = read_metadata(metadata_path)
    base = os.path.dirname(metadata_path)
    paths = []
    for row in rows:
        raw = row.get("snapshot_path", "")
        if not raw:
            continue
        path = raw if os.path.isabs(raw) else os.path.join(base, raw)
        if os.path.isfile(path):
            paths.append(path)
    return paths


def load_spin_snapshots(paths: list[str], mapping: str) -> list[np.ndarray]:
    snapshots = []
    shape = None
    for path in paths:
        lattice = np.load(path)
        spins = lattice_to_spins(lattice, mapping)
        if shape is None:
            shape = spins.shape
        elif spins.shape != shape:
            raise ValueError(f"snapshot shape mismatch: {path} has {spins.shape}, expected {shape}")
        snapshots.append(spins)
    if not snapshots:
        raise ValueError("no readable snapshots found")
    return snapshots


def average_power_spectrum(snapshots: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    n_sites = snapshots[0].size
    power_accum = np.zeros(snapshots[0].shape, dtype=float)
    magnetizations = []
    for spins in snapshots:
        fft = np.fft.fftn(spins)
        power_accum += np.abs(fft) ** 2 / n_sites
        magnetizations.append(float(np.mean(spins)))
    g_hat = power_accum / len(snapshots)
    return g_hat, float(np.mean(magnetizations)), float(np.mean(np.square(magnetizations)))


def radial_mode_series(g_hat: np.ndarray, max_modes: int | None = None) -> list[dict]:
    if g_hat.ndim != 2 or g_hat.shape[0] != g_hat.shape[1]:
        raise ValueError("radial Fourier fit currently expects square 2D snapshots")

    L = g_hat.shape[0]
    n_max = L // 2
    if max_modes is not None:
        n_max = min(n_max, max_modes)

    rows = []
    for n in range(1, n_max + 1):
        k0 = 2.0 * math.pi * n / L
        g_val = float(g_hat[n % L, n % L])
        k_sq = 2.0 * k0 * k0
        rows.append({
            "mode": n,
            "k0": k0,
            "k_sq": k_sq,
            "sin_sq": 2.0 * (math.sin(0.5 * k0) ** 2),
            "g_hat": g_val,
            "inv_g_hat": 1.0 / g_val if g_val > 0 else math.nan,
        })
    return rows


def fit_correlation_length(
    mode_rows: list[dict],
    *,
    L: int,
    min_points: int = 2,
    initial_points: int = 6,
) -> FitResult:
    candidates = [row for row in mode_rows if row["g_hat"] > 0 and np.isfinite(row["inv_g_hat"])]
    if len(candidates) < min_points:
        raise ValueError("not enough positive Fourier modes to fit xi")

    def fit_rows(rows: list[dict]) -> tuple[float, float] | None:
        x_vals = np.asarray([row["sin_sq"] for row in rows], dtype=float)
        y_vals = np.asarray([row["inv_g_hat"] for row in rows], dtype=float)
        slope_val, intercept_val = np.polyfit(x_vals, y_vals, deg=1)
        if intercept_val <= 0 or slope_val <= 0:
            return None
        return float(slope_val), float(intercept_val)

    selected = candidates[:max(min_points, min(initial_points, len(candidates)))]
    initial_fit = fit_rows(selected)
    if initial_fit is None:
        for n_points in range(min_points, min(initial_points, len(candidates)) + 1):
            trial = candidates[:n_points]
            if fit_rows(trial) is not None:
                selected = trial
                initial_fit = fit_rows(trial)
                break
    if initial_fit is None:
        raise ValueError("small-k fit produced nonpositive intercept or slope")

    for _ in range(3):
        fit = fit_rows(selected)
        if fit is None:
            break
        slope, intercept = fit
        xi = float(math.sqrt(slope / intercept))
        filtered = [row for row in candidates if xi * xi * row["sin_sq"] < 1.0]
        if len(filtered) < min_points or fit_rows(filtered) is None:
            break
        if [row["mode"] for row in filtered] == [row["mode"] for row in selected]:
            break
        selected = filtered

    final_fit = fit_rows(selected)
    if final_fit is None:
        raise ValueError("small-k fit produced nonpositive intercept or slope")
    slope, intercept = final_fit
    x = np.asarray([row["sin_sq"] for row in selected], dtype=float)

    xi = float(math.sqrt(slope / intercept))
    chi = float(1.0 / intercept)
    warning = ""
    if xi >= L / 6:
        warning = "xi is not much smaller than L; finite-size effects may be important"

    return FitResult(
        xi=xi,
        chi=chi,
        intercept=float(intercept),
        slope=float(slope),
        n_points=len(selected),
        max_k_sq=float(np.max(x)),
        warning=warning,
    )


def radial_real_space_correlation(
    snapshots: list[np.ndarray],
    *,
    subtract_magnetization: bool = True,
) -> list[dict]:
    shape = snapshots[0].shape
    if len(shape) != 2:
        raise ValueError("real-space radial G(r) currently expects 2D snapshots")

    accum = np.zeros(shape, dtype=float)
    m2_vals = []
    n_sites = snapshots[0].size
    for spins in snapshots:
        fft = np.fft.fftn(spins)
        corr = np.fft.ifftn(np.abs(fft) ** 2).real / n_sites
        accum += corr
        m2_vals.append(float(np.mean(spins)) ** 2)
    corr = accum / len(snapshots)
    if subtract_magnetization:
        corr = corr - float(np.mean(m2_vals))

    Lx, Ly = shape
    bins: dict[int, list[float]] = {}
    for dx in range(Lx):
        mx = min(dx, Lx - dx)
        for dy in range(Ly):
            my = min(dy, Ly - dy)
            r = int(round(math.sqrt(mx * mx + my * my)))
            bins.setdefault(r, []).append(float(corr[dx, dy]))

    return [
        {"r": r, "G_r": float(np.mean(values)), "n_offsets": len(values)}
        for r, values in sorted(bins.items())
    ]


def fit_real_space_xi(
    gr: list[dict],
    *,
    L: int,
    r_min: int = 2,
) -> RealSpaceFitResult:
    """Fit connected G(r) ~ exp(-r/xi) via log(G) vs r in an intermediate distance window."""
    r_max_limit = L / 2
    points = [
        row for row in gr
        if row["r"] > 0
        and row["r"] >= r_min
        and row["r"] <= r_max_limit
        and row["G_r"] > 0
    ]
    if len(points) < 2:
        raise ValueError("not enough positive G(r) points for real-space xi fit")

    r_vals = np.asarray([row["r"] for row in points], dtype=float)
    log_g = np.log(np.asarray([row["G_r"] for row in points], dtype=float))
    slope, _intercept = np.polyfit(r_vals, log_g, deg=1)

    warning = ""
    if len(points) < 4:
        warning = "real-space fit used fewer than 4 points"
    if slope >= 0:
        unphysical = "real-space fit slope is non-negative (unphysical decay)"
        warning = f"{warning}; {unphysical}".strip("; ") if warning else unphysical
        xi = float("nan")
    else:
        xi = float(-1.0 / slope)

    return RealSpaceFitResult(
        xi=xi,
        slope=float(slope),
        n_points=len(points),
        r_min=float(np.min(r_vals)),
        r_max=float(np.max(r_vals)),
        warning=warning,
    )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the GPU real-space estimator; "
            "install torch or use the FFT radial fallback"
        ) from exc
    return torch


def _flat_site_coords(L: int, device) -> "torch.Tensor":
    torch = _require_torch()
    flat = torch.arange(L * L, device=device, dtype=torch.float32)
    y = torch.div(flat, L, rounding_mode="floor")
    x = flat - y * L
    return torch.stack([x, y], dim=-1)


def _pbc_distance_tile(
    coords,
    i0: int,
    i1: int,
    L: int,
) -> "torch.Tensor":
    torch = _require_torch()
    rows = coords[i0:i1]
    diff = rows[:, None, :] - coords[None, :, :]
    diff = diff - L * torch.round(diff / L)
    return diff.norm(dim=-1)


def _accumulate_connected_correlation(
    paths: list[str],
    mapping: str,
    *,
    chunk_size: int,
    compute_device,
    store_device,
) -> tuple["torch.Tensor", int, float]:
    """Tile-wise outer-product accumulation; store the connected G on store_device."""
    torch = _require_torch()
    if not paths:
        raise ValueError("no snapshot paths provided")

    n_snaps = 0
    m_sum = 0.0
    n_sites: int | None = None
    L: int | None = None
    accum = None

    for path in paths:
        spins = lattice_to_spins(np.load(path), mapping)
        if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
            raise ValueError(f"snapshot must be square 2D lattice: {path}")
        L = int(spins.shape[0])
        n = L * L
        if n_sites is None:
            n_sites = n
            accum = torch.zeros(n, n, dtype=torch.float32, device=store_device)
        elif n != n_sites:
            raise ValueError(f"snapshot shape mismatch: {path}")

        sigma = torch.from_numpy(spins.astype(np.float32, copy=False).ravel()).to(compute_device)
        m_sum += float(sigma.sum().item())

        for i0 in range(0, n, chunk_size):
            i1 = min(i0 + chunk_size, n)
            tile = torch.outer(sigma[i0:i1], sigma)
            accum[i0:i1] += tile.to(store_device)

        n_snaps += 1
        del sigma

    assert accum is not None and L is not None and n_sites is not None
    accum /= float(n_snaps)
    m_bar = m_sum / (n_snaps * n_sites)
    accum -= m_bar * m_bar
    return accum, L, m_bar


def _sample_pairs_and_fit(
    G,
    coords,
    L: int,
    *,
    max_pairs: int,
    min_pairs: int,
    seed: int,
) -> RealSpaceGpuFitResult:
    torch = _require_torch()
    n = G.shape[0]
    rng = np.random.default_rng(seed)

    r_vals: list[float] = []
    log_g_vals: list[float] = []
    batch = min(max_pairs * 4, 2_000_000)
    attempts = 0
    max_attempts = max(20, max_pairs // max(1, batch // 4))

    while len(r_vals) < max_pairs and attempts < max_attempts:
        attempts += 1
        ii = rng.integers(0, n, size=batch)
        jj = rng.integers(0, n, size=batch)
        g_vals = G[ii, jj]
        if g_vals.device.type != "cpu":
            g_vals = g_vals.cpu()
        g_np = g_vals.numpy()
        positive = g_np > 0
        if not np.any(positive):
            continue

        ii = ii[positive]
        jj = jj[positive]
        g_np = g_np[positive]

        ci = coords[ii]
        cj = coords[jj]
        diff = ci - cj
        diff = diff - L * torch.round(diff / L)
        dist = diff.norm(dim=-1).cpu().numpy()

        remaining = max_pairs - len(r_vals)
        take = min(remaining, dist.size)
        r_vals.extend(dist[:take].tolist())
        log_g_vals.extend(np.log(g_np[:take]).tolist())

    warning = ""
    if len(r_vals) < 2:
        raise ValueError("not enough positive G(r,r') pairs for GPU fit")
    if len(r_vals) < min_pairs:
        warning = (
            f"GPU fit used only {len(r_vals)} positive pairs "
            f"(recommended >= {min_pairs})"
        )

    r_arr = np.asarray(r_vals, dtype=float)
    log_g_arr = np.asarray(log_g_vals, dtype=float)
    slope, _intercept = np.polyfit(r_arr, log_g_arr, deg=1)

    if slope >= -1e-10:
        unphysical = "GPU real-space fit slope is non-negative (unphysical decay)"
        warning = f"{warning}; {unphysical}".strip("; ") if warning else unphysical
        xi = float("nan")
    else:
        xi = float(-1.0 / slope)

    return RealSpaceGpuFitResult(
        xi=xi,
        slope=float(slope),
        n_pairs_used=len(r_vals),
        warning=warning,
    )


def fit_real_space_xi_gpu(
    paths: list[str],
    *,
    mapping: str = "active_vs_other",
    chunk_size: int = 512,
    max_pairs: int = 5_000_000,
    min_pairs: int = 1000,
    seed: int = 0,
) -> RealSpaceGpuFitResult:
    """
    Brute-force connected real-space correlation length on GPU without FFT.

    Loads snapshots one at a time, accumulates tiled outer products, subtracts
    ⟨m⟩², samples positive pair distances, and fits log G vs r.
    """
    try:
        torch = _require_torch()
    except ImportError:
        print(
            "[correlation_analyzer] WARNING: PyTorch not installed; "
            "using FFT radial real-space path for xi_realspace_gpu",
            flush=True,
        )
        snapshots = load_spin_snapshots(paths, mapping)
        gr = radial_real_space_correlation(snapshots, subtract_magnetization=True)
        rs = fit_real_space_xi(gr, L=snapshots[0].shape[0])
        warning = rs.warning
        prefix = "PyTorch unavailable; used FFT radial fallback"
        warning = f"{prefix}; {warning}".strip("; ") if warning else prefix
        return RealSpaceGpuFitResult(
            xi=rs.xi,
            slope=rs.slope,
            n_pairs_used=rs.n_points,
            warning=warning,
        )

    if not torch.cuda.is_available():
        print(
            "[correlation_analyzer] WARNING: CUDA unavailable; "
            "using FFT radial real-space path for xi_realspace_gpu",
            flush=True,
        )
        snapshots = load_spin_snapshots(paths, mapping)
        gr = radial_real_space_correlation(snapshots, subtract_magnetization=True)
        rs = fit_real_space_xi(gr, L=snapshots[0].shape[0])
        warning = rs.warning
        prefix = "GPU unavailable; used FFT radial fallback"
        warning = f"{prefix}; {warning}".strip("; ") if warning else prefix
        return RealSpaceGpuFitResult(
            xi=rs.xi,
            slope=rs.slope,
            n_pairs_used=rs.n_points,
            warning=warning,
        )

    compute_device = torch.device("cuda")
    store_device = torch.device("cpu")

    G, L, _m_bar = _accumulate_connected_correlation(
        paths,
        mapping,
        chunk_size=chunk_size,
        compute_device=compute_device,
        store_device=store_device,
    )
    coords = _flat_site_coords(L, compute_device)
    return _sample_pairs_and_fit(
        G,
        coords,
        L,
        max_pairs=max_pairs,
        min_pairs=min_pairs,
        seed=seed,
    )


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def analyze_metadata(
    metadata_path: str,
    *,
    mapping: str = "active_vs_other",
    max_modes: int | None = None,
    output_dir: str | None = None,
    gpu_chunk_size: int = 512,
    gpu_max_pairs: int = 5_000_000,
) -> tuple[FitResult, RealSpaceFitResult, RealSpaceGpuFitResult]:
    paths = snapshot_paths(metadata_path)
    snapshots = load_spin_snapshots(paths, mapping)
    Lx, Ly = snapshots[0].shape
    if Lx != Ly:
        raise ValueError(f"correlation analysis expects square snapshots, got {Lx}x{Ly}")

    outdir = output_dir or os.path.dirname(metadata_path)
    g_hat, m_mean, m2_mean = average_power_spectrum(snapshots)
    modes = radial_mode_series(g_hat, max_modes=max_modes)
    fit = fit_correlation_length(modes, L=Lx)
    gr = radial_real_space_correlation(snapshots, subtract_magnetization=True)
    rs_fit = fit_real_space_xi(gr, L=Lx)
    rs_gpu = fit_real_space_xi_gpu(
        paths,
        mapping=mapping,
        chunk_size=gpu_chunk_size,
        max_pairs=gpu_max_pairs,
    )

    write_csv(
        os.path.join(outdir, "fourier_modes.csv"),
        modes,
        ["mode", "k0", "k_sq", "sin_sq", "g_hat", "inv_g_hat"],
    )
    write_csv(
        os.path.join(outdir, "G_r.csv"),
        gr,
        ["r", "G_r", "n_offsets"],
    )
    write_csv(
        os.path.join(outdir, ANALYSIS_CSV),
        [{
            "n_snapshots": len(snapshots),
            "L": Lx,
            "mapping": mapping,
            "m_mean": m_mean,
            "m2_mean": m2_mean,
            "xi": fit.xi,
            "xi_realspace": rs_fit.xi,
            "xi_realspace_gpu": rs_gpu.xi,
            "chi": fit.chi,
            "fit_intercept": fit.intercept,
            "fit_slope": fit.slope,
            "fit_points": fit.n_points,
            "fit_max_k_sq": fit.max_k_sq,
            "realspace_slope": rs_fit.slope,
            "realspace_fit_points": rs_fit.n_points,
            "realspace_r_min": rs_fit.r_min,
            "realspace_r_max": rs_fit.r_max,
            "realspace_gpu_slope": rs_gpu.slope,
            "realspace_gpu_n_pairs": rs_gpu.n_pairs_used,
            "warning": fit.warning,
            "realspace_warning": rs_fit.warning,
            "realspace_gpu_warning": rs_gpu.warning,
        }],
        [
            "n_snapshots",
            "L",
            "mapping",
            "m_mean",
            "m2_mean",
            "xi",
            "xi_realspace",
            "xi_realspace_gpu",
            "chi",
            "fit_intercept",
            "fit_slope",
            "fit_points",
            "fit_max_k_sq",
            "realspace_slope",
            "realspace_fit_points",
            "realspace_r_min",
            "realspace_r_max",
            "realspace_gpu_slope",
            "realspace_gpu_n_pairs",
            "warning",
            "realspace_warning",
            "realspace_gpu_warning",
        ],
    )
    return fit, rs_fit, rs_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate correlation length from chunk snapshots")
    parser.add_argument(
        "path",
        help="Path to snapshot_metadata.csv or a correlation result directory containing it",
    )
    parser.add_argument(
        "--mapping",
        choices=["active_vs_other", "active_vs_empty"],
        default="active_vs_other",
    )
    parser.add_argument("--max-modes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu-chunk-size", type=int, default=512)
    parser.add_argument("--gpu-max-pairs", type=int, default=5_000_000)
    args = parser.parse_args()

    metadata_path = args.path
    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, METADATA_CSV)
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(metadata_path)

    fit, rs_fit, rs_gpu = analyze_metadata(
        metadata_path,
        mapping=args.mapping,
        max_modes=args.max_modes,
        output_dir=args.output_dir,
        gpu_chunk_size=args.gpu_chunk_size,
        gpu_max_pairs=args.gpu_max_pairs,
    )
    print(
        f"[correlation_analyzer] xi_fourier={fit.xi:.6g} xi_realspace={rs_fit.xi:.6g} "
        f"xi_realspace_gpu={rs_gpu.xi:.6g} chi={fit.chi:.6g} "
        f"fit_points={fit.n_points} rs_points={rs_fit.n_points} gpu_pairs={rs_gpu.n_pairs_used}"
    )
    if fit.warning:
        print(f"[correlation_analyzer] Fourier WARNING: {fit.warning}")
    if rs_fit.warning:
        print(f"[correlation_analyzer] Real-space WARNING: {rs_fit.warning}")
    if rs_gpu.warning:
        print(f"[correlation_analyzer] GPU real-space WARNING: {rs_gpu.warning}")


if __name__ == "__main__":
    main()
