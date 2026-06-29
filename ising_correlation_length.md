# Extracting Correlation Length from 2D Ising Configuration

## Overview

Given a single 2D Ising snapshot stored as a `.npy` file with spins 0/1, this guide computes the **connected, isotropically-averaged** spin correlation function and fits an exponential to extract the correlation length ξ.

**Connected correlation function:**

```
G(r) = C(r) - m²
```

where `C(r) = ⟨sᵢ sⱼ⟩` averaged over all pairs at distance `r`, and `m = ⟨s⟩` is the mean magnetization. `G(r) → 0` as `r → ∞`.

---

## Prerequisites

```bash
pip install numpy scipy matplotlib
```

---

## Step 1: Load and Convert Spins

Spins are stored as 0/1; convert to ±1 for the standard Ising convention.

```python
import numpy as np

config = np.load("config.npy")          # shape (L, L), values 0 or 1
spins = 2 * config.astype(float) - 1   # convert to ±1

L = spins.shape[0]
assert spins.shape == (L, L), "Expected a square lattice"

m = np.mean(spins)
print(f"Lattice size: {L}x{L},  magnetization m = {m:.4f}")
```

---

## Step 2: Compute 2D Autocorrelation via FFT

The FFT gives the full 2D spatial correlation map in O(L² log L) time. The FFT implicitly assumes **periodic boundary conditions**, which matches the standard Ising setup.

```python
F = np.fft.fft2(spins)
# C2D[dx, dy] = (1/N) Σ_{x,y} s(x,y) * s(x+dx, y+dy)
C2D = np.fft.ifft2(F * np.conj(F)).real / (L * L)

# Shift origin to center so indices run from -L//2 to L//2
C2D = np.fft.fftshift(C2D)

# Subtract m² to get the connected (subtracted) correlation
G2D = C2D - m**2
```

At `(dx, dy) = (0, 0)`: `C2D = ⟨s²⟩ = 1` (since spins are ±1), so `G(0) = 1 - m²`.

---

## Step 3: Isotropic Radial Average

Bin all `(dx, dy)` pairs by their Euclidean distance `r = √(dx² + dy²)`.

```python
# Build distance array — after fftshift, center is at (L//2, L//2)
cy, cx = L // 2, L // 2
y_idx = np.arange(L) - cy
x_idx = np.arange(L) - cx
dx, dy = np.meshgrid(x_idx, y_idx)
r = np.sqrt(dx**2 + dy**2)

# Radial average using histogram: bin by nearest integer r
r_max = L // 2          # don't go beyond half the box (finite-size artifacts)
r_flat = r.ravel()
G_flat = G2D.ravel()

# Bin edges centered on integer r values: 0, 1, 2, ..., r_max-1
bin_edges = np.arange(0, r_max + 1) - 0.5
bin_edges[0] = -0.5     # include r=0 in first bin

G_sum, _   = np.histogram(r_flat, bins=bin_edges, weights=G_flat)
counts, _  = np.histogram(r_flat, bins=bin_edges)
r_centers  = np.arange(r_max)   # integer distances 0, 1, ..., r_max-1

G_r = G_sum / counts   # mean G at each distance r
```

---

## Step 4: Exponential Fit

Fit `G(r) = A · exp(−r / ξ)` for `r ≥ 1` (exclude `r = 0` self-correlation).

```python
from scipy.optimize import curve_fit

def exp_decay(r, A, xi):
    return A * np.exp(-r / xi)

# Fit range: skip r=0, stop where G goes non-positive (noise floor)
mask = (r_centers >= 1) & (G_r > 0) & (r_centers < r_max)
r_fit = r_centers[mask]
G_fit = G_r[mask]

p0 = [G_r[1], L / 4]   # initial guesses: amplitude and xi
popt, pcov = curve_fit(exp_decay, r_fit, G_fit, p0=p0, maxfev=5000)
A_fit, xi_fit = popt
xi_err = np.sqrt(pcov[1, 1])

print(f"Amplitude A = {A_fit:.4f}")
print(f"Correlation length ξ = {xi_fit:.2f} ± {xi_err:.2f}")
```

---

## Step 5: Plot

```python
import matplotlib.pyplot as plt

r_plot = np.linspace(1, r_fit[-1], 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear scale
ax = axes[0]
ax.plot(r_centers[1:], G_r[1:], "o", ms=4, label="G(r) data")
ax.plot(r_plot, exp_decay(r_plot, *popt), "-", label=f"fit: ξ = {xi_fit:.2f}")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("r")
ax.set_ylabel("G(r)")
ax.set_title("Connected correlation function")
ax.legend()

# Log scale — exponential decay should be a straight line
ax = axes[1]
positive = G_r[1:] > 0
r_pos = r_centers[1:][positive]
ax.semilogy(r_pos, G_r[1:][positive], "o", ms=4, label="G(r) data")
ax.semilogy(r_plot, exp_decay(r_plot, *popt), "-", label=f"fit: ξ = {xi_fit:.2f}")
ax.set_xlabel("r")
ax.set_ylabel("G(r)  [log scale]")
ax.set_title("Log-scale: linearity confirms exponential decay")
ax.legend()

plt.tight_layout()
plt.savefig("correlation_function.png", dpi=150)
plt.show()
```

---

## Full Self-Contained Script

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Load ---
config = np.load("config.npy")
spins = 2 * config.astype(float) - 1
L = spins.shape[0]
m = np.mean(spins)

# --- 2D autocorrelation via FFT ---
F    = np.fft.fft2(spins)
C2D  = np.fft.ifft2(F * np.conj(F)).real / (L * L)
C2D  = np.fft.fftshift(C2D)
G2D  = C2D - m**2

# --- Radial average ---
cy, cx = L // 2, L // 2
y_idx  = np.arange(L) - cy
x_idx  = np.arange(L) - cx
dx, dy = np.meshgrid(x_idx, y_idx)
r      = np.sqrt(dx**2 + dy**2)
r_max  = L // 2

bin_edges      = np.arange(0, r_max + 1) - 0.5
bin_edges[0]   = -0.5
G_sum, _       = np.histogram(r.ravel(), bins=bin_edges, weights=G2D.ravel())
counts, _      = np.histogram(r.ravel(), bins=bin_edges)
r_centers      = np.arange(r_max)
G_r            = G_sum / counts

# --- Exponential fit ---
def exp_decay(r, A, xi):
    return A * np.exp(-r / xi)

mask   = (r_centers >= 1) & (G_r > 0) & (r_centers < r_max)
popt, pcov = curve_fit(exp_decay, r_centers[mask], G_r[mask],
                       p0=[G_r[1], L / 4], maxfev=5000)
A_fit, xi_fit = popt
xi_err = np.sqrt(pcov[1, 1])

print(f"Correlation length ξ = {xi_fit:.2f} ± {xi_err:.2f}")

# --- Plot ---
r_plot = np.linspace(1, r_centers[mask][-1], 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, log in zip(axes, [False, True]):
    plot = ax.semilogy if log else ax.plot
    plot(r_centers[1:], G_r[1:], "o", ms=4, label="G(r)")
    plot(r_plot, exp_decay(r_plot, *popt), "-",
         label=f"ξ = {xi_fit:.2f} ± {xi_err:.2f}")
    ax.set_xlabel("r")
    ax.set_ylabel("G(r)" + (" [log]" if log else ""))
    ax.legend()
plt.tight_layout()
plt.savefig("correlation_function.png", dpi=150)
plt.show()
```

---

## Notes and Caveats

- **Fit range**: The exponential fit is only valid where `G(r) > 0` and `r ≪ L/2`. Near `r_max = L/2`, periodic-boundary wrap-around contaminates the signal.
- **Single snapshot**: With one configuration there is no ensemble average, so `G(r)` will be noisy, especially at large `r`. The fit uncertainty `xi_err` reflects only the statistical quality of the fit, not the thermodynamic variance.
- **Near T_c**: At the critical point the true decay is `G(r) ~ r^{−η} exp(−r/ξ)` with `η = 1/4` for 2D Ising. A pure exponential fit will then underestimate ξ. If you are working near T_c, consider fitting `A · r^{−η} · exp(−r/ξ)` instead.
- **Magnetized phase**: If `|m| ≈ 1`, `G(r)` is nearly flat and ξ diverges — the fit will be unreliable. This indicates you are deep in the ordered phase.
