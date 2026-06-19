# ============================================================
# Q7: HP filter kernel matrix K
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Problem setup
n = 100
lam = 100

# Create second-difference matrix D of size (n-2) x n
D = np.zeros((n - 2, n))

for i in range(n - 2):
    D[i, i] = 1
    D[i, i + 1] = -2
    D[i, i + 2] = 1

# Identity matrix
I = np.eye(n)

# Compute K = (I + lambda D^T D)^(-1)
K = np.linalg.inv(I + lam * D.T @ D)

# Rows to inspect
rows_to_plot = [25, 50, 75]

# Convert to Python zero-based indexing
row_indices = [i - 1 for i in rows_to_plot]

# Plot the selected rows
plt.figure(figsize=(9, 5))

for row, label in zip(row_indices, rows_to_plot):
    plt.plot(
        np.arange(1, n + 1),
        K[row, :],
        linewidth=2,
        label=f"i = {label}"
    )

plt.xlabel("Position j")
plt.ylabel(r"$K_{ij}$")
plt.title("Rows of HP Filter Matrix K")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot instead of showing it
plt.savefig("q7_hp_filter_kernel_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Q11: Empirical verification of spectral autocovariance
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(123)

# Number of time points
n = 1000000
t = np.arange(1, n + 1)

#the error might be empirical vs theoretical at the boundaries (add to discussion)
# At least p = 2 components
omega = np.array([0.05, 0.15])
sigma2 = np.array([4.0, 1.0])
p = len(omega)

# Generate uncorrelated random coefficients
U1 = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=p)
U2 = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=p)

# Generate process
x = np.zeros(n)

for j in range(p):
    x += (
        U1[j] * np.cos(2 * np.pi * omega[j] * t)
        + U2[j] * np.sin(2 * np.pi * omega[j] * t)
    )

# Empirical autocorrelation
max_lag = 50
empirical_acf = acf(x, nlags=max_lag, fft=True)

# Analytic autocovariance:
# gamma(h) = sum_j sigma_j^2 cos(2*pi*omega_j*h)
lags = np.arange(0, max_lag + 1)

gamma_h = np.zeros(len(lags))

for j in range(p):
    gamma_h += sigma2[j] * np.cos(2 * np.pi * omega[j] * lags)

# Convert analytic autocovariance to autocorrelation
analytic_acf = gamma_h / gamma_h[0]

# Plot empirical ACF vs analytic ACF
plt.figure(figsize=(9, 5))

plt.vlines(
    lags,
    ymin=0,
    ymax=empirical_acf,
    linewidth=2,
    label="Empirical ACF"
)

plt.plot(
    lags,
    analytic_acf,
    marker="o",
    linewidth=2,
    label="Analytic ACF"
)

plt.axhline(0, linewidth=1)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Empirical ACF vs Analytic ACF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot instead of showing it
plt.savefig("q11_empirical_vs_analytic_acf.png", dpi=300, bbox_inches="tight")
plt.close()