import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import ccf  # kept since it was already imported in your code

# ----------------------------
# Output directory setup
# ----------------------------
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_current_figure(filename, dpi=300):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ----------------------------
# Q5 helper / Q6
# ----------------------------
def random_walk(n=200, delta=0.0, sigma=1.0, x0=0.0, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, sigma, size=n)
    x = np.empty(n)
    prev = x0
    for i in range(n):
        prev = delta + prev + w[i]
        x[i] = prev
    return x

def drift_t_statistic(x, x0=0.0):
    """
    Computes the one-sample t-statistic for testing H0: delta = 0
    in the random walk model by differencing the series.
    """
    dx = np.diff(np.r_[x0, x])   # [x1-x0, x2-x1, ..., xn-x_{n-1}]
    n = len(dx)
    mean_dx = dx.mean()
    sd_dx = dx.std(ddof=1)
    t_stat = mean_dx / (sd_dx / np.sqrt(n))
    df = n - 1
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=df)
    return t_stat, df, p_value

# Q6
x_no_drift = random_walk(n=200, delta=0.0, sigma=1.0, seed=42)
t0, df0, p0 = drift_t_statistic(x_no_drift)

x_big_drift = random_walk(n=200, delta=3.0, sigma=1.0, seed=43)
t1, df1, p1 = drift_t_statistic(x_big_drift)

print("No drift:")
print(f"t = {t0:.4f}, df = {df0}, p-value = {p0:.4g}")

print("\nLarge nonzero drift:")
print(f"t = {t1:.4f}, df = {df1}, p-value = {p1:.4g}")


# ----------------------------
# Q7
# ----------------------------
n = 200
n_rep = 50
delta = 0.5
sigma = 1.0
seed = 123

rng = np.random.default_rng(seed)

walks = np.column_stack([
    random_walk(n=n, delta=delta, sigma=sigma, seed=rng.integers(1_000_000))
    for _ in range(n_rep)
])

t = np.arange(1, n + 1)
mu_hat = walks.mean(axis=1)
sd_hat = walks.std(axis=1, ddof=1)

plt.figure(figsize=(10, 6))

for j in range(n_rep):
    plt.plot(t, walks[:, j], alpha=0.20)

plt.plot(t, mu_hat, linewidth=2.5, label="Sample mean")
plt.plot(t, mu_hat + sd_hat, linestyle=":", linewidth=2, label="Mean + SD")
plt.plot(t, mu_hat - sd_hat, linestyle=":", linewidth=2, label="Mean - SD")

plt.title("50 Random Walks with Nonzero Drift")
plt.xlabel("Time")
plt.ylabel("x_t")
plt.legend()
save_current_figure("q7_random_walks_with_drift.png")


# ----------------------------
# Q10
# ----------------------------
def simulate_product_process(n=200, mu=0.0, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(mu, sigma, size=n + 1)
    x = w[1:] * w[:-1]
    return x

# Parameters
n = 200
sigma = 1.0
mu0 = 0.0
mu1 = 2.0

# Simulations
x_q8 = simulate_product_process(n=n, mu=mu0, sigma=sigma, seed=100)
x_q9 = simulate_product_process(n=n, mu=mu1, sigma=sigma, seed=101)

# Population mean/variance
pop_mean_q8 = 0.0
pop_var_q8 = sigma**4

pop_mean_q9 = mu1**2
pop_var_q9 = sigma**4 + 2 * mu1**2 * sigma**2

print("Q8 process (mu = 0)")
print(f"Sample mean     = {x_q8.mean():.4f}")
print(f"Population mean = {pop_mean_q8:.4f}")
print(f"Sample variance = {x_q8.var(ddof=1):.4f}")
print(f"Population var  = {pop_var_q8:.4f}")

print("\nQ9 process (mu != 0)")
print(f"Sample mean     = {x_q9.mean():.4f}")
print(f"Population mean = {pop_mean_q9:.4f}")
print(f"Sample variance = {x_q9.var(ddof=1):.4f}")
print(f"Population var  = {pop_var_q9:.4f}")

# Plot the series
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(x_q8)
axes[0].axhline(pop_mean_q8, linestyle="--")
axes[0].set_title("Q8: x_t = w_t w_(t-1), with w_t ~ N(0, sigma^2)")

axes[1].plot(x_q9)
axes[1].axhline(pop_mean_q9, linestyle="--")
axes[1].set_title("Q9: x_t = w_t w_(t-1), with w_t ~ N(mu, sigma^2), mu != 0")

save_current_figure("q10_simulated_series_q8_q9.png")

# Plot sample ACFs
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(x_q8, lags=20, ax=axes[0])
axes[0].set_title("Sample ACF for Q8 process")

plot_acf(x_q9, lags=20, ax=axes[1])
axes[1].set_title("Sample ACF for Q9 process")

save_current_figure("q10_sample_acfs_q8_q9.png")


# ----------------------------
# Q17
# ----------------------------
# NYT historical state-level dataset
url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
df = pd.read_csv(url, parse_dates=["date"])

states = ["Florida", "Georgia", "New York", "Pennsylvania", "Texas"]

df = df[df["state"].isin(states)].copy()
df = df.sort_values(["state", "date"])

# Convert cumulative counts to daily increments
df["new_cases"] = df.groupby("state")["cases"].diff().fillna(0)
df["new_deaths"] = df.groupby("state")["deaths"].diff().fillna(0)

# Replace occasional negative revisions with 0
df["new_cases"] = df["new_cases"].clip(lower=0)
df["new_deaths"] = df["new_deaths"].clip(lower=0)

def standardized(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=0)

def sample_ccf_symmetric(x, y, max_lag=30):
    """
    Returns lags from -max_lag to +max_lag and the corresponding
    sample cross-correlations Corr(x_t, y_{t+k}).
    """
    x = standardized(x)
    y = standardized(y)

    lags = np.arange(-max_lag, max_lag + 1)
    vals = []

    for k in lags:
        if k < 0:
            vals.append(np.corrcoef(x[-k:], y[:len(y)+k])[0, 1])
        elif k > 0:
            vals.append(np.corrcoef(x[:-k], y[k:])[0, 1])
        else:
            vals.append(np.corrcoef(x, y)[0, 1])

    return lags, np.array(vals)

def minmax_scale(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

for state in states:
    d = df[df["state"] == state].copy()

    cases = d["new_cases"].to_numpy()
    deaths = d["new_deaths"].to_numpy()
    dates = d["date"].to_numpy()

    safe_state = state.lower().replace(" ", "_")

    # Cross-correlation plot
    lags, corr = sample_ccf_symmetric(cases, deaths, max_lag=30)

    plt.figure(figsize=(9, 4))
    plt.stem(lags, corr, basefmt=" ")
    plt.axhline(0, linewidth=1)
    plt.title(f"Cross-correlation: Cases vs Deaths ({state})")
    plt.xlabel("Lag k  [Corr(cases_t, deaths_(t+k))]")
    plt.ylabel("Cross-correlation")
    save_current_figure(f"q17_ccf_{safe_state}.png")

    # Overlay plot with common scaling
    cases_scaled = minmax_scale(cases)
    deaths_scaled = minmax_scale(deaths)

    plt.figure(figsize=(10, 4))
    plt.plot(dates, cases_scaled, label="Cases (scaled)")
    plt.plot(dates, deaths_scaled, label="Deaths (scaled)")
    plt.title(f"COVID Cases and Deaths, scaled together ({state})")
    plt.xlabel("Date")
    plt.ylabel("Scaled value")
    plt.legend()
    save_current_figure(f"q17_overlay_{safe_state}.png")

print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")