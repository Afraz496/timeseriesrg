"""
Homework 4 -- Introduction to Time Series
Python solution to the computational questions (Q3, Q12-Q16).

All forecasting is done with statsmodels' SARIMAX. We re-implement the
fable time-series-CV workflow (stretch_tsibble -> model -> forecast ->
accuracy) by hand, using expanding ("stretched") training windows.

Outputs:
  * prints all numerical answers
  * writes results to results.json
  * writes figures to fig_*.png
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

RESULTS = {}
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# Q3 -- verify that polynomials in the backshift operator commute
# ----------------------------------------------------------------------
def q3_backshift_commute():
    """A polynomial in B is encoded by its coefficient vector; multiplying
    two such polynomials is exactly the convolution of their coefficients.
    Convolution is commutative, so the two products are identical."""
    # phi(B) = 1 + 0.5 B - 0.2 B^2   (coeffs ordered B^0, B^1, B^2, ...)
    phi  = np.array([1.0, 0.5, -0.2])
    # varphi(B) = 1 - 0.3 B + 0.4 B^2 + 0.1 B^3
    vphi = np.array([1.0, -0.3, 0.4, 0.1])

    prod_ab = np.convolve(phi, vphi)   # phi(B) * varphi(B)
    prod_ba = np.convolve(vphi, phi)   # varphi(B) * phi(B)

    same = np.allclose(prod_ab, prod_ba)
    RESULTS["q3"] = {
        "phi": phi.tolist(),
        "varphi": vphi.tolist(),
        "phi_times_varphi": np.round(prod_ab, 6).tolist(),
        "varphi_times_phi": np.round(prod_ba, 6).tolist(),
        "identical": bool(same),
    }
    print("\n=== Q3: backshift polynomials commute ===")
    print("phi(B)*varphi(B) =", np.round(prod_ab, 4))
    print("varphi(B)*phi(B) =", np.round(prod_ba, 4))
    print("identical:", same)


# ----------------------------------------------------------------------
# Q12 -- number of rows produced by stretch_tsibble(x, .init = t0)
# ----------------------------------------------------------------------
def stretch_rows(n, t0, step=1):
    """Return (n_ids, total_rows) for stretch_tsibble with .init=t0, .step=step.
    Window lengths are t0, t0+step, t0+2*step, ... up to <= n."""
    lengths = list(range(t0, n + 1, step))
    return len(lengths), int(sum(lengths))


def q12_stretch_row_count():
    """For .step = 1 the window lengths are t0, t0+1, ..., n, so

        rows = sum_{k=t0}^{n} k = (n + t0)(n - t0 + 1) / 2 ,
        #ids = n - t0 + 1 .
    """
    print("\n=== Q12: stretch_tsibble row count ===")
    checks = []
    for n, t0 in [(10, 3), (100, 50), (225, 50), (12, 1)]:
        n_ids, rows = stretch_rows(n, t0)               # brute force
        formula = (n + t0) * (n - t0 + 1) // 2           # closed form
        ok = rows == formula
        checks.append({"n": n, "t0": t0, "n_ids": n_ids,
                       "rows_bruteforce": rows, "rows_formula": formula,
                       "match": bool(ok)})
        print(f"  n={n:>4}, t0={t0:>3}: ids={n_ids:>4}, "
              f"rows(loop)={rows:>6}, rows(formula)={formula:>6}, match={ok}")
    RESULTS["q12"] = {
        "formula_rows": "(n + t0)(n - t0 + 1)/2",
        "formula_ids": "n - t0 + 1",
        "checks": checks,
    }


# ----------------------------------------------------------------------
# Q13 -- RW-with-drift: stretch-based CV == manual rolling-origin CV
# ----------------------------------------------------------------------
def rw_drift_forecast(train, h=1):
    """One-step (or h-step) random-walk-with-drift point forecast.
    drift = (x_T - x_1)/(T-1); forecast_{T+h} = x_T + drift*h."""
    T = len(train)
    drift = (train[-1] - train[0]) / (T - 1)
    return train[-1] + drift * np.arange(1, h + 1)


def q13_rw_drift_equivalence():
    print("\n=== Q13: RW-drift  stretch-CV vs manual-CV ===")
    # data analogous to the homework example (value = 1..10 + noise)
    n = 30
    y = np.arange(1, n + 1) + rng.normal(0, 0.25, n)
    init = 3

    # ---- (a) "fable-style": build stretched ids, forecast h=1 on each,
    #          then average absolute error against the true series ----
    abs_err_stretch = []
    for L in range(init, n):                 # id windows of length L = init..n-1
        train = y[:L]
        fc = rw_drift_forecast(train, h=1)[0]
        actual = y[L]                        # the (L+1)-th observation
        abs_err_stretch.append(abs(actual - fc))
    mae_stretch = float(np.mean(abs_err_stretch))

    # ---- (b) manual rolling-origin loop (classic time-series CV) ----
    abs_err_manual = []
    for t in range(init, n):                 # forecast origin = t observations
        train = y[:t]
        drift = (train[-1] - train[0]) / (len(train) - 1)
        pred = train[-1] + drift             # 1-step ahead
        abs_err_manual.append(abs(y[t] - pred))
    mae_manual = float(np.mean(abs_err_manual))

    print(f"  MAE (stretch workflow): {mae_stretch:.6f}")
    print(f"  MAE (manual loop)     : {mae_manual:.6f}")
    print(f"  match: {np.isclose(mae_stretch, mae_manual)}")
    RESULTS["q13"] = {"mae_stretch": mae_stretch, "mae_manual": mae_manual,
                      "match": bool(np.isclose(mae_stretch, mae_manual))}


# ----------------------------------------------------------------------
# Shared rolling-origin CV engine for SARIMA models (Q14-Q16)
# ----------------------------------------------------------------------
def rolling_cv_sarima(y, order, seasonal_order, trend, init=50, max_h=12,
                      warm_start=True, verbose=False):
    """Expanding-window time-series CV.

    For every origin with L = init, init+1, ..., n-1 observations we fit the
    model on y[:L] and forecast horizons 1..max_h, keeping only horizons whose
    actual value exists.  Returns a long DataFrame of (origin, h, error)."""
    n = len(y)
    rows = []
    prev_params = None
    for L in range(init, n):                       # last usable origin has n-1 obs
        train = y[:L]
        h_max = min(max_h, n - L)
        try:
            mod = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                          trend=trend, enforce_stationarity=False,
                          enforce_invertibility=False)
            if warm_start and prev_params is not None and \
               len(prev_params) == len(mod.start_params):
                res = mod.fit(start_params=prev_params, disp=False, maxiter=200)
            else:
                res = mod.fit(disp=False, maxiter=200)
            prev_params = res.params
            fc = np.asarray(res.get_forecast(h_max).predicted_mean)
        except Exception as e:                     # pragma: no cover
            if verbose:
                print("   fit failed at L=%d: %s" % (L, e))
            continue
        for h in range(1, h_max + 1):
            rows.append((L, h, y[L + h - 1] - fc[h - 1]))
    return pd.DataFrame(rows, columns=["origin", "h", "error"])


def summarise_cv(err_df):
    """Overall MAE (pooled over all origins & horizons) and MAE by horizon."""
    overall = err_df["error"].abs().mean()
    by_h = err_df.groupby("h")["error"].apply(lambda e: e.abs().mean())
    return float(overall), by_h


MODELS = {
    "ARIMA(2,1,0)":              dict(order=(2, 1, 0), seasonal_order=(0, 0, 0, 0),  trend="c"),
    "ARIMA(0,1,2)":              dict(order=(0, 1, 2), seasonal_order=(0, 0, 0, 0),  trend="c"),
    "ARIMA(2,1,0)(1,1,0)[12]":   dict(order=(2, 1, 0), seasonal_order=(1, 1, 0, 12), trend="n"),
    "ARIMA(0,1,2)(0,1,1)[12]":   dict(order=(0, 1, 2), seasonal_order=(0, 1, 1, 12), trend="n"),
}


def q14_q15_leisure_cv(y):
    print("\n=== Q14 / Q15: leisure CV (init=50, horizons 1-12) ===")
    overall_mae, by_h_all, err_store = {}, {}, {}
    for name, spec in MODELS.items():
        t0 = time.time()
        err = rolling_cv_sarima(y, spec["order"], spec["seasonal_order"],
                                spec["trend"], init=50, max_h=12)
        mae, by_h = summarise_cv(err)
        overall_mae[name] = mae
        by_h_all[name] = by_h
        err_store[name] = err
        print(f"  {name:<26} MAE = {mae:.4f}   ({time.time()-t0:4.1f}s, "
              f"{len(err)} forecasts)")

    ranking = sorted(overall_mae, key=overall_mae.get)
    print("\n  Ranking (best -> worst):")
    for i, name in enumerate(ranking, 1):
        print(f"    {i}. {name:<26} {overall_mae[name]:.4f}")

    RESULTS["q14"] = {
        "overall_mae": {k: round(v, 5) for k, v in overall_mae.items()},
        "ranking": ranking,
    }
    RESULTS["q15"] = {
        name: {int(h): round(v, 5) for h, v in by_h_all[name].items()}
        for name in MODELS
    }

    # ---- Q15 figure: MAE vs horizon ----
    plt.figure(figsize=(8, 5))
    markers = {"ARIMA(2,1,0)": "o", "ARIMA(0,1,2)": "s",
               "ARIMA(2,1,0)(1,1,0)[12]": "^", "ARIMA(0,1,2)(0,1,1)[12]": "D"}
    for name in MODELS:
        bh = by_h_all[name]
        plt.plot(bh.index, bh.values, marker=markers[name], label=name)
    plt.xlabel("forecast horizon $h$ (months)")
    plt.ylabel("MAE")
    plt.title("Time-series-CV MAE by forecast horizon (leisure data)")
    plt.xticks(range(1, 13))
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_q15_mae_by_horizon.png", dpi=140)
    plt.close()

    # focused comparison of the two seasonal models
    plt.figure(figsize=(8, 5))
    for name in ["ARIMA(2,1,0)(1,1,0)[12]", "ARIMA(0,1,2)(0,1,1)[12]"]:
        bh = by_h_all[name]
        plt.plot(bh.index, bh.values, marker=markers[name], label=name)
    plt.xlabel("forecast horizon $h$ (months)")
    plt.ylabel("MAE")
    plt.title("Seasonal models: MAE by horizon")
    plt.xticks(range(1, 13))
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_q15_seasonal_compare.png", dpi=140)
    plt.close()
    return err_store


def q16_auto_arima_cv(y, init=50, max_h=12):
    print("\n=== Q16: auto-ARIMA in the CV pipeline (slow) ===")
    try:
        from pmdarima import auto_arima
    except Exception as e:                          # pragma: no cover
        print("  pmdarima unavailable:", e)
        return
    n = len(y)
    rows, t0 = [], time.time()
    for L in range(init, n):
        train = y[:L]
        h_max = min(max_h, n - L)
        try:
            m = auto_arima(train, seasonal=True, m=12,
                           stepwise=True, suppress_warnings=True,
                           error_action="ignore", max_order=6,
                           D=1, d=1)
            fc = np.asarray(m.predict(h_max))
        except Exception:
            continue
        for h in range(1, h_max + 1):
            rows.append((L, h, y[L + h - 1] - fc[h - 1]))
    err = pd.DataFrame(rows, columns=["origin", "h", "error"])
    mae, _ = summarise_cv(err)
    print(f"  auto-ARIMA  MAE = {mae:.4f}   ({time.time()-t0:.0f}s total)")
    RESULTS["q16"] = {"overall_mae": round(mae, 5),
                      "runtime_sec": round(time.time() - t0, 1)}


def plot_leisure(y, dates):
    plt.figure(figsize=(9, 4))
    plt.plot(dates, y, lw=1)
    plt.xlabel("month")
    plt.ylabel("employed (millions)")
    plt.title("US Leisure & Hospitality employment (2001-2019)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_leisure_series.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    leis = pd.read_csv("leisure.csv", parse_dates=["Month"])
    y = leis["Employed"].values.astype(float)
    RESULTS["leisure_n"] = int(len(y))

    plot_leisure(y, leis["Month"])
    q3_backshift_commute()
    q12_stretch_row_count()
    q13_rw_drift_equivalence()
    q14_q15_leisure_cv(y)
    q16_auto_arima_cv(y)

    with open("results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved results.json and figures.")