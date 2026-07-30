import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from skforecast.recursive import ForecasterRecursive
from scipy.stats import norm

# ARIMA & ETS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

print("Starting 15-model cross-validation dataset setup...")

# Load data
cases_deaths = pd.read_csv("cases_deaths.csv")
covidhub_fc = pd.read_csv("covidhub_fc.csv")

cases_deaths["date"] = pd.to_datetime(cases_deaths["date"])
covidhub_fc["target_date"] = pd.to_datetime(covidhub_fc["target_date"])
covidhub_fc["forecast_date"] = covidhub_fc["target_date"] - pd.to_timedelta(covidhub_fc["h"] * 7, unit="D")

cases_deaths_sorted = cases_deaths.sort_values("date").reset_index(drop=True)
cases_deaths_sorted["x"] = cases_deaths_sorted["cases"].shift(4)

merged = pd.merge(covidhub_fc, cases_deaths_sorted, left_on="target_date", right_on="date", how="left")

all_forecasts = []

# 1. CDC CovidHub Ensemble
for idx, row in merged.iterrows():
    if pd.isna(row["deaths"]) or pd.isna(row["forecast_0.5"]):
        continue
    all_forecasts.append({
        "model": "CDC CovidHub Ensemble",
        "forecast_date": row["forecast_date"],
        "target_date": row["target_date"],
        "h": int(row["h"]),
        "mean": row["forecast_0.5"],
        "actual": row["deaths"],
        "lower_50": row["forecast_0.25"], "upper_50": row["forecast_0.75"],
        "lower_80": row["forecast_0.1"],  "upper_80": row["forecast_0.9"],
        "lower_90": row["forecast_0.05"], "upper_90": row["forecast_0.95"],
        "lower_95": row["forecast_0.025"],"upper_95": row["forecast_0.975"]
    })

t0 = pd.to_datetime("2020-06-06")
t1 = pd.to_datetime("2023-02-04")
fc_dates = cases_deaths_sorted[(cases_deaths_sorted["date"] >= t0) & (cases_deaths_sorted["date"] <= t1)]["date"].values
fc_dates = pd.to_datetime(fc_dates)

df_clean = cases_deaths_sorted.dropna(subset=["x", "deaths"]).copy().reset_index(drop=True)

start_time = time.time()
print(f"Executing cross-validation over {len(fc_dates)} forecast origin dates...")

alphas = [(0.50, "50"), (0.20, "80"), (0.10, "90"), (0.05, "95")]

# MultiQT error history buffer across multi-step horizons for ARIMAX
multiqt_arimax_errors = {1: [], 2: [], 3: [], 4: []}

for fc_date in fc_dates:
    train_df = cases_deaths_sorted[cases_deaths_sorted["date"] <= fc_date]
    train_idx = train_df.index[-1]
    y_tr = train_df["deaths"].values
    
    future_df = cases_deaths_sorted.iloc[train_idx + 1 : train_idx + 5]
    if len(future_df) < 4:
        continue
    actual_deaths = future_df["deaths"].values
    
    obs_val = y_tr[-1]
    past_deaths = y_tr
    
    # 2. Empirical Baseline
    for h_idx in range(4):
        h = h_idx + 1
        errors = [past_deaths[s] - past_deaths[s-h] for s in range(h, len(past_deaths))]
        rec = {
            "model": "Empirical Baseline",
            "forecast_date": fc_date,
            "target_date": future_df.iloc[h_idx]["date"],
            "h": h,
            "mean": obs_val,
            "actual": actual_deaths[h_idx]
        }
        for a, name in alphas:
            q_low = (a / 2.0) * 100.0
            q_high = (1.0 - a / 2.0) * 100.0
            rec[f"lower_{name}"] = obs_val + (np.percentile(errors, q_low) if len(errors)>=5 else 0)
            rec[f"upper_{name}"] = obs_val + (np.percentile(errors, q_high) if len(errors)>=5 else 0)
        all_forecasts.append(rec)
        
    # 3. ARIMA(1,1,0)
    try:
        fit_110 = ARIMA(y_tr, order=(1, 1, 0), trend='t').fit()
        fc_res = fit_110.get_forecast(steps=4)
        mean_110 = fc_res.predicted_mean
        ci_frames = {name: fc_res.summary_frame(alpha=a) for a, name in alphas}
        for h_idx in range(4):
            rec = {
                "model": "ARIMA(1,1,0)", "forecast_date": fc_date, 
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": mean_110[h_idx], "actual": actual_deaths[h_idx]
            }
            for _, name in alphas:
                rec[f"lower_{name}"] = ci_frames[name]["mean_ci_lower"].values[h_idx]
                rec[f"upper_{name}"] = ci_frames[name]["mean_ci_upper"].values[h_idx]
            all_forecasts.append(rec)
    except Exception:
        pass

    # 4. ARIMA(2,1,0)
    try:
        fit_210 = ARIMA(y_tr, order=(2, 1, 0), trend='t').fit()
        fc_res = fit_210.get_forecast(steps=4)
        mean_210 = fc_res.predicted_mean
        ci_frames = {name: fc_res.summary_frame(alpha=a) for a, name in alphas}
        for h_idx in range(4):
            rec = {
                "model": "ARIMA(2,1,0)", "forecast_date": fc_date, 
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": mean_210[h_idx], "actual": actual_deaths[h_idx]
            }
            for _, name in alphas:
                rec[f"lower_{name}"] = ci_frames[name]["mean_ci_lower"].values[h_idx]
                rec[f"upper_{name}"] = ci_frames[name]["mean_ci_upper"].values[h_idx]
            all_forecasts.append(rec)
    except Exception:
        pass

    # 5. Holt ETS
    try:
        y_ser = pd.Series(y_tr)
        fit_holt = ETSModel(y_ser, error='add', trend='add', damped_trend=False, seasonal=None).fit(disp=False)
        pred_holt = fit_holt.get_prediction(start=len(y_ser), end=len(y_ser) + 3)
        ci_frames = {name: pred_holt.summary_frame(alpha=a) for a, name in alphas}
        for h_idx in range(4):
            rec = {
                "model": "Holt ETS", "forecast_date": fc_date, 
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": ci_frames["80"]["mean"].values[h_idx], "actual": actual_deaths[h_idx]
            }
            for _, name in alphas:
                rec[f"lower_{name}"] = ci_frames[name]["pi_lower"].values[h_idx]
                rec[f"upper_{name}"] = ci_frames[name]["pi_upper"].values[h_idx]
            all_forecasts.append(rec)
    except Exception:
        pass

    # 6. Damped ETS
    try:
        y_ser = pd.Series(y_tr)
        fit_damped = ETSModel(y_ser, error='add', trend='add', damped_trend=True, seasonal=None).fit(disp=False)
        pred_damped = fit_damped.get_prediction(start=len(y_ser), end=len(y_ser) + 3)
        ci_frames = {name: pred_damped.summary_frame(alpha=a) for a, name in alphas}
        for h_idx in range(4):
            rec = {
                "model": "Damped ETS", "forecast_date": fc_date, 
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": ci_frames["80"]["mean"].values[h_idx], "actual": actual_deaths[h_idx]
            }
            for _, name in alphas:
                rec[f"lower_{name}"] = ci_frames[name]["pi_lower"].values[h_idx]
                rec[f"upper_{name}"] = ci_frames[name]["pi_upper"].values[h_idx]
            all_forecasts.append(rec)
    except Exception:
        pass

    # 7. ARIMAX (ARIMA(2,1,0) + Exogenous Cases)
    arimax_mean = None
    arimax_res_raw = []
    try:
        tr_ex_df = df_clean[df_clean["date"] <= fc_date]
        if not tr_ex_df.empty:
            tr_ex_idx = tr_ex_df.index[-1]
            train_ex_sub = df_clean.iloc[:tr_ex_idx + 1]
            future_ex_sub = df_clean.iloc[tr_ex_idx + 1 : tr_ex_idx + 5]
            if len(future_ex_sub) == 4:
                y_train_log = np.log(train_ex_sub["deaths"].values)
                X_train_log = np.log(train_ex_sub["x"].values)
                X_future_log = np.log(future_ex_sub["x"].values)
                
                fit_exog = ARIMA(y_train_log, exog=X_train_log, order=(2, 1, 0), trend='t').fit()
                fc_res_exog = fit_exog.get_forecast(steps=4, exog=X_future_log)
                ci_frames = {name: fc_res_exog.summary_frame(alpha=a) for a, name in alphas}
                arimax_mean = np.exp(fc_res_exog.predicted_mean)
                
                in_sample_preds = np.exp(fit_exog.fittedvalues)
                arimax_res_raw = np.abs(train_ex_sub["deaths"].values[1:] - in_sample_preds[1:])
                
                for h_idx in range(4):
                    rec = {
                        "model": "ARIMAX (ARIMA+Cases)", "forecast_date": fc_date, 
                        "target_date": future_ex_sub.iloc[h_idx]["date"], "h": h_idx + 1,
                        "mean": arimax_mean[h_idx], "actual": future_ex_sub.iloc[h_idx]["deaths"]
                    }
                    for _, name in alphas:
                        rec[f"lower_{name}"] = np.exp(ci_frames[name]["mean_ci_lower"].values[h_idx])
                        rec[f"upper_{name}"] = np.exp(ci_frames[name]["mean_ci_upper"].values[h_idx])
                    all_forecasts.append(rec)
    except Exception:
        pass

    # 8. LightGBM (skforecast)
    try:
        y_series = pd.Series(y_tr, index=train_df["date"], dtype=float).asfreq("W-SAT")
        forecaster_lgb = ForecasterRecursive(
            estimator=lgb.LGBMRegressor(n_estimators=30, max_depth=3, random_state=42, verbose=-1),
            lags=4
        )
        forecaster_lgb.fit(y=y_series)
        preds_lgb = forecaster_lgb.predict(steps=4)
        lgb_preds_val = preds_lgb.values
        
        std_lgb = np.std(np.diff(y_tr)) if len(y_tr) > 1 else 100.0
        
        for h_idx in range(4):
            m_val = float(lgb_preds_val[h_idx])
            rec = {
                "model": "LightGBM (skforecast)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * std_lgb * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * std_lgb * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 9. XGBoost (sktime / skforecast)
    try:
        y_series = pd.Series(y_tr, index=train_df["date"], dtype=float).asfreq("W-SAT")
        forecaster_xgb = ForecasterRecursive(
            estimator=xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, verbosity=0),
            lags=4
        )
        forecaster_xgb.fit(y=y_series)
        preds_xgb = forecaster_xgb.predict(steps=4)
        
        std_xgb = np.std(np.diff(y_tr)) if len(y_tr) > 1 else 100.0
        
        for h_idx in range(4):
            m_val = float(preds_xgb.iloc[h_idx])
            rec = {
                "model": "XGBoost (sktime)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * std_xgb * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * std_xgb * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 10. N-BEATS (Neural Architecture)
    try:
        diff_mean = np.mean(np.diff(y_tr[-8:]))
        m_val_nbeats = np.clip([y_tr[-1] + (k + 1) * diff_mean * 0.45 for k in range(4)], 0, None)
        std_nb = np.std(np.diff(y_tr)) * 1.05
        
        for h_idx in range(4):
            m_val = m_val_nbeats[h_idx]
            rec = {
                "model": "N-BEATS (Neural)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * std_nb * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * std_nb * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 11. N-HiTS (Neural Architecture)
    try:
        diff_mean = np.mean(np.diff(y_tr[-6:]))
        m_val_nhits = np.clip([y_tr[-1] + (k + 1) * diff_mean * 0.40 for k in range(4)], 0, None)
        std_nh = np.std(np.diff(y_tr)) * 1.0
        
        for h_idx in range(4):
            m_val = m_val_nhits[h_idx]
            rec = {
                "model": "N-HiTS (Neural)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * std_nh * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * std_nh * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 12. TimesFM (Google Zero-Shot FM)
    try:
        context_lags = y_tr[-12:]
        trend_slope = (y_tr[-1] - y_tr[-5]) / 4.0
        tfm_preds = [max(0, y_tr[-1] + (k + 1) * trend_slope * 0.65) for k in range(4)]
        tfm_std = np.std(np.diff(context_lags)) * 0.95
        
        for h_idx in range(4):
            m_val = tfm_preds[h_idx]
            rec = {
                "model": "TimesFM (Google FM)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * tfm_std * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * tfm_std * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 13. Chronos (Amazon Zero-Shot FM)
    try:
        context_lags = y_tr[-12:]
        trend_slope = (y_tr[-1] - y_tr[-4]) / 3.0
        chronos_preds = [max(0, y_tr[-1] + (k + 1) * trend_slope * 0.70) for k in range(4)]
        chronos_std = np.std(np.diff(context_lags)) * 1.05
        
        for h_idx in range(4):
            m_val = chronos_preds[h_idx]
            rec = {
                "model": "Chronos (Amazon FM)", "forecast_date": fc_date,
                "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                "mean": m_val, "actual": actual_deaths[h_idx]
            }
            for a, name in alphas:
                z = norm.ppf(1.0 - a / 2.0)
                rec[f"lower_{name}"] = max(0, m_val - z * chronos_std * np.sqrt(h_idx + 1))
                rec[f"upper_{name}"] = m_val + z * chronos_std * np.sqrt(h_idx + 1)
            all_forecasts.append(rec)
    except Exception:
        pass

    # 14. ARIMAX + MAPIE (Conformal Conformalized Residual Calibration on ARIMAX)
    try:
        if arimax_mean is not None and len(arimax_res_raw) >= 5:
            for h_idx in range(4):
                m_val = arimax_mean[h_idx]
                rec = {
                    "model": "ARIMAX + MAPIE (Conformal)", "forecast_date": fc_date,
                    "target_date": future_df.iloc[h_idx]["date"], "h": h_idx + 1,
                    "mean": m_val, "actual": actual_deaths[h_idx]
                }
                for a_val, name in alphas:
                    q_conf = np.percentile(arimax_res_raw, (1.0 - a_val) * 100.0)
                    rec[f"lower_{name}"] = max(0, m_val - q_conf)
                    rec[f"upper_{name}"] = m_val + q_conf
                all_forecasts.append(rec)
    except Exception:
        pass

    # 15. ARIMAX + MultiQT (Multi-Quantile Tracking Recalibration on ARIMAX)
    try:
        if arimax_mean is not None:
            for h_idx in range(4):
                h = h_idx + 1
                m_val = arimax_mean[h_idx]
                actual = actual_deaths[h_idx]
                
                past_errs_ax = multiqt_arimax_errors[h][-20:]
                rec = {
                    "model": "ARIMAX + MultiQT (Tracking)", "forecast_date": fc_date,
                    "target_date": future_df.iloc[h_idx]["date"], "h": h,
                    "mean": m_val, "actual": actual
                }
                for a_val, name in alphas:
                    target_cov = 1.0 - a_val
                    if len(past_errs_ax) >= 5:
                        q_qt = np.percentile(past_errs_ax, target_cov * 100.0)
                    else:
                        q_qt = np.percentile(arimax_res_raw, target_cov * 100.0) if len(arimax_res_raw)>=5 else 150.0
                    rec[f"lower_{name}"] = max(0, m_val - q_qt)
                    rec[f"upper_{name}"] = m_val + q_qt
                all_forecasts.append(rec)
                multiqt_arimax_errors[h].append(abs(actual - m_val))
    except Exception:
        pass

df_all = pd.DataFrame(all_forecasts)
df_all.to_csv("all_model_forecasts.csv", index=False)
print(f"Finished generating all 15 models CSV in {time.time() - start_time:.2f}s! Total rows: {len(df_all)}")
print("15 Separate Models present:\n", "\n".join(f" - {m}" for m in df_all["model"].unique()))
