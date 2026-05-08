"""
Homework 2 — Python Simulation Questions Only

Includes:
- Q10: Simulate data, plot yhat1/yhat2, compute MAE and MAPE
- Q11: Create yhat3, plot yhat2/yhat3, compute MAE, MAPE, and MARE
- Q17: Optional empirical example showing perfect training fit with enough features

Outputs:
- plots/q10_predictions.png
- plots/q11_predictions.png
- plots/q17_perfect_training_fit.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------

def mae(y_true, y_pred):
    """
    Mean Absolute Error.

    MAE = mean(|y - yhat|)
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.

    MAPE = 100 * mean(|y - yhat| / |y|)

    Note:
    This assumes y_true has no zero values.
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / np.abs(y_true))


def mare(y_true, y_pred):
    """
    Mean Absolute Relative Error.

    MARE = 100 * mean(|y - yhat| / |yhat|)

    This is different from MAPE because the denominator is the prediction,
    not the observed value.
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / np.abs(y_pred))


# ---------------------------------------------------------------------
# Shared data simulation for Q10 and Q11
# ---------------------------------------------------------------------

def simulate_metrics_matter_data(seed=0):
    """
    Simulates the data used in Q10 and Q11.

    This is the Python version of the R code:

        set.seed(0)
        x = 1:50
        y = rpois(n = 50, lambda = c(rep(5, 25), 5 + exp(0:24 * 0.2)))
        yhat1 = c(rep(6.5, 25), 6.5 + exp(0:24 * 0.2))
        yhat2 = c(rep(5, 25), 5 + exp(0:24 * 0.18))

    Note:
    Python and R use different random number generators, so the exact y values
    may differ from R even with the same seed.
    """

    np.random.seed(seed)

    x = np.arange(1, 51)

    lam = np.concatenate([
        np.repeat(5, 25),
        5 + np.exp(np.arange(25) * 0.2)
    ])

    y = np.random.poisson(lam=lam)

    yhat1 = np.concatenate([
        np.repeat(6.5, 25),
        6.5 + np.exp(np.arange(25) * 0.2)
    ])

    yhat2 = np.concatenate([
        np.repeat(5, 25),
        5 + np.exp(np.arange(25) * 0.18)
    ])

    return x, y, yhat1, yhat2


# ---------------------------------------------------------------------
# Q10
# ---------------------------------------------------------------------

def solve_q10():
    """
    Q10:

    Plot predictions from yhat1 and yhat2 as lines overlaid on the original data.
    Compute and report MAE and MAPE for each model.
    Discuss what the metrics show.
    """

    print("\n" + "=" * 80)
    print("Q10 — Metrics matter: Model 1 vs Model 2")
    print("=" * 80)

    x, y, yhat1, yhat2 = simulate_metrics_matter_data(seed=0)

    # Compute metrics
    results = pd.DataFrame({
        "Model": ["Model 1", "Model 2"],
        "MAE": [
            mae(y, yhat1),
            mae(y, yhat2)
        ],
        "MAPE": [
            mape(y, yhat1),
            mape(y, yhat2)
        ]
    })

    print("\nQ10 metrics:")
    print(results.to_string(index=False))

    # Plot observed data and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="-", label="Observed y")
    plt.plot(x, yhat1, linewidth=2, label="Model 1 predictions")
    plt.plot(x, yhat2, linewidth=2, label="Model 2 predictions")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Q10: Observed Data and Predictions from Models 1 and 2")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "q10_predictions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved Q10 plot to: {output_path}")

    print(
        "\nQ10 discussion:\n"
        "Model 1 has the lower MAE, meaning it has the smaller average absolute error. "
        "Model 2 has the lower MAPE, meaning it has the smaller average percentage error. "
        "This difference happens because MAE and MAPE emphasize different parts of the time series. "
        "MAE measures raw absolute error, so large errors during the high-count part of the series "
        "can dominate. MAPE divides by the observed value, so errors during the low-count part of "
        "the series can be heavily penalized. Model 1 is shifted upward early in the series, which "
        "creates relatively large percentage errors when y is small. Therefore, Model 1 can look "
        "better by MAE while Model 2 looks better by MAPE."
    )


# ---------------------------------------------------------------------
# Q11
# ---------------------------------------------------------------------

def solve_q11():
    """
    Q11:

    Define yhat3 by starting with yhat2 and changing the exponent multiplier
    in the last 25 predictions from 0.18 to 0.22.

    Plot yhat2 and yhat3 overlaid on the data.
    Compute MAE, MAPE, and MARE for yhat2 and yhat3.
    Discuss the comparison.
    """

    print("\n" + "=" * 80)
    print("Q11 — Model 2 vs Model 3 with MAE, MAPE, and MARE")
    print("=" * 80)

    x, y, _, yhat2 = simulate_metrics_matter_data(seed=0)

    # Model 3:
    # First 25 predictions are the same as yhat2.
    # Last 25 predictions use exponent multiplier 0.22 instead of 0.18.
    yhat3 = np.concatenate([
        np.repeat(5, 25),
        5 + np.exp(np.arange(25) * 0.22)
    ])

    # Compute metrics
    results = pd.DataFrame({
        "Model": ["Model 2", "Model 3"],
        "MAE": [
            mae(y, yhat2),
            mae(y, yhat3)
        ],
        "MAPE": [
            mape(y, yhat2),
            mape(y, yhat3)
        ],
        "MARE": [
            mare(y, yhat2),
            mare(y, yhat3)
        ]
    })

    print("\nQ11 metrics:")
    print(results.to_string(index=False))

    # Plot observed data and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="-", label="Observed y")
    plt.plot(x, yhat2, linewidth=2, label="Model 2 predictions")
    plt.plot(x, yhat3, linewidth=2, label="Model 3 predictions")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Q11: Observed Data and Predictions from Models 2 and 3")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "q11_predictions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved Q11 plot to: {output_path}")

    print(
        "\nQ11 discussion:\n"
        "Model 3 is worse than Model 2 according to both MAE and MAPE. "
        "This is because Model 3 grows too quickly in the second half of the series after "
        "changing the exponent multiplier from 0.18 to 0.22. As a result, it overshoots the "
        "observed values more strongly. The MARE comparison is more subtle because MARE divides "
        "by the prediction rather than by the observed value. Since Model 3 makes larger predictions, "
        "the denominator in MARE is also larger, which can make overprediction look less severe. "
        "This shows that metric choice matters: different error metrics can lead to different "
        "interpretations of model performance."
    )


# ---------------------------------------------------------------------
# Q17
# ---------------------------------------------------------------------

def solve_q17():
    """
    Q17 bonus:

    Implement an empirical example verifying that with enough linearly independent
    features, a regression model can achieve perfect training accuracy.

    Since this question asks for an empirical verification, we simulate arbitrary
    data and create an identity-matrix feature design.

    With X = I_n and no intercept, linear regression can exactly reproduce y,
    because each row has its own feature.
    """

    print("\n" + "=" * 80)
    print("Q17 — Empirical verification of perfect training fit")
    print("=" * 80)

    np.random.seed(0)

    n = 50
    t = np.arange(n)

    # Simulate arbitrary response values
    y = np.sin(t / 5) + np.random.normal(scale=0.2, size=n)

    # Create n linearly independent features using the identity matrix
    X = np.eye(n)

    # Fit without intercept so that the identity basis can exactly reconstruct y
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    y_fitted = model.predict(X)

    training_mse = mean_squared_error(y, y_fitted)

    print(f"\nTraining MSE: {training_mse:.12f}")

    # Plot observed y and fitted values
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, marker="o", linestyle="-", label="Observed y")
    plt.plot(t, y_fitted, linewidth=2, label="Fitted values")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Q17: Perfect Training Fit with Enough Features\nTraining MSE = {training_mse:.12f}")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "q17_perfect_training_fit.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved Q17 plot to: {output_path}")

    print(
        "\nQ17 discussion:\n"
        "The training MSE is essentially zero because the identity matrix gives the model one "
        "linearly independent feature per observation. This lets the regression memorize the "
        "training data exactly. This verifies empirically that with enough linearly independent "
        "features, training error can be driven to zero. However, this does not imply good "
        "forecasting or generalization performance. It is an example of overfitting."
    )


# ---------------------------------------------------------------------
# Run all simulation questions
# ---------------------------------------------------------------------

if __name__ == "__main__":
    solve_q10()
    solve_q11()
    solve_q17()

    print("\n" + "=" * 80)
    print("Finished running simulation questions.")
    print(f"All plots saved in: {OUTPUT_DIR}/")
    print("=" * 80)