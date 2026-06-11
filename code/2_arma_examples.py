import os
import numpy as np
import pandas as pd
import altair as alt
from scipy.signal import lfilter

# Set up reliable paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "../figs")
os.makedirs(output_dir, exist_ok=True)

# Simulation, ACF, PACF, and property helper functions

def simulate_arma(phi=None, theta=None, n=1000, sigma=1.0, burn_in=200, seed=42):
    """
    Simulate an ARMA(p, q) process:
    phi(B) x_t = theta(B) w_t
    where w_t ~ N(0, sigma^2)
    """
    np.random.seed(seed)
    phi = np.array(phi) if phi is not None else np.array([])
    theta = np.array(theta) if theta is not None else np.array([])
    
    # Set up lfilter coefficients:
    # Denominator a: [1, -phi_1, -phi_2, ..., -phi_p]
    a = np.ones(len(phi) + 1)
    a[1:] = -phi
    
    # Numerator b: [1, theta_1, theta_2, ..., theta_q]
    b = np.ones(len(theta) + 1)
    b[1:] = theta
    
    # Generate white noise sequence
    total_n = n + burn_in
    w = np.random.normal(0, sigma, total_n)
    
    # Filter using scipy's lfilter
    x = lfilter(b, a, w)
    
    # Drop burn-in to avoid starting from zero transients
    return x[burn_in:], w[burn_in:]

def compute_acf(x, nlags=20):
    """
    Compute sample Autocorrelation Function (ACF)
    """
    n = len(x)
    x_mean = np.mean(x)
    x_var = np.var(x)
    if x_var == 0:
        return np.zeros(nlags + 1)
    acf = []
    for h in range(nlags + 1):
        if h == 0:
            acf.append(1.0)
        else:
            cov = np.sum((x[h:] - x_mean) * (x[:-h] - x_mean)) / n
            acf.append(cov / x_var)
    return np.array(acf)

def compute_autocovariance(x, nlags=20):
    """
    Compute sample Autocovariance Function
    """
    n = len(x)
    x_mean = np.mean(x)
    autocov = []
    for h in range(nlags + 1):
        if h == 0:
            autocov.append(np.var(x))
        else:
            cov = np.sum((x[h:] - x_mean) * (x[:-h] - x_mean)) / n
            autocov.append(cov)
    return np.array(autocov)

def compute_pacf(x, nlags=20):
    """
    Compute sample Partial Autocorrelation Function (PACF) using Levinson-Durbin recursion
    """
    acf = compute_acf(x, nlags=nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    if nlags >= 1:
        pacf[1] = acf[1]
    
    phi = np.zeros(nlags + 1)
    phi[1] = acf[1]
    
    for k in range(2, nlags + 1):
        numerator = acf[k] - np.sum(phi[1:k] * acf[1:k][::-1])
        denominator = 1.0 - np.sum(phi[1:k] * acf[1:k])
        
        if np.abs(denominator) < 1e-12:
            pacf[k] = 0.0
        else:
            pacf[k] = numerator / denominator
            
        new_phi = np.zeros(nlags + 1)
        new_phi[k] = pacf[k]
        for j in range(1, k):
            new_phi[j] = phi[j] - pacf[k] * phi[k - j]
        phi = new_phi
        
    return pacf

def get_acf_pacf_df(x, type_name, nlags=25):
    """
    Helper to bundle ACF and PACF into a pandas DataFrame
    """
    acf = compute_acf(x, nlags=nlags)
    pacf = compute_pacf(x, nlags=nlags)
    lags = np.arange(nlags + 1)
    return pd.DataFrame({
        'lag': lags,
        'acf': acf,
        'pacf': pacf,
        'type': type_name
    })

def analyze_properties(phi=None, theta=None, J=10):
    """
    Analyze roots of AR and MA operators to check causality and invertibility.
    Also computes the J-order linear filter coefficients psi_j and pi_j.
    """
    phi = np.array(phi) if phi is not None else np.array([])
    theta = np.array(theta) if theta is not None else np.array([])
    
    p = len(phi)
    q = len(theta)
    
    # AR roots (Causality check)
    if p > 0:
        # phi(z) = 1 - phi_1 z - ... - phi_p z^p = 0
        # Roots in np.roots expects coefficients of decreasing powers: [-phi_p, ..., -phi_1, 1]
        ar_poly = np.append(-phi[::-1], 1.0)
        ar_roots = np.roots(ar_poly)
        ar_roots_moduli = np.abs(ar_roots)
        is_causal = np.all(ar_roots_moduli > 1.0001)
    else:
        ar_roots = np.array([])
        ar_roots_moduli = np.array([])
        is_causal = True
        
    # MA roots (Invertibility check)
    if q > 0:
        # theta(z) = 1 + theta_1 z + ... + theta_q z^q = 0
        # Coefficients of decreasing powers: [theta_q, ..., theta_1, 1]
        ma_poly = np.append(theta[::-1], 1.0)
        ma_roots = np.roots(ma_poly)
        ma_roots_moduli = np.abs(ma_roots)
        is_invertible = np.all(ma_roots_moduli > 1.0001)
    else:
        ma_roots = np.array([])
        ma_roots_moduli = np.array([])
        is_invertible = True
        
    # Recursive computation of psi_j coefficients:
    # psi_j = theta_j + sum_{k=1}^{min(j, p)} phi_k * psi_{j-k}
    psi = np.zeros(J + 1)
    psi[0] = 1.0
    for j in range(1, J + 1):
        theta_j = theta[j - 1] if j <= q else 0.0
        sum_term = 0.0
        for k in range(1, min(j, p) + 1):
            sum_term += phi[k - 1] * psi[j - k]
        psi[j] = theta_j + sum_term
        
    # Recursive computation of pi_j coefficients:
    # pi_j = -phi_j - sum_{k=1}^{min(j, q)} theta_k * pi_{j-k}
    pi_coefs = np.zeros(J + 1)
    pi_coefs[0] = 1.0
    for j in range(1, J + 1):
        phi_j = phi[j - 1] if j <= p else 0.0
        sum_term = 0.0
        for k in range(1, min(j, q) + 1):
            sum_term += theta[k - 1] * pi_coefs[j - k]
        pi_coefs[j] = -phi_j - sum_term
        
    return {
        'ar_roots': ar_roots,
        'ar_roots_moduli': ar_roots_moduli,
        'is_causal': is_causal,
        'ma_roots': ma_roots,
        'ma_roots_moduli': ma_roots_moduli,
        'is_invertible': is_invertible,
        'psi': psi,
        'pi': pi_coefs
    }

def compute_theoretical_autocovariance(phi=None, theta=None, sigma=1.0, J=500, nlags=20):
    """
    Compute theoretical autocovariance using the psi filter representation:
    gamma(h) = sigma^2 * sum_{j=0}^{J} psi_j * psi_{j+h}
    """
    phi = np.array(phi) if phi is not None else np.array([])
    theta = np.array(theta) if theta is not None else np.array([])
    
    p = len(phi)
    q = len(theta)
    
    # Generate psi coefficients up to J + nlags
    psi = np.zeros(J + nlags + 1)
    psi[0] = 1.0
    for j in range(1, J + nlags + 1):
        theta_j = theta[j - 1] if j <= q else 0.0
        sum_term = 0.0
        for k in range(1, min(j, p) + 1):
            sum_term += phi[k - 1] * psi[j - k]
        psi[j] = theta_j + sum_term
        
    # Compute autocovariances
    gamma = np.zeros(nlags + 1)
    for h in range(nlags + 1):
        gamma[h] = (sigma**2) * np.sum(psi[:J + 1] * psi[h:J + h + 1])
        
    return gamma

# Plotting helper functions

def create_diagnostics_row(df_traj, df_acf_pacf, type_name, color, n_traj_plot=150, title_prefix=""):
    """
    Creates a row containing: [Trajectory, ACF, PACF]
    """
    # Trajectory chart (first n_traj_plot points)
    traj_data = df_traj[df_traj['time'] < n_traj_plot]
    chart_traj = alt.Chart(traj_data).mark_line(strokeWidth=1.2).encode(
        x=alt.X('time:Q', title='Time (t)'),
        y=alt.Y('value:Q', title='Value (x_t)'),
        color=alt.value(color)
    ).properties(
        title=f"{title_prefix} Trajectory (First {n_traj_plot} pts)",
        width=350,
        height=140
    )
    
    # ACF chart (ignoring lag 0 to emphasize lag structure)
    df_no_zero = df_acf_pacf[df_acf_pacf['lag'] > 0]
    ci = 1.96 / np.sqrt(len(df_traj))
    
    chart_acf = alt.layer(
        alt.Chart(df_no_zero).mark_bar(size=6).encode(
            x=alt.X('lag:O', title='Lag (h)'),
            y=alt.Y('acf:Q', title='ACF', scale=alt.Scale(domain=[-1.0, 1.0])),
            color=alt.value(color)
        ),
        alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(ci)),
        alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(-ci))
    ).properties(
        title=f"Sample ACF",
        width=200,
        height=140
    )
    
    # PACF chart
    chart_pacf = alt.layer(
        alt.Chart(df_no_zero).mark_bar(size=6).encode(
            x=alt.X('lag:O', title='Lag (h)'),
            y=alt.Y('pacf:Q', title='PACF', scale=alt.Scale(domain=[-1.0, 1.0])),
            color=alt.value(color)
        ),
        alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(ci)),
        alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(-ci))
    ).properties(
        title=f"Sample PACF",
        width=200,
        height=140
    )
    
    return alt.hconcat(chart_traj, chart_acf, chart_pacf)

# Part 1: AR(1) model analysis (phi = +0.9 vs phi = -0.9)

def analyze_and_plot_ar1():
    print("\n--- AR(1) Models Comparison ---")
    n = 1000
    phi_pos = [0.9]
    phi_neg = [-0.9]
    
    # Simulate
    x_pos, _ = simulate_arma(phi=phi_pos, n=n, seed=42)
    x_neg, _ = simulate_arma(phi=phi_neg, n=n, seed=42)
    
    # Get properties and theoretical comparison
    prop_pos = analyze_properties(phi=phi_pos)
    prop_neg = analyze_properties(phi=phi_neg)
    
    gamma_pos = compute_theoretical_autocovariance(phi=phi_pos, nlags=5)
    gamma_neg = compute_theoretical_autocovariance(phi=phi_neg, nlags=5)
    
    print(f"AR(1) phi = +0.9 (Causal: {prop_pos['is_causal']}):")
    print(f"  Root: {prop_pos['ar_roots'][0]:.4f} (Modulus: {prop_pos['ar_roots_moduli'][0]:.4f})")
    print(f"  First 5 Theoretical Autocovariances: {gamma_pos}")
    print(f"  First 5 Sample Autocovariances:      {compute_autocovariance(x_pos, nlags=5)}")
    print(f"  First 5 Psi coefficients:            {prop_pos['psi'][:5]}")
    print(f"\nAR(1) phi = -0.9 (Causal: {prop_neg['is_causal']}):")
    print(f"  Root: {prop_neg['ar_roots'][0]:.4f} (Modulus: {prop_neg['ar_roots_moduli'][0]:.4f})")
    print(f"  First 5 Theoretical Autocovariances: {gamma_neg}")
    print(f"  First 5 Sample Autocovariances:      {compute_autocovariance(x_neg, nlags=5)}")
    print(f"  First 5 Psi coefficients:            {prop_neg['psi'][:5]}")
    
    # DataFrames
    df_traj_pos = pd.DataFrame({'time': np.arange(n), 'value': x_pos})
    df_acf_pacf_pos = get_acf_pacf_df(x_pos, 'AR(1) phi = +0.9')
    
    df_traj_neg = pd.DataFrame({'time': np.arange(n), 'value': x_neg})
    df_acf_pacf_neg = get_acf_pacf_df(x_neg, 'AR(1) phi = -0.9')
    
    # Create combined row plots
    row_pos = create_diagnostics_row(df_traj_pos, df_acf_pacf_pos, 'AR(1) phi = +0.9', '#2980b9', title_prefix="AR(1) phi = +0.9")
    row_neg = create_diagnostics_row(df_traj_neg, df_acf_pacf_neg, 'AR(1) phi = -0.9', '#e67e22', title_prefix="AR(1) phi = -0.9")
    
    chart = alt.vconcat(row_pos, row_neg).properties(
        title=alt.TitleParams(
            text="AR(1) Processes Comparison (phi = +0.9 vs. -0.9)",
            subtitle="Causal AR(1) shows exponential decay (phi=0.9) or oscillating decay (phi=-0.9) in ACF, and a cut-off at lag 1 in PACF",
            fontSize=16,
            anchor="start",
            offset=15
        )
    )
    
    chart.save(os.path.join(output_dir, '1_ar_phi_comparison.png'))
    print("\nSaved 1_ar_phi_comparison.png successfully to figs/.")

# Part 2: MA(1) model analysis (theta = +0.9 vs theta = -0.9)

def analyze_and_plot_ma1():
    print("\n--- MA(1) Models Comparison ---")
    n = 1000
    theta_pos = [0.9]
    theta_neg = [-0.9]
    
    # Simulate
    x_pos, _ = simulate_arma(theta=theta_pos, n=n, seed=42)
    x_neg, _ = simulate_arma(theta=theta_neg, n=n, seed=42)
    
    # Get properties and theoretical comparison
    prop_pos = analyze_properties(theta=theta_pos)
    prop_neg = analyze_properties(theta=theta_neg)
    
    gamma_pos = compute_theoretical_autocovariance(theta=theta_pos, nlags=3)
    gamma_neg = compute_theoretical_autocovariance(theta=theta_neg, nlags=3)
    
    print(f"MA(1) theta = +0.9 (Invertible: {prop_pos['is_invertible']}):")
    print(f"  Root: {prop_pos['ma_roots'][0]:.4f} (Modulus: {prop_pos['ma_roots_moduli'][0]:.4f})")
    print(f"  First 3 Theoretical Autocovariances: {gamma_pos}")
    print(f"  First 3 Sample Autocovariances:      {compute_autocovariance(x_pos, nlags=3)}")
    print(f"  First 5 Pi coefficients (Invertible): {prop_pos['pi'][:5]}")
    print(f"\nMA(1) theta = -0.9 (Invertible: {prop_neg['is_invertible']}):")
    print(f"  Root: {prop_neg['ma_roots'][0]:.4f} (Modulus: {prop_neg['ma_roots_moduli'][0]:.4f})")
    print(f"  First 3 Theoretical Autocovariances: {gamma_neg}")
    print(f"  First 3 Sample Autocovariances:      {compute_autocovariance(x_neg, nlags=3)}")
    print(f"  First 5 Pi coefficients (Invertible): {prop_neg['pi'][:5]}")
    
    # DataFrames
    df_traj_pos = pd.DataFrame({'time': np.arange(n), 'value': x_pos})
    df_acf_pacf_pos = get_acf_pacf_df(x_pos, 'MA(1) theta = +0.9')
    
    df_traj_neg = pd.DataFrame({'time': np.arange(n), 'value': x_neg})
    df_acf_pacf_neg = get_acf_pacf_df(x_neg, 'MA(1) theta = -0.9')
    
    # Create combined row plots
    row_pos = create_diagnostics_row(df_traj_pos, df_acf_pacf_pos, 'MA(1) theta = +0.9', '#27ae60', title_prefix="MA(1) theta = +0.9")
    row_neg = create_diagnostics_row(df_traj_neg, df_acf_pacf_neg, 'MA(1) theta = -0.9', '#8e44ad', title_prefix="MA(1) theta = -0.9")
    
    chart = alt.vconcat(row_pos, row_neg).properties(
        title=alt.TitleParams(
            text="MA(1) Processes Comparison (theta = +0.9 vs. -0.9)",
            subtitle="Invertible MA(1) shows a cut-off at lag 1 in ACF, and exponential/oscillating decay in PACF",
            fontSize=16,
            anchor="start",
            offset=15
        )
    )
    
    chart.save(os.path.join(output_dir, '2_ma_theta_comparison.png'))
    print("\nSaved 2_ma_theta_comparison.png successfully to figs/.")

# Part 3: MA(1) non-uniqueness analysis

def analyze_and_plot_ma_non_uniqueness():
    print("\n--- MA(1) Non-Uniqueness Comparison ---")
    n = 1000
    
    # Process A: theta = 5.0, var_w = 1.0 (Non-invertible representation)
    theta_a = [5.0]
    sigma_a = 1.0
    
    # Process B: theta = 1/5 = 0.2, var_w = 25.0 (Invertible representation)
    theta_b = [0.2]
    sigma_b = 5.0  # std deviation is sqrt(25.0) = 5.0
    
    # Simulate both
    x_a, _ = simulate_arma(theta=theta_a, n=n, sigma=sigma_a, seed=42)
    x_b, _ = simulate_arma(theta=theta_b, n=n, sigma=sigma_b, seed=42)
    
    prop_a = analyze_properties(theta=theta_a)
    prop_b = analyze_properties(theta=theta_b)
    
    print("Theoretical autocovariance calculations:")
    print("  Process A: gamma(0) = (1 + 25)*1.0 = 26.0, gamma(1) = 5.0*1.0 = 5.0")
    print("  Process B: gamma(0) = (1 + 0.04)*25.0 = 26.0, gamma(1) = 0.2*25.0 = 5.0")
    print(f"Process A (theta=5.0): Invertible={prop_a['is_invertible']}, Root={prop_a['ma_roots'][0]:.4f}")
    print(f"Process B (theta=0.2): Invertible={prop_b['is_invertible']}, Root={prop_b['ma_roots'][0]:.4f}")
    
    df_a = get_non_uniqueness_df(x_a, 'Process A (theta = 5.0, var = 1.0)')
    df_b = get_non_uniqueness_df(x_b, 'Process B (theta = 0.2, var = 25.0)')
    
    print(f"  Process A Sample Autocovariance at lag 0, 1, 2: {compute_autocovariance(x_a, nlags=2)}")
    print(f"  Process B Sample Autocovariance at lag 0, 1, 2: {compute_autocovariance(x_b, nlags=2)}")
    
    # Build charts
    def make_non_uniqueness_column(df, title, color):
        cov_chart = alt.Chart(df[df['lag'] < 8]).mark_bar(size=12).encode(
            x=alt.X('lag:O', title='Lag (h)'),
            y=alt.Y('cov:Q', title='Autocovariance (gamma)', scale=alt.Scale(domain=[0, 30])),
            color=alt.value(color)
        ).properties(
            title=f"{title} - Autocovariance",
            width=280,
            height=140
        )
        
        acf_chart = alt.Chart(df[df['lag'] > 0]).mark_bar(size=12).encode(
            x=alt.X('lag:O', title='Lag (h)'),
            y=alt.Y('acf:Q', title='ACF (rho)', scale=alt.Scale(domain=[-0.2, 1.0])),
            color=alt.value(color)
        ).properties(
            title=f"{title} - ACF (Lag >= 1)",
            width=280,
            height=140
        )
        
        return alt.vconcat(cov_chart, acf_chart)
        
    col_a = make_non_uniqueness_column(df_a, "Non-Invertible (theta=5.0)", "#1abc9c")
    col_b = make_non_uniqueness_column(df_b, "Invertible (theta=0.2)", "#34495e")
    
    chart = alt.hconcat(col_a, col_b).properties(
        title=alt.TitleParams(
            text="MA(1) Parameter Non-uniqueness & Redundancy",
            subtitle="Both models exhibit the identical autocovariance and ACF structure, showing observational equivalence.",
            fontSize=16,
            anchor="start",
            offset=15
        )
    )
    
    chart.save(os.path.join(output_dir, '3_ma_non_uniqueness.png'))
    print("\nSaved 3_ma_non_uniqueness.png successfully to figs/.")

def get_non_uniqueness_df(x, name, nlags=10):
    acf = compute_acf(x, nlags=nlags)
    cov = compute_autocovariance(x, nlags=nlags)
    lags = np.arange(nlags + 1)
    return pd.DataFrame({
        'lag': lags,
        'acf': acf,
        'cov': cov,
        'process': name
    })

# Part 4: Parameter redundancy in ARMA(1,1)

def analyze_and_plot_parameter_redundancy():
    print("\n--- Parameter Redundancy in ARMA(1,1) ---")
    n = 1000
    phi = [0.5]
    theta = [-0.5]
    
    # Simulate ARMA(1,1) with phi=0.5, theta=-0.5 (redundant factor 1-0.5B on both sides)
    x_redundant, _ = simulate_arma(phi=phi, theta=theta, n=n, seed=42)
    # Simulate pure white noise for comparison
    np.random.seed(42)
    x_wn = np.random.normal(0, 1, n)
    
    df_redundant = get_acf_pacf_df(x_redundant, 'ARMA(1,1) Redundant')
    df_wn = get_acf_pacf_df(x_wn, 'Pure White Noise')
    
    # Output properties to console
    prop = analyze_properties(phi=phi, theta=theta)
    print("Roots of Redundant ARMA(1,1) (1 - 0.5B) x_t = (1 - 0.5B) w_t:")
    print(f"  AR Polynomial Root: {prop['ar_roots'][0]:.4f} (Modulus: {prop['ar_roots_moduli'][0]:.4f})")
    print(f"  MA Polynomial Root: {prop['ma_roots'][0]:.4f} (Modulus: {prop['ma_roots_moduli'][0]:.4f})")
    print("  Note: The roots cancel out (pole-zero cancellation), leaving a pure white noise process.")
    
    # Build ACF plots
    ci = 1.96 / np.sqrt(n)
    
    def make_acf_chart(df, title, color):
        # Omit lag 0 for readability
        df_no_zero = df[df['lag'] > 0]
        return alt.layer(
            alt.Chart(df_no_zero).mark_bar(size=8).encode(
                x=alt.X('lag:O', title='Lag (h)'),
                y=alt.Y('acf:Q', title='ACF', scale=alt.Scale(domain=[-0.2, 0.4])),
                color=alt.value(color)
            ),
            alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(ci)),
            alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(-ci))
        ).properties(
            title=title,
            width=280,
            height=140
        )
        
    chart_red = make_acf_chart(df_redundant, "ARMA(1,1) with phi=0.5, theta=-0.5 (ACF)", "#d35400")
    chart_wn = make_acf_chart(df_wn, "Pure White Noise (ACF)", "#7f8c8d")
    
    chart = alt.hconcat(chart_red, chart_wn).properties(
        title=alt.TitleParams(
            text="Parameter Redundancy: ARMA(1,1) Pole-Zero Cancellation",
            subtitle="The redundant ARMA(1,1) process has the same sample ACF characteristics as a pure white noise process.",
            fontSize=16,
            anchor="start",
            offset=15
        )
    )
    
    chart.save(os.path.join(output_dir, '4_parameter_redundancy.png'))
    print("\nSaved 4_parameter_redundancy.png successfully to figs/.")

# Part 5: AR(2) vs MA(3) comparison (Figure 3 reproduction)

def analyze_and_plot_fig3():
    print("\n--- Figure 3 Reproduction (AR(2) vs MA(3)) ---")
    n = 1000
    phi_ar2 = [1.5, -0.75]
    theta_ma3 = [0.9, 0.4, 0.1]
    
    # Simulate processes
    x_ar2, _ = simulate_arma(phi=phi_ar2, n=n, seed=42)
    x_ma3, _ = simulate_arma(theta=theta_ma3, n=n, seed=42)
    
    # Get properties
    prop_ar2 = analyze_properties(phi=phi_ar2)
    prop_ma3 = analyze_properties(theta=theta_ma3)
    
    print("AR(2) properties:")
    print(f"  Causal: {prop_ar2['is_causal']}")
    print(f"  Roots:  {prop_ar2['ar_roots']}")
    print(f"  Moduli: {prop_ar2['ar_roots_moduli']}")
    print("\nMA(3) properties:")
    print(f"  Invertible: {prop_ma3['is_invertible']}")
    print(f"  Roots:      {prop_ma3['ma_roots']}")
    print(f"  Moduli:     {prop_ma3['ma_roots_moduli']}")
    
    # Compute DataFrames
    df_ar2 = get_acf_pacf_df(x_ar2, 'AR(2)', nlags=20)
    df_ma3 = get_acf_pacf_df(x_ma3, 'MA(3)', nlags=20)
    
    ci = 1.96 / np.sqrt(n)
    
    # Create subplot layered function
    def make_plot(df, value_col, title, y_title, color):
        # Remove lag 0 to emphasize decay & cutoff
        df_no_zero = df[df['lag'] > 0]
        
        return alt.layer(
            alt.Chart(df_no_zero).mark_bar(size=8).encode(
                x=alt.X('lag:O', title='Lag (h)'),
                y=alt.Y(f'{value_col}:Q', title=y_title, scale=alt.Scale(domain=[-1.0, 1.0])),
                color=alt.value(color)
            ),
            alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(ci)),
            alt.Chart(df_no_zero).mark_rule(color='red', strokeDash=[4, 4]).encode(y=alt.datum(-ci))
        ).properties(
            title=title,
            width=280,
            height=160
        )
        
    ar2_acf = make_plot(df_ar2, 'acf', 'AR(2) Autocorrelation (ACF)', 'ACF', '#2980b9')
    ar2_pacf = make_plot(df_ar2, 'pacf', 'AR(2) Partial Autocorrelation (PACF)', 'PACF', '#2980b9')
    
    ma3_acf = make_plot(df_ma3, 'acf', 'MA(3) Autocorrelation (ACF)', 'ACF', '#27ae60')
    ma3_pacf = make_plot(df_ma3, 'pacf', 'MA(3) Partial Autocorrelation (PACF)', 'PACF', '#27ae60')
    
    # Combine charts into a 2x2 layout
    col_ar2 = alt.vconcat(ar2_acf, ar2_pacf)
    col_ma3 = alt.vconcat(ma3_acf, ma3_pacf)
    
    chart = alt.hconcat(col_ar2, col_ma3).properties(
        title=alt.TitleParams(
            text="Figure 3 Reproduction: ACF and PACF Comparison",
            subtitle="Comparing a causal AR(2) process (left) and an invertible MA(3) process (right)",
            fontSize=16,
            anchor="start",
            offset=15
        )
    )
    
    chart.save(os.path.join(output_dir, '5_ar2_ma3_comparison.png'))
    print("\nSaved 5_ar2_ma3_comparison.png successfully to figs/.")

# Main script execution

if __name__ == '__main__':
    print("--- Running ARMA Time Series Examples Script ---")
    
    # Run all analyses and plot generation
    analyze_and_plot_ar1()
    analyze_and_plot_ma1()
    analyze_and_plot_ma_non_uniqueness()
    analyze_and_plot_parameter_redundancy()
    analyze_and_plot_fig3()
    
    print(f"\nAll outputs and figures saved successfully to: {output_dir}")
