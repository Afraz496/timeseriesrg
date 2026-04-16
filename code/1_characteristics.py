import numpy as np
import pandas as pd
import altair as alt
import os
from epidatpy import CovidcastEpidata, EpiDataContext, EpiRange

# Set up reliable paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "../figs")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# PART 1: Simulated White Noise Processes
# ---------------------------------------------------------

# Parameters
n = 1000
np.random.seed(42)

# Generate Data
gaussian_wn = np.random.normal(0, 1, n)
iid_non_gaussian_wn = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
e = np.random.normal(0, 1, n + 1)
dependent_wn = e[1:] * e[:-1]

# Summary Statistics Verification
processes = [
    ("Gaussian White Noise", gaussian_wn),
    ("IID White Noise (Uniform)", iid_non_gaussian_wn),
    ("Dependent White Noise", dependent_wn)
]

print("\n--- Summary Statistics (Simulated WN) ---")
for name, data in processes:
    # Calculate Lag-1 Autocorrelation
    lag1_corr = np.corrcoef(data[1:], data[:-1])[0, 1]
    print(f"{name}:")
    print(f"  Mean:     {np.mean(data):.4f}")
    print(f"  Variance: {np.var(data):.4f}")
    print(f"  Lag-1 Correlation: {lag1_corr:.4f}\n")

# Create a DataFrame for Trajectories and Distributions
df_sim = pd.concat([
    pd.DataFrame({'time': np.arange(n), 'value': gaussian_wn, 'type': 'Gaussian White Noise'}),
    pd.DataFrame({'time': np.arange(n), 'value': iid_non_gaussian_wn, 'type': 'IID White Noise (Uniform)'}),
    pd.DataFrame({'time': np.arange(n), 'value': dependent_wn, 'type': 'Dependent White Noise'})
])

# Trajectories Plot with Mean Line and Var Label
base_sim = alt.Chart(df_sim).encode(
    y=alt.Y('value:Q', title='Value'),
    color=alt.Color('type:N', legend=None)
)

# Use joinaggregate for stats to keep all layers on the same dataframe
stats_base = base_sim.transform_joinaggregate(
    mean_val='mean(value)',
    var_val='variance(value)',
    groupby=['type']
)

trajectories = alt.layer(
    base_sim.mark_line(strokeWidth=1).encode(x=alt.X('time:Q', title='Time')),
    stats_base.mark_rule(color='red', strokeDash=[5, 5]).encode(y='mean_val:Q'),
    stats_base.mark_text(align='right', dx=280, dy=-60, fontWeight='bold', color='black').encode(
        y='mean_val:Q',
        text=alt.Text('var_val:Q', format='.3f')
    ).transform_filter('datum.time == 0')
).properties(
    width=600,
    height=150
).facet(
    row=alt.Row('type:N', title=None, header=alt.Header(labelFontSize=12, labelFontWeight='bold'))
)

# Histograms with Mean Line
histograms = alt.layer(
    alt.Chart(df_sim).mark_bar(opacity=0.7).encode(
        y=alt.Y('value:Q', bin=alt.Bin(maxbins=30), title='Value Range'),
        x=alt.X('count():Q', title='Frequency'),
        color=alt.Color('type:N', legend=None)
    ),
    alt.Chart(df_sim).transform_aggregate(
        mean_val='mean(value)',
        groupby=['type']
    ).mark_rule(color='red', strokeDash=[5, 5]).encode(y='mean_val:Q')
).properties(
    width=200,
    height=150
).facet(
    row=alt.Row('type:N', title=None, header=alt.Header(labels=False))
)

# Combine Time Plots and Histograms
combined_stats = alt.hconcat(trajectories, histograms).resolve_scale(y='shared')

# ACF Plot
# Calculate ACF data
def get_acf_df(data, type_name):
    lags = np.arange(21)
    acf = [np.corrcoef(data[i:], data[:len(data)-i])[0, 1] if i > 0 else 1 for i in lags]
    variance = np.var(data)
    return pd.DataFrame({'lag': lags, 'correlation': acf, 'type': type_name, 'var': variance})

acf_df = pd.concat([
    get_acf_df(gaussian_wn, 'Gaussian White Noise'),
    get_acf_df(iid_non_gaussian_wn, 'IID White Noise (Uniform)'),
    get_acf_df(dependent_wn, 'Dependent White Noise')
])

ci = 1.96 / np.sqrt(n)

# Combine ACF bars and CI rules, then facet
acf_final = alt.layer(
    alt.Chart().mark_bar(size=5).encode(
        x=alt.X('lag:O', title='Lag'),
        y=alt.Y('correlation:Q', title='Autocorrelation (Cor)', scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color('type:N', legend=alt.Legend(orient='bottom'))
    ),
    alt.Chart().mark_rule(color='red', strokeDash=[5, 5]).encode(y=alt.datum(ci)),
    alt.Chart().mark_rule(color='red', strokeDash=[5, 5]).encode(y=alt.datum(-ci)),
    # Add Variance text (which is Covariance at lag 0)
    alt.Chart().mark_text(align='right', dx=280, dy=-60, fontWeight='bold', color='black').encode(
        text=alt.condition(alt.datum.lag == 0, alt.Text('var:Q', format='.3f'), alt.value(''))
    ).transform_calculate(label='"Cov(0)=Var: " + datum.var')
).facet(
    row=alt.Row('type:N', title=None, header=alt.Header(labelFontSize=12)),
    data=acf_df
)

# ---------------------------------------------------------
# PART 2: Real-World Data and Moving Averages (epidatpy)
# ---------------------------------------------------------

# Initialize context
epidata = EpiDataContext(use_cache=True, cache_max_age_days=7)

# Fetch JHU confirmed cumulative numbers for the US
apicall = epidata.pub_covidcast(
    data_source="jhu-csse",
    signals="confirmed_cumulative_num",
    geo_type="nation",
    time_type="day",
    geo_values="us",
    time_values=EpiRange(20210301, 20210601),
)

print("--- Real-World Analysis ---")
print("Fetching JHU data from Delphi API...")
df_real = apicall.df()

# Processing: Calculate daily new cases from cumulative
df_real['time_value'] = pd.to_datetime(df_real['time_value'])
df_real = df_real.sort_values('time_value')
df_real['daily_cases'] = df_real['value'].diff()

# Moving Averages
# 7-day Trailing Average
df_real['trailing_avg_7d'] = df_real['daily_cases'].rolling(window=7).mean()

# 7-day Centered Moving Average
df_real['centered_avg_7d'] = df_real['daily_cases'].rolling(window=7, center=True).mean()

# Drop rows with NAs
df_real_clean = df_real.dropna(subset=['daily_cases', 'trailing_avg_7d', 'centered_avg_7d'])

# Visualization
plot_df_real = df_real_clean.melt(
    id_vars=['time_value'], 
    value_vars=['daily_cases', 'trailing_avg_7d', 'centered_avg_7d'],
    var_name='Signal Type', 
    value_name='Cases'
)

name_map = {
    'daily_cases': 'Raw Daily Cases',
    'trailing_avg_7d': '7-day Trailing Average',
    'centered_avg_7d': '7-day Centered Average'
}
plot_df_real['Signal Type'] = plot_df_real['Signal Type'].map(name_map)

chart_ma = alt.Chart(plot_df_real).mark_line().encode(
    x=alt.X('time_value:T', title='Date'),
    y=alt.Y('Cases:Q', title='Number of Cases'),
    color=alt.Color('Signal Type:N', scale=alt.Scale(
        domain=['Raw Daily Cases', '7-day Trailing Average', '7-day Centered Average'],
        range=['#cccccc', '#2c3e50', '#e67e22']
    ), legend=alt.Legend(orient='bottom')),
    strokeDash=alt.condition(
        alt.datum['Signal Type'] == 'Raw Daily Cases',
        alt.value([3, 3]),
        alt.value([0])
    ),
    strokeWidth=alt.condition(
        alt.datum['Signal Type'] == 'Raw Daily Cases',
        alt.value(1),
        alt.value(2.5)
    )
).properties(
    title='US COVID-19 Daily Cases: Moving Average Comparison (Spring 2021)',
    width=800,
    height=400
).interactive()

# ---------------------------------------------------------
# PART 3: Autoregressive Model (AR1)
# ---------------------------------------------------------

# Simulate AR(1): x_t = phi * x_{t-1} + w_t
phi = 0.9
w = np.random.normal(0, 1, n)
ar1 = np.zeros(n)
for t in range(1, n):
    ar1[t] = phi * ar1[t-1] + w[t]

# Random Walks (Non-Stationary)
rw = np.cumsum(w) # Random Walk
delta = 0.2
rwd = np.cumsum(w + delta) # Random Walk with Drift

df_non_stationary = pd.concat([
    pd.DataFrame({'time': np.arange(n), 'value': rw, 'type': 'Random Walk'}),
    pd.DataFrame({'time': np.arange(n), 'value': rwd, 'type': 'Random Walk with Drift'})
])

# Visualization for Non-Stationary Processes
chart_rw = alt.Chart(df_non_stationary).mark_line(strokeWidth=1.5).encode(
    x=alt.X('time:Q', title='Time'),
    y=alt.Y('value:Q', title='Value'),
    color=alt.Color('type:N', scale=alt.Scale(range=['#8e44ad', '#27ae60']), legend=alt.Legend(orient='bottom'))
).properties(
    title='Random Walk vs. Random Walk with Drift',
    width=800,
    height=300
)

# Existing AR Analysis
df_ar = pd.DataFrame({'time': np.arange(n), 'value': ar1, 'type': f'AR(1) Process (phi={phi})'})

# AR Trajectory
chart_ar_traj = alt.Chart(df_ar).mark_line(strokeWidth=1).encode(
    x=alt.X('time:Q', title='Time'),
    y=alt.Y('value:Q', title='Value'),
    color=alt.value('#2980b9')
).properties(
    title=f'AR(1) Process Trajectory (phi={phi})',
    width=800,
    height=200
)

# AR ACF
ar_acf_df = get_acf_df(ar1, f'AR(1) phi={phi}')
chart_ar_acf = alt.Chart(ar_acf_df).mark_bar(size=10).encode(
    x=alt.X('lag:O', title='Lag'),
    y=alt.Y('correlation:Q', title='Autocorrelation', scale=alt.Scale(domain=[-1, 1])),
    color=alt.value('#2980b9')
).properties(
    title='ACF of AR(1) Process (Geometric Decay)',
    width=800,
    height=200
)

chart_ar_final = alt.vconcat(chart_ar_traj, chart_ar_acf)
# Combine with Random Walk chart
chart_part3_combined = alt.vconcat(chart_ar_final, chart_rw)

# ---------------------------------------------------------
# PART 4: Signal + Noise
# ---------------------------------------------------------

# Define a periodic signal + noise: y_t = 2*sin(2*pi*t/50) + noise
t_idx = np.arange(200) # Shorter for better visual clarity
signal = 2 * np.sin(2 * np.pi * t_idx / 50)
noise = np.random.normal(0, 1, 200)
y = signal + noise

df_sn = pd.DataFrame({
    'time': t_idx,
    'Raw (Signal + Noise)': y,
    'Pure Signal': signal
})

plot_df_sn = df_sn.melt(id_vars=['time'], var_name='Type', value_name='Value')

chart_sn = alt.Chart(plot_df_sn).mark_line().encode(
    x=alt.X('time:Q', title='Time'),
    y=alt.Y('Value:Q', title='Value'),
    color=alt.Color('Type:N', scale=alt.Scale(
        domain=['Raw (Signal + Noise)', 'Pure Signal'],
        range=['#bdc3c7', '#c0392b']
    ), legend=alt.Legend(orient='bottom')),
    strokeWidth=alt.condition(alt.datum.Type == 'Pure Signal', alt.value(3), alt.value(1.5)),
    strokeDash=alt.condition(alt.datum.Type == 'Raw (Signal + Noise)', alt.value([2, 1]), alt.value([0]))
).properties(
    title='Signal + Noise Example (Periodic Signal)',
    width=800,
    height=400
)

# Save Outputs
try:
    # Part 1
    combined_stats.save(os.path.join(output_dir, 'white_noise_stats.png'))
    acf_final.save(os.path.join(output_dir, 'white_noise_acf_stats.png'))
    
    # Part 2
    chart_ma.save(os.path.join(output_dir, 'covid_moving_averages.png'))
    
    # Part 3
    chart_part3_combined.save(os.path.join(output_dir, 'ar_process_analysis.png'))
    
    # Part 4
    chart_sn.save(os.path.join(output_dir, 'signal_plus_noise.png'))
    
    print(f"\nAll Altair plots (4 sections) saved successfully to {output_dir}")
except Exception as e:
    print(f"Error saving Altair plots: {e}")
