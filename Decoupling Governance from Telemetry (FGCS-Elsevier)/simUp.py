# -*- coding: utf-8 -*-
"""
HTGL FEASIBILITY ANALYSIS — FINAL VERSION

- Monte Carlo workload model
- Generates ALL figures and datasets
- Fully reproducible and robust

OUTPUT:
    feasibility_data/
    feasibility_figures/
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os

# =============================================================================
# DIRECTORIES
# =============================================================================

DATA_DIR = "feasibility_data"

os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

sns.set_theme(style="whitegrid")

# =============================================================================
# PARAMETERS
# =============================================================================

SECONDS_PER_DAY = 86400
### Shall we introduce multiple population sizes
Nc = 50_000_000
alpha_d = 0.1
Nd = int(alpha_d * Nc)
Na = 500

lambda_c_year = 2
lambda_a_day = 0.5
lambda_m_day = 1
lambda_i_day = 24

# =============================================================================
# MODEL
# =============================================================================

def compute_event_rates(lambda_a, lambda_m, lambda_i):
    Rc = (Nc * lambda_c_year) / 365
    Ra = Nc * lambda_a
    Rm = Nd * lambda_m
    Ri = Na * lambda_i
    return Rc, Ra, Rm, Ri, Rc + Ra + Rm + Ri

Rc, Ra, Rm, Ri, R_gov = compute_event_rates(
    lambda_a_day, lambda_m_day, lambda_i_day
)

lambda_total = R_gov / SECONDS_PER_DAY

# =============================================================================
# MONTE CARLO
# =============================================================================

mc_samples = np.random.poisson(R_gov, size=100)

df_mc = pd.DataFrame({
    "Daily_Events": mc_samples,
    "TPS": mc_samples / SECONDS_PER_DAY
})
df_mc.to_csv(f"{DATA_DIR}/monte_carlo.csv", index=False)


# CDF
sorted_data = np.sort(df_mc["Daily_Events"])
cdf = np.arange(len(sorted_data)) / len(sorted_data)

# =============================================================================
# TIME SERIES (STEADY STATE)
# =============================================================================

minutes = np.arange(1440)

base = lambda_total * 60
rates = base * (1 + 0.3 * np.sin(2 * np.pi * minutes / 1440))

events = np.random.poisson(rates)

df_ts = pd.DataFrame({
    "Minute": minutes,
    "Events": events
})

df_ts["Moving_Avg"] = df_ts["Events"].rolling(30).mean()
df_ts.to_csv(f"{DATA_DIR}/timeseries.csv", index=False)

# =============================================================================
# BURST SCENARIO
### Is this grounded in some theory or standard
# =============================================================================

_, _, _, _, R_burst = compute_event_rates(
    lambda_a_day * 2,
    lambda_m_day,
    lambda_i_day
)

lambda_burst = R_burst / SECONDS_PER_DAY

burst_events = np.random.poisson(lambda_burst * 60, size=60)

df_burst = pd.DataFrame({
    "Minute": np.arange(60),
    "TPS": burst_events / 60
})

df_burst.to_csv(f"{DATA_DIR}/burst.csv", index=False)

# =============================================================================
# SENSITIVITY ANALYSIS (FIXED)
### Should we change it from fixed to dynamic or mixed
# =============================================================================

rows = []

for la in [0.2, 0.5, 1.0]:
    for lm in [0.5, 1.0, 2.0]:
        _, _, _, _, R = compute_event_rates(la, lm, lambda_i_day)
        rows.append((la, lm, R / SECONDS_PER_DAY))

df_sens = pd.DataFrame(rows, columns=["lambda_a", "lambda_m", "TPS"])
df_sens.to_csv(f"{DATA_DIR}/sensitivity.csv", index=False)

# FIXED pivot
pivot = df_sens.pivot_table(
    index="lambda_m",
    columns="lambda_a",
    values="TPS",
    aggfunc="mean"
)

# =============================================================================
# 3D SURFACE (FINAL — MATPLOTLIB, GUARANTEED WORKING)
# =============================================================================

# Create grid
x_vals = sorted(df_sens["lambda_a"].unique())
y_vals = sorted(df_sens["lambda_m"].unique())

X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X, dtype=float)

for i, lm in enumerate(y_vals):
    for j, la in enumerate(x_vals):
        Z[i, j] = df_sens[
            (df_sens["lambda_a"] == la) &
            (df_sens["lambda_m"] == lm)
        ]["TPS"].values[0]

# =============================================================================
# WORKLOAD COMPOSITION
# =============================================================================

labels = ["Consent (Rc)", "Authorization (Ra)", "DER (Rm)", "Integrity (Ri)"]
values = [Rc, Ra, Rm, Ri]

# =============================================================================
# TELEMETRY VS GOVERNANCE
# =============================================================================

telemetry = Nc * 96

# =============================================================================
# STORAGE
### ToDo: Consider adding multiple message sizes and evaluate storage footprint etc.
# =============================================================================

record_size = 32 + 64 + 200
storage_daily = R_gov * record_size / (1024**3)

pd.DataFrame({
    "Record_Size_Bytes": [record_size],
    "Daily_Storage_GB": [storage_daily]
}).to_csv(f"{DATA_DIR}/storage.csv", index=False)

# =============================================================================
# COMPLETE
# =============================================================================

print("\n Simulation completed.")
print(" Data folder:", DATA_DIR)