# -*- coding: utf-8 -*-
"""
HTGL FEASIBILITY ANALYSIS — HYBRID PUBLICATION VISUALIZATIONS (PLOTLY 3D)

Generates 9 high-fidelity figures. 
- 2D plots (1-5, 7-9) use matplotlib to mimic strict LaTeX pgfplots aesthetic.
- 3D plot (6) uses Plotly for superior surface rendering, exported statically via Kaleido.
Includes high-precision axis formatting to resolve Poisson clustering artifacts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.graph_objects as go
import os

# =============================================================================
# DIRECTORIES & CONSTANTS (DATA PRESERVED EXACTLY)
# =============================================================================
DATA_DIR = "feasibility_data"
PUB_FIG_DIR = "feasibility_figures"
os.makedirs(PUB_FIG_DIR, exist_ok=True)

SECONDS_PER_DAY = 86400
Nc = 50_000_000
alpha_d = 0.1
Nd = int(alpha_d * Nc)
Na = 500

lambda_c_year = 2
lambda_a_day = 0.5
lambda_m_day = 1
lambda_i_day = 24

Rc = (Nc * lambda_c_year) / 365
Ra = Nc * lambda_a_day
Rm = Nd * lambda_m_day
Ri = Na * lambda_i_day
R_gov = Rc + Ra + Rm + Ri
telemetry = Nc * 96

# =============================================================================
# GLOBAL STYLE: MATPLOTLIB PGFPLOTS AESTHETIC
# =============================================================================
SINGLE_COL_WIDTH = 4.0
DOUBLE_COL_WIDTH = 7.16

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Times New Roman"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "-",
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

PGF_FACECOLOR = "#b3c6ff"
PGF_EDGECOLOR = "blue"
PGF_TEXTCOLOR = "blue"

def save_fig(name):
    """Saves matplotlib figures in PDF and PNG."""
    plt.savefig(f"{PUB_FIG_DIR}/{name}.pdf", format='pdf')
    #plt.savefig(f"{PUB_FIG_DIR}/{name}.png", format='png')
    plt.close()

# =============================================================================
# FIGURE GENERATION: 2D PLOTS (MATPLOTLIB)
# =============================================================================

def plot_distributions():
    df_mc = pd.read_csv(f"{DATA_DIR}/monte_carlo.csv")
    
    # 1. Histogram of Daily Events
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))
    ax.hist(df_mc["Daily_Events"], bins=20, color=PGF_FACECOLOR, edgecolor=PGF_EDGECOLOR)
    ax.set_xlabel(r"Daily Governance Events ($\times 10^6$)")
    ax.set_ylabel(r"Frequency")
    # Increased precision to .3f to prevent repeated ticks due to tight Poisson clustering
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*1e-6:.3f}"))
    ax.grid(True)
    save_fig("fig_01_events_distribution")

    # 2. Histogram of TPS Distribution (Restored)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))
    ax.hist(df_mc["TPS"], bins=20, color=PGF_FACECOLOR, edgecolor=PGF_EDGECOLOR)
    ax.set_xlabel(r"Throughput (TPS)")
    ax.set_ylabel(r"Frequency")
    # Precision set to .2f to separate tightly clustered TPS values (~350.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True)
    save_fig("fig_02_tps_distribution")

    # 3. CDF of Daily Events
    sorted_data = np.sort(df_mc["Daily_Events"])
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))
    ax.step(sorted_data, cdf, where='post', color=PGF_EDGECOLOR, linewidth=1.5)
    ax.plot(sorted_data[::5], cdf[::5], marker='o', color=PGF_EDGECOLOR, linestyle='None', markersize=3, markerfacecolor='white')
    ax.set_xlabel(r"Daily Governance Events ($\times 10^6$)")
    ax.set_ylabel(r"Cumulative Probability")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*1e-6:.3f}"))
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    save_fig("fig_03_cdf")

def plot_timeseries():
    df_ts = pd.read_csv(f"{DATA_DIR}/timeseries.csv")
    
    # 4. Steady State
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.0))
    ax.plot(df_ts["Minute"], df_ts["Events"], alpha=0.4, color="gray", label=r"Raw Events $\lambda(t)$")
    ax.plot(df_ts["Minute"], df_ts["Moving_Avg"], color=PGF_EDGECOLOR, linestyle="-", 
            linewidth=1.5, marker='s', markersize=4, markerfacecolor='white', markevery=60, 
            label=r"$30$-min SMA")
    ax.set_xlabel(r"Simulation Time (Minutes)")
    ax.set_ylabel(r"Throughput (Events/Min)")
    ax.margins(x=0)
    ax.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
    ax.grid(True)
    save_fig("fig_04_steady_state")

    # 5. Burst Scenario
    df_burst = pd.read_csv(f"{DATA_DIR}/burst.csv")
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))
    ax.plot(df_burst["Minute"], df_burst["TPS"], color=PGF_EDGECOLOR, marker='o', 
            linestyle="-", markerfacecolor='white', markersize=4, markevery=5)
    ax.set_xlabel(r"Time (Minutes during burst)")
    ax.set_ylabel(r"Throughput (TPS)")
    ax.margins(x=0.02)
    ax.grid(True)
    save_fig("fig_05_burst")

def plot_heatmap():
    df_sens = pd.read_csv(f"{DATA_DIR}/sensitivity.csv")
    
    # 6. Sensitivity Heatmap
    pivot = df_sens.pivot_table(index="lambda_m", columns="lambda_a", values="TPS", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", 
                cbar_kws={'label': r'Throughput (TPS)'}, ax=ax, linecolor='white', linewidths=0.5)
    ax.set_xlabel(r"Access Rate $\lambda_a$ (per Consumer/Day)")
    ax.set_ylabel(r"DER Event Rate $\lambda_m$ (per Asset/Day)")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    save_fig("fig_06_heatmap")

# =============================================================================
# FIGURE GENERATION: 3D PLOT (PLOTLY)
# =============================================================================

def plot_3d_surface_plotly():
    df_sens = pd.read_csv(f"{DATA_DIR}/sensitivity.csv")
    
    # 7. 3D Surface Plot using Plotly
    pivot = df_sens.pivot_table(index="lambda_m", columns="lambda_a", values="TPS", aggfunc="mean")
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    z_vals = pivot.values

    fig = go.Figure(data=[go.Surface(
        z=z_vals, 
        x=x_vals, 
        y=y_vals,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title="Throughput (TPS)", 
            len=0.7,
            thickness=15,
            tickfont=dict(family="Times New Roman", size=12)
        )
    )])

    # Configure the scene to match the target aesthetic
    fig.update_layout(
        scene=dict(
            xaxis_title="λa (Access Rate)",
            yaxis_title="λm (DER Events)",
            zaxis_title="Throughput (TPS)",
            xaxis=dict(gridcolor='white', showbackground=True, backgroundcolor='rgb(235, 239, 245)', gridwidth=2),
            yaxis=dict(gridcolor='white', showbackground=True, backgroundcolor='rgb(235, 239, 245)', gridwidth=2),
            zaxis=dict(gridcolor='white', showbackground=True, backgroundcolor='rgb(235, 239, 245)', gridwidth=2),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.6)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        font=dict(family="Times New Roman", size=14, color="black"),
        width=800,
        height=600
    )
    
    # Export static images via Kaleido (High Res for IEEE)
    print(" Exporting Plotly 3D Surface (this may take a moment due to Kaleido engine)...")
    fig.write_image(f"{PUB_FIG_DIR}/fig_07_3d_surface.pdf", engine="kaleido")
    #fig.write_image(f"{PUB_FIG_DIR}/fig_07_3d_surface.png", engine="kaleido", scale=3)

# =============================================================================
# FIGURE GENERATION: COMPOSITIONAL PLOTS (MATPLOTLIB)
# =============================================================================

def plot_comparisons():
    # 8. Workload Composition
    labels = [r"Consent", r"Auth", r"DER", r"Integrity"]
    values = [Rc, Ra, Rm, Ri]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 3.0))
    bars = ax.bar(labels, values, color=PGF_FACECOLOR, edgecolor=PGF_EDGECOLOR, width=0.5)
    
    ax.set_yscale('log')
    ax.set_ylabel(r"Events per day")
    
    for bar in bars:
        yval = bar.get_height()
        if yval >= 1e6:
            text_val = f"{yval*1e-6:.1f}M"
        else:
            text_val = f"{yval*1e-3:.1f}K"
            
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.3, text_val, 
                ha='center', va='bottom', fontsize=8, color=PGF_TEXTCOLOR)

    ax.set_ylim(bottom=10**3, top=10**8)
    ax.grid(True, axis='y', which='major')
    save_fig("fig_08_composition")

    # 9. Telemetry vs Governance
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    bars = ax.bar(["Telemetry", "Governance"], [telemetry, R_gov], 
                  color=PGF_FACECOLOR, edgecolor=PGF_EDGECOLOR, width=0.25)
    
    ax.set_yscale("log")
    ax.set_ylabel(r"Events per day")
    ax.set_title(r"Telemetry vs. Governance Workload")
    
    telemetry_label = f"{telemetry*1e-9:.1f}B"
    gov_label = f"{R_gov*1e-6:.1f}M"
    
    ax.text(bars[0].get_x() + bars[0].get_width()/2, telemetry * 1.3, telemetry_label, 
            ha='center', va='bottom', fontsize=8, color=PGF_TEXTCOLOR)
    ax.text(bars[1].get_x() + bars[1].get_width()/2, R_gov * 1.3, gov_label, 
            ha='center', va='bottom', fontsize=8, color=PGF_TEXTCOLOR)
            
    ax.text(1, telemetry * 0.1, r"$\approx 158\times$" + "\n" + r"smaller", 
            ha='center', va='top', fontsize=9, color="black")
            
    ax.set_ylim(bottom=10**7, top=10**10 + 5e9)
    ax.grid(True, axis='y', which='both', color="#e0e0e0", linestyle="-")
    save_fig("fig_09_comparison")

if __name__ == "__main__":
   # print("Generating figures (Matplotlib + Plotly)...")
    plot_distributions()
    plot_timeseries()
    plot_heatmap()
    plot_3d_surface_plotly()
    plot_comparisons()
    print(f"Success! All 9 graphics saved to: {PUB_FIG_DIR}/")