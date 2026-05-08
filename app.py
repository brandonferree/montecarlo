"""
Becker Capital — Monte Carlo Portfolio Analysis
Streamlit web app

Run locally:
    pip install streamlit numpy matplotlib reportlab
    streamlit run app.py
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, NextPageTemplate,
)
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY


# =============================================================================
# Brand palette (shared between matplotlib and reportlab)
# =============================================================================
NAVY_HEX        = "#1F3A5F"
NAVY_DARK_HEX   = "#152844"
GOLD_HEX        = "#B8924D"
TEAL_HEX        = "#3A8B8B"
LIGHT_BG_HEX    = "#F2F2F2"
ALT_BG_HEX      = "#FAFAFA"
TEXT_DARK_HEX   = "#222222"
TEXT_MED_HEX    = "#5A5A5A"
RULE_GREY_HEX   = "#CCCCCC"

NAVY        = colors.HexColor(NAVY_HEX)
NAVY_DARK   = colors.HexColor(NAVY_DARK_HEX)
GOLD        = colors.HexColor(GOLD_HEX)
TEAL        = colors.HexColor(TEAL_HEX)
LIGHT_BG    = colors.HexColor(LIGHT_BG_HEX)
ALT_BG      = colors.HexColor(ALT_BG_HEX)
TEXT_DARK   = colors.HexColor(TEXT_DARK_HEX)
TEXT_MED    = colors.HexColor(TEXT_MED_HEX)
RULE_GREY   = colors.HexColor(RULE_GREY_HEX)

SCENARIO_COLOR_HEX = [NAVY_HEX, TEAL_HEX, GOLD_HEX]


# =============================================================================
# Historical return data — Damodaran long-run U.S. asset class series
# Equity: S&P 500 total return | Fixed Income: 10-Yr Treasury total return
# Source-style series (annual, decimal returns), 1928–2024 (97 years).
# These are the de facto reference figures for long-run U.S. asset class returns.
# =============================================================================
HISTORICAL_RETURNS_1928_2024: List[Tuple[int, float, float]] = [
    # (year, S&P 500 total return, 10Y Treasury total return)
    (1928,  0.4388,  0.0084), (1929, -0.0825,  0.0420), (1930, -0.2512,  0.0454),
    (1931, -0.4384, -0.0256), (1932, -0.0864,  0.0879), (1933,  0.4998,  0.0186),
    (1934, -0.0119,  0.0796), (1935,  0.4674,  0.0447), (1936,  0.3194,  0.0502),
    (1937, -0.3534,  0.0138), (1938,  0.2928,  0.0421), (1939, -0.0110,  0.0441),
    (1940, -0.1067,  0.0540), (1941, -0.1277, -0.0202), (1942,  0.1917,  0.0229),
    (1943,  0.2506,  0.0249), (1944,  0.1903,  0.0258), (1945,  0.3582,  0.0380),
    (1946, -0.0843,  0.0313), (1947,  0.0520,  0.0092), (1948,  0.0570,  0.0195),
    (1949,  0.1830,  0.0466), (1950,  0.3081,  0.0043), (1951,  0.2368, -0.0030),
    (1952,  0.1815,  0.0227), (1953, -0.0121,  0.0414), (1954,  0.5256,  0.0329),
    (1955,  0.3260, -0.0134), (1956,  0.0744, -0.0226), (1957, -0.1046,  0.0680),
    (1958,  0.4372, -0.0210), (1959,  0.1206, -0.0265), (1960,  0.0034,  0.1164),
    (1961,  0.2664,  0.0206), (1962, -0.0881,  0.0564), (1963,  0.2261,  0.0182),
    (1964,  0.1642,  0.0399), (1965,  0.1245,  0.0470), (1966, -0.1006,  0.0286),
    (1967,  0.2398,  0.0177), (1968,  0.1106,  0.0357), (1969, -0.0850, -0.0500),
    (1970,  0.0401,  0.1675), (1971,  0.1431,  0.0979), (1972,  0.1898,  0.0282),
    (1973, -0.1466,  0.0366), (1974, -0.2647,  0.0199), (1975,  0.3720,  0.0361),
    (1976,  0.2384,  0.1598), (1977, -0.0718,  0.0129), (1978,  0.0656, -0.0078),
    (1979,  0.1844,  0.0067), (1980,  0.3242, -0.0299), (1981, -0.0491,  0.0820),
    (1982,  0.2155,  0.3281), (1983,  0.2256,  0.0320), (1984,  0.0627,  0.1373),
    (1985,  0.3173,  0.2571), (1986,  0.1867,  0.2428), (1987,  0.0525, -0.0496),
    (1988,  0.1661,  0.0822), (1989,  0.3169,  0.1769), (1990, -0.0310,  0.0624),
    (1991,  0.3047,  0.1500), (1992,  0.0762,  0.0936), (1993,  0.1008,  0.1421),
    (1994,  0.0132, -0.0804), (1995,  0.3758,  0.2348), (1996,  0.2296,  0.0143),
    (1997,  0.3336,  0.0994), (1998,  0.2858,  0.1492), (1999,  0.2104, -0.0825),
    (2000, -0.0910,  0.1666), (2001, -0.1189,  0.0557), (2002, -0.2210,  0.1512),
    (2003,  0.2868,  0.0038), (2004,  0.1088,  0.0449), (2005,  0.0491,  0.0287),
    (2006,  0.1579,  0.0196), (2007,  0.0549,  0.1021), (2008, -0.3700,  0.2003),
    (2009,  0.2646, -0.1112), (2010,  0.1506,  0.0846), (2011,  0.0211,  0.1604),
    (2012,  0.1600,  0.0297), (2013,  0.3239, -0.0910), (2014,  0.1369,  0.1075),
    (2015,  0.0138,  0.0128), (2016,  0.1196,  0.0069), (2017,  0.2183,  0.0280),
    (2018, -0.0438, -0.0002), (2019,  0.3149,  0.0964), (2020,  0.1840,  0.1133),
    (2021,  0.2871, -0.0442), (2022, -0.1811, -0.1777), (2023,  0.2629,  0.0305),
    (2024,  0.2502,  0.0131),
]

# Pre-computed historical statistics for display purposes
def _historical_stats():
    eq = np.array([r[1] for r in HISTORICAL_RETURNS_1928_2024])
    fi = np.array([r[2] for r in HISTORICAL_RETURNS_1928_2024])
    return {
        "eq_mu": float(eq.mean()),
        "eq_sigma": float(eq.std(ddof=1)),
        "fi_mu": float(fi.mean()),
        "fi_sigma": float(fi.std(ddof=1)),
        "worst_eq": float(eq.min()),
        "worst_fi": float(fi.min()),
        "correlation": float(np.corrcoef(eq, fi)[0, 1]),
        "n_years": len(HISTORICAL_RETURNS_1928_2024),
        "first_year": HISTORICAL_RETURNS_1928_2024[0][0],
        "last_year": HISTORICAL_RETURNS_1928_2024[-1][0],
    }

HIST_STATS = _historical_stats()


# =============================================================================
# Data classes for inputs
# =============================================================================
@dataclass
class ReturnAssumptions:
    """
    Return-generation specification.

    Two methods supported:
      - method="parametric": draw N(mu, sigma) each period, asset classes independent.
      - method="bootstrap": resample matched historical (eq, fi) year pairs,
                            re-centered so the sample mean equals (eq_mu, fi_mu).

    eq_mu / fi_mu always represent the *forward-looking expected annual return*.
    eq_sigma / fi_sigma are only used when method="parametric"; in bootstrap mode
    the historical volatility, skew, and equity/FI correlation are preserved
    automatically through resampling.
    """
    eq_mu: float
    eq_sigma: float
    fi_mu: float
    fi_sigma: float
    label: str
    method: str = "bootstrap"            # "bootstrap" | "parametric"
    historical_period: str = "1928–2024" # used only for bootstrap labelling
    worst_eq: float = -0.4384
    worst_fi: float = -0.1777


@dataclass
class Scenario:
    name: str
    eq_weight: float                      # 0–1
    fi_weight: float                      # 0–1
    annual_distribution: float            # in today's (Year-1) dollars
    contribution_years: int = 0           # number of years to contribute (0 = none)
    annual_contribution: float = 0.0      # in today's (Year-1) dollars

    @property
    def distribution_start_year(self) -> int:
        """The first year in which a distribution is paid (1-indexed)."""
        return self.contribution_years + 1


@dataclass
class SimInputs:
    initial: float
    horizon_years: int
    inflation: float                  # decimal
    distribution_frequency: str       # "Annual" | "Quarterly" | "Monthly"
    return_assumptions: ReturnAssumptions
    scenarios: List[Scenario]
    n_paths: int = 10_000
    seed: int = 20260501


# Preset packages — bootstrap mode uses forward-looking expected returns
# while preserving the historical volatility, skew, and correlation structure
# from the 1928–2024 series.
PRESETS = {
    "Bootstrap — Forward-looking (Aggressive)": ReturnAssumptions(
        eq_mu=0.0950, eq_sigma=HIST_STATS["eq_sigma"],
        fi_mu=0.0450, fi_sigma=HIST_STATS["fi_sigma"],
        label="Bootstrap • Forward-Looking Aggressive",
        method="bootstrap",
        historical_period="1928–2024",
        worst_eq=HIST_STATS["worst_eq"], worst_fi=HIST_STATS["worst_fi"],
    ),
    "Bootstrap — Forward-looking (Moderate)": ReturnAssumptions(
        eq_mu=0.0800, eq_sigma=HIST_STATS["eq_sigma"],
        fi_mu=0.0450, fi_sigma=HIST_STATS["fi_sigma"],
        label="Bootstrap • Forward-Looking Moderate",
        method="bootstrap",
        historical_period="1928–2024",
        worst_eq=HIST_STATS["worst_eq"], worst_fi=HIST_STATS["worst_fi"],
    ),
    "Bootstrap — Forward-looking (Conservative)": ReturnAssumptions(
        eq_mu=0.0700, eq_sigma=HIST_STATS["eq_sigma"],
        fi_mu=0.0400, fi_sigma=HIST_STATS["fi_sigma"],
        label="Bootstrap • Forward-Looking Conservative",
        method="bootstrap",
        historical_period="1928–2024",
        worst_eq=HIST_STATS["worst_eq"], worst_fi=HIST_STATS["worst_fi"],
    ),
    "Bootstrap — Historical means (1928–2024)": ReturnAssumptions(
        eq_mu=HIST_STATS["eq_mu"], eq_sigma=HIST_STATS["eq_sigma"],
        fi_mu=HIST_STATS["fi_mu"], fi_sigma=HIST_STATS["fi_sigma"],
        label="Bootstrap • Historical 1928–2024",
        method="bootstrap",
        historical_period="1928–2024",
        worst_eq=HIST_STATS["worst_eq"], worst_fi=HIST_STATS["worst_fi"],
    ),
    "Parametric — 1960–2024 (Legacy Becker)": ReturnAssumptions(
        eq_mu=0.1179, eq_sigma=0.1667,
        fi_mu=0.0615, fi_sigma=0.0879,
        label="Parametric • 1960–2024",
        method="parametric",
        worst_eq=-0.370, worst_fi=-0.131,
    ),
    "Parametric — Forward-looking (Conservative)": ReturnAssumptions(
        eq_mu=0.0800, eq_sigma=0.1500,
        fi_mu=0.0450, fi_sigma=0.0700,
        label="Parametric • Forward-Looking",
        method="parametric",
        worst_eq=-0.370, worst_fi=-0.131,
    ),
}


FREQ_TO_PER_YEAR = {"Annual": 1, "Quarterly": 4, "Monthly": 12}


# =============================================================================
# Simulation
# =============================================================================
def blended_params(eq_w: float, fi_w: float, ra: ReturnAssumptions) -> Tuple[float, float]:
    """
    Blended annual mean and std for display purposes only.
    For parametric mode: classical formula (asset classes assumed uncorrelated).
    For bootstrap mode: computed from the historical series, preserving the
        natural eq/fi correlation captured in matched-pair sampling.
    """
    if ra.method == "parametric":
        mu = eq_w * ra.eq_mu + fi_w * ra.fi_mu
        sig = np.sqrt((eq_w * ra.eq_sigma) ** 2 + (fi_w * ra.fi_sigma) ** 2)
        return mu, sig
    # Bootstrap: forward-looking mean is the weighted forward μ.
    # Volatility uses the actual historical eq/fi series (with their correlation)
    # since re-centering only shifts the mean, not the dispersion.
    mu = eq_w * ra.eq_mu + fi_w * ra.fi_mu
    eq_hist = np.array([r[1] for r in HISTORICAL_RETURNS_1928_2024])
    fi_hist = np.array([r[2] for r in HISTORICAL_RETURNS_1928_2024])
    blended_series = eq_w * eq_hist + fi_w * fi_hist
    sig = float(blended_series.std(ddof=1))
    return mu, sig


def _build_recentered_pairs(ra: ReturnAssumptions) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the re-centered historical (eq, fi) return series.
    For each year: r_recentered = r_historical - mean_historical + mean_forward.
    Volatility, skew, and the eq/fi correlation are preserved exactly.
    """
    eq_hist = np.array([r[1] for r in HISTORICAL_RETURNS_1928_2024])
    fi_hist = np.array([r[2] for r in HISTORICAL_RETURNS_1928_2024])
    eq_recentered = eq_hist - eq_hist.mean() + ra.eq_mu
    fi_recentered = fi_hist - fi_hist.mean() + ra.fi_mu
    return eq_recentered, fi_recentered


def simulate_scenario(scen: Scenario, inputs: SimInputs, seed_offset: int) -> dict:
    """
    Run Monte Carlo for a single scenario.

    Cash-flow model (all amounts are entered in Year-1 'today's' dollars):
      - Years 1 .. contribution_years:
            CONTRIBUTION of `annual_contribution * (1+infl)^(y-1)` is ADDED at start
            of each period (split evenly across k periods).
      - Years contribution_years+1 .. horizon:
            DISTRIBUTION of `annual_distribution * (1+infl)^(y-1)` is REMOVED at
            start of each period.
      - Cash flows applied AT THE START of each period; returns applied to
        post-cashflow balance.

    Return generation (one of):
      - method="parametric": each period's return drawn from N(μ/k, σ/√k)
        with asset classes treated as independent.
      - method="bootstrap": each YEAR, sample one historical (eq, fi) pair from
        the re-centered 1928–2024 series. Combine by portfolio weights to get
        an annual portfolio return. Decompose to k periods using
        per_period = (1 + annual)^(1/k) - 1 (same return each quarter within
        a year — preserves the historical-year semantics).
    """
    rng = np.random.default_rng(inputs.seed + seed_offset)
    ra = inputs.return_assumptions
    k = FREQ_TO_PER_YEAR[inputs.distribution_frequency]

    n = inputs.n_paths
    yrs = inputs.horizon_years
    yearly = np.zeros((n, yrs + 1))
    yearly[:, 0] = inputs.initial
    bal = np.full(n, float(inputs.initial))

    contrib_years = max(0, int(scen.contribution_years))

    # Pre-compute return generators based on method
    if ra.method == "bootstrap":
        eq_recentered, fi_recentered = _build_recentered_pairs(ra)
        n_hist = len(eq_recentered)
        # Annual portfolio return for any sampled year:
        portfolio_returns = scen.eq_weight * eq_recentered + scen.fi_weight * fi_recentered
    else:
        mu_a, sig_a = blended_params(scen.eq_weight, scen.fi_weight, ra)
        mu_p = mu_a / k
        sig_p = sig_a / np.sqrt(k)

    for y in range(1, yrs + 1):
        infl_factor = (1 + inputs.inflation) ** (y - 1)

        if y <= contrib_years:
            annual_contrib = scen.annual_contribution * infl_factor
            per_period_cf = -annual_contrib / k
        else:
            annual_dist = scen.annual_distribution * infl_factor
            per_period_cf = annual_dist / k

        if ra.method == "bootstrap":
            # Sample one historical year per path (matched eq/fi correlation
            # is preserved automatically because we sample year indices, not
            # eq and fi independently).
            year_idx = rng.integers(0, n_hist, size=n)
            annual_r = portfolio_returns[year_idx]
            # Decompose into k equal per-period returns. Using geometric
            # decomposition: (1+r_period)^k = (1+r_annual)
            with np.errstate(invalid="ignore"):
                per_period_r = np.sign(1 + annual_r) * np.power(
                    np.abs(1 + annual_r), 1.0 / k
                ) - 1.0
            for _ in range(k):
                bal = bal - per_period_cf
                bal = np.maximum(bal, 0.0)
                bal = np.maximum(bal * (1 + per_period_r), 0.0)
        else:
            for _ in range(k):
                bal = bal - per_period_cf
                bal = np.maximum(bal, 0.0)
                r = rng.normal(mu_p, sig_p, size=n)
                bal = np.maximum(bal * (1 + r), 0.0)

        yearly[:, y] = bal

    yr_final = yearly[:, -1]
    yr10 = yearly[:, min(10, yrs)]
    yr20 = yearly[:, min(20, yrs)]
    mu_a, sig_a = blended_params(scen.eq_weight, scen.fi_weight, ra)

    return {
        "scenario": scen,
        "mu_a": mu_a,
        "sig_a": sig_a,
        "balances": yearly,
        "median_path": np.median(yearly, axis=0),
        "p20_path": np.percentile(yearly, 20, axis=0),
        "p80_path": np.percentile(yearly, 80, axis=0),
        "median_y10": float(np.median(yr10)),
        "median_y20": float(np.median(yr20)),
        "median_yfinal": float(np.median(yr_final)),
        "mean_yfinal": float(np.mean(yr_final)),
        "p20_yfinal": float(np.percentile(yr_final, 20)),
        "p25_yfinal": float(np.percentile(yr_final, 25)),
        "p75_yfinal": float(np.percentile(yr_final, 75)),
        "p80_yfinal": float(np.percentile(yr_final, 80)),
        "iqr_yfinal": float(np.percentile(yr_final, 75) - np.percentile(yr_final, 25)),
        "p_ruin": float(np.mean(yr_final <= 0.0)),
        "p_above_init": float(np.mean(yr_final > inputs.initial)),
        "total_contributed": total_contributed(scen, inputs.inflation),
        "total_distributed": total_distributed(scen, inputs.inflation, yrs),
    }


def run_all_simulations(inputs: SimInputs) -> List[dict]:
    return [simulate_scenario(s, inputs, i * 1000) for i, s in enumerate(inputs.scenarios)]


def total_contributed(scen: Scenario, inflation: float) -> float:
    """Sum of contributions in nominal dollars across the contribution phase."""
    if scen.contribution_years <= 0 or scen.annual_contribution <= 0:
        return 0.0
    return sum(
        scen.annual_contribution * (1 + inflation) ** y
        for y in range(int(scen.contribution_years))
    )


def total_distributed(scen: Scenario, inflation: float, total_years: int) -> float:
    """
    Sum of distributions in nominal dollars across the distribution phase only.
    Distributions begin at year `contribution_years + 1` and run through year `total_years`.
    """
    start = int(scen.contribution_years) + 1
    if start > total_years or scen.annual_distribution <= 0:
        return 0.0
    return sum(
        scen.annual_distribution * (1 + inflation) ** (y - 1)
        for y in range(start, total_years + 1)
    )


# =============================================================================
# Charts (return BytesIO PNGs, no file system)
# =============================================================================
def _setup_mpl():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": NAVY_HEX,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def chart_paths_with_bands(results: List[dict], inputs: SimInputs) -> io.BytesIO:
    _setup_mpl()
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    years_axis = np.arange(0, inputs.horizon_years + 1)

    for r, color in zip(results, SCENARIO_COLOR_HEX):
        ax.fill_between(years_axis, r["p20_path"] / 1e6, r["p80_path"] / 1e6,
                        color=color, alpha=0.15, linewidth=0)
        scen = r["scenario"]
        label = (f"{scen.name} — {int(scen.eq_weight*100)}/{int(scen.fi_weight*100)} "
                 f"(Median)")
        ax.plot(years_axis, r["median_path"] / 1e6, color=color, linewidth=2.4, label=label)

    ax.axhline(inputs.initial / 1e6, color="#888888", linestyle="--", linewidth=1,
               alpha=0.7, label="Initial Investment")
    ax.set_title(
        f"{inputs.horizon_years}-Year Portfolio Value — Monte Carlo Median & Percentile Bands\n"
        "(Shaded: 20th–80th percentiles)",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value ($M)")
    ax.set_xlim(0, inputs.horizon_years)

    # Pick reasonable y-axis ceiling: ~max p80 across scenarios, rounded up
    ceiling = max(np.max(r["p80_path"]) for r in results) / 1e6
    step = 20 if ceiling > 60 else 10 if ceiling > 30 else 5
    ymax = int(np.ceil(ceiling / step) * step)
    yticks = list(range(0, ymax + 1, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y}M" for y in yticks])
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_yfinal_distributions(results: List[dict], inputs: SimInputs) -> io.BytesIO:
    _setup_mpl()
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(11, 3.6), dpi=180)
    if n == 1:
        axes = [axes]

    xmax_global = max(np.percentile(r["balances"][:, -1], 99) for r in results) / 1e6
    xmax_global = min(xmax_global, 250)

    for ax, r, color in zip(axes, results, SCENARIO_COLOR_HEX):
        data_m = r["balances"][:, -1] / 1e6
        median_m = r["median_yfinal"] / 1e6
        ax.hist(data_m, bins=60, range=(0, xmax_global), color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(median_m, color="#C0392B", linestyle="--", linewidth=1.6,
                   label=f"Median: ${median_m:.1f}M")
        scen = r["scenario"]
        ax.set_title(f"{scen.name}\n{int(scen.eq_weight*100)}/{int(scen.fi_weight*100)}",
                     fontsize=11)
        ax.set_xlabel(f"Year-{inputs.horizon_years} Value ($M)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25, linestyle=":", axis="y")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"Year-{inputs.horizon_years} Portfolio Value Distribution — Monte Carlo",
                 fontsize=12, color=NAVY_HEX, fontweight="bold", y=1.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_allocations(results: List[dict]) -> io.BytesIO:
    _setup_mpl()
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(9, 3.0), dpi=180)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        scen = r["scenario"]
        sizes = [scen.eq_weight * 100, scen.fi_weight * 100]
        ax.pie(sizes, colors=[NAVY_HEX, GOLD_HEX], startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title(
            f"{scen.name} — {int(scen.eq_weight*100)}% / {int(scen.fi_weight*100)}%",
            fontsize=10, color=NAVY_HEX, fontweight="bold", pad=8,
        )
        ax.text(0, 0.15, f"{int(scen.eq_weight*100)}%", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
        ax.text(0, -0.25, "Equity", ha="center", va="center",
                fontsize=8, color="white")

    legend_elems = [Patch(color=NAVY_HEX, label="Equity"),
                    Patch(color=GOLD_HEX, label="Fixed Income")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
# PDF generation
# =============================================================================
def fmt_m(x: float) -> str:
    return f"${x/1e6:.2f}M"


def fmt_pct(x: float, d: int = 2) -> str:
    return f"{x*100:.{d}f}%"


def build_pdf(results: List[dict], inputs: SimInputs,
              prep_date: str | None = None) -> bytes:
    """Build the full Becker-styled PDF in memory and return bytes."""
    if prep_date is None:
        prep_date = datetime.now().strftime("%B %d, %Y")
    footer_date = "December 31, 2025"

    PAGE_W, PAGE_H = LETTER
    buf = io.BytesIO()

    # Render charts to BytesIO
    img_paths_buf = chart_paths_with_bands(results, inputs)
    img_dist_buf = chart_yfinal_distributions(results, inputs)
    img_alloc_buf = chart_allocations(results)

    # ----- Styles -----
    H_TITLE = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=26,
                             leading=30, textColor=NAVY, alignment=TA_LEFT,
                             spaceAfter=6)
    H_TAGLINE = ParagraphStyle("Tagline", fontName="Helvetica-Bold", fontSize=10,
                               leading=14, textColor=GOLD, alignment=TA_LEFT,
                               spaceAfter=10)
    H_SECTION = ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=14,
                               leading=18, textColor=NAVY, alignment=TA_LEFT,
                               spaceBefore=10, spaceAfter=6)
    P_KICKER = ParagraphStyle("Kicker", fontName="Helvetica-Bold", fontSize=8.5,
                              leading=11, textColor=GOLD, alignment=TA_LEFT,
                              spaceAfter=4)
    P_BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, leading=13.5,
                            textColor=TEXT_DARK, alignment=TA_JUSTIFY, spaceAfter=6)
    P_FIGCAP = ParagraphStyle("FigCap", fontName="Helvetica-Oblique", fontSize=8.5,
                              leading=11, textColor=TEXT_MED, alignment=TA_LEFT,
                              spaceAfter=8)
    P_DISCLAIM = ParagraphStyle("Disclaim", fontName="Helvetica", fontSize=8.5,
                                leading=12, textColor=TEXT_DARK, alignment=TA_JUSTIFY,
                                spaceAfter=6)
    P_KEY_LABEL = ParagraphStyle("KeyLabel", fontName="Helvetica", fontSize=8,
                                 leading=10, textColor=TEXT_MED, alignment=TA_LEFT)
    P_COVER_FIELD_LABEL = ParagraphStyle("CovLabel", fontName="Helvetica", fontSize=8.5,
                                         leading=10, textColor=TEXT_MED, alignment=TA_LEFT)
    P_COVER_FIELD_VAL = ParagraphStyle("CovVal", fontName="Helvetica-Bold", fontSize=15,
                                       leading=17, textColor=NAVY, alignment=TA_LEFT)

    # ----- Page decorations -----
    def cover_decoration(canv, doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, 0, 2.5 * inch, PAGE_H, stroke=0, fill=1)
        canv.setFillColor(GOLD)
        canv.rect(0, 0, PAGE_W, 0.35 * inch, stroke=0, fill=1)
        canv.setFillColor(NAVY_DARK)
        canv.setFont("Helvetica-Bold", 9)
        canv.drawString(0.4 * inch, 0.13 * inch, "BECKER CAPITAL MANAGEMENT")
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Oblique", 7.5)
        canv.drawRightString(
            PAGE_W - 0.4 * inch, 0.13 * inch,
            "This report is hypothetical and for illustrative purposes only. Not investment advice.",
        )
        # 50-B monogram: a "5" then a gold-ringed circle (the "0") with "B" inside.
        # The circle IS the zero, not a separate element overlapping a printed 0.
        cx_5 = 0.85 * inch    # center of the "5"
        cx_0 = 1.75 * inch    # center of the "0" / B-circle
        cy = 4.7 * inch       # vertical baseline-ish
        # The "5" — large, flat dark navy on the panel
        canv.setFillColor(NAVY_DARK)
        canv.setFont("Helvetica-Bold", 110)
        canv.drawCentredString(cx_5, cy - 0.15 * inch, "5")
        # The "0" — drawn as a thick gold ring (no number 0 character)
        canv.setStrokeColor(GOLD)
        canv.setLineWidth(8)
        canv.setFillColor(NAVY)         # interior matches panel
        canv.circle(cx_0, cy + 0.20 * inch, 0.62 * inch, stroke=1, fill=1)
        # The "B" — gold, centered inside the circle
        canv.setFillColor(GOLD)
        canv.setFont("Helvetica-Bold", 56)
        canv.drawCentredString(cx_0, cy - 0.02 * inch, "B")
        # Established + URL — positioned below the monogram with space
        canv.setFillColor(GOLD)
        canv.setFont("Helvetica-Bold", 10)
        canv.drawCentredString(1.25 * inch, 3.30 * inch, "Established in 1976")
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica", 9)
        canv.drawCentredString(1.25 * inch, 3.05 * inch, "BECKERCAP.COM")
        canv.drawCentredString(1.25 * inch, 2.87 * inch, "503.223.1720")
        canv.restoreState()

    def standard_decoration(canv, doc):
        canv.saveState()
        canv.setStrokeColor(RULE_GREY)
        canv.setLineWidth(0.5)
        canv.line(0.6 * inch, 0.65 * inch, PAGE_W - 0.6 * inch, 0.65 * inch)
        canv.setFillColor(TEXT_MED)
        canv.setFont("Helvetica", 8)
        canv.drawString(0.6 * inch, 0.45 * inch,
                        "Becker Capital Management | BECKERCAP.COM | 503.223.1720")
        canv.drawRightString(PAGE_W - 0.6 * inch, 0.45 * inch,
                             f"{footer_date}  |  Pg. {doc.page - 1}")
        canv.restoreState()

    doc = BaseDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.55 * inch, bottomMargin=0.85 * inch,
        title="Monte Carlo Portfolio Analysis — Becker Capital Management",
        author="Becker Capital Management",
    )
    cover_frame = Frame(2.9 * inch, 0.85 * inch,
                        PAGE_W - 2.9 * inch - 0.6 * inch,
                        PAGE_H - 0.85 * inch - 0.6 * inch,
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0, id="cover")
    content_frame = Frame(0.6 * inch, 0.85 * inch,
                          PAGE_W - 1.2 * inch, PAGE_H - 0.85 * inch - 0.55 * inch,
                          leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0, id="content")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=cover_decoration),
        PageTemplate(id="content", frames=[content_frame], onPage=standard_decoration),
    ])

    def section_header(text):
        rule = Table([[""]], colWidths=[PAGE_W - 1.2 * inch], rowHeights=[2])
        rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.5, GOLD),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return [Paragraph(text, H_SECTION), rule, Spacer(1, 6)]

    story = []

    # ==================== COVER ====================
    n_scen = len(inputs.scenarios)
    eq_strs = " • ".join([f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)}"
                          for s in inputs.scenarios])
    dist_strs = ", ".join([f"${int(s.annual_distribution/1000)}K"
                           for s in inputs.scenarios])
    mu_strs = [blended_params(s.eq_weight, s.fi_weight, inputs.return_assumptions)[0]
               for s in inputs.scenarios]

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"{inputs.horizon_years}-YEAR PROJECTION  •  "
        f"{n_scen} ALLOCATION{'S' if n_scen > 1 else ''}  •  "
        f"${inputs.initial/1e6:.1f}M",
        H_TAGLINE,
    ))
    story.append(Paragraph("Monte Carlo Portfolio<br/>Analysis Report", H_TITLE))
    underline = Table([[""]], colWidths=[3.5 * inch], rowHeights=[3])
    underline.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 2, GOLD)]))
    story.append(underline)
    story.append(Spacer(1, 0.25 * inch))

    def cover_field(value, label):
        t = Table([[Paragraph(value, P_COVER_FIELD_VAL)],
                   [Paragraph(label, P_COVER_FIELD_LABEL)]],
                  colWidths=[4.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("LINEBEFORE", (0, 0), (0, -1), 3, GOLD),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (0, 0), 8),
            ("BOTTOMPADDING", (0, 0), (0, 0), 0),
            ("TOPPADDING", (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (0, 1), 8),
        ]))
        return t

    if len(set([s.annual_distribution for s in inputs.scenarios])) == 1:
        dist_label = f"${inputs.scenarios[0].annual_distribution:,.0f} / yr"
    else:
        dist_label = dist_strs + " / yr"

    # Build a contribution-phase summary if any scenario has contributions
    has_contrib = any(s.contribution_years > 0 and s.annual_contribution > 0
                      for s in inputs.scenarios)
    if has_contrib:
        contrib_amounts = sorted(set(s.annual_contribution for s in inputs.scenarios
                                     if s.contribution_years > 0))
        contrib_years_set = sorted(set(s.contribution_years for s in inputs.scenarios
                                       if s.contribution_years > 0))
        if len(contrib_amounts) == 1:
            contrib_amt_str = f"${contrib_amounts[0]:,.0f} / yr"
        else:
            contrib_amt_str = " – ".join(f"${a:,.0f}" for a in contrib_amounts)
        if len(contrib_years_set) == 1:
            contrib_yrs = contrib_years_set[0]
            phase_str = f"Yrs 1–{contrib_yrs} contribute, Yrs {contrib_yrs+1}–{inputs.horizon_years} distribute"
        else:
            phase_str = "Per-scenario contribution period"

    cover_fields = [
        (f"${inputs.initial:,.0f}", "Initial Investment"),
        (eq_strs, f"Equity / Fixed Income ({n_scen} Scenario{'s' if n_scen > 1 else ''})"),
    ]
    if has_contrib:
        cover_fields.append(
            (contrib_amt_str,
             f"Annual Contribution (today's $, {phase_str})")
        )
    cover_fields += [
        (dist_label, f"Annual Distribution (today's $, paid {inputs.distribution_frequency.lower()}, "
                     f"+{inputs.inflation*100:.1f}% / yr)"),
        (f"{min(mu_strs)*100:.2f}% – {max(mu_strs)*100:.2f}%"
         if n_scen > 1 else f"{mu_strs[0]*100:.2f}%",
         f"Blended Expected Return ({inputs.return_assumptions.label})"),
        (f"{inputs.horizon_years} Years", "Time Horizon"),
        (f"{inputs.inflation*100:.2f}%", "Inflation Rate"),
    ]
    for v, l in cover_fields:
        story.append(cover_field(v, l))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 0.3 * inch))
    prep_table = Table(
        [[Paragraph("<b>Prepared by:</b> Becker Capital Management", P_KEY_LABEL)],
         [Paragraph(f"<b>Date:</b> {prep_date}", P_KEY_LABEL)]],
        colWidths=[4.5 * inch],
    )
    prep_table.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(prep_table)
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())

    # ==================== DISCLAIMER ====================
    story.extend(section_header("Disclaimer"))
    disclaim = [
        "The following report is a diagnostic tool intended to review the inputs provided "
        "and illustrate potential planning concepts that may be of benefit. The purpose of "
        "the report is to illustrate how accepted financial and investment planning principles "
        "may apply to the assumptions provided.",
        "This report is based upon assumptions provided for illustrative purposes only. It "
        "provides broad and general guidelines on the advantages of certain financial planning "
        "concepts and does not constitute a recommendation of any particular technique or "
        "investment strategy. We recommend that you review your plan annually, or when "
        "circumstances change.",
        "The term 'plan' or 'planning,' when used within this report, does not imply that a "
        "recommendation has been made to implement any financial plan or make a particular "
        "investment. The reports provide projections based on various assumptions and are "
        "hypothetical in nature and not guarantees of investment returns. Consult your tax "
        "and/or legal advisors before implementing any transactions.",
        "Past performance is no guarantee of future performance. Actual results may differ "
        "from the projections contained in this report. The presentation of investment returns "
        "does not reflect the deduction of any commissions or advisory fees. Deduction of such "
        "charges will result in a lower rate of return.",
        f"Monte Carlo Analysis is a mathematical process used to implement complex statistical "
        f"methods that chart the probability of certain financial outcomes at certain times in "
        f"the future. This charting is accomplished by generating {inputs.n_paths:,} possible "
        f"economic scenarios. " + (
            f"Each scenario draws annual return data via <b>matched-pair bootstrap "
            f"resampling</b> from {inputs.return_assumptions.historical_period} historical "
            f"S&amp;P 500 and 10-Year Treasury total returns, re-centered so the long-run mean "
            f"equals the forward-looking expected return."
            if inputs.return_assumptions.method == "bootstrap"
            else f"Each scenario randomly draws return data from a normal distribution based "
                 f"on the means and standard deviations specified in the assumption set "
                 f"({inputs.return_assumptions.label})."
        ),
        f"The Monte Carlo simulation uses {inputs.n_paths:,} scenarios to determine the "
        f"probability of outcomes resulting from the asset allocation choices and underlying "
        "return and volatility assumptions. Some scenarios will assume very favorable financial "
        "market returns; some will conform to the worst periods in investing history; most will "
        "fall somewhere in between.",
        "<b>IMPORTANT:</b> The projections generated by this Monte Carlo simulation are "
        "hypothetical in nature, do not reflect actual investment results, and are not "
        "guarantees of future results. Results may vary with each use and over time. This "
        "report is prepared by Becker Capital Management for informational purposes only.",
    ]
    for p in disclaim:
        story.append(Paragraph(p, P_DISCLAIM))
    story.append(PageBreak())

    # ==================== EXEC SUMMARY + RETURN ASSUMPTIONS ====================
    story.append(Paragraph("Portfolio Monte Carlo Analysis", H_SECTION))
    story.append(Paragraph(
        f"{inputs.horizon_years}-Year Scenario Analysis — "
        f"{n_scen} Allocation Strateg{'ies' if n_scen > 1 else 'y'}",
        P_KICKER,
    ))

    # Top fact strip
    fact_data = [
        [f"${inputs.initial/1e6:.1f}M",
         eq_strs,
         dist_label.replace(" / yr", " / yr"),
         inputs.distribution_frequency,
         f"{inputs.horizon_years} Years",
         f"{inputs.inflation*100:.2f}%"],
        ["Initial Investment", "Equity / Fixed Income", "Annual Distribution",
         "Distribution Frequency", "Time Horizon", "Inflation"],
    ]
    fact_tbl = Table(fact_data, colWidths=[1.15 * inch] * 6)
    fact_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("FONT", (0, 1), (-1, 1), "Helvetica", 7.5),
        ("TEXTCOLOR", (0, 1), (-1, 1), TEXT_MED),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, RULE_GREY),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, RULE_GREY),
    ]))
    story.append(fact_tbl)
    story.append(Spacer(1, 12))

    story.extend(section_header("Executive Summary"))
    scen_descs = ", ".join(
        f"<b>{int(s.eq_weight*100)}/{int(s.fi_weight*100)} ({s.name})</b>"
        for s in inputs.scenarios
    )

    # Optional contribution-phase paragraph
    contrib_phase_text = ""
    if has_contrib:
        if len(contrib_years_set) == 1 and len(contrib_amounts) == 1:
            cy = contrib_years_set[0]
            ca = contrib_amounts[0]
            contrib_phase_text = (
                f"<br/><br/>"
                f"<b>Contribution phase (Years 1–{cy}):</b> "
                f"${ca:,.0f}/yr in today's dollars is added to the portfolio, "
                f"escalating {inputs.inflation*100:.1f}% annually so that real purchasing "
                f"power is preserved. Distributions begin in Year {cy+1}. "
                f"All amounts entered are stated in today's (Year-1) dollars; the "
                f"simulation translates them into nominal cash flows by compounding "
                f"inflation forward. For example, a ${ca:,.0f} contribution in Year 1 "
                f"becomes ${ca * (1 + inputs.inflation) ** (cy-1):,.0f} in Year {cy} "
                f"to maintain the same real value."
            )
        else:
            contrib_phase_text = (
                "<br/><br/>"
                "<b>Contribution phase:</b> Each scenario adds capital to the portfolio "
                "for an initial period before distributions begin. Both contributions and "
                f"distributions are entered in today's dollars and inflated forward at "
                f"{inputs.inflation*100:.1f}% annually."
            )

    ra = inputs.return_assumptions
    if ra.method == "bootstrap":
        method_text = (
            f"Expected returns are <b>forward-looking</b> "
            f"(equity {ra.eq_mu*100:.2f}%, fixed income {ra.fi_mu*100:.2f}%), but the "
            f"<b>volatility, fat tails, skew, and equity/fixed-income correlation are "
            f"derived from actual historical experience</b> over {ra.historical_period} "
            f"({HIST_STATS['n_years']} years of S&amp;P 500 and 10-Year Treasury total returns). "
            f"The simulation uses <b>matched-pair bootstrap resampling</b>: each year of "
            f"each path randomly draws a real historical year, then re-centers it so the "
            f"long-run mean equals the forward-looking assumption. This preserves real-"
            f"world properties — including 2008-style left-tail events and the way bonds "
            f"and stocks tend to move together — that a normal distribution would erase."
        )
    else:
        method_text = (
            f"Expected returns and volatility are derived from the "
            f"<b>{ra.label}</b> assumption set: equity mean {ra.eq_mu*100:.2f}% "
            f"(σ = {ra.eq_sigma*100:.2f}%), fixed income mean {ra.fi_mu*100:.2f}% "
            f"(σ = {ra.fi_sigma*100:.2f}%). Per-period returns are drawn from a normal "
            f"distribution parameterized to those annual figures, with asset classes "
            f"treated as independent."
        )

    exec_text = (
        f"This report presents a {inputs.horizon_years}-year Monte Carlo simulation for a "
        f"${inputs.initial:,.0f} portfolio, evaluating {n_scen} Equity / Fixed Income "
        f"allocation strateg{'ies' if n_scen > 1 else 'y'} — {scen_descs}. "
        f"Distributions are paid {inputs.distribution_frequency.lower()}, escalating "
        f"{inputs.inflation*100:.1f}% annually to maintain real purchasing power. "
        f"{contrib_phase_text}"
        f"<br/><br/>"
        f"{method_text} "
        f"The simulation runs {inputs.n_paths:,} independent paths per scenario."
    )
    story.append(Paragraph(exec_text, P_BODY))

    # Return assumptions table
    story.extend(section_header(f"Return Assumptions — {ra.label}"))
    blended_data = [(blended_params(s.eq_weight, s.fi_weight, ra)) for s in inputs.scenarios]
    header = ["Parameter", "Equity", "Fixed Income"] + [
        f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)} Blend" for s in inputs.scenarios
    ]

    if ra.method == "bootstrap":
        # Show both forward-looking μ AND historical σ separately
        ra_data = [
            header,
            ["Forward-looking Mean (μ)", f"{ra.eq_mu*100:.2f}%", f"{ra.fi_mu*100:.2f}%"]
            + [f"{m*100:.2f}%" for m, _ in blended_data],
            [f"Historical Std. Dev (σ)",
             f"{HIST_STATS['eq_sigma']*100:.2f}%", f"{HIST_STATS['fi_sigma']*100:.2f}%"]
            + [f"{s*100:.2f}%" for _, s in blended_data],
            ["Eq/FI Correlation (preserved)",
             f"ρ = {HIST_STATS['correlation']:.3f}",
             f"ρ = {HIST_STATS['correlation']:.3f}"] + ["—"] * n_scen,
            ["Worst Historical Year",
             f"{HIST_STATS['worst_eq']*100:.1f}%", f"{HIST_STATS['worst_fi']*100:.1f}%"]
            + ["—"] * n_scen,
            ["Inflation on Cash Flows", "—", "—"]
            + [f"{inputs.inflation*100:.2f}%"] * n_scen,
            ["Method", "Bootstrap", "Bootstrap"]
            + ["Bootstrap"] * n_scen,
        ]
    else:
        ra_data = [
            header,
            ["Mean Return (μ)", f"{ra.eq_mu*100:.2f}%", f"{ra.fi_mu*100:.2f}%"]
            + [f"{m*100:.2f}%" for m, _ in blended_data],
            ["Annual Std. Deviation (σ)", f"{ra.eq_sigma*100:.2f}%", f"{ra.fi_sigma*100:.2f}%"]
            + [f"{s*100:.2f}%" for _, s in blended_data],
            ["Worst Calendar Year", f"{ra.worst_eq*100:.1f}%", f"{ra.worst_fi*100:.1f}%"]
            + ["—"] * n_scen,
            ["Inflation on Cash Flows", "—", "—"]
            + [f"{inputs.inflation*100:.2f}%"] * n_scen,
            ["Method", "Parametric (normal)", "Parametric (normal)"]
            + ["Parametric"] * n_scen,
        ]
    base_w = 1.85
    blend_w = max(0.65, (PAGE_W - 1.2 * inch - (base_w + 0.85 + 0.95) * inch)
                  / max(n_scen, 1) / inch)
    col_w = [base_w * inch, 0.85 * inch, 0.95 * inch] + [blend_w * inch] * n_scen
    ra_tbl = Table(ra_data, colWidths=col_w)
    ra_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (0, -1), "Helvetica-Bold", 8.5),
        ("FONT", (1, 1), (-1, -1), "Helvetica", 8.5),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_BG]),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE_GREY),
    ]))
    story.append(ra_tbl)
    story.append(PageBreak())

    # ==================== PATHS CHART + SUMMARY TABLE ====================
    story.extend(section_header("Monte Carlo Simulation — Scenario Comparison"))
    if inputs.return_assumptions.method == "bootstrap":
        method_blurb = (
            "Each year of each path samples a real historical year (matched "
            "equity / fixed-income pair) from the re-centered "
            f"{inputs.return_assumptions.historical_period} series. Annual returns "
            "are decomposed into equal per-period returns to honor the "
            f"{inputs.distribution_frequency.lower()} cash-flow schedule."
        )
    else:
        method_blurb = (
            f"Returns are drawn at the {inputs.distribution_frequency.lower()} "
            "frequency from normal distributions parameterized by the chosen "
            "annual return assumptions."
        )
    story.append(Paragraph(
        f"Each scenario was simulated across <b>{inputs.n_paths:,} independent paths</b> "
        f"over {inputs.horizon_years} years. {method_blurb} "
        "The lines below show median outcomes; shaded regions show the 20th–80th percentile bands "
        "of portfolio value.",
        P_BODY,
    ))
    story.append(Image(img_paths_buf, width=7.0 * inch, height=3.7 * inch))
    story.append(Paragraph(
        f"Figure 1 — Median portfolio value with 20th–80th percentile bands across {n_scen} "
        f"allocation scenario{'s' if n_scen > 1 else ''}.",
        P_FIGCAP,
    ))

    story.extend(section_header(f"{inputs.horizon_years}-Year Outcome Summary"))
    summary_header = ["Metric"] + [
        f"{r['scenario'].name}\n({int(r['scenario'].eq_weight*100)}% / "
        f"{int(r['scenario'].fi_weight*100)}%)"
        for r in results
    ]

    rows = []

    # Contribution rows (only shown if any scenario has contributions)
    if has_contrib:
        rows.append(
            ["Annual Contribution (today's $)"]
            + [(f"${r['scenario'].annual_contribution:,.0f}"
                if r['scenario'].contribution_years > 0 else "—")
               for r in results]
        )
        rows.append(
            ["Contribution Years"]
            + [(f"Yrs 1–{r['scenario'].contribution_years}"
                if r['scenario'].contribution_years > 0 else "—")
               for r in results]
        )
        rows.append(
            ["Total Contributed (nominal)"]
            + [(fmt_m(r["total_contributed"])
                if r['scenario'].contribution_years > 0 else "—")
               for r in results]
        )

    # Distribution rows
    rows.append(
        ["Annual Distribution (today's $)"]
        + [f"${r['scenario'].annual_distribution:,.0f}" for r in results]
    )
    if has_contrib:
        rows.append(
            ["Distribution Start Year"]
            + [f"Yr {r['scenario'].distribution_start_year}" for r in results]
        )
    rows.append(
        [f"Per-Period Distribution ({inputs.distribution_frequency}, today's $)"]
        + [f"${r['scenario'].annual_distribution/FREQ_TO_PER_YEAR[inputs.distribution_frequency]:,.0f}"
           for r in results]
    )
    rows.append(["Annual Escalation"] + [f"{inputs.inflation*100:.1f}%"] * n_scen)
    rows.append(
        [f"Total Distributed (nominal, {inputs.horizon_years} yrs)"]
        + [fmt_m(r["total_distributed"]) for r in results]
    )

    # Outcome rows
    rows += [
        ["Median Value — Year 10"] + [fmt_m(r["median_y10"]) for r in results],
        ["Median Value — Year 20"] + [fmt_m(r["median_y20"]) for r in results],
        [f"Median Value — Year {inputs.horizon_years}"]
        + [fmt_m(r["median_yfinal"]) for r in results],
        [f"20th Percentile (Year {inputs.horizon_years})"]
        + [fmt_m(r["p20_yfinal"]) for r in results],
        [f"80th Percentile (Year {inputs.horizon_years})"]
        + [fmt_m(r["p80_yfinal"]) for r in results],
        ["Probability of Ruin"] + [fmt_pct(r["p_ruin"], 1) for r in results],
        ["Prob. Exceeds Initial Investment"]
        + [fmt_pct(r["p_above_init"], 1) for r in results],
    ]
    sum_data = [summary_header] + rows
    metric_w = 2.45 * inch
    scen_w = (PAGE_W - 1.2 * inch - metric_w) / max(n_scen, 1)
    def _make_data_table_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONT", (0, 1), (0, -1), "Helvetica-Bold", 8.5),
            ("FONT", (1, 1), (-1, -1), "Helvetica", 8.5),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_BG]),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE_GREY),
            ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE_GREY),
        ])

    sum_tbl = Table(sum_data, colWidths=[metric_w] + [scen_w] * n_scen)
    sum_tbl.setStyle(_make_data_table_style())
    story.append(sum_tbl)
    story.append(PageBreak())

    # ==================== DISTRIBUTIONS + DETAILED STATS ====================
    story.extend(section_header(f"Year-{inputs.horizon_years} Outcome Distributions"))
    story.append(Paragraph(
        f"The histograms below show the full distribution of Year-{inputs.horizon_years} portfolio "
        f"values across all {inputs.n_paths:,} simulated paths for each scenario. The dashed red "
        "line marks the median outcome.",
        P_BODY,
    ))
    story.append(Image(img_dist_buf, width=7.0 * inch, height=2.3 * inch))
    story.append(Paragraph(
        f"Figure 2 — Year-{inputs.horizon_years} portfolio value distributions.",
        P_FIGCAP,
    ))

    story.extend(section_header("Detailed Monte Carlo Statistics"))
    det_header = ["Statistic"] + [
        f"{r['scenario'].name} ({int(r['scenario'].eq_weight*100)}/"
        f"{int(r['scenario'].fi_weight*100)})" for r in results
    ]
    det_rows = [
        ["Mean Final Value"] + [fmt_m(r["mean_yfinal"]) for r in results],
        [f"Median (50th Pct) — Yr {inputs.horizon_years}"]
        + [fmt_m(r["median_yfinal"]) for r in results],
        [f"20th Percentile — Yr {inputs.horizon_years}"]
        + [fmt_m(r["p20_yfinal"]) for r in results],
        [f"25th Percentile — Yr {inputs.horizon_years}"]
        + [fmt_m(r["p25_yfinal"]) for r in results],
        [f"75th Percentile — Yr {inputs.horizon_years}"]
        + [fmt_m(r["p75_yfinal"]) for r in results],
        [f"80th Percentile — Yr {inputs.horizon_years}"]
        + [fmt_m(r["p80_yfinal"]) for r in results],
        [f"Interquartile Range (Yr {inputs.horizon_years})"]
        + [fmt_m(r["iqr_yfinal"]) for r in results],
        ["Probability of Portfolio Ruin"] + [fmt_pct(r["p_ruin"], 2) for r in results],
        [f"Prob. Above Initial Inv. (Yr {inputs.horizon_years})"]
        + [fmt_pct(r["p_above_init"], 1) for r in results],
        ["Blended Annual Mean Return"] + [fmt_pct(r["mu_a"], 2) for r in results],
        ["Blended Annual Volatility (σ)"] + [fmt_pct(r["sig_a"], 2) for r in results],
    ]
    det_tbl = Table([det_header] + det_rows,
                    colWidths=[metric_w] + [scen_w] * n_scen)
    det_tbl.setStyle(_make_data_table_style())
    story.append(det_tbl)
    story.append(PageBreak())

    # ==================== ALLOCATION COMPARISON + KEY FINDINGS ====================
    story.extend(section_header("Allocation Comparison"))
    story.append(Paragraph(
        "Each allocation maintains the same underlying asset classes — U.S. equity and "
        "U.S. fixed income — and the same distribution policy. The variable across scenarios "
        "is the equity weighting and/or distribution amount. Higher equity weights raise both "
        "expected return and expected volatility.",
        P_BODY,
    ))
    story.append(Image(img_alloc_buf, width=7.0 * inch, height=2.3 * inch))
    story.append(Paragraph(
        f"Figure 3 — {n_scen} target allocation{'s' if n_scen > 1 else ''} evaluated.",
        P_FIGCAP,
    ))

    story.extend(section_header("Key Findings"))
    for r in results:
        s = r["scenario"]
        if s.contribution_years > 0 and s.annual_contribution > 0:
            cf_desc = (
                f"${s.annual_contribution:,.0f}/yr contributed Yrs 1–{s.contribution_years}, "
                f"then ${s.annual_distribution:,.0f}/yr distributed Yrs "
                f"{s.distribution_start_year}–{inputs.horizon_years}"
            )
        else:
            cf_desc = f"${s.annual_distribution:,.0f}/yr distributed"
        finding = (
            f"<b>{s.name} — {int(s.eq_weight*100)}% Equity / {int(s.fi_weight*100)}% "
            f"Fixed Income, {cf_desc}:</b> "
            f"At the median, this allocation projects a Year-{inputs.horizon_years} portfolio "
            f"value of <b>{fmt_m(r['median_yfinal'])}</b>, with a 20th–80th percentile range of "
            f"{fmt_m(r['p20_yfinal'])} to {fmt_m(r['p80_yfinal'])}. "
            f"Probability of portfolio ruin: <b>{r['p_ruin']*100:.2f}%</b>. "
            f"Probability of ending above the initial ${inputs.initial/1e6:.1f}M investment: "
            f"<b>{r['p_above_init']*100:.1f}%</b>. "
            f"Blended expected return: {r['mu_a']*100:.2f}%; annual volatility: {r['sig_a']*100:.2f}%."
        )
        story.append(Paragraph(finding, P_BODY))

    story.append(Spacer(1, 8))
    contrib_note = ""
    if has_contrib:
        contrib_note = (
            "Contribution and distribution amounts are entered in today's (Year-1) dollars "
            "and inflated forward at the inflation rate to preserve real purchasing power | "
        )
    ra2 = inputs.return_assumptions
    if ra2.method == "bootstrap":
        ret_text = (
            f"Equity forward-looking μ = {ra2.eq_mu*100:.2f}%, fixed income μ = {ra2.fi_mu*100:.2f}%; "
            f"historical σ and eq/FI correlation preserved from {ra2.historical_period} "
            f"({HIST_STATS['n_years']}-year matched-pair bootstrap, re-centered) | "
        )
        cor_text = (
            f"Equity/fixed income correlation captured naturally via matched-pair bootstrap "
            f"sampling | "
        )
    else:
        ret_text = (
            f"Equity expected return {ra2.eq_mu*100:.2f}% (σ = {ra2.eq_sigma*100:.2f}%); "
            f"fixed income expected return {ra2.fi_mu*100:.2f}% (σ = {ra2.fi_sigma*100:.2f}%); "
            f"source: {ra2.label} | "
            f"Per-period returns drawn independently from N(μ/k, σ/√k) where k = periods per year | "
        )
        cor_text = "Asset-class returns assumed uncorrelated for blended-volatility calculation | "

    assump = (
        f"<b>Key Assumptions & Disclosures:</b> Initial investment ${inputs.initial:,.0f} | "
        f"{contrib_note}"
        f"Distribution frequency: {inputs.distribution_frequency} | "
        f"Annual escalation: {inputs.inflation*100:.1f}% | Time horizon: {inputs.horizon_years} years | "
        f"{ret_text}"
        f"Cash flows applied at the start of each period; returns applied to post-cashflow balance | "
        f"{cor_text}"
        f"No tax drag, advisory fees, or rebalancing costs modeled | "
        f"Monte Carlo simulation: {inputs.n_paths:,} paths per scenario | "
        "Past performance is not indicative of future results. This analysis is for illustrative "
        "purposes only and does not constitute investment advice."
    )
    story.append(Paragraph(assump, P_DISCLAIM))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# Streamlit UI
# =============================================================================
def _parse_dollar_input(raw: str) -> Optional[int]:
    """
    Parse a freeform dollar string into an integer.

    Accepts:
      "5400000", "5,400,000", "$5,400,000"
      "5.4M", "5.4m", "$5.4M"
      "225K", "225k"
      "1.5B", "1.5b"
      "" or whitespace -> 0
      anything malformed -> None (caller can show an error)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return 0
    # Strip $, commas, spaces
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Detect suffix
    multiplier = 1
    if s and s[-1].lower() == "k":
        multiplier = 1_000
        s = s[:-1]
    elif s and s[-1].lower() == "m":
        multiplier = 1_000_000
        s = s[:-1]
    elif s and s[-1].lower() == "b":
        multiplier = 1_000_000_000
        s = s[:-1]
    try:
        value = float(s) * multiplier
    except ValueError:
        return None
    return int(round(value))


def _format_dollar_callback(widget_key: str, last_good_key: str,
                            min_value: int, max_value: int):
    """
    Streamlit on_change callback for dollar_input.

    Runs after the user types and presses Enter (or clicks away). Parses the
    raw text in session_state[widget_key], clamps to bounds, and writes the
    nicely-formatted version back to session_state[widget_key] so the widget
    re-renders with commas. Stores the parsed integer value in
    session_state[last_good_key] for the main flow to read.
    """
    raw = st.session_state.get(widget_key, "")
    parsed = _parse_dollar_input(raw)
    if parsed is None:
        # Bad input — leave the raw text alone so the user can fix it; the
        # main flow will detect this and surface an error.
        st.session_state[f"{widget_key}__error"] = (
            f'Couldn\'t read "{raw}" as a dollar amount. '
            f"Try formats like 5,400,000 or $5.4M."
        )
        return
    # Clear any prior error
    st.session_state.pop(f"{widget_key}__error", None)
    parsed = max(min_value, min(max_value, parsed))
    st.session_state[widget_key] = f"{parsed:,}"
    st.session_state[last_good_key] = parsed


def dollar_input(label: str, default: int, key: str,
                 min_value: int = 0, max_value: int = 1_000_000_000,
                 help: str = None, disabled: bool = False) -> int:
    """
    Streamlit text input for dollar amounts that auto-formats with commas.

    Behavior:
      - User can type "5400000", "5,400,000", "$5.4M", "225K", etc.
      - On Enter (or blur), the value reformats to "5,400,000" with commas.
      - Returns the integer dollar amount.
      - Bad input shows an inline error and the prior value is kept.
    """
    last_good_key = f"{key}__last_good"

    # First render: seed both the display string and the last-good integer
    if key not in st.session_state:
        st.session_state[key] = f"{default:,}"
        st.session_state[last_good_key] = default

    st.text_input(
        label,
        key=key,
        help=help,
        disabled=disabled,
        on_change=_format_dollar_callback,
        args=(key, last_good_key, min_value, max_value),
    )

    # Surface any error from the callback
    err = st.session_state.get(f"{key}__error")
    if err:
        st.error(f"⚠️  {err}")

    # The current valid integer value — even if user typed garbage, this is the
    # last successfully-parsed amount.
    current = st.session_state.get(last_good_key, default)

    # Edge case: user is mid-edit (display doesn't yet match a parsed integer)
    # but the raw text is parseable — return the parsed value so live preview
    # stays in sync without needing the user to press Enter.
    raw = st.session_state.get(key, "")
    parsed = _parse_dollar_input(raw)
    if parsed is not None:
        parsed = max(min_value, min(max_value, parsed))
        return parsed
    return current


def _inject_becker_css():
    """Apply Becker brand styling to all Streamlit components."""
    st.markdown(
        f"""
        <style>
        /* ===== Becker Capital Management — Brand Theme ===== */

        /* Slider track + thumb in Becker gold */
        div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
            background-color: {GOLD_HEX} !important;
            border-color: {GOLD_HEX} !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
            background: {GOLD_HEX} !important;
        }}
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"],
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
            color: {GOLD_HEX} !important;
            font-weight: 600 !important;
        }}

        /* Primary buttons — navy fill, gold border on hover */
        button[kind="primary"] {{
            background-color: {NAVY_HEX} !important;
            color: white !important;
            border: 1.5px solid {GOLD_HEX} !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
        }}
        button[kind="primary"]:hover {{
            background-color: {GOLD_HEX} !important;
            color: {NAVY_DARK_HEX} !important;
            border-color: {GOLD_HEX} !important;
        }}

        /* Secondary buttons (download) — gold-on-navy outline style */
        button[kind="secondary"] {{
            border: 1.5px solid {GOLD_HEX} !important;
            color: {GOLD_HEX} !important;
            font-weight: 600 !important;
        }}
        button[kind="secondary"]:hover {{
            background-color: {GOLD_HEX} !important;
            color: {NAVY_DARK_HEX} !important;
        }}

        /* Metric values — gold for emphasis */
        div[data-testid="stMetricValue"] {{
            color: {GOLD_HEX} !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: #CCCCCC !important;
            font-size: 12px !important;
        }}

        /* Section headers — gold underline accent */
        h1, h2, h3 {{
            color: white !important;
            font-weight: 600 !important;
        }}

        /* Sidebar headers — gold */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {GOLD_HEX} !important;
            border-bottom: 1.5px solid {GOLD_HEX};
            padding-bottom: 6px;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}

        /* Radio buttons in gold when selected */
        div[role="radiogroup"] label[data-checked="true"] {{
            color: {GOLD_HEX} !important;
        }}

        /* Custom Becker header card */
        .becker-header {{
            background: linear-gradient(135deg, {NAVY_DARK_HEX} 0%, {NAVY_HEX} 100%);
            border-left: 6px solid {GOLD_HEX};
            border-radius: 4px;
            padding: 22px 28px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }}
        .becker-monogram {{
            flex-shrink: 0;
            width: 72px;
            height: 72px;
            position: relative;
        }}
        .becker-monogram .fifty {{
            position: absolute;
            top: 50%;
            left: 0;
            transform: translateY(-50%);
            font-family: Georgia, serif;
            font-size: 48px;
            font-weight: 900;
            color: rgba(184, 146, 77, 0.35);
            letter-spacing: -3px;
        }}
        .becker-monogram .b-circle {{
            position: absolute;
            top: 50%;
            right: 0;
            transform: translateY(-50%);
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 2.5px solid {GOLD_HEX};
            background: {NAVY_DARK_HEX};
            display: flex;
            align-items: center;
            justify-content: center;
            color: {GOLD_HEX};
            font-family: Georgia, serif;
            font-size: 24px;
            font-weight: 900;
        }}
        .becker-header-text {{
            flex: 1;
        }}
        .becker-eyebrow {{
            color: {GOLD_HEX};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 4px;
        }}
        .becker-title {{
            color: white;
            font-size: 28px;
            font-weight: 700;
            line-height: 1.15;
        }}
        .becker-subtitle {{
            color: #B8C4D6;
            font-size: 13px;
            margin-top: 4px;
        }}

        /* Scenario card headers */
        .becker-scenario-card {{
            background: rgba(31, 58, 95, 0.45);
            border-left: 4px solid var(--scenario-color);
            border-radius: 3px;
            padding: 10px 14px;
            margin-bottom: 8px;
            color: white;
            font-weight: 700;
            letter-spacing: 0.3px;
        }}

        /* Footer */
        .becker-footer {{
            margin-top: 32px;
            padding: 14px 0;
            border-top: 1px solid rgba(184, 146, 77, 0.3);
            color: #8A99B0;
            font-size: 11px;
            text-align: center;
            letter-spacing: 0.3px;
        }}
        .becker-footer .firm {{
            color: {GOLD_HEX};
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    """Becker-branded header with 50-year monogram."""
    st.markdown(
        """
        <div class="becker-header">
          <div class="becker-monogram">
            <div class="fifty">50</div>
            <div class="b-circle">B</div>
          </div>
          <div class="becker-header-text">
            <div class="becker-eyebrow">BECKER CAPITAL MANAGEMENT  •  EST. 1976</div>
            <div class="becker-title">Monte Carlo Portfolio Analysis</div>
            <div class="becker-subtitle">
              Configure inputs, preview Monte Carlo outcomes, and download a branded
              PDF report for client review.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer():
    """Becker-branded footer line matching the PDF disclaimer style."""
    st.markdown(
        """
        <div class="becker-footer">
          <span class="firm">BECKER CAPITAL MANAGEMENT</span>
          &nbsp; | &nbsp; BECKERCAP.COM &nbsp; | &nbsp; 503.223.1720
          &nbsp; | &nbsp; Established 1976
          <br/>
          <span style="font-style:italic;">
            This report is hypothetical and for illustrative purposes only.
            Not investment advice. Past performance is no guarantee of future results.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Becker Capital — Monte Carlo Portfolio Analysis",
        page_icon="📊",
        layout="wide",
    )

    _inject_becker_css()
    _render_header()

    # ----- Sidebar: inputs -----
    with st.sidebar:
        st.header("Portfolio Inputs")

        initial = dollar_input(
            "Initial Investment ($)",
            default=5_400_000,
            key="initial",
            min_value=100_000,
            max_value=1_000_000_000,
            help="Type freely — e.g., 5400000, 5,400,000, $5.4M, or 5.4m. "
                 "Commas are added automatically.",
        )
        horizon = st.slider("Time Horizon (years)", 5, 50, 30, step=1)
        inflation = st.slider("Inflation Rate (%)", 0.0, 8.0, 3.0, step=0.25) / 100
        freq = st.selectbox(
            "Distribution Frequency", ["Annual", "Quarterly", "Monthly"], index=1
        )

        st.divider()
        st.subheader("Return Assumptions")

        method = st.radio(
            "Method",
            ["Bootstrap (recommended)", "Parametric (normal distribution)"],
            index=0,
            help=(
                "Bootstrap resamples actual historical years (1928–2024) and "
                "re-centers them to your forward-looking mean — preserving real-"
                "world volatility, fat tails, and the equity/fixed-income "
                "correlation. Parametric draws from a normal distribution, which "
                "is faster but understates downside risk."
            ),
        )

        if method.startswith("Bootstrap"):
            preset_choice = st.selectbox(
                "Preset",
                ["Forward-looking (Aggressive): 9.5% / 4.5%",
                 "Forward-looking (Moderate): 8% / 4.5%",
                 "Forward-looking (Conservative): 7% / 4%",
                 "Historical means (1928–2024): 11.87% / 4.89%",
                 "Custom forward-looking μ"],
                index=1,
            )
            if preset_choice.startswith("Forward-looking (Aggressive)"):
                ra = PRESETS["Bootstrap — Forward-looking (Aggressive)"]
            elif preset_choice.startswith("Forward-looking (Moderate)"):
                ra = PRESETS["Bootstrap — Forward-looking (Moderate)"]
            elif preset_choice.startswith("Forward-looking (Conservative)"):
                ra = PRESETS["Bootstrap — Forward-looking (Conservative)"]
            elif preset_choice.startswith("Historical means"):
                ra = PRESETS["Bootstrap — Historical means (1928–2024)"]
            else:
                # Custom forward-looking
                colA, colB = st.columns(2)
                with colA:
                    st.caption("**Equity**")
                    eq_mu_in = st.number_input(
                        "Forward μ (%)", value=8.0, step=0.25,
                        min_value=0.0, max_value=20.0, key="boot_eq_mu",
                    ) / 100
                with colB:
                    st.caption("**Fixed Income**")
                    fi_mu_in = st.number_input(
                        "Forward μ (%)", value=4.5, step=0.25,
                        min_value=0.0, max_value=12.0, key="boot_fi_mu",
                    ) / 100
                ra = ReturnAssumptions(
                    eq_mu=eq_mu_in,
                    eq_sigma=HIST_STATS["eq_sigma"],
                    fi_mu=fi_mu_in,
                    fi_sigma=HIST_STATS["fi_sigma"],
                    label=f"Bootstrap • Custom (μ={eq_mu_in*100:.1f}%/{fi_mu_in*100:.1f}%)",
                    method="bootstrap",
                    historical_period="1928–2024",
                    worst_eq=HIST_STATS["worst_eq"],
                    worst_fi=HIST_STATS["worst_fi"],
                )
            st.caption(
                f"**Forward μ:** Eq {ra.eq_mu*100:.2f}% • FI {ra.fi_mu*100:.2f}%  \n"
                f"**Historical σ (preserved):** Eq {HIST_STATS['eq_sigma']*100:.2f}% • "
                f"FI {HIST_STATS['fi_sigma']*100:.2f}%  \n"
                f"**Eq/FI ρ (preserved):** {HIST_STATS['correlation']:.3f}  \n"
                f"**Worst historical year:** Eq {HIST_STATS['worst_eq']*100:.1f}% • "
                f"FI {HIST_STATS['worst_fi']*100:.1f}%"
            )
        else:
            # Parametric mode — keep legacy presets
            param_choice = st.selectbox(
                "Preset",
                ["1960–2024 (Legacy Becker)",
                 "Forward-looking Conservative (8%/4.5%)",
                 "Custom"],
                index=0,
            )
            if param_choice == "1960–2024 (Legacy Becker)":
                ra = PRESETS["Parametric — 1960–2024 (Legacy Becker)"]
            elif param_choice.startswith("Forward-looking"):
                ra = PRESETS["Parametric — Forward-looking (Conservative)"]
            else:
                colA, colB = st.columns(2)
                with colA:
                    st.caption("**Equity**")
                    eq_mu_in = st.number_input("Mean (%)", value=11.79, step=0.1,
                                            key="param_eq_mu") / 100
                    eq_sig_in = st.number_input("Std Dev (%)", value=16.67, step=0.1,
                                             key="param_eq_sig") / 100
                with colB:
                    st.caption("**Fixed Income**")
                    fi_mu_in = st.number_input("Mean (%)", value=6.15, step=0.1,
                                            key="param_fi_mu") / 100
                    fi_sig_in = st.number_input("Std Dev (%)", value=8.79, step=0.1,
                                             key="param_fi_sig") / 100
                ra = ReturnAssumptions(
                    eq_mu=eq_mu_in, eq_sigma=eq_sig_in,
                    fi_mu=fi_mu_in, fi_sigma=fi_sig_in,
                    label="Parametric • Custom", method="parametric",
                )
            st.caption(
                f"Eq μ = {ra.eq_mu*100:.2f}%, σ = {ra.eq_sigma*100:.2f}%  |  "
                f"FI μ = {ra.fi_mu*100:.2f}%, σ = {ra.fi_sigma*100:.2f}%"
            )

        st.divider()
        st.subheader("Simulation")
        n_paths = st.select_slider(
            "Number of paths", options=[1_000, 2_500, 5_000, 10_000, 25_000],
            value=10_000,
        )

    # ----- Main area: scenario builder -----
    st.subheader("Scenarios")
    st.caption(
        "Configure 1–3 scenarios. Each scenario can have a contribution phase "
        "(money added) followed by a distribution phase (money withdrawn). "
        "All amounts are entered in today's dollars and inflated forward automatically."
    )

    n_scen = st.radio("Number of scenarios", [1, 2, 3], index=2, horizontal=True)

    default_scenarios = [
        ("Scenario A", 60, 225_000, 0,  0),
        ("Scenario B", 70, 225_000, 0,  0),
        ("Scenario C", 80, 225_000, 0,  0),
    ]

    scenarios: List[Scenario] = []
    cols = st.columns(n_scen)
    for i, col in enumerate(cols):
        name_def, eq_def, dist_def, cy_def, ca_def = default_scenarios[i]
        with col:
            st.markdown(
                f"<div class='becker-scenario-card' "
                f"style='--scenario-color:{SCENARIO_COLOR_HEX[i]};'>"
                f"{name_def}</div>",
                unsafe_allow_html=True,
            )
            name = st.text_input("Name", value=name_def, key=f"name_{i}")
            eq_pct = st.slider(
                "Equity %", 0, 100, eq_def, step=5, key=f"eq_{i}",
            )
            st.caption(f"Fixed Income: **{100 - eq_pct}%**")

            # ----- Contribution phase -----
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:8px; padding-top:6px; "
                f"border-top:1px solid rgba(184,146,77,0.3);'>"
                f"CONTRIBUTION PHASE (OPTIONAL)</div>",
                unsafe_allow_html=True,
            )
            contrib_yrs = st.number_input(
                "Contribution Years (0 = none)",
                min_value=0, max_value=int(horizon) - 1,
                value=cy_def, step=1, key=f"cy_{i}",
                help="Number of years to contribute before distributions begin. "
                     "Set to 0 if there is no contribution phase.",
            )
            contrib_amt = dollar_input(
                "Annual Contribution ($, today's dollars)",
                default=ca_def, key=f"ca_{i}",
                min_value=0, max_value=10_000_000,
                disabled=(contrib_yrs == 0),
                help="In today's dollars. Type 50000, 50,000, or 50K. "
                     "Will be inflated forward automatically.",
            )

            # ----- Distribution phase -----
            dist_start_year = int(contrib_yrs) + 1
            dist_label_suffix = (
                f" — starts Yr {dist_start_year}" if contrib_yrs > 0 else ""
            )
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:10px; padding-top:6px; "
                f"border-top:1px solid rgba(184,146,77,0.3);'>"
                f"DISTRIBUTION PHASE{dist_label_suffix.upper()}</div>",
                unsafe_allow_html=True,
            )
            dist = dollar_input(
                "Annual Distribution ($, today's dollars)",
                default=dist_def, key=f"dist_{i}",
                min_value=0, max_value=10_000_000,
                help="In today's dollars. Type 225000, 225,000, or 225K. "
                     "Will be inflated forward to maintain real purchasing power.",
            )

            # Show what the first nominal distribution will be (for transparency)
            if contrib_yrs > 0:
                nominal_first = dist * ((1 + inflation) ** (dist_start_year - 1))
                st.caption(
                    f"Nominal Year-{dist_start_year} distribution: "
                    f"**${nominal_first:,.0f}**"
                )

            scenarios.append(Scenario(
                name=name,
                eq_weight=eq_pct / 100,
                fi_weight=(100 - eq_pct) / 100,
                annual_distribution=float(dist),
                contribution_years=int(contrib_yrs),
                annual_contribution=float(contrib_amt),
            ))

    inputs = SimInputs(
        initial=float(initial),
        horizon_years=int(horizon),
        inflation=inflation,
        distribution_frequency=freq,
        return_assumptions=ra,
        scenarios=scenarios,
        n_paths=int(n_paths),
    )

    # ----- Run simulation (cached) -----
    @st.cache_data(show_spinner="Running Monte Carlo simulation...")
    def run_cached(_inputs_key: str, inputs: SimInputs):
        return run_all_simulations(inputs)

    inputs_key = repr(inputs)
    results = run_cached(inputs_key, inputs)

    st.divider()

    # ----- Results -----
    st.subheader("Live Preview")
    metric_cols = st.columns(n_scen)
    for col, r in zip(metric_cols, results):
        s = r["scenario"]
        with col:
            if s.contribution_years > 0 and s.annual_contribution > 0:
                phase_desc = (
                    f"+${s.annual_contribution/1000:,.0f}K/yr × {s.contribution_years} yrs, "
                    f"then −${s.annual_distribution/1000:,.0f}K/yr"
                )
            else:
                phase_desc = f"−${s.annual_distribution/1000:,.0f}K/yr"

            st.markdown(
                f"**{s.name}** — {int(s.eq_weight*100)}/{int(s.fi_weight*100)}<br/>"
                f"<span style='color:#B8C4D6; font-size:12px;'>{phase_desc}</span>",
                unsafe_allow_html=True,
            )
            st.metric(
                f"Median Yr {horizon}",
                f"${r['median_yfinal']/1e6:,.2f}M",
            )
            st.metric(
                "20th–80th Pct",
                f"${r['p20_yfinal']/1e6:,.2f}M – ${r['p80_yfinal']/1e6:,.2f}M",
            )
            st.metric("Probability of Ruin", f"{r['p_ruin']*100:.2f}%")
            st.metric("P(Exceeds Initial)", f"{r['p_above_init']*100:.1f}%")

    # Path chart preview
    st.markdown("**Median Path with 20th–80th Percentile Bands**")
    fig_buf = chart_paths_with_bands(results, inputs)
    st.image(fig_buf, use_container_width=True)

    # ----- Generate PDF -----
    st.divider()
    st.subheader("Generate PDF Report")
    colp1, colp2 = st.columns([1, 3])
    with colp1:
        if st.button("📄 Build PDF Report", type="primary", use_container_width=True):
            with st.spinner("Building PDF..."):
                pdf_bytes = build_pdf(results, inputs)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_filename"] = (
                    f"Becker_MonteCarlo_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                )
            st.success("PDF ready — click below to download.")
    with colp2:
        if "pdf_bytes" in st.session_state:
            st.download_button(
                "⬇️  Download PDF",
                data=st.session_state["pdf_bytes"],
                file_name=st.session_state["pdf_filename"],
                mime="application/pdf",
                use_container_width=True,
            )

    st.caption(
        "Tip: change the equity sliders to compare allocations side by side, "
        "or adjust the annual distribution amount to test different withdrawal levels."
    )

    _render_footer()


if __name__ == "__main__":
    main()
