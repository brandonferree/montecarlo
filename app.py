"""
Becker Capital — Cashflow Portfolio Analysis
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
# Brand palette — Becker Capital Management official identity
# Sourced from BCM brand guidelines (anthropic-skills:bcm-branding).
#
# Official brand colors below; the legacy aliases (NAVY_HEX, GOLD_HEX, …)
# preserve the existing call sites across CSS, matplotlib, and reportlab
# without a sweeping rename. Note GOLD_HEX is now Canyon orange (#AD5E25),
# NOT the previous warm gold (#B8924D) — the name is historical.
# =============================================================================
# Official BCM palette
MIDNIGHT_HEX      = "#0C2331"   # Dark backgrounds
CHARCOAL_HEX      = "#2B2D24"   # Primary text on light bg
LIGHT_GREY_HEX    = "#EEEEED"   # Light backgrounds, text on dark
DARK_GREY_HEX     = "#8C8985"   # Secondary text / elements
DEACTIVATED_HEX   = "#D5D8D9"   # Subtle backgrounds, rules, disabled
DARK_WATER_HEX    = "#1D5D78"   # Primary accent (blue)
ACTIVE_HEX        = "#41858F"   # Secondary accent (teal)
CANYON_HEX        = "#AD5E25"   # Tertiary accent (orange / copper)

# RGB-tuple strings — for use inside rgba() expressions where the legacy
# hardcoded "184,146,77" / "31,58,95" tuples once lived. Updating the hex
# above automatically threads through every rgba(...) site that uses these.
MIDNIGHT_RGB      = "12,35,49"
CANYON_RGB        = "173,94,37"
DARK_WATER_RGB    = "29,93,120"

# Legacy aliases — used pervasively across CSS / matplotlib / reportlab.
NAVY_HEX        = MIDNIGHT_HEX
NAVY_DARK_HEX   = "#061520"          # Slightly darker than Midnight (for the header gradient)
GOLD_HEX        = CANYON_HEX         # ⚠ Canyon orange, not actually gold
TEAL_HEX        = ACTIVE_HEX
LIGHT_BG_HEX    = LIGHT_GREY_HEX
ALT_BG_HEX      = "#F8F8F7"          # Soft tint between LIGHT_GREY and white
TEXT_DARK_HEX   = CHARCOAL_HEX
TEXT_MED_HEX    = DARK_GREY_HEX
RULE_GREY_HEX   = DEACTIVATED_HEX

NAVY        = colors.HexColor(NAVY_HEX)
NAVY_DARK   = colors.HexColor(NAVY_DARK_HEX)
GOLD        = colors.HexColor(GOLD_HEX)
TEAL        = colors.HexColor(TEAL_HEX)
LIGHT_BG    = colors.HexColor(LIGHT_BG_HEX)
ALT_BG      = colors.HexColor(ALT_BG_HEX)
TEXT_DARK   = colors.HexColor(TEXT_DARK_HEX)
TEXT_MED    = colors.HexColor(TEXT_MED_HEX)
RULE_GREY   = colors.HexColor(RULE_GREY_HEX)

# Scenario rotation — the three official BCM accents in order
# (Dark Water → Active → Canyon = blue → teal → orange).
SCENARIO_COLOR_HEX = [DARK_WATER_HEX, ACTIVE_HEX, CANYON_HEX]


# =============================================================================
# Data classes for inputs
# =============================================================================
@dataclass
class ReturnAssumptions:
    """
    Return-generation specification — parametric normal-distribution draws.

    eq_mu / fi_mu are forward-looking expected annual returns; eq_sigma /
    fi_sigma are annual volatilities. Each period of the simulation draws
    from N(mu/k, sigma/sqrt(k)) where k is periods per year (asset classes
    treated as independent).

    Values are sourced from Becker Capital Management's published 2026
    Capital Market Assumptions (10-year estimates) — see BCM_CMA_2026.

    worst_eq / worst_fi are static historical worst-calendar-year reference
    figures shown on the PDF assumption table; they are not used by the
    simulation itself.
    """
    eq_mu: float
    eq_sigma: float
    fi_mu: float
    fi_sigma: float
    label: str
    worst_eq: float = -0.4384
    worst_fi: float = -0.1777


@dataclass
class Scenario:
    name: str
    eq_weight: float                      # 0–1, accumulation-phase equity weight
    fi_weight: float                      # 0–1, accumulation-phase fixed-income weight
    annual_distribution: float            # in today's (Year-1) dollars
    contribution_years: int = 0           # number of years to contribute (0 = none)
    annual_contribution: float = 0.0      # in today's (Year-1) dollars
    # ----- Glide path (optional) -----
    # If set, the portfolio rebalances ONCE at the start of the distribution phase
    # to the retirement allocation. If left None, the same eq/fi weights apply
    # throughout (i.e., a static allocation).
    retirement_eq_weight: Optional[float] = None    # 0–1, post-rebalance equity
    retirement_fi_weight: Optional[float] = None    # 0–1, post-rebalance FI

    @property
    def distribution_start_year(self) -> int:
        """The first year in which a distribution is paid (1-indexed)."""
        return self.contribution_years + 1

    @property
    def has_glide_path(self) -> bool:
        """True iff the retirement allocation is set AND differs from accumulation."""
        if self.retirement_eq_weight is None:
            return False
        return abs(self.retirement_eq_weight - self.eq_weight) > 1e-9

    def eq_weight_in_year(self, year_1_indexed: int) -> float:
        """Equity weight applicable in a given year (1-indexed)."""
        if self.has_glide_path and year_1_indexed >= self.distribution_start_year:
            return float(self.retirement_eq_weight)
        return self.eq_weight

    def fi_weight_in_year(self, year_1_indexed: int) -> float:
        """Fixed-income weight applicable in a given year (1-indexed)."""
        return 1.0 - self.eq_weight_in_year(year_1_indexed)


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


# =============================================================================
# BCM 2026 Capital Market Assumptions (10-year estimates)
# Source: Becker Capital Management, Inc. — "2026 CMAs.pdf"
# All values are annual decimals: (mu, sigma).
#
# Only the two asset classes the single preset uses are kept here. The full
# 19-asset CMA reference table lives in memory/reference_bcm_cma_2026.md.
# =============================================================================
BCM_CMA_2026 = {
    "EQ_US_LARGE": (0.0600, 0.1700),  # Russell 1000
    "FI_IT_US":    (0.0500, 0.0550),  # Intermediate-Term U.S. Corp & Govt (5-10Y)
}


def _bcm_preset(label: str, eq_key: str, fi_key: str) -> ReturnAssumptions:
    """Build a ReturnAssumptions from two BCM_CMA_2026 entries."""
    eq_mu, eq_sigma = BCM_CMA_2026[eq_key]
    fi_mu, fi_sigma = BCM_CMA_2026[fi_key]
    return ReturnAssumptions(
        eq_mu=eq_mu, eq_sigma=eq_sigma,
        fi_mu=fi_mu, fi_sigma=fi_sigma,
        label=label,
    )


# Single preset sourced from BCM's 2026 10-year CMAs (Large Cap +
# Intermediate U.S. Bonds — the same combination that was previously
# labeled "Moderate" before the risk-tier framing was dropped).
PRESETS = {
    "BCM 2026 CMAs": _bcm_preset(
        "BCM 2026 CMAs (Large Cap + Intermediate Bonds)",
        eq_key="EQ_US_LARGE", fi_key="FI_IT_US",
    ),
}


FREQ_TO_PER_YEAR = {"Annual": 1, "Quarterly": 4, "Monthly": 12}


# =============================================================================
# Simulation
# =============================================================================
def blended_params(eq_w: float, fi_w: float, ra: ReturnAssumptions) -> Tuple[float, float]:
    """
    Blended annual mean and std for display purposes only. Asset classes
    are assumed uncorrelated for the variance term.
    """
    mu = eq_w * ra.eq_mu + fi_w * ra.fi_mu
    sig = np.sqrt((eq_w * ra.eq_sigma) ** 2 + (fi_w * ra.fi_sigma) ** 2)
    return mu, sig


def simulate_scenario(scen: Scenario, inputs: SimInputs, seed_offset: int) -> dict:
    """
    Run the simulation for a single scenario.

    Cash-flow model (all amounts are entered in Year-1 'today's' dollars):
      - Years 1 .. contribution_years:
            CONTRIBUTION of `annual_contribution * (1+infl)^(y-1)` is ADDED at start
            of each period (split evenly across k periods).
      - Years contribution_years+1 .. horizon:
            DISTRIBUTION of `annual_distribution * (1+infl)^(y-1)` is REMOVED at
            start of each period.
      - Cash flows applied AT THE START of each period; returns applied to
        post-cashflow balance.

    Return generation: each period's return drawn from N(μ/k, σ/√k) with
    asset classes treated as independent.
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

    for y in range(1, yrs + 1):
        infl_factor = (1 + inputs.inflation) ** (y - 1)

        # Allocation weights for this year (handles glide-path transition).
        eq_w_y = scen.eq_weight_in_year(y)
        fi_w_y = scen.fi_weight_in_year(y)

        if y <= contrib_years:
            annual_contrib = scen.annual_contribution * infl_factor
            per_period_cf = -annual_contrib / k
        else:
            annual_dist = scen.annual_distribution * infl_factor
            per_period_cf = annual_dist / k

        # Parametric draw — recompute (mu, sigma) for this year's weights.
        mu_a_y, sig_a_y = blended_params(eq_w_y, fi_w_y, ra)
        mu_p = mu_a_y / k
        sig_p = sig_a_y / np.sqrt(k)
        for _ in range(k):
            bal = bal - per_period_cf
            bal = np.maximum(bal, 0.0)
            r = rng.normal(mu_p, sig_p, size=n)
            bal = np.maximum(bal * (1 + r), 0.0)

        yearly[:, y] = bal

    yr_final = yearly[:, -1]
    yr10 = yearly[:, min(10, yrs)]
    yr20 = yearly[:, min(20, yrs)]
    # For display: show the *retirement-phase* blended params if glide path,
    # else the (unchanged) static blend. We compute both so the report can
    # surface the transition explicitly.
    mu_a_acc, sig_a_acc = blended_params(scen.eq_weight, scen.fi_weight, ra)
    if scen.has_glide_path:
        mu_a_ret, sig_a_ret = blended_params(
            scen.retirement_eq_weight, 1.0 - scen.retirement_eq_weight, ra
        )
    else:
        mu_a_ret, sig_a_ret = mu_a_acc, sig_a_acc
    # The "primary" mu/sigma for tables uses the retirement blend (the regime
    # that holds for most of a typical horizon). Static scenarios are unaffected.
    mu_a, sig_a = mu_a_ret, sig_a_ret

    return {
        "scenario": scen,
        "mu_a": mu_a,
        "sig_a": sig_a,
        "mu_a_acc": mu_a_acc,         # accumulation-phase blended mean
        "sig_a_acc": sig_a_acc,       # accumulation-phase blended sigma
        "mu_a_ret": mu_a_ret,         # retirement-phase blended mean
        "sig_a_ret": sig_a_ret,       # retirement-phase blended sigma
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
        # Disable matplotlib's LaTeX-style math-mode parsing — without this,
        # any text containing two $ signs (e.g., "Save $129K/yr × ... → $200K/yr")
        # would have everything between them rendered as italic math text with
        # the $ signs stripped. Affects legends, titles, tick labels.
        "text.parse_math": False,
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
        f"{inputs.horizon_years}-Year Portfolio Value — Median & Percentile Bands\n"
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

    fig.suptitle(f"Year-{inputs.horizon_years} Portfolio Value Distribution",
                 fontsize=12, color=NAVY_HEX, fontweight="bold", y=1.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_allocations(results: List[dict]) -> io.BytesIO:
    """
    Render the allocation comparison.

    If ANY scenario has a glide path, render a dual-row layout:
      - Top row    = accumulation-phase allocation
      - Arrow row  = gold transition arrow on glide-path scenarios
      - Bottom row = retirement-phase allocation (or "(no change)" for static)

    If no scenario has a glide path, render the original single-row pies.

    Layout uses an explicit GridSpec with margins reserved for the title,
    legend, and arrow annotations — this AVOIDS the `tight_layout` warning
    that fires when matplotlib can't squeeze suptitles + bbox_to_anchor
    legends + cross-axes annotations into auto-computed padding.
    """
    _setup_mpl()
    n = len(results)
    any_glide = any(r["scenario"].has_glide_path for r in results)

    # ---------------- Single-row layout (no glide paths anywhere) -----------
    if not any_glide:
        fig = plt.figure(figsize=(9.0, 3.2), dpi=180, facecolor="white")
        # Reserve top space for titles, bottom for legend, sides for breathing room.
        gs = fig.add_gridspec(
            nrows=1, ncols=n,
            left=0.04, right=0.96, top=0.82, bottom=0.20,
            wspace=0.30,
        )
        for i, r in enumerate(results):
            ax = fig.add_subplot(gs[0, i])
            scen = r["scenario"]
            sizes = [scen.eq_weight * 100, scen.fi_weight * 100]
            ax.pie(sizes, colors=[NAVY_HEX, GOLD_HEX], startangle=90,
                   wedgeprops=dict(edgecolor="white", linewidth=2))
            ax.set_title(
                f"{scen.name} — {int(scen.eq_weight*100)}% / {int(scen.fi_weight*100)}%",
                fontsize=10, color=NAVY_HEX, fontweight="bold", pad=8,
            )
            ax.text(0, 0.10, f"{int(scen.eq_weight*100)}%", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
            ax.text(0, -0.20, "Equity", ha="center", va="center",
                    fontsize=8, color="white")

        legend_elems = [Patch(color=NAVY_HEX, label="Equity"),
                        Patch(color=GOLD_HEX, label="Fixed Income")]
        fig.legend(handles=legend_elems, loc="lower center", ncol=2,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02))
        # NOTE: deliberately NO tight_layout/constrained_layout call here —
        # GridSpec margins are set explicitly above.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf

    # ---------------- Dual-row layout (glide paths present) ------------------
    # Top row: accumulation; bottom row: retirement; arrows in between for
    # glide scenarios. Heights are biased toward the pies so the arrow band
    # in the middle is a thin visual divider.
    fig = plt.figure(figsize=(9.0, 5.7), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=n,
        left=0.04, right=0.96, top=0.92, bottom=0.13,
        hspace=0.85, wspace=0.30,
        height_ratios=[1, 1],
    )

    def _draw_pie(ax, eq_pct, fi_pct, label_eq=True):
        """Render one pie with equity centered label."""
        sizes = [eq_pct, fi_pct]
        ax.pie(sizes, colors=[NAVY_HEX, GOLD_HEX], startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2),
               radius=0.95)
        if label_eq:
            ax.text(0, 0.10, f"{int(round(eq_pct))}%",
                    ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
            ax.text(0, -0.20, "Equity", ha="center", va="center",
                    fontsize=8, color="white")
        ax.set_aspect("equal")

    def _draw_no_change_placeholder(ax):
        """Italic '(no change)' centered in the cell — for static scenarios."""
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.text(0, 0.0, "(no change)", ha="center", va="center",
                fontsize=11, color=TEXT_MED_HEX, style="italic")

    for i, r in enumerate(results):
        scen = r["scenario"]

        # ----- Top row: accumulation pie + title above -----
        ax_top = fig.add_subplot(gs[0, i])
        _draw_pie(ax_top, scen.eq_weight * 100, scen.fi_weight * 100)
        if scen.has_glide_path:
            top_label = (f"{scen.name}\nAccumulation — "
                         f"{int(scen.eq_weight*100)}/{int(scen.fi_weight*100)}")
        else:
            top_label = (f"{scen.name} — "
                         f"{int(scen.eq_weight*100)}/{int(scen.fi_weight*100)}\n"
                         f"All Years")
        ax_top.set_title(top_label, fontsize=10, color=NAVY_HEX,
                         fontweight="bold", pad=8)

        # ----- Bottom row: retirement pie OR placeholder -----
        # Critical: the LABEL goes BELOW the pie, not above (no set_title call).
        # This keeps the space above the bottom pie clear so the glide arrow
        # can terminate directly on top of the bottom pie without colliding
        # with text. For static scenarios we just show "(no change)" centered
        # in the cell — no second label.
        ax_bot = fig.add_subplot(gs[1, i])
        if scen.has_glide_path:
            ret_eq = float(scen.retirement_eq_weight) * 100
            ret_fi = (1.0 - float(scen.retirement_eq_weight)) * 100
            _draw_pie(ax_bot, ret_eq, ret_fi)
            # Label goes BELOW the pie (using ax.text in axes coords).
            ax_bot.text(
                0, -1.20,
                f"Retirement (Yr {scen.distribution_start_year}+) — "
                f"{int(round(ret_eq))}/{int(round(ret_fi))}",
                ha="center", va="top",
                fontsize=9.5, color=TEXT_DARK_HEX, fontweight="bold",
                transform=ax_bot.transData,
            )
        else:
            _draw_no_change_placeholder(ax_bot)

        # ----- Glide-path arrow between the rows (figure coords) ---
        # Goes from below the TOP pie to above the BOTTOM pie. Because we
        # moved the bottom label BELOW the pie, the area above the bottom
        # pie is clear and the arrow lands on it cleanly.
        if scen.has_glide_path:
            top_bbox = ax_top.get_position()
            bot_bbox = ax_bot.get_position()
            x_center = (top_bbox.x0 + top_bbox.x1) / 2
            # Start just below the top axis (top pie sits in upper part of axis,
            # so leave some clearance below the pie).
            y_arrow_top = top_bbox.y0 + 0.005
            # End just above the bottom axis (top of bottom pie).
            y_arrow_bot = bot_bbox.y1 - 0.005
            fig.add_artist(
                plt.matplotlib.patches.FancyArrowPatch(
                    (x_center, y_arrow_top),
                    (x_center, y_arrow_bot),
                    transform=fig.transFigure,
                    arrowstyle="-|>", mutation_scale=18,
                    color=GOLD_HEX, linewidth=2.0, alpha=0.95,
                )
            )

    # Single bottom legend
    legend_elems = [Patch(color=NAVY_HEX, label="Equity"),
                    Patch(color=GOLD_HEX, label="Fixed Income")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.02))

    # GridSpec margins were set explicitly above. NO tight_layout() call —
    # it would reset our margins and warn about the FancyArrowPatch
    # annotations falling outside its computed bbox.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white")
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
        title="Cashflow Portfolio Analysis — Becker Capital Management",
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

    def _alloc_str(s: Scenario) -> str:
        """e.g. '60/40' for static, '80/20→60/40' for glide path."""
        base = f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)}"
        if s.has_glide_path:
            ret = (f"{int(s.retirement_eq_weight*100)}/"
                   f"{int((1.0-s.retirement_eq_weight)*100)}")
            return f"{base}→{ret}"
        return base

    eq_strs = " • ".join([_alloc_str(s) for s in inputs.scenarios])
    dist_strs = ", ".join([f"${int(s.annual_distribution/1000)}K"
                           for s in inputs.scenarios])
    # Use retirement-phase blended μ for glide scenarios (the regime that
    # holds for the majority of a typical horizon).
    def _scen_mu(s: Scenario) -> float:
        if s.has_glide_path:
            return blended_params(s.retirement_eq_weight,
                                  1.0 - s.retirement_eq_weight,
                                  inputs.return_assumptions)[0]
        return blended_params(s.eq_weight, s.fi_weight,
                              inputs.return_assumptions)[0]
    mu_strs = [_scen_mu(s) for s in inputs.scenarios]
    any_glide = any(s.has_glide_path for s in inputs.scenarios)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"{inputs.horizon_years}-YEAR PROJECTION  •  "
        f"{n_scen} ALLOCATION{'S' if n_scen > 1 else ''}  •  "
        f"${inputs.initial/1e6:.1f}M",
        H_TAGLINE,
    ))
    story.append(Paragraph("Cashflow Portfolio<br/>Analysis Report", H_TITLE))
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
        f"This probabilistic analysis is a mathematical process used to implement complex statistical "
        f"methods that chart the probability of certain financial outcomes at certain times in "
        f"the future. This charting is accomplished by generating {inputs.n_paths:,} possible "
        f"economic scenarios. Each scenario randomly draws return data from a normal "
        f"distribution based on the means and standard deviations specified in the "
        f"assumption set ({inputs.return_assumptions.label}).",
        f"The simulation uses {inputs.n_paths:,} scenarios to determine the "
        f"probability of outcomes resulting from the asset allocation choices and underlying "
        "return and volatility assumptions. Some scenarios will assume very favorable financial "
        "market returns; some will conform to the worst periods in investing history; most will "
        "fall somewhere in between.",
        "<b>IMPORTANT:</b> The projections generated by this simulation are "
        "hypothetical in nature, do not reflect actual investment results, and are not "
        "guarantees of future results. Results may vary with each use and over time. This "
        "report is prepared by Becker Capital Management for informational purposes only.",
    ]
    for p in disclaim:
        story.append(Paragraph(p, P_DISCLAIM))
    story.append(PageBreak())

    # ==================== EXEC SUMMARY + RETURN ASSUMPTIONS ====================
    story.append(Paragraph("Portfolio Cashflow Analysis", H_SECTION))
    story.append(Paragraph(
        f"{inputs.horizon_years}-Year Scenario Analysis — "
        f"{n_scen} Allocation Strateg{'ies' if n_scen > 1 else 'y'}",
        P_KICKER,
    ))

    # Top fact strip — use variable column widths because the equity column
    # may carry a long glide notation like "60/40 • 80/20→60/40 • 80/20".
    # Total available: PAGE_W - 1.2*inch (margins) = 7.3 inch.
    fact_data = [
        [f"${inputs.initial/1e6:.1f}M",
         eq_strs,
         dist_label,
         inputs.distribution_frequency,
         f"{inputs.horizon_years} Years",
         f"{inputs.inflation*100:.2f}%"],
        ["Initial Investment", "Equity / Fixed Income", "Annual Distribution",
         "Distribution Frequency", "Time Horizon", "Inflation"],
    ]
    # Give the equity column more room when glide path notation is in play
    if any_glide:
        fact_widths = [0.95 * inch, 2.10 * inch, 1.20 * inch,
                       1.15 * inch, 0.95 * inch, 0.95 * inch]
    else:
        fact_widths = [1.15 * inch] * 6
    fact_tbl = Table(fact_data, colWidths=fact_widths)
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
        f"<b>{_alloc_str(s)} ({s.name})</b>"
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
    method_text = (
        f"Expected returns and volatility are derived from the "
        f"<b>{ra.label}</b> assumption set: equity mean {ra.eq_mu*100:.2f}% "
        f"(σ = {ra.eq_sigma*100:.2f}%), fixed income mean {ra.fi_mu*100:.2f}% "
        f"(σ = {ra.fi_sigma*100:.2f}%). Per-period returns are drawn from a normal "
        f"distribution parameterized to those annual figures, with asset classes "
        f"treated as independent."
    )

    # Optional glide-path paragraph
    glide_text = ""
    if any_glide:
        glide_scens = [s for s in inputs.scenarios if s.has_glide_path]
        # Most reports have just one glide scenario, but handle the general case
        glide_descs = "; ".join(
            f"{s.name} rebalances from "
            f"{int(s.eq_weight*100)}% equity / {int(s.fi_weight*100)}% fixed income "
            f"during accumulation to "
            f"{int(s.retirement_eq_weight*100)}% equity / "
            f"{int((1-s.retirement_eq_weight)*100)}% fixed income at the start of "
            f"the distribution phase (Year {s.distribution_start_year})"
            for s in glide_scens
        )
        glide_text = (
            "<br/><br/>"
            "<b>Glide path:</b> One or more scenarios use a glide-path allocation that "
            "shifts from a more aggressive accumulation-phase mix to a more conservative "
            "retirement-phase mix when distributions begin. The transition is sharp — "
            "the portfolio is rebalanced once at the start of the retirement phase. "
            f"Specifically, {glide_descs}. This pattern reflects the common practice "
            "of de-risking around retirement to reduce sequence-of-returns risk during "
            "the early withdrawal years."
        )

    exec_text = (
        f"This report presents a {inputs.horizon_years}-year simulation for a "
        f"${inputs.initial:,.0f} portfolio, evaluating {n_scen} allocation "
        f"strateg{'ies' if n_scen > 1 else 'y'} — {scen_descs}. "
        f"Distributions are paid {inputs.distribution_frequency.lower()}, escalating "
        f"{inputs.inflation*100:.1f}% annually to maintain real purchasing power. "
        f"{contrib_phase_text}"
        f"{glide_text}"
        f"<br/><br/>"
        f"{method_text} "
        f"The simulation runs {inputs.n_paths:,} independent paths per scenario."
    )
    story.append(Paragraph(exec_text, P_BODY))

    # Return assumptions table
    story.extend(section_header(f"Return Assumptions — {ra.label}"))
    # For glide scenarios, use retirement weights for the displayed blend
    # (the regime that holds for the majority of the horizon).
    def _blend_for_display(s: Scenario):
        if s.has_glide_path:
            return blended_params(s.retirement_eq_weight,
                                  1.0 - s.retirement_eq_weight, ra)
        return blended_params(s.eq_weight, s.fi_weight, ra)
    blended_data = [_blend_for_display(s) for s in inputs.scenarios]
    header = ["Parameter", "Equity", "Fixed Income"] + [
        f"{_alloc_str(s)} Blend" for s in inputs.scenarios
    ]

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
    story.extend(section_header("Scenario Comparison"))
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
    def _scen_col_header(s: Scenario) -> str:
        if s.has_glide_path:
            return (f"{s.name}\n({int(s.eq_weight*100)}/{int(s.fi_weight*100)} → "
                    f"{int(s.retirement_eq_weight*100)}/"
                    f"{int((1-s.retirement_eq_weight)*100)})")
        return (f"{s.name}\n({int(s.eq_weight*100)}% / {int(s.fi_weight*100)}%)")
    summary_header = ["Metric"] + [_scen_col_header(r['scenario']) for r in results]

    rows = []

    # Allocation rows (only shown if any scenario has a glide path — for static
    # scenarios the allocation is already in the column header).
    if any_glide:
        rows.append(
            ["Accumulation Allocation (Eq / FI)"]
            + [f"{int(r['scenario'].eq_weight*100)}% / "
               f"{int(r['scenario'].fi_weight*100)}%" for r in results]
        )
        rows.append(
            ["Retirement Allocation (Eq / FI)"]
            + [(f"{int(r['scenario'].retirement_eq_weight*100)}% / "
                f"{int((1-r['scenario'].retirement_eq_weight)*100)}%"
                if r['scenario'].has_glide_path else "(no change)")
               for r in results]
        )
        rows.append(
            ["Glide Transition Year"]
            + [(f"Yr {r['scenario'].distribution_start_year}"
                if r['scenario'].has_glide_path else "—")
               for r in results]
        )

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

    story.extend(section_header("Detailed Statistics"))
    det_header = ["Statistic"] + [
        f"{r['scenario'].name} ({_alloc_str(r['scenario'])})" for r in results
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
    if any_glide:
        alloc_intro = (
            "Each allocation maintains the same underlying asset classes — U.S. equity and "
            "U.S. fixed income — and the same distribution policy. Scenarios with a "
            "<b>glide path</b> rebalance from a more aggressive accumulation-phase "
            "allocation to a more conservative retirement-phase allocation when "
            "distributions begin. The transition is sharp — the portfolio rebalances "
            "once at the start of the retirement phase."
        )
        fig3_caption = (
            f"Figure 3 — {n_scen} allocation strateg"
            f"{'ies' if n_scen > 1 else 'y'} evaluated. Top row shows the "
            f"accumulation-phase allocation; bottom row shows the post-retirement "
            f"allocation. Gold arrows mark scenarios that rebalance at the start of "
            f"distributions."
        )
    else:
        alloc_intro = (
            "Each allocation maintains the same underlying asset classes — U.S. equity and "
            "U.S. fixed income — and the same distribution policy. The variable across scenarios "
            "is the equity weighting and/or distribution amount. Higher equity weights raise both "
            "expected return and expected volatility."
        )
        fig3_caption = (
            f"Figure 3 — {n_scen} target allocation"
            f"{'s' if n_scen > 1 else ''} evaluated."
        )
    story.append(Paragraph(alloc_intro, P_BODY))

    # Dual-pie image needs more vertical space when glide path is present.
    img_h = 3.6 * inch if any_glide else 2.3 * inch
    story.append(Image(img_alloc_buf, width=7.0 * inch, height=img_h))
    story.append(Paragraph(fig3_caption, P_FIGCAP))

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

        # Allocation header: "60% Eq / 40% FI" for static, glide-aware for glide
        if s.has_glide_path:
            alloc_header = (
                f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)} accumulation → "
                f"{int(s.retirement_eq_weight*100)}/"
                f"{int((1-s.retirement_eq_weight)*100)} retirement"
            )
            ret_text = (
                f"Blended expected return: {r['mu_a_acc']*100:.2f}% (accumulation) → "
                f"{r['mu_a_ret']*100:.2f}% (retirement)."
            )
        else:
            alloc_header = (f"{int(s.eq_weight*100)}% Equity / "
                            f"{int(s.fi_weight*100)}% Fixed Income")
            ret_text = (f"Blended expected return: {r['mu_a']*100:.2f}%; "
                        f"annual volatility: {r['sig_a']*100:.2f}%.")

        finding = (
            f"<b>{s.name} — {alloc_header}, {cf_desc}:</b> "
            f"At the median, this allocation projects a Year-{inputs.horizon_years} portfolio "
            f"value of <b>{fmt_m(r['median_yfinal'])}</b>, with a 20th–80th percentile range of "
            f"{fmt_m(r['p20_yfinal'])} to {fmt_m(r['p80_yfinal'])}. "
            f"Probability of portfolio ruin: <b>{r['p_ruin']*100:.2f}%</b>. "
            f"Probability of ending above the initial ${inputs.initial/1e6:.1f}M investment: "
            f"<b>{r['p_above_init']*100:.1f}%</b>. "
            f"{ret_text}"
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
        f"simulation: {inputs.n_paths:,} paths per scenario | "
        "Past performance is not indicative of future results. This analysis is for illustrative "
        "purposes only and does not constitute investment advice."
    )
    story.append(Paragraph(assump, P_DISCLAIM))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# Savings Goal Calculator — inverse simulation
# =============================================================================
# This section implements the *inverse* of the distribution simulator:
# instead of "given my savings, what outcomes can I expect?", it answers
# "given the retirement income I want, how much do I need to save annually
# to hit a target probability of success?"
#
# The methodology reuses the existing simulation engine (simulate_scenario)
# inside a bisection loop over candidate annual savings amounts. "Success"
# is defined as: the portfolio survives the entire retirement distribution
# period (final balance > $0) — i.e., the goal is supported.
# =============================================================================

@dataclass
class SavingsGoalScenario:
    """
    A single goal-planning scenario for the savings calculator.

    All amounts are entered in today's (Year-1) dollars. The solver inflates
    them forward at the inflation rate, identical to the distribution-phase
    semantics in the main simulator.
    """
    name: str
    years_to_retirement: int             # accumulation period
    years_in_retirement: int             # distribution period
    desired_annual_income: float         # in today's $, inflated forward
    accumulation_eq_weight: float        # 0–1, equity weight while saving
    retirement_eq_weight: float          # 0–1, equity weight in retirement
                                         #   (set equal to accumulation for static)

    @property
    def total_horizon(self) -> int:
        return self.years_to_retirement + self.years_in_retirement

    @property
    def has_glide_path(self) -> bool:
        return abs(self.accumulation_eq_weight - self.retirement_eq_weight) > 1e-9


def _build_scenario_from_goal(
    goal: SavingsGoalScenario, annual_savings: float
) -> Scenario:
    """
    Translate a SavingsGoalScenario + candidate annual savings into the
    standard Scenario shape that simulate_scenario consumes.

    This is the bridge between the two analyses: the savings calculator's
    'years_to_retirement / years_in_retirement / income target' inputs
    become the simulator's 'contribution_years / annual_contribution /
    annual_distribution' inputs.
    """
    return Scenario(
        name=goal.name,
        eq_weight=goal.accumulation_eq_weight,
        fi_weight=1.0 - goal.accumulation_eq_weight,
        annual_distribution=goal.desired_annual_income,
        contribution_years=goal.years_to_retirement,
        annual_contribution=annual_savings,
        # Glide path is engaged whenever the two weights differ.
        retirement_eq_weight=(goal.retirement_eq_weight
                              if goal.has_glide_path else None),
        retirement_fi_weight=((1.0 - goal.retirement_eq_weight)
                              if goal.has_glide_path else None),
    )


def simulate_goal_scenario(
    goal: SavingsGoalScenario,
    annual_savings: float,
    initial_savings: float,
    inflation: float,
    return_assumptions: ReturnAssumptions,
    n_paths: int = 5_000,
    seed: int = 20260501,
    seed_offset: int = 0,
) -> dict:
    """
    Run a single simulation for a (goal, savings) pair.

    Returns the same dict shape as simulate_scenario, plus:
      - "success_prob": fraction of paths where portfolio survived all
                        retirement years (i.e., final balance > 0)
      - "median_at_retirement": median portfolio balance at end of
                                accumulation phase (start of retirement)
      - "p20_at_retirement", "p80_at_retirement": same percentiles
    """
    s = _build_scenario_from_goal(goal, annual_savings)
    inputs = SimInputs(
        initial=initial_savings,
        horizon_years=goal.total_horizon,
        inflation=inflation,
        distribution_frequency="Annual",  # annual cash flow for goal-planning
        return_assumptions=return_assumptions,
        scenarios=[s],
        n_paths=n_paths,
        seed=seed,
    )
    result = simulate_scenario(s, inputs, seed_offset=seed_offset)
    # Augment with goal-specific stats
    yr_at_retirement = result["balances"][:, goal.years_to_retirement]
    yr_final = result["balances"][:, -1]
    result["success_prob"] = 1.0 - result["p_ruin"]
    result["median_at_retirement"] = float(np.median(yr_at_retirement))
    result["p20_at_retirement"] = float(np.percentile(yr_at_retirement, 20))
    result["p80_at_retirement"] = float(np.percentile(yr_at_retirement, 80))
    # Per-path probability that the final balance ended above the path's own
    # retirement-start balance — i.e., that the portfolio grew (or at least
    # didn't shrink) across the distribution phase despite withdrawals.
    # NOT compared against `initial_savings`; the user-facing question on the
    # savings-goal page is whether retirement preserves the accumulation
    # endpoint, not whether it beats the starting deposit.
    result["p_above_retirement"] = float(np.mean(yr_final > yr_at_retirement))
    result["goal"] = goal
    result["annual_savings"] = annual_savings
    result["initial_savings"] = initial_savings
    return result


def find_required_annual_savings(
    goal: SavingsGoalScenario,
    initial_savings: float,
    target_success_prob: float,
    inflation: float,
    return_assumptions: ReturnAssumptions,
    n_paths: int = 5_000,
    max_iters: int = 18,
    rel_tol: float = 0.015,
    seed: int = 20260501,
) -> dict:
    """
    Bisect over the annual savings amount to find the MINIMUM amount such
    that the simulated success probability >= target_success_prob.

    Bisection is sound here because success is monotonically non-decreasing
    in annual savings (more saved → more starting capital at retirement →
    higher survival probability), modulo sampling noise. To control
    that noise we use a fixed seed across iterations (same RNG state for
    each candidate savings) so the function is deterministic.

    Returns:
        {
          "required_savings": float,   # minimum that hits the target
          "achieved_prob":    float,   # actual P(survival) at that level
          "result":           dict,    # full simulate_goal_scenario output
          "converged":        bool,
          "iterations":       int,
          # Diagnostic: every (savings, prob) pair tried during search.
          "search_trace":     [(savings, prob), ...],
        }

    Edge cases:
      - If $0/yr already meets target (e.g., huge initial savings): return 0.
      - If even the upper bound doesn't meet target: return upper bound
        with `converged=False` so the caller can warn the user.
    """
    # Deterministic noise control: same seed for every candidate savings
    # makes prob() monotone non-decreasing in savings (sample-path-wise),
    # which is what bisection needs.
    def prob_at(savings: float) -> Tuple[float, dict]:
        r = simulate_goal_scenario(
            goal, savings, initial_savings, inflation,
            return_assumptions, n_paths=n_paths, seed=seed, seed_offset=0,
        )
        return r["success_prob"], r

    trace: List[Tuple[float, float]] = []

    # Reasonable upper bound for bisection: enough to cover even pessimistic
    # cases. 5x the desired income is a ~500% savings rate — should always
    # be sufficient unless the time horizon is degenerate.
    hi = max(5.0 * goal.desired_annual_income, 100_000.0)
    lo = 0.0

    # Edge case: $0 already works (typically if initial_savings is huge).
    p0, r0 = prob_at(0.0)
    trace.append((0.0, p0))
    if p0 >= target_success_prob:
        return {
            "required_savings": 0.0,
            "achieved_prob": p0,
            "result": r0,
            "converged": True,
            "iterations": 0,
            "search_trace": trace,
        }

    # Edge case: even the upper bound can't hit target. Likely means the
    # time horizon is too short or the desired income is too aggressive
    # relative to expected returns.
    pmax, rmax = prob_at(hi)
    trace.append((hi, pmax))
    if pmax < target_success_prob:
        return {
            "required_savings": hi,
            "achieved_prob": pmax,
            "result": rmax,
            "converged": False,
            "iterations": 1,
            "search_trace": trace,
        }

    # Bisection — narrow the bracket until it's tight (relative tol) or
    # we hit the iteration cap.
    best_meeting_result = rmax  # last result that met or exceeded target
    best_meeting_savings = hi
    best_meeting_prob = pmax

    for it in range(max_iters):
        mid = 0.5 * (lo + hi)
        p_mid, r_mid = prob_at(mid)
        trace.append((mid, p_mid))
        if p_mid >= target_success_prob:
            hi = mid
            best_meeting_result = r_mid
            best_meeting_savings = mid
            best_meeting_prob = p_mid
        else:
            lo = mid
        # Tight enough?
        if hi > 0 and (hi - lo) / hi < rel_tol:
            break

    return {
        "required_savings": best_meeting_savings,
        "achieved_prob": best_meeting_prob,
        "result": best_meeting_result,
        "converged": True,
        "iterations": len(trace) - 2,  # subtract the two boundary checks
        "search_trace": trace,
    }


def required_savings_at_confidence_levels(
    goal: SavingsGoalScenario,
    initial_savings: float,
    confidence_levels: List[float],
    inflation: float,
    return_assumptions: ReturnAssumptions,
    n_paths: int = 5_000,
    seed: int = 20260501,
) -> List[dict]:
    """
    Run find_required_annual_savings at each requested confidence level.

    Useful for the sensitivity table in the PDF: shows the user the
    cost (in additional savings) of asking for a higher success probability.
    Returns a list of result dicts, one per confidence level.
    """
    out = []
    for cl in confidence_levels:
        res = find_required_annual_savings(
            goal=goal,
            initial_savings=initial_savings,
            target_success_prob=cl,
            inflation=inflation,
            return_assumptions=return_assumptions,
            n_paths=n_paths,
            seed=seed,
        )
        res["confidence_level"] = cl
        out.append(res)
    return out


# =============================================================================
# Savings-goal charts
# =============================================================================
def chart_lifecycle_paths(
    goal_results: List[dict], inflation: float
) -> io.BytesIO:
    """
    Median lifecycle path for each goal scenario, with a vertical line
    marking the retirement year (transition from accumulation to
    distribution). 20–80 percentile bands shaded.
    """
    _setup_mpl()
    n = len(goal_results)
    max_horizon = max(r["goal"].total_horizon for r in goal_results)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)

    for r, color in zip(goal_results, SCENARIO_COLOR_HEX):
        goal = r["goal"]
        years_axis = np.arange(0, goal.total_horizon + 1)
        ax.fill_between(years_axis, r["p20_path"] / 1e6, r["p80_path"] / 1e6,
                        color=color, alpha=0.15, linewidth=0)
        label = (f"{goal.name} — Save ${r['annual_savings']/1000:,.0f}K/yr × "
                 f"{goal.years_to_retirement} yrs  →  "
                 f"${goal.desired_annual_income/1000:,.0f}K/yr × "
                 f"{goal.years_in_retirement} yrs")
        ax.plot(years_axis, r["median_path"] / 1e6, color=color,
                linewidth=2.4, label=label)
        # Vertical line marking the retirement transition for this scenario,
        # in the scenario's color but lighter.
        ax.axvline(goal.years_to_retirement, color=color, linestyle=":",
                   linewidth=1.2, alpha=0.55)

    ax.set_title(
        "Lifecycle Portfolio Value — Median & 20th–80th Percentile Bands\n"
        "(Dotted vertical line = start of retirement / distributions)",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value ($M)")
    ax.set_xlim(0, max_horizon)

    # Y-axis: scale by max p80 across scenarios
    ceiling = max(np.max(r["p80_path"]) for r in goal_results) / 1e6
    step = 20 if ceiling > 60 else 10 if ceiling > 30 else 5
    ymax = int(np.ceil(ceiling / step) * step)
    yticks = list(range(0, ymax + 1, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y}M" for y in yticks])
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_required_savings_by_confidence(
    sensitivity_results: List[List[dict]],
    scenario_names: List[str],
) -> io.BytesIO:
    """
    Bar chart: required annual savings on the y-axis, grouped by scenario,
    with side-by-side bars for each confidence level.

    sensitivity_results: list of (one per scenario) lists of result dicts
                         (one per confidence level), as returned by
                         required_savings_at_confidence_levels.
    """
    _setup_mpl()
    n_scen = len(sensitivity_results)
    confidence_levels = [r["confidence_level"] for r in sensitivity_results[0]]
    n_lvl = len(confidence_levels)

    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=180)
    bar_w = 0.8 / n_lvl
    x_centers = np.arange(n_scen)
    # Use a navy → teal → gold ramp for the confidence levels (low → high).
    colors_ramp = [NAVY_HEX, TEAL_HEX, GOLD_HEX][:n_lvl]
    if n_lvl > 3:
        colors_ramp = (colors_ramp + ["#7BA8B8", "#A09480"])[:n_lvl]

    for i, cl in enumerate(confidence_levels):
        vals = [sens[i]["required_savings"] / 1000.0
                for sens in sensitivity_results]
        offsets = x_centers - 0.4 + bar_w * (i + 0.5)
        bars = ax.bar(offsets, vals, bar_w, color=colors_ramp[i],
                      edgecolor="white", linewidth=1.2,
                      label=f"{int(cl*100)}% confidence")
        # Value labels above each bar
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    f"${v:,.0f}K", ha="center", va="bottom",
                    fontsize=8, color=TEXT_DARK_HEX)

    ax.set_title(
        "Required Annual Savings — by Scenario and Target Confidence",
        fontsize=12, pad=12,
    )
    ax.set_ylabel("Required Annual Savings ($K, today's $)")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(scenario_names)
    ax.grid(True, alpha=0.25, linestyle=":", axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    # Headroom for value labels
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(0, cur_top * 1.15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_portfolio_at_retirement_distributions(
    goal_results: List[dict],
) -> io.BytesIO:
    """
    Histograms of portfolio value at the END OF ACCUMULATION (start of
    retirement) for each scenario. Shows the size of the nest egg the
    required savings produces.
    """
    _setup_mpl()
    n = len(goal_results)
    fig, axes = plt.subplots(1, n, figsize=(11, 3.6), dpi=180)
    if n == 1:
        axes = [axes]

    # Pull "balance at retirement year" for each scenario
    nest_eggs = []
    for r in goal_results:
        goal = r["goal"]
        nest_eggs.append(r["balances"][:, goal.years_to_retirement] / 1e6)

    xmax_global = max(np.percentile(ne, 99) for ne in nest_eggs)
    xmax_global = min(xmax_global, 100)  # cap for readability

    for ax, r, ne, color in zip(axes, goal_results, nest_eggs,
                                SCENARIO_COLOR_HEX):
        median_m = float(np.median(ne))
        ax.hist(ne, bins=60, range=(0, xmax_global), color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(median_m, color="#C0392B", linestyle="--", linewidth=1.6,
                   label=f"Median: ${median_m:.1f}M")
        goal = r["goal"]
        ax.set_title(f"{goal.name}\nat Yr {goal.years_to_retirement} (retirement)",
                     fontsize=10)
        ax.set_xlabel("Portfolio Value ($M)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25, linestyle=":", axis="y")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Portfolio Value at Retirement",
        fontsize=12, color=NAVY_HEX, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# =============================================================================
# Savings-goal PDF builder
# =============================================================================
def build_savings_goal_pdf(
    goal_results: List[dict],
    sensitivity_results: List[List[dict]],
    initial_savings: float,
    target_success_prob: float,
    inflation: float,
    return_assumptions: ReturnAssumptions,
    n_paths: int,
    confidence_levels: List[float],
    prep_date: str | None = None,
) -> bytes:
    """
    Build the Becker-styled Savings Goal PDF in memory and return bytes.

    Mirrors the structure of build_pdf (cover / disclaimer / exec summary /
    return assumptions / lifecycle chart + summary / nest-egg distribution /
    sensitivity / key findings).
    """
    if prep_date is None:
        prep_date = datetime.now().strftime("%B %d, %Y")
    footer_date = "December 31, 2025"

    PAGE_W, PAGE_H = LETTER
    buf = io.BytesIO()

    # Charts
    img_paths_buf = chart_lifecycle_paths(goal_results, inflation)
    img_nest_buf = chart_portfolio_at_retirement_distributions(goal_results)
    scen_names = [r["goal"].name for r in goal_results]
    img_sens_buf = chart_required_savings_by_confidence(
        sensitivity_results, scen_names
    )

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

    # ----- Page decorations (identical to build_pdf — kept inline for simplicity) -----
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
        cx_5 = 0.85 * inch
        cx_0 = 1.75 * inch
        cy = 4.7 * inch
        canv.setFillColor(NAVY_DARK)
        canv.setFont("Helvetica-Bold", 110)
        canv.drawCentredString(cx_5, cy - 0.15 * inch, "5")
        canv.setStrokeColor(GOLD)
        canv.setLineWidth(8)
        canv.setFillColor(NAVY)
        canv.circle(cx_0, cy + 0.20 * inch, 0.62 * inch, stroke=1, fill=1)
        canv.setFillColor(GOLD)
        canv.setFont("Helvetica-Bold", 56)
        canv.drawCentredString(cx_0, cy - 0.02 * inch, "B")
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
        title="Savings Goal Calculator — Becker Capital Management",
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
    n_scen = len(goal_results)
    incomes = sorted(set(r["goal"].desired_annual_income for r in goal_results))
    yrs_to_ret = sorted(set(r["goal"].years_to_retirement for r in goal_results))
    yrs_in_ret = sorted(set(r["goal"].years_in_retirement for r in goal_results))

    if len(incomes) == 1:
        income_str = f"${incomes[0]:,.0f} / yr"
    else:
        income_str = (f"${min(incomes)/1000:,.0f}K – "
                      f"${max(incomes)/1000:,.0f}K / yr")

    if len(yrs_to_ret) == 1:
        yrs_to_ret_str = f"{yrs_to_ret[0]} years"
    else:
        yrs_to_ret_str = f"{min(yrs_to_ret)} – {max(yrs_to_ret)} years"

    if len(yrs_in_ret) == 1:
        yrs_in_ret_str = f"{yrs_in_ret[0]} years"
    else:
        yrs_in_ret_str = f"{min(yrs_in_ret)} – {max(yrs_in_ret)} years"

    def _alloc_str_goal(g: SavingsGoalScenario) -> str:
        # Use round() not int() — int(0.20*100) == 19 due to FP representation.
        eq_a = round(g.accumulation_eq_weight * 100)
        base = f"{eq_a}/{100 - eq_a}"
        if g.has_glide_path:
            eq_r = round(g.retirement_eq_weight * 100)
            return f"{base}→{eq_r}/{100 - eq_r}"
        return base

    # Collapse the allocation string when all scenarios share the same alloc.
    # Otherwise the fact strip and cover field overflow with redundant content
    # like "80/20→60/40 • 80/20→60/40 • 80/20→60/40".
    individual_alloc_strs = [_alloc_str_goal(r["goal"]) for r in goal_results]
    if len(set(individual_alloc_strs)) == 1:
        alloc_strs = individual_alloc_strs[0]
    else:
        alloc_strs = " • ".join(individual_alloc_strs)
    any_glide = any(r["goal"].has_glide_path for r in goal_results)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"SAVINGS GOAL CALCULATOR  •  "
        f"{n_scen} SCENARIO{'S' if n_scen > 1 else ''}  •  "
        f"{int(target_success_prob*100)}% TARGET CONFIDENCE",
        H_TAGLINE,
    ))
    story.append(Paragraph("Required Annual<br/>Savings Analysis", H_TITLE))
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

    cover_fields = [
        (f"${initial_savings:,.0f}", "Current Savings (today's $)"),
        (income_str, "Desired Retirement Income (today's $, escalates with inflation)"),
        (yrs_to_ret_str, "Years to Retirement (accumulation)"),
        (yrs_in_ret_str, "Years in Retirement (distribution)"),
        (alloc_strs, f"Equity / Fixed Income ({n_scen} Scenario"
                     f"{'s' if n_scen > 1 else ''})"),
        (f"{int(target_success_prob*100)}%",
         "Target Success Probability (portfolio survives all retirement years)"),
        (f"{inflation*100:.2f}%", "Inflation Rate"),
    ]
    for v, l in cover_fields:
        story.append(cover_field(v, l))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 0.25 * inch))
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
        "and illustrate potential savings concepts that may be of benefit. The purpose of "
        "the report is to illustrate how accepted financial and investment planning principles "
        "may apply to the assumptions provided.",
        "This report estimates the annual savings amount required to support a desired "
        "retirement income stream with a chosen probability of success, based on "
        "probabilistic simulation. Required savings are computed via bisection search: candidate savings "
        "amounts are evaluated by simulating the full lifecycle (accumulation + retirement) "
        "and measuring the fraction of paths in which the portfolio survives the entire "
        "distribution phase. The reported figure is the minimum annual savings amount whose "
        "success probability meets or exceeds the target.",
        "This report is based upon assumptions provided for illustrative purposes only. It "
        "does not constitute a recommendation of any particular technique or investment "
        "strategy. We recommend that you review your plan annually, or when circumstances "
        "change.",
        "Past performance is no guarantee of future performance. Actual results may differ "
        "from the projections contained in this report. The presentation of investment returns "
        "does not reflect the deduction of any commissions or advisory fees. Deduction of such "
        "charges will result in a lower required savings figure being insufficient.",
        f"This probabilistic analysis is a mathematical process used to implement complex statistical "
        f"methods that chart the probability of certain financial outcomes at certain times in "
        f"the future. This charting is accomplished by generating {n_paths:,} possible "
        f"economic scenarios. Each scenario randomly draws return data from a normal "
        f"distribution based on the means and standard deviations specified in the "
        f"assumption set ({return_assumptions.label}).",
        "<b>IMPORTANT:</b> The required savings figures generated by this "
        "simulation are hypothetical in nature, do not reflect actual investment results, and "
        "are not guarantees of future results. Results may vary with each use and over time. "
        "This report is prepared by Becker Capital Management for informational purposes only.",
    ]
    for p in disclaim:
        story.append(Paragraph(p, P_DISCLAIM))
    story.append(PageBreak())

    # ==================== EXEC SUMMARY ====================
    story.append(Paragraph("Required Savings Analysis", H_SECTION))
    story.append(Paragraph(
        f"{n_scen} Goal-Planning Scenario{'s' if n_scen > 1 else ''} — "
        f"{int(target_success_prob*100)}% Target Success Probability",
        P_KICKER,
    ))

    # Top fact strip — wrap each value cell in a Paragraph so long
    # multi-scenario allocation strings ("60/40 • 80/20→60/40 • 80/20")
    # word-wrap onto two lines rather than overflowing into adjacent cells.
    P_FACT_VAL = ParagraphStyle(
        "FactVal", fontName="Helvetica-Bold", fontSize=10, leading=12,
        textColor=NAVY, alignment=1,  # 1 = TA_CENTER
    )
    P_FACT_LBL = ParagraphStyle(
        "FactLbl", fontName="Helvetica", fontSize=7.5, leading=9,
        textColor=TEXT_MED, alignment=1,
    )
    fact_data = [
        [Paragraph(f"${initial_savings/1e6:.2f}M", P_FACT_VAL),
         Paragraph(alloc_strs, P_FACT_VAL),
         Paragraph(income_str, P_FACT_VAL),
         Paragraph(yrs_to_ret_str, P_FACT_VAL),
         Paragraph(yrs_in_ret_str, P_FACT_VAL),
         Paragraph(f"{int(target_success_prob*100)}%", P_FACT_VAL)],
        [Paragraph("Current Savings", P_FACT_LBL),
         Paragraph("Equity / Fixed Income", P_FACT_LBL),
         Paragraph("Income Target", P_FACT_LBL),
         Paragraph("Yrs to Retirement", P_FACT_LBL),
         Paragraph("Yrs in Retirement", P_FACT_LBL),
         Paragraph("Target Success", P_FACT_LBL)],
    ]
    # When ANY scenario has a glide path AND scenarios differ, the equity
    # column needs the most room; otherwise equal-width columns work fine.
    if any_glide and len(set(individual_alloc_strs)) > 1:
        fact_widths = [0.95 * inch, 2.30 * inch, 1.10 * inch,
                       1.05 * inch, 1.05 * inch, 0.85 * inch]
    else:
        fact_widths = [1.15 * inch] * 6
    fact_tbl = Table(fact_data, colWidths=fact_widths)
    fact_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, RULE_GREY),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, RULE_GREY),
    ]))
    story.append(fact_tbl)
    story.append(Spacer(1, 12))

    story.extend(section_header("Executive Summary"))

    # Build the per-scenario savings descriptor
    scen_descs = ", ".join(
        f"<b>{r['goal'].name}</b> (save ${r['annual_savings']:,.0f}/yr "
        f"× {r['goal'].years_to_retirement} yrs)"
        for r in goal_results
    )
    glide_text = ""
    if any_glide:
        glide_text = (
            "<br/><br/><b>Glide path:</b> One or more scenarios use a glide-path allocation "
            "that shifts from a more aggressive accumulation-phase mix to a more "
            "conservative retirement-phase mix at the start of distributions, reflecting "
            "the common practice of de-risking around retirement to reduce sequence-of-"
            "returns risk."
        )
    ra = return_assumptions
    method_text = (
        f"Expected returns and volatility are derived from the "
        f"<b>{ra.label}</b> assumption set: equity mean {ra.eq_mu*100:.2f}% "
        f"(σ = {ra.eq_sigma*100:.2f}%), fixed income mean {ra.fi_mu*100:.2f}% "
        f"(σ = {ra.fi_sigma*100:.2f}%). Per-period returns are drawn independently "
        f"from a normal distribution parameterized to those annual figures."
    )

    exec_text = (
        f"This report calculates the <b>minimum annual savings</b> required to support a "
        f"desired retirement income stream with a target success probability of "
        f"<b>{int(target_success_prob*100)}%</b>, evaluated across {n_scen} planning "
        f"scenario{'s' if n_scen > 1 else ''}: {scen_descs}. "
        f"<b>Success</b> is defined as the portfolio surviving the entire retirement "
        f"distribution phase (final balance &gt; $0) — i.e., the income goal is "
        f"supported throughout retirement. "
        f"All amounts are entered in today's (Year-1) dollars and inflated forward at "
        f"{inflation*100:.1f}% annually to preserve real purchasing power."
        f"{glide_text}"
        f"<br/><br/>"
        f"{method_text} "
        f"For each scenario, the simulator runs {n_paths:,} independent paths and a "
        f"bisection search identifies the minimum annual savings amount whose success "
        f"probability meets or exceeds the target. Results at additional confidence "
        f"levels are also reported as a sensitivity analysis."
    )
    story.append(Paragraph(exec_text, P_BODY))

    # ==================== HEADLINE RESULT TABLE ====================
    story.extend(section_header("Required Annual Savings — Headline Result"))
    headline_rows = [
        ["Scenario"] + [r["goal"].name for r in goal_results],
        ["Years to Retirement"]
        + [f"{r['goal'].years_to_retirement} yrs" for r in goal_results],
        ["Years in Retirement"]
        + [f"{r['goal'].years_in_retirement} yrs" for r in goal_results],
        ["Desired Annual Income (today's $)"]
        + [f"${r['goal'].desired_annual_income:,.0f}" for r in goal_results],
        ["Allocation"]
        + [_alloc_str_goal(r["goal"]) for r in goal_results],
        [f"<b>Required Annual Savings</b><br/>"
         f"<font size='8' color='{TEXT_MED_HEX}'>(to hit "
         f"{int(target_success_prob*100)}% target)</font>"]
        + [f"<b><font color='{NAVY_HEX}' size='13'>"
           f"${r['annual_savings']:,.0f}</font></b>" for r in goal_results],
        ["Achieved Success Probability"]
        + [f"{r['success_prob']*100:.1f}%" for r in goal_results],
        ["Median Portfolio at Retirement"]
        + [fmt_m(r["median_at_retirement"]) for r in goal_results],
        ["20th Pct at Retirement"]
        + [fmt_m(r["p20_at_retirement"]) for r in goal_results],
        ["80th Pct at Retirement"]
        + [fmt_m(r["p80_at_retirement"]) for r in goal_results],
    ]
    # Wrap each cell in Paragraph so HTML formatting renders
    headline_data = [[Paragraph(str(c), P_KEY_LABEL) if isinstance(c, str) else c
                      for c in row] for row in headline_rows]
    metric_w = 2.2 * inch
    scen_w = (PAGE_W - 1.2 * inch - metric_w) / max(n_scen, 1)
    headline_tbl = Table(headline_data, colWidths=[metric_w] + [scen_w] * n_scen)
    headline_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_BG]),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE_GREY),
    ]))
    story.append(headline_tbl)
    story.append(PageBreak())

    # ==================== LIFECYCLE CHART ====================
    story.extend(section_header("Lifecycle Portfolio Path"))
    story.append(Paragraph(
        f"At the required savings level for each scenario, the chart below shows the "
        f"<b>median portfolio path</b> with 20th–80th percentile bands across "
        f"{n_paths:,} simulations. The dotted vertical line for each scenario "
        f"marks the start of retirement — the point at which contributions stop and "
        f"distributions begin.",
        P_BODY,
    ))
    story.append(Image(img_paths_buf, width=7.0 * inch, height=3.7 * inch))
    story.append(Paragraph(
        "Figure 1 — Median lifecycle portfolio value with 20th–80th percentile bands.",
        P_FIGCAP,
    ))
    story.append(PageBreak())

    # ==================== NEST EGG DISTRIBUTION ====================
    story.extend(section_header("Portfolio Value at Retirement"))
    story.append(Paragraph(
        "The histograms below show the distribution of portfolio values at the start "
        "of retirement (end of accumulation) for each scenario, given the required "
        "annual savings. This is the 'nest egg' from which retirement distributions "
        "will be drawn.",
        P_BODY,
    ))
    story.append(Image(img_nest_buf, width=7.0 * inch, height=2.4 * inch))
    story.append(Paragraph(
        "Figure 2 — Portfolio value at retirement, scenario distributions.",
        P_FIGCAP,
    ))

    # ==================== SENSITIVITY ====================
    story.extend(section_header("Sensitivity to Target Confidence Level"))
    story.append(Paragraph(
        f"The required savings figure depends on how confident you want to be that the "
        f"portfolio survives the full retirement period. The chart and table below show "
        f"how the required annual savings amount changes as the target confidence level "
        f"shifts. Higher confidence demands more savings — there is a real cost to "
        f"reducing the chance of failure.",
        P_BODY,
    ))
    story.append(Image(img_sens_buf, width=7.0 * inch, height=3.0 * inch))
    story.append(Paragraph(
        "Figure 3 — Required annual savings at multiple target confidence levels.",
        P_FIGCAP,
    ))

    # Sensitivity table — rows = confidence levels, cols = scenarios
    sens_header = ["Target Confidence"] + [r["goal"].name for r in goal_results]
    sens_rows = [sens_header]
    for i, cl in enumerate(confidence_levels):
        row = [f"{int(cl*100)}%"]
        for sens in sensitivity_results:
            res = sens[i]
            note = "" if res["converged"] else " (capped)"
            row.append(f"${res['required_savings']:,.0f}/yr{note}")
        sens_rows.append(row)
    sens_tbl = Table(sens_rows,
                     colWidths=[2.2 * inch] + [scen_w] * n_scen)
    sens_tbl.setStyle(TableStyle([
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
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_BG]),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE_GREY),
    ]))
    story.append(sens_tbl)
    story.append(PageBreak())

    # ==================== KEY FINDINGS ====================
    story.extend(section_header("Key Findings"))
    for r in goal_results:
        g = r["goal"]
        eq_a = round(g.accumulation_eq_weight * 100)
        if g.has_glide_path:
            eq_r = round(g.retirement_eq_weight * 100)
            alloc_phrase = (
                f"{eq_a}/{100 - eq_a} during accumulation, glide to "
                f"{eq_r}/{100 - eq_r} in retirement"
            )
        else:
            alloc_phrase = f"static {eq_a}/{100 - eq_a} throughout"
        finding = (
            f"<b>{g.name}:</b> "
            f"To support ${g.desired_annual_income:,.0f}/yr (today's $) of retirement "
            f"income for {g.years_in_retirement} years with "
            f"<b>{int(target_success_prob*100)}% confidence</b>, the required annual "
            f"savings is <b>${r['annual_savings']:,.0f}</b> for each of the "
            f"{g.years_to_retirement} years prior to retirement. "
            f"At this savings level, the median portfolio value at retirement is "
            f"<b>{fmt_m(r['median_at_retirement'])}</b> "
            f"(20th–80th percentile range: {fmt_m(r['p20_at_retirement'])} to "
            f"{fmt_m(r['p80_at_retirement'])}). "
            f"Allocation: {alloc_phrase}. "
            f"Achieved success probability: {r['success_prob']*100:.1f}%."
        )
        story.append(Paragraph(finding, P_BODY))

    story.append(Spacer(1, 8))
    ret_text = (
        f"Equity expected return {ra.eq_mu*100:.2f}% (σ = {ra.eq_sigma*100:.2f}%); "
        f"fixed income expected return {ra.fi_mu*100:.2f}% "
        f"(σ = {ra.fi_sigma*100:.2f}%); source: {ra.label} | "
    )

    assump = (
        f"<b>Key Assumptions &amp; Disclosures:</b> Current savings ${initial_savings:,.0f} | "
        f"Income and savings amounts are entered in today's (Year-1) dollars and inflated "
        f"forward at the inflation rate to preserve real purchasing power | "
        f"Annual cash-flow frequency | Annual escalation: {inflation*100:.1f}% | "
        f"Target success probability: {int(target_success_prob*100)}% (P[portfolio "
        f"survives all retirement years] ≥ target) | "
        f"{ret_text}"
        f"Required savings determined via bisection search "
        f"(15–18 iterations, relative tolerance ~1.5%) | "
        f"No tax drag, advisory fees, or rebalancing costs modeled | "
        f"simulation: {n_paths:,} paths per candidate savings level | "
        f"Past performance is not indicative of future results. This analysis is for "
        f"illustrative purposes only and does not constitute investment advice."
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
            color: rgba({CANYON_RGB}, 0.35);
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
            background: rgba({MIDNIGHT_RGB}, 0.45);
            border-left: 4px solid var(--scenario-color);
            border-radius: 3px;
            padding: 10px 14px;
            margin-bottom: 8px;
            color: white;
            font-weight: 700;
            letter-spacing: 0.3px;
        }}

        /* ===== Live Preview scenario cards =====
           Custom HTML replaces st.metric so:
           - Dollar signs render as $ (st.metric was triggering LaTeX math
             mode on paired $..$ values, producing green monospace runs)
           - Font / size / weight stay consistent across every row
           - The distribution-amount line is visually prominent in gold */
        .becker-preview-card {{
            background: rgba({MIDNIGHT_RGB}, 0.30);
            border: 1px solid rgba({CANYON_RGB}, 0.20);
            border-left: 3px solid {GOLD_HEX};
            border-radius: 3px;
            padding: 14px 16px 10px;
            margin-bottom: 10px;
            font-family: inherit;
        }}
        .becker-preview-name {{
            color: white;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.2px;
            line-height: 1.2;
        }}
        .becker-preview-dist {{
            color: {GOLD_HEX};
            font-size: 17px;
            font-weight: 700;
            letter-spacing: 0.3px;
            margin: 4px 0 10px 0;
            line-height: 1.2;
        }}
        .becker-preview-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 8px;
            padding: 7px 0;
            border-top: 1px solid rgba({CANYON_RGB}, 0.18);
        }}
        .becker-preview-label {{
            color: #B8C4D6;
            font-size: 10.5px;
            font-weight: 500;
            letter-spacing: 0.7px;
            text-transform: uppercase;
            flex-shrink: 0;
        }}
        .becker-preview-value {{
            color: {GOLD_HEX};
            font-size: 18px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            text-align: right;
        }}

        /* Footer */
        .becker-footer {{
            margin-top: 32px;
            padding: 14px 0;
            border-top: 1px solid rgba({CANYON_RGB}, 0.3);
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
            <div class="becker-title">Cashflow Portfolio Analysis</div>
            <div class="becker-subtitle">
              Configure inputs, preview outcomes, and download a branded
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
        page_title="Becker Capital — Cashflow Portfolio Analysis",
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
        st.caption(
            "Forward-looking μ and σ are sourced from Becker Capital "
            "Management's published 2026 Capital Market Assumptions "
            "(10-year estimates)."
        )

        preset_choice = st.selectbox(
            "Preset",
            ["BCM 2026 CMAs", "Custom"],
            index=0,
        )
        if preset_choice == "BCM 2026 CMAs":
            ra = PRESETS["BCM 2026 CMAs"]
        else:
            colA, colB = st.columns(2)
            with colA:
                st.caption("**Equity**")
                eq_mu_in = st.number_input(
                    "Mean (%)", value=6.00, step=0.1,
                    key="param_eq_mu") / 100
                eq_sig_in = st.number_input(
                    "Std Dev (%)", value=17.00, step=0.1,
                    key="param_eq_sig") / 100
            with colB:
                st.caption("**Fixed Income**")
                fi_mu_in = st.number_input(
                    "Mean (%)", value=5.00, step=0.1,
                    key="param_fi_mu") / 100
                fi_sig_in = st.number_input(
                    "Std Dev (%)", value=5.50, step=0.1,
                    key="param_fi_sig") / 100
            ra = ReturnAssumptions(
                eq_mu=eq_mu_in, eq_sigma=eq_sig_in,
                fi_mu=fi_mu_in, fi_sigma=fi_sig_in,
                label="Custom",
            )
        st.caption(
            f"Eq μ = {ra.eq_mu*100:.2f}%, σ = {ra.eq_sigma*100:.2f}%  |  "
            f"FI μ = {ra.fi_mu*100:.2f}%, σ = {ra.fi_sigma*100:.2f}%"
        )

    # Path count was previously a user-facing select_slider; hard-coded now
    # for consistency across runs. 7500 keeps run time well under 5 seconds
    # for the 3-scenario case while still giving tight percentile bands.
    n_paths = 7500

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
                f"border-top:1px solid rgba({CANYON_RGB},0.3);'>"
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
                f"border-top:1px solid rgba({CANYON_RGB},0.3);'>"
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

            # ----- Glide Path (optional) -----
            # When enabled, the portfolio rebalances ONCE at the start of the
            # distribution phase to the retirement-phase equity weight. Useful
            # for de-risking around retirement to reduce sequence-of-returns
            # risk. If disabled, the equity weight stays constant for the
            # entire horizon (static allocation).
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:10px; padding-top:6px; "
                f"border-top:1px solid rgba({CANYON_RGB},0.3);'>"
                f"GLIDE PATH (OPTIONAL)</div>",
                unsafe_allow_html=True,
            )
            glide_enabled = st.checkbox(
                "Enable glide path (rebalance at retirement)",
                value=False, key=f"glide_on_{i}",
                help=(
                    "When enabled, the portfolio rebalances once at the start "
                    "of the distribution phase to a different equity weight. "
                    "Common pattern: hold a higher equity weight during "
                    "accumulation, then de-risk to a lower equity weight "
                    "when distributions begin."
                ),
            )
            # Default the retirement weight to a sensible de-risked value:
            # 20 points lower than accumulation, floored at 30%.
            ret_eq_default = max(30, eq_pct - 20)
            ret_eq_pct = st.slider(
                "Retirement Equity %", 0, 100, ret_eq_default, step=5,
                key=f"ret_eq_{i}",
                disabled=not glide_enabled,
                help=(
                    "The equity weight applied from the distribution start "
                    "year onward. Fixed income is set to 100% minus this."
                ),
            )
            if glide_enabled:
                # Confirmation caption — make the transition explicit so the
                # user can see exactly what will happen and when.
                trans_yr = int(contrib_yrs) + 1
                if abs(ret_eq_pct - eq_pct) < 1e-6:
                    st.caption(
                        ":warning: Retirement equity matches accumulation "
                        "equity — no glide will be applied. Adjust the "
                        "slider to enable a real transition."
                    )
                else:
                    st.caption(
                        f"Glide: **{eq_pct}/{100-eq_pct}** during accumulation "
                        f"→ **{ret_eq_pct}/{100-ret_eq_pct}** from Year {trans_yr} "
                        f"onward (rebalanced once)."
                    )

            scenarios.append(Scenario(
                name=name,
                eq_weight=eq_pct / 100,
                fi_weight=(100 - eq_pct) / 100,
                annual_distribution=float(dist),
                contribution_years=int(contrib_yrs),
                annual_contribution=float(contrib_amt),
                retirement_eq_weight=(ret_eq_pct / 100) if glide_enabled else None,
                retirement_fi_weight=((100 - ret_eq_pct) / 100) if glide_enabled else None,
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

    # ----- Run simulation (cached in session_state) -----
    # We previously used @st.cache_data here, but Streamlit's cache_data
    # serializes the return value via pickle and stricter validation in
    # recent Streamlit / Python combinations rejects our result dict as
    # "unserializable" even though it contains only plain types and numpy
    # arrays. Storing in st.session_state avoids the serialization round-trip
    # entirely — Python objects live in memory between reruns and we manually
    # invalidate when inputs change.
    inputs_key = repr(inputs)
    cached = st.session_state.get("_main_sim_cache")
    if cached is None or cached.get("key") != inputs_key:
        with st.spinner("Running simulation..."):
            results = run_all_simulations(inputs)
        st.session_state["_main_sim_cache"] = {
            "key": inputs_key,
            "results": results,
        }
    else:
        results = cached["results"]

    st.divider()

    # ----- Results -----
    st.subheader("Live Preview")
    metric_cols = st.columns(n_scen)
    for col, r in zip(metric_cols, results):
        s = r["scenario"]
        with col:
            if s.contribution_years > 0 and s.annual_contribution > 0:
                phase_desc = (
                    f"+&#36;{s.annual_contribution/1000:,.0f}K/yr × {s.contribution_years} yrs, "
                    f"then −&#36;{s.annual_distribution/1000:,.0f}K/yr"
                )
            else:
                phase_desc = f"−&#36;{s.annual_distribution/1000:,.0f}K/yr"

            # Glide-aware allocation label for the preview header
            if s.has_glide_path:
                alloc_label = (
                    f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)} → "
                    f"{int(s.retirement_eq_weight*100)}/"
                    f"{int((1-s.retirement_eq_weight)*100)}"
                )
            else:
                alloc_label = f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)}"

            # Use &#36; (HTML entity) for dollar signs so Streamlit's markdown
            # parser doesn't pair them up and render the middle as LaTeX math.
            median_y_str = f"&#36;{r['median_yfinal']/1e6:,.2f}M"
            band_str = (
                f"&#36;{r['p20_yfinal']/1e6:,.2f}M – "
                f"&#36;{r['p80_yfinal']/1e6:,.2f}M"
            )
            p_ruin_str = f"{r['p_ruin']*100:.2f}%"
            p_above_str = f"{r['p_above_init']*100:.1f}%"

            st.markdown(
                f"""
                <div class="becker-preview-card">
                  <div class="becker-preview-name">{s.name} — {alloc_label}</div>
                  <div class="becker-preview-dist">{phase_desc}</div>
                  <div class="becker-preview-row">
                    <div class="becker-preview-label">Median Yr {horizon}</div>
                    <div class="becker-preview-value">{median_y_str}</div>
                  </div>
                  <div class="becker-preview-row">
                    <div class="becker-preview-label">20th–80th Pct</div>
                    <div class="becker-preview-value">{band_str}</div>
                  </div>
                  <div class="becker-preview-row">
                    <div class="becker-preview-label">Probability of Ruin</div>
                    <div class="becker-preview-value">{p_ruin_str}</div>
                  </div>
                  <div class="becker-preview-row">
                    <div class="becker-preview-label">P(Exceeds Initial)</div>
                    <div class="becker-preview-value">{p_above_str}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
    # Streamlit multi-page navigation via st.navigation(). This explicitly
    # labels each page in the sidebar — without it, the home page would
    # show as "app" (derived from the entry-point filename), and the only
    # way to change that label would be to rename app.py and reconfigure
    # Streamlit Cloud's main-file path.
    #
    # Using st.navigation also disables Streamlit's automatic discovery of
    # the pages/ folder, so we list every page explicitly here.
    home_page = st.Page(
        main,
        title="Plan your retirement",
        icon="📊",
        default=True,
    )
    savings_page = st.Page(
        "pages/2_Savings_Goal_Calculator.py",
        title="Savings Goal Calculator",
        icon="🎯",
    )
    st.navigation([home_page, savings_page]).run()
