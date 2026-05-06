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
from typing import List, Tuple

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
# Data classes for inputs
# =============================================================================
@dataclass
class ReturnAssumptions:
    eq_mu: float          # annual mean (decimal)
    eq_sigma: float       # annual std (decimal)
    fi_mu: float
    fi_sigma: float
    label: str            # e.g., "1960–2024" or "Custom"
    worst_eq: float = -0.37
    worst_fi: float = -0.131


@dataclass
class Scenario:
    name: str
    eq_weight: float        # 0–1
    fi_weight: float        # 0–1
    annual_distribution: float


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


# Preset return assumption packages
PRESETS = {
    "1960–2024 (Becker default)": ReturnAssumptions(
        eq_mu=0.1179, eq_sigma=0.1667,
        fi_mu=0.0615, fi_sigma=0.0879,
        label="1960–2024",
        worst_eq=-0.370, worst_fi=-0.131,
    ),
    "1930–2024 (Long history)": ReturnAssumptions(
        eq_mu=0.1100, eq_sigma=0.1950,
        fi_mu=0.0520, fi_sigma=0.0760,
        label="1930–2024",
        worst_eq=-0.430, worst_fi=-0.131,
    ),
    "Forward-looking (Conservative)": ReturnAssumptions(
        eq_mu=0.0800, eq_sigma=0.1500,
        fi_mu=0.0450, fi_sigma=0.0700,
        label="Forward-looking",
        worst_eq=-0.370, worst_fi=-0.131,
    ),
}


FREQ_TO_PER_YEAR = {"Annual": 1, "Quarterly": 4, "Monthly": 12}


# =============================================================================
# Simulation
# =============================================================================
def blended_params(eq_w: float, fi_w: float, ra: ReturnAssumptions) -> Tuple[float, float]:
    """Annual mean and std, assuming zero correlation between asset classes."""
    mu = eq_w * ra.eq_mu + fi_w * ra.fi_mu
    sig = np.sqrt((eq_w * ra.eq_sigma) ** 2 + (fi_w * ra.fi_sigma) ** 2)
    return mu, sig


def simulate_scenario(scen: Scenario, inputs: SimInputs, seed_offset: int) -> dict:
    """
    Run Monte Carlo for a single scenario.
    - Distribution paid at the start of each period (annual/quarterly/monthly).
    - Annual distribution escalates by inflation.
    - Per-period return drawn from N(mu/k, sig/sqrt(k)) where k = periods/year.
    Returns balances of shape (N_PATHS, horizon_years + 1) — year-end values.
    """
    rng = np.random.default_rng(inputs.seed + seed_offset)
    mu_a, sig_a = blended_params(scen.eq_weight, scen.fi_weight, inputs.return_assumptions)
    k = FREQ_TO_PER_YEAR[inputs.distribution_frequency]
    mu_p = mu_a / k
    sig_p = sig_a / np.sqrt(k)

    n = inputs.n_paths
    yrs = inputs.horizon_years
    yearly = np.zeros((n, yrs + 1))
    yearly[:, 0] = inputs.initial
    bal = np.full(n, float(inputs.initial))

    for y in range(1, yrs + 1):
        annual_dist = scen.annual_distribution * (1 + inputs.inflation) ** (y - 1)
        per_period = annual_dist / k
        for _ in range(k):
            bal = np.maximum(bal - per_period, 0.0)
            r = rng.normal(mu_p, sig_p, size=n)
            bal = np.maximum(bal * (1 + r), 0.0)
        yearly[:, y] = bal

    yr_final = yearly[:, -1]
    yr10 = yearly[:, min(10, yrs)]
    yr20 = yearly[:, min(20, yrs)]

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
    }


def run_all_simulations(inputs: SimInputs) -> List[dict]:
    return [simulate_scenario(s, inputs, i * 1000) for i, s in enumerate(inputs.scenarios)]


def total_distributed(scen: Scenario, inflation: float, years: int) -> float:
    return sum(scen.annual_distribution * (1 + inflation) ** y for y in range(years))


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
        canv.rect(0, 0, PAGE_W, 0.45 * inch, stroke=0, fill=1)
        canv.setFillColor(NAVY_DARK)
        canv.setFont("Helvetica-Bold", 9)
        canv.drawString(0.4 * inch, 0.18 * inch, "BECKER CAPITAL MANAGEMENT")
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Oblique", 7.5)
        canv.drawRightString(
            PAGE_W - 0.4 * inch, 0.18 * inch,
            "This report is hypothetical and for illustrative purposes only. Not investment advice.",
        )
        # 50-B monogram
        cx = 1.25 * inch
        cy = 4.7 * inch
        canv.setFillColor(NAVY_DARK)
        canv.setFont("Helvetica-Bold", 90)
        canv.drawCentredString(cx - 0.45 * inch, cy, "50")
        canv.setStrokeColor(GOLD)
        canv.setLineWidth(4)
        canv.setFillColor(NAVY_DARK)
        canv.circle(cx + 0.55 * inch, cy + 0.35 * inch, 0.55 * inch, stroke=1, fill=1)
        canv.setFillColor(GOLD)
        canv.setFont("Helvetica-Bold", 56)
        canv.drawCentredString(cx + 0.55 * inch, cy + 0.05 * inch, "B")
        # Established + URL
        canv.setFillColor(GOLD)
        canv.setFont("Helvetica-Bold", 10)
        canv.drawCentredString(1.25 * inch, 3.5 * inch, "Established in 1976")
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica", 9)
        canv.drawCentredString(1.25 * inch, 3.25 * inch, "BECKERCAP.COM")
        canv.drawCentredString(1.25 * inch, 3.07 * inch, "503.223.1720")
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

    cover_fields = [
        (f"${inputs.initial:,.0f}", "Initial Investment"),
        (eq_strs, f"Equity / Fixed Income ({n_scen} Scenario{'s' if n_scen > 1 else ''})"),
        (dist_label, f"Annual Distribution (paid {inputs.distribution_frequency.lower()}, "
                     f"+{inputs.inflation*100:.1f}% / yr)"),
        (f"{min(mu_strs)*100:.2f}% – {max(mu_strs)*100:.2f}%"
         if n_scen > 1 else f"{mu_strs[0]*100:.2f}%",
         f"Blended Expected Return ({inputs.return_assumptions.label})"),
        (f"{inputs.horizon_years} Years", "Time Horizon"),
        (f"{inputs.inflation*100:.2f}%", "Inflation Rate (Distributions)"),
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
        f"economic scenarios. Each scenario randomly draws return data from a distribution "
        f"based on means and standard deviations for equity and fixed income asset classes "
        f"({inputs.return_assumptions.label}).",
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
    exec_text = (
        f"This report presents a {inputs.horizon_years}-year Monte Carlo simulation for a "
        f"${inputs.initial:,.0f} portfolio, evaluating {n_scen} Equity / Fixed Income "
        f"allocation strateg{'ies' if n_scen > 1 else 'y'} — {scen_descs}. "
        f"Distributions are paid {inputs.distribution_frequency.lower()}, escalating "
        f"{inputs.inflation*100:.1f}% annually to maintain real purchasing power. "
        f"<br/><br/>"
        f"Expected returns and volatility are derived from the <b>{inputs.return_assumptions.label}</b> "
        f"assumption set: equity mean {inputs.return_assumptions.eq_mu*100:.2f}% "
        f"(σ = {inputs.return_assumptions.eq_sigma*100:.2f}%), fixed income mean "
        f"{inputs.return_assumptions.fi_mu*100:.2f}% "
        f"(σ = {inputs.return_assumptions.fi_sigma*100:.2f}%). "
        f"The simulation runs {inputs.n_paths:,} independent paths per scenario, with "
        f"per-period returns drawn from a normal distribution parameterized to those annual figures."
    )
    story.append(Paragraph(exec_text, P_BODY))

    # Return assumptions table
    story.extend(section_header(f"Return Assumptions — {inputs.return_assumptions.label}"))
    ra = inputs.return_assumptions
    blended_data = [(blended_params(s.eq_weight, s.fi_weight, ra)) for s in inputs.scenarios]
    header = ["Parameter", "Equity", "Fixed Income"] + [
        f"{int(s.eq_weight*100)}/{int(s.fi_weight*100)} Blend" for s in inputs.scenarios
    ]
    ra_data = [
        header,
        ["Mean Return", f"{ra.eq_mu*100:.2f}%", f"{ra.fi_mu*100:.2f}%"]
        + [f"{m*100:.2f}%" for m, _ in blended_data],
        ["Annual Std. Deviation (σ)", f"{ra.eq_sigma*100:.2f}%", f"{ra.fi_sigma*100:.2f}%"]
        + [f"{s*100:.2f}%" for _, s in blended_data],
        ["Worst Calendar Year", f"{ra.worst_eq*100:.1f}%", f"{ra.worst_fi*100:.1f}%"]
        + ["—"] * n_scen,
        ["Inflation on Distributions", "—", "—"]
        + [f"{inputs.inflation*100:.2f}%"] * n_scen,
        ["Source / Period", ra.label, ra.label] + [ra.label] * n_scen,
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
    story.append(Paragraph(
        f"Each scenario was simulated across <b>{inputs.n_paths:,} independent paths</b> "
        f"over {inputs.horizon_years} years. Returns are drawn at the {inputs.distribution_frequency.lower()} "
        "frequency from normal distributions parameterized by the chosen annual return assumptions. "
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
    rows = [
        ["Annual Distribution (Yr 1)"]
        + [f"${r['scenario'].annual_distribution:,.0f}" for r in results],
        [f"Per-Period Distribution (Yr 1, {inputs.distribution_frequency})"]
        + [f"${r['scenario'].annual_distribution/FREQ_TO_PER_YEAR[inputs.distribution_frequency]:,.0f}"
           for r in results],
        ["Annual Escalation"] + [f"{inputs.inflation*100:.1f}%"] * n_scen,
        [f"Total Distributed ({inputs.horizon_years} yrs)"]
        + [fmt_m(total_distributed(r['scenario'], inputs.inflation, inputs.horizon_years))
           for r in results],
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
        finding = (
            f"<b>{s.name} — {int(s.eq_weight*100)}% Equity / {int(s.fi_weight*100)}% "
            f"Fixed Income, ${s.annual_distribution:,.0f}/yr:</b> "
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
    assump = (
        f"<b>Key Assumptions & Disclosures:</b> Initial investment ${inputs.initial:,.0f} | "
        f"Distribution frequency: {inputs.distribution_frequency} | "
        f"Annual escalation: {inputs.inflation*100:.1f}% | Time horizon: {inputs.horizon_years} years | "
        f"Equity expected return {inputs.return_assumptions.eq_mu*100:.2f}% "
        f"(σ = {inputs.return_assumptions.eq_sigma*100:.2f}%); "
        f"fixed income expected return {inputs.return_assumptions.fi_mu*100:.2f}% "
        f"(σ = {inputs.return_assumptions.fi_sigma*100:.2f}%); source: {inputs.return_assumptions.label} | "
        f"Per-period returns drawn independently from N(μ/k, σ/√k) where k = periods per year | "
        f"Distributions paid at the start of each period; returns applied to post-distribution balance | "
        f"Asset-class returns assumed uncorrelated for blended-volatility calculation | "
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

        initial = st.number_input(
            "Initial Investment ($)", min_value=100_000, max_value=1_000_000_000,
            value=5_400_000, step=100_000, format="%d",
        )
        horizon = st.slider("Time Horizon (years)", 5, 50, 30, step=1)
        inflation = st.slider("Inflation Rate (%)", 0.0, 8.0, 3.0, step=0.25) / 100
        freq = st.selectbox(
            "Distribution Frequency", ["Annual", "Quarterly", "Monthly"], index=1
        )

        st.divider()
        st.subheader("Return Assumptions")
        preset_name = st.selectbox(
            "Source", list(PRESETS.keys()) + ["Custom"], index=0
        )
        if preset_name == "Custom":
            colA, colB = st.columns(2)
            with colA:
                st.caption("**Equity**")
                eq_mu = st.number_input("Mean (%)", value=11.79, step=0.1,
                                        key="eq_mu") / 100
                eq_sig = st.number_input("Std Dev (%)", value=16.67, step=0.1,
                                         key="eq_sig") / 100
            with colB:
                st.caption("**Fixed Income**")
                fi_mu = st.number_input("Mean (%)", value=6.15, step=0.1,
                                        key="fi_mu") / 100
                fi_sig = st.number_input("Std Dev (%)", value=8.79, step=0.1,
                                         key="fi_sig") / 100
            ra = ReturnAssumptions(eq_mu, eq_sig, fi_mu, fi_sig, "Custom")
        else:
            ra = PRESETS[preset_name]
            st.caption(
                f"Equity μ = {ra.eq_mu*100:.2f}%, σ = {ra.eq_sigma*100:.2f}%  |  "
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
        "Configure 1–3 scenarios. Vary the equity allocation, the distribution amount, "
        "or both to compare strategies side by side."
    )

    n_scen = st.radio("Number of scenarios", [1, 2, 3], index=2, horizontal=True)

    default_scenarios = [
        ("Scenario A", 60, 225_000),
        ("Scenario B", 70, 225_000),
        ("Scenario C", 80, 225_000),
    ]

    scenarios: List[Scenario] = []
    cols = st.columns(n_scen)
    for i, col in enumerate(cols):
        name_def, eq_def, dist_def = default_scenarios[i]
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
            dist = st.number_input(
                "Annual Distribution ($)", min_value=0, max_value=10_000_000,
                value=dist_def, step=5_000, key=f"dist_{i}", format="%d",
            )
            scenarios.append(Scenario(
                name=name, eq_weight=eq_pct / 100,
                fi_weight=(100 - eq_pct) / 100, annual_distribution=float(dist),
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
            st.markdown(
                f"**{s.name}** — {int(s.eq_weight*100)}/{int(s.fi_weight*100)}, "
                f"${s.annual_distribution:,.0f}/yr"
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
