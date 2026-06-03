"""
Becker Capital — Savings Goal Calculator

Inverse simulation: given a desired retirement income, time horizon, and
allocation, compute the minimum annual savings required to support that
income with a chosen target probability of success.

This page lives alongside the main Cashflow Portfolio Analysis simulator
(app.py) in a Streamlit multipage app. It imports the engine + UI helpers
from app.py rather than duplicating them.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List

import streamlit as st

# Make the parent directory (where app.py lives) importable regardless of
# how Streamlit launches this script. Streamlit normally adds the entry
# point's directory to sys.path automatically; this is belt-and-suspenders.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
from app import (  # noqa: E402
    # Brand
    GOLD_HEX, NAVY_HEX, NAVY_DARK_HEX, SCENARIO_COLOR_HEX,
    MIDNIGHT_RGB, CANYON_RGB,
    # Engine — return assumptions
    BCM_CMA_2026, ReturnAssumptions,
    # Engine — shared inflows
    InflowEvent,
    # Engine — savings goal
    SavingsGoalScenario,
    find_required_annual_savings,
    required_savings_at_confidence_levels,
    build_savings_goal_pdf,
    # Charts (rendered inline on the page, identical to the PDF version)
    chart_lifecycle_paths,
    # UI helpers (already brand-styled)
    dollar_input, _inject_becker_css, _render_header, _render_footer,
)


def _render_savings_header():
    """Becker-branded header specific to the savings goal calculator."""
    st.markdown(
        """
        <div class="becker-header">
          <div class="becker-monogram">
            <div class="fifty">50</div>
            <div class="b-circle">B</div>
          </div>
          <div class="becker-header-text">
            <div class="becker-eyebrow">BECKER CAPITAL MANAGEMENT  •  EST. 1976</div>
            <div class="becker-title">Savings Goal Calculator</div>
            <div class="becker-subtitle">
              Calculate the minimum annual savings needed to support a desired
              retirement income at your chosen probability of success.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _return_assumptions_picker() -> ReturnAssumptions:
    """
    Mirror of the return-assumptions picker on the main page. Forward-looking
    μ/σ are sourced from BCM's 2026 Capital Market Assumptions.
    """
    st.caption(
        "Forward-looking μ and σ are sourced from Becker Capital "
        "Management's 2026 Capital Market Assumptions (10-year estimates)."
    )

    preset_choice = st.selectbox(
        "Preset",
        ["BCM 2026 CMAs", "Custom"],
        index=0,
        key="sg_preset",
    )
    if preset_choice == "BCM 2026 CMAs":
        # Construct the preset inline from BCM_CMA_2026 rather than reading
        # from app.PRESETS. This avoids stale-import KeyErrors on Streamlit
        # Cloud when sys.modules['app'] is cached from a previous deploy
        # where PRESETS had different keys.
        eq_mu, eq_sigma = BCM_CMA_2026["EQ_US_LARGE"]
        fi_mu, fi_sigma = BCM_CMA_2026["FI_IT_US"]
        ra = ReturnAssumptions(
            eq_mu=eq_mu, eq_sigma=eq_sigma,
            fi_mu=fi_mu, fi_sigma=fi_sigma,
            label="BCM 2026 CMAs (Large Cap + Intermediate Bonds)",
        )
    else:
        colA, colB = st.columns(2)
        with colA:
            st.caption("**Equity**")
            eq_mu = st.number_input("Mean (%)", value=6.00, step=0.1,
                                    key="sg_par_eq_mu") / 100
            eq_sig = st.number_input("Std Dev (%)", value=17.00, step=0.1,
                                     key="sg_par_eq_sig") / 100
        with colB:
            st.caption("**Fixed Income**")
            fi_mu = st.number_input("Mean (%)", value=5.00, step=0.1,
                                    key="sg_par_fi_mu") / 100
            fi_sig = st.number_input("Std Dev (%)", value=5.50, step=0.1,
                                     key="sg_par_fi_sig") / 100
        ra = ReturnAssumptions(
            eq_mu=eq_mu, eq_sigma=eq_sig,
            fi_mu=fi_mu, fi_sigma=fi_sig,
            label="Custom",
        )
    st.caption(
        f"Eq μ = {ra.eq_mu*100:.2f}%, σ = {ra.eq_sigma*100:.2f}%  |  "
        f"FI μ = {ra.fi_mu*100:.2f}%, σ = {ra.fi_sigma*100:.2f}%"
    )
    return ra


def main():
    st.set_page_config(
        page_title="Becker Capital — Savings Goal Calculator",
        page_icon="🎯",
        layout="wide",
    )
    _inject_becker_css()
    _render_savings_header()

    # Client-facing guide — opens the static HTML walkthrough in a new tab.
    # Served from ./static/ via enableStaticServing (see .streamlit/config.toml).
    st.markdown(
        f"""
        <a href="app/static/savings_guide.html" target="_blank" rel="noopener"
           style="display:inline-block;text-decoration:none;
                  background:{GOLD_HEX};color:#0C2331;font-weight:700;
                  font-size:13px;letter-spacing:0.3px;padding:9px 18px;
                  border-radius:4px;margin:4px 0 14px;">
           📖 How to read this tool — client guide (opens in a new window)
        </a>
        """,
        unsafe_allow_html=True,
    )

    # Invalidate any stale session-state results carried over from an older
    # deploy. The version tag is bumped whenever simulate_goal_scenario adds
    # a new field that the renderer expects; without this guard the page
    # crashes (or shows "—") on first load after a deploy with no input
    # change to trigger a recompute. The cache_data bump on the same key
    # handles the cache layer; this handles the session-state layer.
    _CURRENT_SCHEMA_VERSION = "v4-mid-band"
    if (st.session_state.get("sg_schema_version")
            != _CURRENT_SCHEMA_VERSION):
        for k in ("sg_results", "sg_sensitivity", "sg_inputs_snapshot",
                  "sg_pdf_bytes", "sg_pdf_filename"):
            st.session_state.pop(k, None)
        st.session_state["sg_schema_version"] = _CURRENT_SCHEMA_VERSION

    # ----- Sidebar: shared inputs -----
    with st.sidebar:
        st.header("Plan Inputs")

        initial_savings = dollar_input(
            "Current Savings ($)",
            default=250_000, key="sg_initial",
            min_value=0, max_value=500_000_000,
            help=(
                "Starting balance. Type freely — e.g., 250000, 250,000, "
                "$250K, or 1.5M."
            ),
        )
        inflation = st.slider(
            "Inflation Rate (%)", 0.0, 8.0, 3.0, step=0.25, key="sg_infl"
        ) / 100
        amount_basis = st.radio(
            "Enter income amounts as",
            ["Annual", "Monthly"], index=0, horizontal=True,
            key="sg_amount_basis",
            help=(
                "How you'd like to type the desired retirement income. 'Monthly' "
                "multiplies by 12 to get the annual figure used in the analysis — "
                "handy since many clients think in monthly income terms."
            ),
        )
        monthly_entry = (amount_basis == "Monthly")
        unit_div = 12 if monthly_entry else 1
        unit_word = "Monthly" if monthly_entry else "Annual"
        target_pct = st.slider(
            "Target Success Probability (%)", 50, 99, 80, step=1,
            key="sg_target",
            help=(
                "Probability that the portfolio survives all retirement "
                "years (final balance > $0). Higher target → more required "
                "savings."
            ),
        )
        target_success = target_pct / 100.0

        st.divider()
        st.subheader("Return Assumptions")
        ra = _return_assumptions_picker()

    # Path count was previously a user-facing select_slider; hard-coded now
    # for consistency. 2500 keeps a single Calculate click under ~60 seconds
    # for the 3-scenario case (each click runs ~12 bisection searches × ~14
    # sims each), while leaving the bisection converged to within ~1.5%.
    n_paths = 2500

    # ----- Main area: scenario builder -----
    st.subheader("Goal-Planning Scenarios")
    st.caption(
        "Configure 1–3 scenarios. Each scenario can vary the retirement age, "
        "income target, allocation, or all three. The solver computes the "
        "minimum annual savings needed for each."
    )

    n_scen = st.radio(
        "Number of scenarios", [1, 2, 3], index=2, horizontal=True,
        key="sg_n_scen",
    )

    # Defaults illustrate "compare retirement ages" — same income, different
    # accumulation periods, all glide 80→60.
    DEFAULTS = [
        ("Retire in 30 yrs", 30, 30, 200_000, 80, True,  60),
        ("Retire in 20 yrs", 20, 30, 200_000, 80, True,  60),
        ("Retire in 10 yrs", 10, 30, 200_000, 80, True,  60),
    ]

    # ----- Shared one-time / installment inflows (note receivable, etc.) -----
    # Applied identically to every scenario; they reduce the savings required.
    st.markdown(
        "##### Note / Other Income &nbsp;·&nbsp; optional, shared across all scenarios"
    )
    st.caption(
        "Model one-time or installment inflows — e.g. a note receivable, property "
        "sale, or inheritance. Amounts are in today's dollars, inflated forward, "
        "and added to every scenario. Year 1 is the first accumulation year; "
        "retirement begins after the Years-to-Retirement period."
    )
    n_inflows = st.number_input(
        "Number of income streams", min_value=0, max_value=3, value=0, step=1,
        key="sg_n_inflows", help="Set to 0 if there is no extra income to model.",
    )
    extra_inflows: List[InflowEvent] = []
    if int(n_inflows) > 0:
        icols = st.columns(int(n_inflows))
        for j, icol in enumerate(icols):
            with icol:
                in_label = st.text_input(
                    "Label", value=f"Income {j + 1}", key=f"sg_inflow_label_{j}",
                )
                in_amt = dollar_input(
                    "Annual Amount ($, today's dollars)",
                    default=50_000, key=f"sg_inflow_amt_{j}",
                    min_value=0, max_value=10_000_000,
                    help="Amount received per year (today's dollars). For a "
                         "one-time payment, set Number of Years to 1.",
                )
                in_start = st.number_input(
                    "Start Year", min_value=1, max_value=99,
                    value=1, step=1, key=f"sg_inflow_start_{j}",
                )
                in_years = st.number_input(
                    "Number of Years (1 = one-time)",
                    min_value=1, max_value=99,
                    value=1, step=1, key=f"sg_inflow_years_{j}",
                )
                extra_inflows.append(InflowEvent(
                    amount=float(in_amt),
                    start_year=int(in_start),
                    years=int(in_years),
                    label=in_label.strip() or f"Income {j + 1}",
                ))
    st.divider()

    goals: List[SavingsGoalScenario] = []
    cols = st.columns(n_scen)
    for i, col in enumerate(cols):
        (_name_def, ytr_def, yir_def, inc_def,
         eq_def, glide_def, ret_eq_def) = DEFAULTS[i]
        with col:
            # The scenario name is derived from Years to Retirement rather than
            # entered separately (the dedicated Name field was redundant). Read
            # the live value from session_state so the card title reflects the
            # current input on rerun; fall back to the default on first render.
            ytr_for_title = int(st.session_state.get(f"sg_ytr_{i}", ytr_def))
            name = f"Retire in {ytr_for_title} yrs"
            st.markdown(
                f"<div class='becker-scenario-card' "
                f"style='--scenario-color:{SCENARIO_COLOR_HEX[i]};'>"
                f"{name}</div>",
                unsafe_allow_html=True,
            )

            # ----- Time horizon -----
            ytr = st.number_input(
                "Years to Retirement",
                min_value=1, max_value=60, value=ytr_def, step=1,
                key=f"sg_ytr_{i}",
                help="Number of accumulation years (during which you save).",
            )
            # Authoritative name from the actual widget value.
            name = f"Retire in {int(ytr)} yrs"
            yir = st.number_input(
                "Years in Retirement",
                min_value=1, max_value=60, value=yir_def, step=1,
                key=f"sg_yir_{i}",
                help="Number of distribution years (during which you draw income).",
            )

            # ----- Income target -----
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:6px; padding-top:6px; "
                f"border-top:1px solid rgba({CANYON_RGB},0.3);'>"
                f"INCOME TARGET</div>",
                unsafe_allow_html=True,
            )
            income_entered = dollar_input(
                f"Desired {unit_word} Income ($, today's dollars)",
                default=int(round(inc_def / unit_div)),
                key=f"sg_inc_{i}_{'m' if monthly_entry else 'a'}",
                min_value=0, max_value=10_000_000,
                help=(
                    "Income you want to draw in retirement, in today's dollars. "
                    "Will be inflated forward."
                ),
            )
            income = income_entered * unit_div  # annualize for the model

            # ----- Allocation -----
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:8px; padding-top:6px; "
                f"border-top:1px solid rgba({CANYON_RGB},0.3);'>"
                f"ALLOCATION</div>",
                unsafe_allow_html=True,
            )
            eq_pct = st.slider(
                "Accumulation Equity %", 0, 100, eq_def, step=5,
                key=f"sg_eq_{i}",
            )
            st.caption(f"Fixed Income: **{100 - eq_pct}%**")

            # Glide path toggle
            glide_on = st.checkbox(
                "Enable glide path (de-risk at retirement)",
                value=glide_def, key=f"sg_glide_{i}",
                help=(
                    "When enabled, the portfolio rebalances once at the "
                    "start of retirement to a different equity weight. "
                    "Common pattern: higher equity while saving, lower "
                    "in retirement."
                ),
            )
            ret_default = max(30, eq_pct - 20)
            ret_eq_pct = st.slider(
                "Retirement Equity %", 0, 100,
                ret_eq_def if glide_def else ret_default,
                step=5, key=f"sg_ret_eq_{i}",
                disabled=not glide_on,
            )
            if glide_on:
                if abs(ret_eq_pct - eq_pct) < 1e-6:
                    st.caption(
                        ":warning: Retirement equity matches accumulation. "
                        "No glide will be applied."
                    )
                else:
                    st.caption(
                        f"Glide: **{eq_pct}/{100-eq_pct}** during accumulation "
                        f"→ **{ret_eq_pct}/{100-ret_eq_pct}** in retirement."
                    )

            goals.append(SavingsGoalScenario(
                name=name,
                years_to_retirement=int(ytr),
                years_in_retirement=int(yir),
                desired_annual_income=float(income),
                accumulation_eq_weight=eq_pct / 100,
                retirement_eq_weight=(ret_eq_pct / 100) if glide_on else (eq_pct / 100),
            ))

    # ----- Calculate button -----
    st.divider()
    colp1, colp2 = st.columns([1, 3])
    with colp1:
        run_clicked = st.button(
            "🎯 Calculate Required Savings",
            type="primary", use_container_width=True,
        )

    # We cache the result keyed on the inputs. This way the user can
    # tweak the PDF or re-render charts without re-solving.
    # NOTE: the `key` argument MUST NOT have a leading underscore. Streamlit
    # treats leading-underscore parameters as "don't hash this" — that would
    # silently break cache invalidation when goals change.
    @st.cache_data(show_spinner=False)
    def _solve_all(key: str, initial_savings: float,
                   target_success: float, inflation: float,
                   ra_repr: str, n_paths: int):
        # `key` and `ra_repr` exist purely to participate in the cache hash.
        # The real Scenario objects come from the `goals` closure, which is
        # rebuilt fresh on every Streamlit rerun from the current widget state.
        results = []
        sensitivity = []
        for g in goals:
            res = find_required_annual_savings(
                goal=g, initial_savings=initial_savings,
                target_success_prob=target_success, inflation=inflation,
                return_assumptions=ra, extra_inflows=extra_inflows,
                n_paths=n_paths,
            )
            sens = required_savings_at_confidence_levels(
                goal=g, initial_savings=initial_savings,
                confidence_levels=[0.70, 0.80, 0.90],
                inflation=inflation, return_assumptions=ra,
                extra_inflows=extra_inflows, n_paths=n_paths,
            )
            results.append(res)
            sensitivity.append(sens)
        return results, sensitivity

    if run_clicked:
        # The version tag is bumped whenever the result dict gains new fields,
        # so stale @st.cache_data entries from older deploys (which lack the
        # new field) don't satisfy a cache hit and crash the renderer.
        # Bump this any time simulate_goal_scenario adds/renames a field.
        _RESULT_SCHEMA_VERSION = "v4-mid-band"
        cache_key = repr((
            _RESULT_SCHEMA_VERSION,
            tuple((g.name, g.years_to_retirement, g.years_in_retirement,
                   g.desired_annual_income, g.accumulation_eq_weight,
                   g.retirement_eq_weight) for g in goals),
            tuple((ev.label, ev.amount, ev.start_year, ev.years)
                  for ev in extra_inflows),
            initial_savings, target_success, inflation,
            ra.label, ra.eq_mu, ra.fi_mu, n_paths,
        ))
        # Show progress: 3 scenarios × (1 headline + 3 sensitivity) = ~12
        # bisection runs at 12–18 sims each. Display a spinner with hint.
        n_sols = n_scen * 4
        approx_sims = n_sols * 14
        with st.spinner(
            f"Running ~{approx_sims:,} simulations across "
            f"{n_sols} bisection searches… (this can take 20–60 seconds)"
        ):
            results, sensitivity = _solve_all(
                cache_key, initial_savings, target_success,
                inflation, ra.label, n_paths,
            )
        st.session_state["sg_results"] = results
        st.session_state["sg_sensitivity"] = sensitivity
        st.session_state["sg_inputs_snapshot"] = {
            "initial_savings": initial_savings,
            "target_success_prob": target_success,
            "inflation": inflation,
            "return_assumptions": ra,
            "n_paths": n_paths,
            "confidence_levels": [0.70, 0.80, 0.90],
            "goals": goals,
            "extra_inflows": extra_inflows,
        }

    # ----- Results display -----
    if "sg_results" in st.session_state:
        results = st.session_state["sg_results"]
        sensitivity = st.session_state["sg_sensitivity"]
        snapshot = st.session_state["sg_inputs_snapshot"]

        st.subheader("Required Savings — Headline Result")
        st.caption(
            "Each card shows the bisection-solved required savings at your "
            "target probability, the achieved success and median portfolio "
            "value at that level, plus the same answer at 70/80/90% "
            "confidence for comparison."
        )
        conf_levels_pct = [
            int(round(c * 100)) for c in snapshot["confidence_levels"]
        ]

        metric_cols = st.columns(len(results))
        for col, res, sens, g in zip(metric_cols, results, sensitivity,
                                      snapshot["goals"]):
            with col:
                if g.has_glide_path:
                    eq_a = round(g.accumulation_eq_weight * 100)
                    eq_r = round(g.retirement_eq_weight * 100)
                    glide_str = (f" • {eq_a}/{100-eq_a} → "
                                 f"{eq_r}/{100-eq_r}")
                else:
                    eq_a = round(g.accumulation_eq_weight * 100)
                    glide_str = f" • {eq_a}/{100-eq_a} static"

                phase_desc = (
                    f"{g.years_to_retirement}yr accum / "
                    f"{g.years_in_retirement}yr ret{glide_str}"
                )

                # Use &#36; for $ so Streamlit's markdown parser doesn't pair
                # them up and render the middle as LaTeX math (the bug that
                # produced green monospace runs on the main page).
                required_str = (
                    f"&#36;{res['required_savings']:,.0f}/yr"
                    + ("" if res["converged"] else " (capped)")
                )
                achieved_str = f"{res['achieved_prob']*100:.1f}%"
                median_str = (
                    f"&#36;{res['result']['median_at_retirement']/1e6:,.2f}M"
                )
                # Mirrors the two headline risk metrics from the main Monte
                # Carlo page. p_above_retirement compares each path's final
                # balance against THAT path's retirement-start balance — i.e.,
                # "did retirement preserve the accumulation endpoint", not
                # "did it beat the starting deposit".
                #
                # Computed inline (rather than read from a result dict field)
                # so it ALWAYS works regardless of which deploy's cached
                # entry produced the result. simulate_scenario has always
                # returned balances; everything we need is there.
                balances = res['result'].get('balances')
                if balances is not None:
                    yr_ret = balances[:, g.years_to_retirement]
                    yr_final = balances[:, -1]
                    p_above_ret_val = float((yr_final > yr_ret).mean())
                    p_above_ret_str = f"{p_above_ret_val*100:.1f}%"
                else:
                    p_above_ret_str = "—"

                # Inline styles back up the .becker-preview-* classes in
                # _inject_becker_css(). Belt-and-suspenders against any
                # markdown-indentation / class-stripping quirks — the previous
                # iteration rendered plain white text on this page even though
                # the same classes worked on the main page.
                card_css = (
                    f"background:rgba({MIDNIGHT_RGB},0.30);"
                    f"border:1px solid rgba({CANYON_RGB},0.20);"
                    f"border-left:3px solid {GOLD_HEX};"
                    f"border-radius:3px;padding:14px 16px 10px;"
                    f"margin-bottom:10px;font-family:inherit;"
                )
                name_css = (
                    "color:white;font-size:15px;font-weight:700;"
                    "letter-spacing:0.2px;line-height:1.2;"
                )
                dist_css = (
                    f"color:{GOLD_HEX};font-size:17px;font-weight:700;"
                    f"letter-spacing:0.3px;margin:4px 0 10px 0;line-height:1.2;"
                )
                row_css = (
                    f"display:flex;justify-content:space-between;"
                    f"align-items:baseline;gap:8px;padding:7px 0;"
                    f"border-top:1px solid rgba({CANYON_RGB},0.18);"
                )
                label_css = (
                    "color:#B8C4D6;font-size:10.5px;font-weight:500;"
                    "letter-spacing:0.7px;text-transform:uppercase;"
                    "flex-shrink:0;"
                )
                value_css = (
                    f"color:{GOLD_HEX};font-size:18px;font-weight:700;"
                    f"font-variant-numeric:tabular-nums;text-align:right;"
                )
                plan_value_css = value_css + "font-size:13px;font-weight:600;"

                # Sub-section header introducing the rolled-in confidence
                # breakdown (formerly its own standalone table).
                sens_header_css = (
                    f"color:{GOLD_HEX};font-size:10px;font-weight:700;"
                    f"letter-spacing:1.2px;text-transform:uppercase;"
                    f"margin:14px 0 0 0;padding:8px 0 2px 0;"
                    f"border-top:1px solid rgba({CANYON_RGB},0.35);"
                )
                # Build one row per confidence level (70 / 80 / 90 by default).
                # All rows formatted identically — the headline number at the
                # top of the card already shows the answer at the user's target
                # confidence, so an extra row-level highlight is redundant.
                sens_val_css = value_css + "font-size:15px;"
                sens_rows_html = ""
                for s, conf_pct in zip(sens, conf_levels_pct):
                    sens_val = (
                        f"&#36;{s['required_savings']:,.0f}/yr"
                        + ("" if s["converged"] else " (capped)")
                    )
                    sens_rows_html += (
                        f'<div class="becker-preview-row" style="{row_css}">'
                        f'<div class="becker-preview-label" style="{label_css}">{conf_pct}% Confidence</div>'
                        f'<div class="becker-preview-value" style="{sens_val_css}">{sens_val}</div>'
                        f'</div>'
                    )

                # HTML written with no leading whitespace so the markdown
                # parser never treats a line as an indented code block.
                card_html = (
                    f'<div class="becker-preview-card" style="{card_css}">'
                    f'<div class="becker-preview-name" style="{name_css}">{g.name}</div>'
                    f'<div class="becker-preview-dist" style="{dist_css}">{required_str}</div>'
                    f'<div class="becker-preview-row" style="{row_css}">'
                    f'<div class="becker-preview-label" style="{label_css}">Plan</div>'
                    f'<div class="becker-preview-value" style="{plan_value_css}">{phase_desc}</div>'
                    f'</div>'
                    f'<div class="becker-preview-row" style="{row_css}">'
                    f'<div class="becker-preview-label" style="{label_css}">Chance of Success</div>'
                    f'<div class="becker-preview-value" style="{value_css}">{achieved_str}</div>'
                    f'</div>'
                    f'<div class="becker-preview-row" style="{row_css}">'
                    f'<div class="becker-preview-label" style="{label_css}">Median at Retirement</div>'
                    f'<div class="becker-preview-value" style="{value_css}">{median_str}</div>'
                    f'</div>'
                    f'<div class="becker-preview-row" style="{row_css}">'
                    f'<div class="becker-preview-label" style="{label_css}">P(End &gt; Ret-Start)</div>'
                    f'<div class="becker-preview-value" style="{value_css}">{p_above_ret_str}</div>'
                    f'</div>'
                    f'<div style="{sens_header_css}">Required Savings by Confidence</div>'
                    f'{sens_rows_html}'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if not res["converged"]:
                    st.warning(
                        "⚠️ Bisection capped at the upper bound. The target "
                        "may be infeasible for this scenario — consider a "
                        "longer horizon, lower income, or higher equity."
                    )

        # ----- Lifecycle chart (same renderer used in the PDF report) -----
        # Visualizes the median portfolio path across both phases for each
        # scenario at the required-savings level, with 20th–80th percentile
        # bands shaded. Mirrors the path chart on the main Cashflow Portfolio Analysis page.
        st.subheader("Lifecycle Portfolio Path")
        st.caption(
            "Median portfolio value across the full lifecycle (accumulation "
            "+ retirement) at the required-savings level, with 20th–80th "
            "percentile bands shaded. The dotted vertical line for each "
            "scenario marks the start of retirement."
        )
        goal_results_for_chart = [r["result"] for r in results]
        try:
            chart_buf = chart_lifecycle_paths(
                goal_results_for_chart, snapshot["inflation"]
            )
            st.image(chart_buf, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render lifecycle chart: {e}")

        # ----- PDF generation -----
        st.divider()
        st.subheader("Generate PDF Report")
        colp1, colp2 = st.columns([1, 3])
        with colp1:
            if st.button("📄 Build PDF Report", type="primary",
                         use_container_width=True, key="sg_build_pdf"):
                with st.spinner("Building PDF..."):
                    goal_results = [r["result"] for r in results]
                    pdf_bytes = build_savings_goal_pdf(
                        goal_results=goal_results,
                        sensitivity_results=sensitivity,
                        initial_savings=snapshot["initial_savings"],
                        target_success_prob=snapshot["target_success_prob"],
                        inflation=snapshot["inflation"],
                        return_assumptions=snapshot["return_assumptions"],
                        n_paths=snapshot["n_paths"],
                        confidence_levels=snapshot["confidence_levels"],
                        extra_inflows=snapshot.get("extra_inflows", []),
                    )
                    st.session_state["sg_pdf_bytes"] = pdf_bytes
                    st.session_state["sg_pdf_filename"] = (
                        f"bcmsavings_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    )
                st.success("PDF ready — click below to download.")
        with colp2:
            if "sg_pdf_bytes" in st.session_state:
                st.download_button(
                    "⬇️  Download PDF",
                    data=st.session_state["sg_pdf_bytes"],
                    file_name=st.session_state["sg_pdf_filename"],
                    mime="application/pdf",
                    use_container_width=True,
                    key="sg_download",
                )
    else:
        st.info(
            "👆 Configure your scenarios above and click "
            "**Calculate Required Savings** to run the analysis."
        )

    _render_footer()


# Always call main() — when Streamlit invokes this script as a Page via
# st.navigation(), __name__ is not always "__main__", so we don't guard
# the call. (Importing this file as a module would be unusual and would
# also trigger main(), but in practice nothing imports this page.)
main()
