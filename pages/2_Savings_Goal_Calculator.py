"""
Becker Capital — Savings Goal Calculator

Inverse Monte Carlo: given a desired retirement income, time horizon, and
allocation, compute the minimum annual savings required to support that
income with a chosen target probability of success.

This page lives alongside the main Monte Carlo Distribution simulator
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

from app import (  # noqa: E402
    # Brand
    GOLD_HEX, NAVY_HEX, NAVY_DARK_HEX, SCENARIO_COLOR_HEX,
    # Engine — return assumptions
    PRESETS, ReturnAssumptions, HIST_STATS,
    # Engine — savings goal
    SavingsGoalScenario,
    find_required_annual_savings,
    required_savings_at_confidence_levels,
    build_savings_goal_pdf,
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
    Mirror of the return-assumptions picker on the main page. Kept
    inline rather than imported because it's tightly coupled to
    Streamlit widget state (and the main page's version contains
    main-page-specific session keys).
    """
    method = st.radio(
        "Method",
        ["Bootstrap (recommended)", "Parametric (normal distribution)"],
        index=0,
        help=(
            "Bootstrap resamples actual historical years (1928–2024) and "
            "re-centers them to your forward-looking mean — preserving real-"
            "world volatility, fat tails, and the equity/fixed-income "
            "correlation. Parametric draws from a normal distribution."
        ),
        key="sg_method",
    )

    if method.startswith("Bootstrap"):
        preset_choice = st.selectbox(
            "Preset",
            ["Forward-looking (Aggressive): 9.5% / 4.5%",
             "Forward-looking (Moderate): 8% / 4.5%",
             "Forward-looking (Conservative): 7% / 4%",
             "Historical means (1928–2024)",
             "Custom forward-looking μ"],
            index=1,
            key="sg_boot_preset",
        )
        if preset_choice.startswith("Forward-looking (Aggressive)"):
            ra = PRESETS["Bootstrap — Forward-looking (Aggressive)"]
        elif preset_choice.startswith("Forward-looking (Moderate)"):
            ra = PRESETS["Bootstrap — Forward-looking (Moderate)"]
        elif preset_choice.startswith("Forward-looking (Conservative)"):
            ra = PRESETS["Bootstrap — Forward-looking (Conservative)"]
        elif preset_choice.startswith("Historical"):
            ra = PRESETS["Bootstrap — Historical means (1928–2024)"]
        else:
            colA, colB = st.columns(2)
            with colA:
                st.caption("**Equity**")
                eq_mu = st.number_input(
                    "Forward μ (%)", value=8.0, step=0.25,
                    min_value=0.0, max_value=20.0, key="sg_boot_eq_mu",
                ) / 100
            with colB:
                st.caption("**Fixed Income**")
                fi_mu = st.number_input(
                    "Forward μ (%)", value=4.5, step=0.25,
                    min_value=0.0, max_value=12.0, key="sg_boot_fi_mu",
                ) / 100
            ra = ReturnAssumptions(
                eq_mu=eq_mu, eq_sigma=HIST_STATS["eq_sigma"],
                fi_mu=fi_mu, fi_sigma=HIST_STATS["fi_sigma"],
                label=f"Bootstrap • Custom (μ={eq_mu*100:.1f}%/{fi_mu*100:.1f}%)",
                method="bootstrap", historical_period="1928–2024",
                worst_eq=HIST_STATS["worst_eq"],
                worst_fi=HIST_STATS["worst_fi"],
            )
        st.caption(
            f"**Forward μ:** Eq {ra.eq_mu*100:.2f}% • FI {ra.fi_mu*100:.2f}%  \n"
            f"**Historical σ (preserved):** Eq {HIST_STATS['eq_sigma']*100:.2f}% • "
            f"FI {HIST_STATS['fi_sigma']*100:.2f}%"
        )
    else:
        param_choice = st.selectbox(
            "Preset",
            ["1960–2024 (Legacy Becker)",
             "Forward-looking Conservative (8%/4.5%)",
             "Custom"],
            index=0, key="sg_param_preset",
        )
        if param_choice == "1960–2024 (Legacy Becker)":
            ra = PRESETS["Parametric — 1960–2024 (Legacy Becker)"]
        elif param_choice.startswith("Forward-looking"):
            ra = PRESETS["Parametric — Forward-looking (Conservative)"]
        else:
            colA, colB = st.columns(2)
            with colA:
                st.caption("**Equity**")
                eq_mu = st.number_input("Mean (%)", value=11.79, step=0.1,
                                        key="sg_par_eq_mu") / 100
                eq_sig = st.number_input("Std Dev (%)", value=16.67, step=0.1,
                                         key="sg_par_eq_sig") / 100
            with colB:
                st.caption("**Fixed Income**")
                fi_mu = st.number_input("Mean (%)", value=6.15, step=0.1,
                                        key="sg_par_fi_mu") / 100
                fi_sig = st.number_input("Std Dev (%)", value=8.79, step=0.1,
                                         key="sg_par_fi_sig") / 100
            ra = ReturnAssumptions(
                eq_mu=eq_mu, eq_sigma=eq_sig,
                fi_mu=fi_mu, fi_sigma=fi_sig,
                label="Parametric • Custom", method="parametric",
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

        st.divider()
        st.subheader("Simulation")
        n_paths = st.select_slider(
            "Paths per candidate",
            options=[1_000, 2_500, 5_000, 10_000],
            value=2_500, key="sg_n_paths",
            help=(
                "More paths = more accurate solver, but slower. "
                "The bisection runs ~12–18 simulations per scenario."
            ),
        )

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

    goals: List[SavingsGoalScenario] = []
    cols = st.columns(n_scen)
    for i, col in enumerate(cols):
        (name_def, ytr_def, yir_def, inc_def,
         eq_def, glide_def, ret_eq_def) = DEFAULTS[i]
        with col:
            st.markdown(
                f"<div class='becker-scenario-card' "
                f"style='--scenario-color:{SCENARIO_COLOR_HEX[i]};'>"
                f"{name_def}</div>",
                unsafe_allow_html=True,
            )
            name = st.text_input("Name", value=name_def, key=f"sg_name_{i}")

            # ----- Time horizon -----
            ytr = st.number_input(
                "Years to Retirement",
                min_value=1, max_value=60, value=ytr_def, step=1,
                key=f"sg_ytr_{i}",
                help="Number of accumulation years (during which you save).",
            )
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
                f"border-top:1px solid rgba(184,146,77,0.3);'>"
                f"INCOME TARGET</div>",
                unsafe_allow_html=True,
            )
            income = dollar_input(
                "Desired Annual Income ($, today's dollars)",
                default=inc_def, key=f"sg_inc_{i}",
                min_value=0, max_value=10_000_000,
                help=(
                    "Income you want to draw each year in retirement, in "
                    "today's dollars. Will be inflated forward."
                ),
            )

            # ----- Allocation -----
            st.markdown(
                f"<div style='color:{GOLD_HEX}; font-size:12px; font-weight:700; "
                f"letter-spacing:0.5px; margin-top:8px; padding-top:6px; "
                f"border-top:1px solid rgba(184,146,77,0.3);'>"
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
    @st.cache_data(show_spinner=False)
    def _solve_all(_key: str, goals_repr, initial_savings: float,
                   target_success: float, inflation: float,
                   ra_repr: str, n_paths: int):
        # `goals_repr` and `ra_repr` are used for cache-key purposes only;
        # the real objects come from the closure.
        results = []
        sensitivity = []
        for g in goals:
            res = find_required_annual_savings(
                goal=g, initial_savings=initial_savings,
                target_success_prob=target_success, inflation=inflation,
                return_assumptions=ra, n_paths=n_paths,
            )
            sens = required_savings_at_confidence_levels(
                goal=g, initial_savings=initial_savings,
                confidence_levels=[0.70, 0.80, 0.90],
                inflation=inflation, return_assumptions=ra,
                n_paths=n_paths,
            )
            results.append(res)
            sensitivity.append(sens)
        return results, sensitivity

    if run_clicked:
        cache_key = repr((
            tuple((g.name, g.years_to_retirement, g.years_in_retirement,
                   g.desired_annual_income, g.accumulation_eq_weight,
                   g.retirement_eq_weight) for g in goals),
            initial_savings, target_success, inflation,
            ra.label, ra.eq_mu, ra.fi_mu, ra.method, n_paths,
        ))
        # Show progress: 3 scenarios × (1 headline + 3 sensitivity) = ~12
        # bisection runs at 12–18 sims each. Display a spinner with hint.
        n_sols = n_scen * 4
        approx_sims = n_sols * 14
        with st.spinner(
            f"Running ~{approx_sims:,} Monte Carlo simulations across "
            f"{n_sols} bisection searches… (this can take 20–60 seconds)"
        ):
            results, sensitivity = _solve_all(
                cache_key, None, initial_savings, target_success,
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
        }

    # ----- Results display -----
    if "sg_results" in st.session_state:
        results = st.session_state["sg_results"]
        sensitivity = st.session_state["sg_sensitivity"]
        snapshot = st.session_state["sg_inputs_snapshot"]

        st.subheader("Required Savings — Headline Result")
        metric_cols = st.columns(len(results))
        for col, res, g in zip(metric_cols, results, snapshot["goals"]):
            with col:
                glide_str = ""
                if g.has_glide_path:
                    eq_a = round(g.accumulation_eq_weight * 100)
                    eq_r = round(g.retirement_eq_weight * 100)
                    glide_str = (f" • {eq_a}/{100-eq_a} → "
                                 f"{eq_r}/{100-eq_r}")
                else:
                    eq_a = round(g.accumulation_eq_weight * 100)
                    glide_str = f" • {eq_a}/{100-eq_a} static"
                st.markdown(
                    f"**{g.name}**<br/>"
                    f"<span style='color:#B8C4D6; font-size:12px;'>"
                    f"{g.years_to_retirement}yr accum / "
                    f"{g.years_in_retirement}yr ret"
                    f"{glide_str}</span>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Required Savings",
                    f"${res['required_savings']:,.0f}/yr",
                )
                st.metric(
                    "Achieved Success",
                    f"{res['achieved_prob']*100:.1f}%",
                )
                st.metric(
                    "Median at Retirement",
                    f"${res['result']['median_at_retirement']/1e6:,.2f}M",
                )
                if not res["converged"]:
                    st.warning(
                        "⚠️ Bisection capped at the upper bound. The target "
                        "may be infeasible for this scenario — consider a "
                        "longer horizon, lower income, or higher equity."
                    )

        # Sensitivity table
        st.subheader("Sensitivity — Required Savings by Confidence Level")
        sens_table = {"Confidence": ["70%", "80%", "90%"]}
        for sens, g in zip(sensitivity, snapshot["goals"]):
            sens_table[g.name] = [
                f"${s['required_savings']:,.0f}/yr"
                + ("" if s["converged"] else " (capped)")
                for s in sens
            ]
        st.table(sens_table)

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
                    )
                    st.session_state["sg_pdf_bytes"] = pdf_bytes
                    st.session_state["sg_pdf_filename"] = (
                        f"Becker_SavingsGoal_"
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


if __name__ == "__main__":
    main()
