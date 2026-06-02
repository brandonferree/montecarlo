"""
Becker Capital — Social Security Optimizer

Finds the Social Security claiming strategy that maximizes PLAN SUCCESS (not a
fixed-life-expectancy breakeven) by sweeping every claim-age combination and
scoring each through the same Monte Carlo engine used by the main Cashflow
Portfolio Analysis page.

Why plan-success and not breakeven? Breakeven math collapses a longevity
distribution into a single point and — for couples — ignores that the higher
earner's claim age sets the SURVIVOR'S income floor for the rest of their life.
This page scores on portfolio survival and surfaces the survivor floor
explicitly, the two things eMoney/RightCapital-style breakeven tools miss.

Lives alongside app.py (main) and the Savings Goal Calculator in a Streamlit
multipage app; it imports the engine + brand-styled UI helpers from app.py and
the Social Security math from ss_engine.py.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, List

import streamlit as st

# Make the parent directory (where app.py lives) importable regardless of how
# Streamlit launches this script.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import ss_engine as se  # noqa: E402
from app import (  # noqa: E402
    GOLD_HEX, NAVY_HEX, MIDNIGHT_RGB, CANYON_RGB,
    BCM_CMA_2026, ReturnAssumptions,
    dollar_input, _inject_becker_css, _render_footer,
)


def _render_ss_header():
    st.markdown(
        """
        <div class="becker-header">
          <div class="becker-monogram">
            <div class="fifty">50</div>
            <div class="b-circle">B</div>
          </div>
          <div class="becker-header-text">
            <div class="becker-eyebrow">BECKER CAPITAL MANAGEMENT  •  EST. 1976</div>
            <div class="becker-title">Social Security Optimizer</div>
            <div class="becker-subtitle">
              Find the claiming strategy that maximizes the probability your
              plan succeeds — scored through Monte Carlo, with the survivor
              income floor surfaced explicitly.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _return_assumptions_picker() -> ReturnAssumptions:
    st.caption(
        "Forward-looking μ and σ are sourced from Becker Capital "
        "Management's 2026 Capital Market Assumptions (10-year estimates)."
    )
    preset_choice = st.selectbox("Preset", ["BCM 2026 CMAs", "Custom"],
                                 index=0, key="ss_preset")
    if preset_choice == "BCM 2026 CMAs":
        eq_mu, eq_sigma = BCM_CMA_2026["EQ_US_LARGE"]
        fi_mu, fi_sigma = BCM_CMA_2026["FI_IT_US"]
        ra = ReturnAssumptions(eq_mu=eq_mu, eq_sigma=eq_sigma,
                               fi_mu=fi_mu, fi_sigma=fi_sigma,
                               label="BCM 2026 CMAs (Large Cap + Intermediate Bonds)")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.caption("**Equity**")
            eq_mu = st.number_input("Mean (%)", value=6.00, step=0.1, key="ss_eq_mu") / 100
            eq_sig = st.number_input("Std Dev (%)", value=17.00, step=0.1, key="ss_eq_sig") / 100
        with colB:
            st.caption("**Fixed Income**")
            fi_mu = st.number_input("Mean (%)", value=5.00, step=0.1, key="ss_fi_mu") / 100
            fi_sig = st.number_input("Std Dev (%)", value=5.50, step=0.1, key="ss_fi_sig") / 100
        ra = ReturnAssumptions(eq_mu=eq_mu, eq_sigma=eq_sig,
                               fi_mu=fi_mu, fi_sigma=fi_sig, label="Custom")
    st.caption(
        f"Eq μ = {ra.eq_mu*100:.2f}%, σ = {ra.eq_sigma*100:.2f}%  |  "
        f"FI μ = {ra.fi_mu*100:.2f}%, σ = {ra.fi_sigma*100:.2f}%"
    )
    return ra


def _person_inputs(idx: int, defaults: dict) -> se.Person:
    """Render one claimant's inputs and return a Person."""
    st.markdown(
        f"<div class='becker-scenario-card' "
        f"style='--scenario-color:{NAVY_HEX};'>{defaults['label']}</div>",
        unsafe_allow_html=True,
    )
    label = st.text_input("Name", value=defaults["label"], key=f"ss_label_{idx}")
    birth_year = st.number_input(
        "Birth year", min_value=1940, max_value=2005,
        value=defaults["birth_year"], step=1, key=f"ss_by_{idx}",
        help="Determines Full Retirement Age (66–67).",
    )
    current_age = st.number_input(
        "Current age", min_value=40, max_value=70,
        value=defaults["current_age"], step=1, key=f"ss_age_{idx}",
        help="Age today. Year 1 of the projection starts at the primary "
             "person's current age.",
    )
    pia_monthly = dollar_input(
        "FRA benefit / PIA ($/mo)", default=defaults["pia"],
        key=f"ss_pia_{idx}", min_value=0, max_value=10_000,
        help="Monthly benefit at Full Retirement Age, from the SSA statement "
             "(the 'Primary Insurance Amount').",
    )
    fra = se.full_retirement_age_months(int(birth_year))
    st.caption(f"Full Retirement Age: **{fra // 12}y {fra % 12}m**")
    return se.Person(
        label=label.strip() or defaults["label"],
        birth_year=int(birth_year),
        pia_monthly=float(pia_monthly),
        current_age=int(current_age),
    )


def main():
    st.set_page_config(
        page_title="Becker Capital — Social Security Optimizer",
        page_icon="🧾", layout="wide",
    )
    _inject_becker_css()
    _render_ss_header()

    _SCHEMA = "v1"
    if st.session_state.get("ss_schema") != _SCHEMA:
        for k in ("ss_results", "ss_snapshot", "ss_pdf_bytes", "ss_pdf_filename"):
            st.session_state.pop(k, None)
        st.session_state["ss_schema"] = _SCHEMA

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Plan Inputs")
        household = st.radio("Household", ["Married couple", "Single"],
                             index=0, horizontal=True, key="ss_household")
        is_couple = (household == "Married couple")

        initial_portfolio = dollar_input(
            "Investable Portfolio ($)", default=2_000_000, key="ss_initial",
            min_value=0, max_value=500_000_000,
            help="Liquid investable assets available to fund the spending gap "
                 "before and alongside Social Security.",
        )
        spend_basis = st.radio("Enter spending need as", ["Annual", "Monthly"],
                               index=0, horizontal=True, key="ss_spend_basis")
        unit_div = 12 if spend_basis == "Monthly" else 1
        unit_word = "Monthly" if spend_basis == "Monthly" else "Annual"
        spend_entered = dollar_input(
            f"{unit_word} Spending Need ($, today's dollars)",
            default=int(round(120_000 / unit_div)),
            key=f"ss_spend_{'m' if unit_div == 12 else 'a'}",
            min_value=0, max_value=10_000_000,
            help="Total household spending need in retirement (today's dollars). "
                 "Social Security offsets it; the portfolio funds the rest.",
        )
        gross_need = spend_entered * unit_div

        eq_pct = st.slider("Portfolio Equity %", 0, 100, 60, step=5, key="ss_eq")
        st.caption(f"Fixed Income: **{100 - eq_pct}%**")
        inflation = st.slider("Inflation / COLA (%)", 0.0, 8.0, 2.5, step=0.25,
                              key="ss_infl") / 100
        horizon = st.slider("Projection horizon (years)", 20, 45, 35, step=1,
                            key="ss_horizon",
                            help="Length of the projection from the primary "
                                 "person's current age.")

        st.divider()
        st.subheader("Longevity")
        st.caption("Assumed age at death drives when the survivor benefit "
                   "begins. A longer life favors delaying — that's the "
                   "longevity-insurance value breakeven math hides.")
        death_primary = st.slider("Primary — age at death", 75, 100, 90,
                                   step=1, key="ss_death_0")
        if is_couple:
            death_spouse = st.slider("Spouse — age at death", 75, 100, 93,
                                     step=1, key="ss_death_1")

        st.divider()
        st.subheader("Tax / IRMAA")
        tax_pct = st.slider(
            "Effective tax gross-up on portfolio withdrawals (%)",
            0, 35, 0, step=1, key="ss_tax",
            help="Approximates income tax on portfolio withdrawals + taxable "
                 "Social Security (the 'tax torpedo'). Inflates the "
                 "portfolio-funded portion of the spending need. Set 0 to "
                 "ignore taxes.",
        )
        tax_gross_up = tax_pct / 100.0

        st.divider()
        st.subheader("Return Assumptions")
        ra = _return_assumptions_picker()

    # ----- People inputs -----
    st.subheader("Household")
    if is_couple:
        c1, c2 = st.columns(2)
        with c1:
            p1 = _person_inputs(0, {"label": "Higher earner",
                                    "birth_year": 1962, "current_age": 64,
                                    "pia": 3000})
        with c2:
            p2 = _person_inputs(1, {"label": "Spouse",
                                    "birth_year": 1964, "current_age": 62,
                                    "pia": 1400})
        people = [p1, p2]
        death_age = {p1.label: int(death_primary), p2.label: int(death_spouse)}
    else:
        p1 = _person_inputs(0, {"label": "Claimant", "birth_year": 1962,
                                "current_age": 64, "pia": 2600})
        people = [p1]
        death_age = {p1.label: int(death_primary)}

    st.divider()
    colb1, _ = st.columns([1, 3])
    with colb1:
        run = st.button("🧾 Optimize Claiming Strategy", type="primary",
                        use_container_width=True)

    if run:
        hh = se.HouseholdSSInputs(
            people=people, horizon_years=int(horizon), inflation=inflation,
            death_age=death_age,
        )
        n_combos = (9 if not is_couple else 81)
        with st.spinner(
            f"Scoring {n_combos} claiming strategies through Monte Carlo…"
        ):
            results = se.optimize_claim_strategies(
                hh, gross_spending_need_annual=gross_need,
                initial_portfolio=initial_portfolio, eq_weight=eq_pct / 100,
                return_assumptions=ra, n_paths_grid=2000,
                tax_gross_up=tax_gross_up,
            )
        st.session_state["ss_results"] = results
        st.session_state["ss_snapshot"] = {
            "hh": hh, "people": people, "is_couple": is_couple,
            "gross_need": gross_need, "initial_portfolio": initial_portfolio,
            "eq_pct": eq_pct, "ra": ra, "inflation": inflation,
            "tax_gross_up": tax_gross_up,
        }

    # ----- Results -----
    if "ss_results" in st.session_state:
        results = st.session_state["ss_results"]
        snap = st.session_state["ss_snapshot"]
        best = se.best_strategy(results)
        people = snap["people"]

        # Recommended strategy card.
        st.subheader("Recommended Strategy")
        claim_str = " · ".join(
            f"{p.label}: claim at {best.claim_ages[p.label]}" for p in people)
        baseline = next(
            r for r in results
            if all(r.claim_ages[p.label] == 62 for p in people))
        lift = (best.prob_success - baseline.prob_success) * 100

        card_css = (
            f"background:rgba({MIDNIGHT_RGB},0.30);"
            f"border:1px solid rgba({CANYON_RGB},0.20);"
            f"border-left:3px solid {GOLD_HEX};border-radius:3px;"
            f"padding:16px 18px;margin-bottom:12px;"
        )
        st.markdown(
            f'<div style="{card_css}">'
            f'<div style="color:white;font-size:16px;font-weight:700;">{claim_str}</div>'
            f'<div style="color:{GOLD_HEX};font-size:24px;font-weight:700;margin:6px 0;">'
            f'{best.prob_success*100:.1f}% chance of success</div>'
            f'<div style="color:#B8C4D6;font-size:13px;">'
            f'+{lift:.1f} pts vs. both claiming at 62 · '
            f'Median ending portfolio &#36;{best.median_terminal/1e6:,.2f}M</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if snap["is_couple"]:
            st.info(
                f"**Survivor income floor:** under this strategy the surviving "
                f"spouse receives **&#36;{best.survivor_floor_annual:,.0f}/yr** "
                f"(today's dollars) in Social Security after the first death — "
                f"set primarily by the higher earner's claim age. This is the "
                f"longevity protection breakeven analysis overlooks."
            )

        # Grid chart.
        st.subheader("Plan Success by Claiming Combination")
        st.caption(
            "Each cell is the Monte Carlo probability the portfolio survives "
            "the full horizon under that claiming combination. The orange box "
            "marks the recommended strategy. Common random numbers are used so "
            "differences reflect the claiming decision, not sampling noise."
        )
        try:
            st.image(se.chart_strategy_grid(results, people),
                     use_container_width=True)
        except Exception as e:
            st.error(f"Could not render grid: {e}")

        # SS income stream chart.
        st.subheader("Social Security Income Stream — Recommended Strategy")
        try:
            st.image(se.chart_ss_income_stream(snap["hh"], best.claim_ages),
                     use_container_width=True)
        except Exception as e:
            st.error(f"Could not render income stream: {e}")

        # IRMAA / tax note.
        with st.expander("Tax, IRMAA & 2025 law notes"):
            st.markdown(
                "- **Taxation of benefits / tax torpedo.** Up to 85% of Social "
                "Security becomes taxable as provisional income rises, which can "
                "push the marginal rate on ordinary withdrawals to 40%+ in that "
                "band. Use the *tax gross-up* slider to stress this.\n"
                "- **IRMAA.** Medicare Part B/D premiums step up at MAGI "
                f"thresholds (2025: ~&#36;{se.IRMAA_MAGI_TIER1_SINGLE:,} single / "
                f"&#36;{se.IRMAA_MAGI_TIER1_MARRIED:,} married). A delay strategy "
                "funded by large IRA withdrawals or Roth conversions can trip a "
                "tier — worth checking against the recommended drawdown.\n"
                "- **Gap-year Roth conversions.** The years between retirement "
                "and the higher earner's claim are typically a low-income window "
                "ideal for Roth conversions.\n"
                "- **Social Security Fairness Act (Jan 2025).** WEP and GPO are "
                "repealed; no windfall-elimination haircut is applied to clients "
                "with non-covered pensions."
            )

        # PDF.
        st.divider()
        st.subheader("Generate PDF Report")
        cpdf1, cpdf2 = st.columns([1, 3])
        with cpdf1:
            if st.button("📄 Build PDF Report", type="primary",
                         use_container_width=True, key="ss_build_pdf"):
                with st.spinner("Building PDF..."):
                    from app import build_ss_pdf  # lazy: defined in app.py
                    st.session_state["ss_pdf_bytes"] = build_ss_pdf(
                        results=results, best=best, snapshot=snap,
                    )
                    st.session_state["ss_pdf_filename"] = (
                        f"bcmsocialsecurity_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    )
                st.success("PDF ready — click below to download.")
        with cpdf2:
            if "ss_pdf_bytes" in st.session_state:
                st.download_button(
                    "⬇️  Download PDF",
                    data=st.session_state["ss_pdf_bytes"],
                    file_name=st.session_state["ss_pdf_filename"],
                    mime="application/pdf", use_container_width=True,
                    key="ss_download",
                )
    else:
        st.info("👆 Set the household, spending need, and longevity, then click "
                "**Optimize Claiming Strategy**.")

    _render_footer()


main()
