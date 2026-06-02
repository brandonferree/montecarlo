"""
Becker Capital — Social Security benefit engine.

Pure, Streamlit-free functions so the math is unit-testable and reusable by
both the page UI and the PDF builder. Two layers live here:

  1. Benefit math — translate a worker's PIA (Primary Insurance Amount, the
     benefit payable at Full Retirement Age) into the *actual* monthly benefit
     at any claim age, plus spousal and survivor benefits.

  2. Claim-age optimizer — sweep every claiming-age combination, translate each
     into Social Security income streams, hold the household's gross spending
     need constant, and score each strategy by Monte Carlo plan-success using
     the existing engine in app.py. Scoring on plan-success (rather than a
     fixed-life-expectancy breakeven) is the differentiator: it values Social
     Security as longevity insurance on a survival-weighted basis, and — for
     couples — captures that the higher earner's claim age sets the survivor's
     income floor for the rest of their life.

Conventions
-----------
* All ages are handled in whole months internally for accuracy, exposed in
  whole years at the UI boundary (62..70 claim ages).
* PIA and benefits are entered/returned as MONTHLY dollars unless a name ends
  in `_annual`.
* COLA is modeled as equal to the simulation's inflation assumption (a standard
  planning simplification — Social Security's COLA tracks CPI-W, which over long
  horizons is close to the general inflation rate used elsewhere in the model).

Rules reflected
---------------
* Early-claim reduction: 5/9 of 1% per month for the first 36 months before
  FRA, 5/12 of 1% per month beyond that (max ~30% at 62 when FRA is 67).
* Delayed Retirement Credits: 2/3 of 1% per month (8%/yr) from FRA to age 70.
* Spousal benefit: up to 50% of the higher earner's PIA; the *excess* spousal
  portion is reduced 25/36 of 1% per month for the first 36 months early and
  5/12 of 1% beyond. No Delayed Retirement Credits accrue on spousal benefits.
* Survivor benefit: based on the deceased's *actual* (claim-adjusted) benefit,
  but never less than 82.5% of the deceased's PIA if the deceased had claimed
  early (the "widow(er)'s limit"). Reduced if the survivor claims before their
  own FRA.
* Retirement Earnings Test: pre-FRA benefits are withheld $1 for every $2 of
  earnings above an annual exempt amount (informational / optional).
* Social Security Fairness Act (signed Jan 2025): WEP and GPO are REPEALED, so
  no windfall-elimination haircut is applied to workers with non-covered
  pensions. This is reflected by simply not modeling a WEP/GPO reduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARLIEST_CLAIM_AGE = 62
LATEST_CLAIM_AGE = 70           # no benefit to delaying past 70

# 2025 Retirement Earnings Test annual exempt amount (under-FRA, full year).
# Used only when the still-working earnings test is enabled.
EARNINGS_TEST_EXEMPT_2025 = 23_400

# 2025 IRMAA MAGI thresholds (first surcharge tier). Informational flags only.
IRMAA_MAGI_TIER1_SINGLE = 106_000
IRMAA_MAGI_TIER1_MARRIED = 212_000


def full_retirement_age_months(birth_year: int) -> int:
    """
    Full Retirement Age in whole months, per the SSA schedule.

    1943–1954 -> 66y0m; then +2 months per birth year through 1959; 1960+ ->
    67y0m.
    """
    if birth_year <= 1954:
        return 66 * 12
    if birth_year >= 1960:
        return 67 * 12
    # 1955..1959: 66y + 2 months per year past 1954
    return 66 * 12 + 2 * (birth_year - 1954)


# ---------------------------------------------------------------------------
# Worker benefit (own record): early reduction & delayed retirement credits
# ---------------------------------------------------------------------------
def worker_benefit_factor(claim_age_months: int, fra_months: int) -> float:
    """
    Multiplier applied to PIA for a worker claiming on their own record.

    < FRA  -> reduction (5/9 %/mo first 36, 5/12 %/mo beyond)
    = FRA  -> 1.0
    > FRA  -> +2/3 %/mo (8%/yr), capped at age 70
    """
    if claim_age_months < fra_months:
        months_early = fra_months - claim_age_months
        first = min(months_early, 36)
        beyond = max(0, months_early - 36)
        reduction = first * (5.0 / 9.0) / 100.0 + beyond * (5.0 / 12.0) / 100.0
        return 1.0 - reduction
    # Delayed: cap credit accrual at age 70.
    capped = min(claim_age_months, LATEST_CLAIM_AGE * 12)
    months_late = max(0, capped - fra_months)
    return 1.0 + months_late * (2.0 / 3.0) / 100.0


def worker_monthly_benefit(pia_monthly: float, claim_age_months: int,
                           fra_months: int) -> float:
    """Actual monthly worker benefit at a given claim age."""
    return pia_monthly * worker_benefit_factor(claim_age_months, fra_months)


# ---------------------------------------------------------------------------
# Spousal benefit
# ---------------------------------------------------------------------------
def spousal_excess_factor(claim_age_months: int, fra_months: int) -> float:
    """
    Reduction multiplier for the *spousal excess* portion when claimed early.
    No delayed credits accrue on spousal benefits, so at/after FRA -> 1.0.
    """
    if claim_age_months >= fra_months:
        return 1.0
    months_early = fra_months - claim_age_months
    first = min(months_early, 36)
    beyond = max(0, months_early - 36)
    reduction = first * (25.0 / 36.0) / 100.0 + beyond * (5.0 / 12.0) / 100.0
    return max(0.0, 1.0 - reduction)


def spousal_monthly_benefit(own_pia_monthly: float, higher_pia_monthly: float,
                            claim_age_months: int, fra_months: int) -> float:
    """
    Total monthly benefit for the lower earner including any spousal top-up.

    The worker first receives their own reduced/credited benefit; if half the
    higher earner's PIA exceeds the lower earner's PIA, the *excess* is added
    (reduced for early claiming). The spouse must have filed for the top-up to
    be payable; we assume the higher earner has filed by the time the spousal
    benefit is evaluated in the simulation.
    """
    own = worker_monthly_benefit(own_pia_monthly, claim_age_months, fra_months)
    excess = max(0.0, 0.5 * higher_pia_monthly - own_pia_monthly)
    return own + excess * spousal_excess_factor(claim_age_months, fra_months)


# ---------------------------------------------------------------------------
# Survivor benefit
# ---------------------------------------------------------------------------
def survivor_factor(survivor_claim_age_months: int, survivor_fra_months: int) -> float:
    """
    Survivor benefit reduction multiplier. Survivor benefits can start as early
    as age 60 (survivor_fra here is the survivor benefit FRA, ~equal to the
    retirement FRA for these cohorts). Max reduction at 60 is 28.5%.
    """
    if survivor_claim_age_months >= survivor_fra_months:
        return 1.0
    months_early = survivor_fra_months - survivor_claim_age_months
    # Linear from 0% at FRA to 28.5% at age 60 (the statutory floor).
    max_months_early = survivor_fra_months - 60 * 12
    if max_months_early <= 0:
        return 1.0
    reduction = 0.285 * (months_early / max_months_early)
    return max(0.0, 1.0 - reduction)


def survivor_monthly_benefit(deceased_pia_monthly: float,
                             deceased_actual_monthly: float,
                             survivor_claim_age_months: int,
                             survivor_fra_months: int) -> float:
    """
    Monthly survivor benefit. Based on the deceased's *actual* benefit
    (including any delayed credits — the key reason for the higher earner to
    delay), with the widow(er)'s-limit floor of 82.5% of the deceased's PIA
    when the deceased had claimed early. Reduced if the survivor claims the
    survivor benefit before their own survivor FRA.
    """
    base = deceased_actual_monthly
    if deceased_actual_monthly < deceased_pia_monthly:
        # Deceased had claimed early -> apply the 82.5%-of-PIA floor.
        base = max(deceased_actual_monthly, 0.825 * deceased_pia_monthly)
    return base * survivor_factor(survivor_claim_age_months, survivor_fra_months)


def earnings_test_withholding(gross_annual_benefit: float, annual_earnings: float,
                              exempt: float = EARNINGS_TEST_EXEMPT_2025) -> float:
    """
    Annual benefit withheld under the Retirement Earnings Test (pre-FRA only):
    $1 withheld per $2 of earnings above the exempt amount. Returns the dollar
    amount of benefit withheld (capped at the gross benefit).
    """
    if annual_earnings <= exempt:
        return 0.0
    return min(gross_annual_benefit, 0.5 * (annual_earnings - exempt))


# ---------------------------------------------------------------------------
# Person & household specification
# ---------------------------------------------------------------------------
@dataclass
class Person:
    """A worker/claimant. `pia_monthly` is the benefit at FRA (from the SSA
    statement). `current_age` is today's age in whole years; the simulation
    treats year 1 of the horizon as starting at current_age."""
    label: str
    birth_year: int
    pia_monthly: float
    current_age: int
    still_working_until_age: Optional[int] = None  # earnings-test years (optional)
    annual_earnings: float = 0.0                    # used only with the above

    @property
    def fra_months(self) -> int:
        return full_retirement_age_months(self.birth_year)

    @property
    def fra_years(self) -> float:
        return self.fra_months / 12.0


@dataclass
class HouseholdSSInputs:
    """Everything the optimizer needs that is independent of the claim ages."""
    people: List[Person]                 # 1 (single) or 2 (married)
    horizon_years: int                   # simulation length (from year-1 = primary's current age)
    inflation: float                     # decimal; also used as the SS COLA
    # Per-person assumed death age (whole years). Drives when the survivor
    # benefit kicks in. For a deterministic plan use point estimates; the
    # mortality layer can override per-path.
    death_age: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Household SS income stream for a given claiming strategy
# ---------------------------------------------------------------------------
def household_ss_real_by_year(hh: HouseholdSSInputs,
                              claim_ages: Dict[str, int]) -> np.ndarray:
    """
    Annual household Social Security income in TODAY'S (real) dollars for each
    simulation year 1..horizon, under a given {person_label: claim_age} map.

    Real dollars because the caller (the MC engine) inflates by COLA = inflation
    downstream — see ss_inflow_events(). This keeps the engine's "today's
    dollars" convention.

    Models, per year:
      * Each living person's own/spousal benefit once they have claimed.
      * On the first death, the survivor receives the greater of their own
        benefit and the survivor benefit; the deceased's own benefit stops.
    """
    n = hh.horizon_years
    out = np.zeros(n)
    primary = hh.people[0]

    # Identify the higher-PIA earner for spousal/survivor logic.
    higher_pia = max(p.pia_monthly for p in hh.people)

    for idx in range(n):
        sim_year = idx + 1  # 1-indexed
        # Calendar age of each person in this sim year. Year 1 = current age.
        living = []
        for p in hh.people:
            age_this_year = p.current_age + (sim_year - 1)
            death_age = hh.death_age.get(p.label, 200)
            if age_this_year < death_age:
                living.append((p, age_this_year))

        if not living:
            continue

        # Determine if there has been a death (for survivor switching).
        someone_died = len(living) < len(hh.people)

        monthly_total = 0.0

        if someone_died and len(hh.people) == 2:
            # Survivor case: one person alive. They receive the greater of their
            # own benefit and the survivor benefit on the deceased's record.
            survivor, surv_age = living[0]
            deceased = next(p for p in hh.people if p.label != survivor.label)

            own = _claimed_own_or_spousal(survivor, surv_age, claim_ages, higher_pia)

            # Deceased's actual benefit had they claimed at their chosen age.
            dec_claim = claim_ages[deceased.label]
            dec_actual = worker_monthly_benefit(
                deceased.pia_monthly, dec_claim * 12, deceased.fra_months)
            surv_ben = survivor_monthly_benefit(
                deceased.pia_monthly, dec_actual,
                max(surv_age, 60) * 12, survivor.fra_months)
            monthly_total = max(own, surv_ben)
        else:
            # Both alive (or single): each gets their own/spousal benefit if claimed.
            for p, age_this_year in living:
                monthly_total += _claimed_own_or_spousal(
                    p, age_this_year, claim_ages, higher_pia)

        # Earnings-test withholding (pre-FRA, while working).
        annual = monthly_total * 12
        for p, age_this_year in living:
            if (p.still_working_until_age is not None
                    and age_this_year < p.still_working_until_age
                    and age_this_year * 12 < p.fra_months
                    and age_this_year >= claim_ages.get(p.label, 200)):
                annual -= earnings_test_withholding(annual, p.annual_earnings)
        out[idx] = max(0.0, annual)

    return out


def _claimed_own_or_spousal(person: Person, age_this_year: int,
                            claim_ages: Dict[str, int], higher_pia: float) -> float:
    """Monthly benefit for a living person this year (0 before they claim)."""
    claim_age = claim_ages[person.label]
    if age_this_year < claim_age:
        return 0.0
    claim_months = claim_age * 12
    # Lower earner may get a spousal top-up; higher earner just gets own.
    if person.pia_monthly < higher_pia:
        return spousal_monthly_benefit(
            person.pia_monthly, higher_pia, claim_months, person.fra_months)
    return worker_monthly_benefit(person.pia_monthly, claim_months, person.fra_months)


def survivor_income_floor_annual(hh: HouseholdSSInputs,
                                 claim_ages: Dict[str, int]) -> float:
    """
    The survivor's annual SS income (today's dollars) after the FIRST death,
    assuming the lower earner is the survivor (the common planning case). This
    is the number the breakeven-style tools miss: the higher earner's claim age
    sets this floor for the rest of the survivor's life.
    """
    if len(hh.people) < 2:
        return 0.0
    # Survivor = lower-PIA earner; deceased = higher-PIA earner.
    deceased = max(hh.people, key=lambda p: p.pia_monthly)
    survivor = min(hh.people, key=lambda p: p.pia_monthly)
    dec_claim = claim_ages[deceased.label]
    dec_actual = worker_monthly_benefit(
        deceased.pia_monthly, dec_claim * 12, deceased.fra_months)
    # Survivor assumed at/after their FRA when the death occurs (typical).
    surv = survivor_monthly_benefit(
        deceased.pia_monthly, dec_actual,
        survivor.fra_months, survivor.fra_months)
    own = worker_monthly_benefit(
        survivor.pia_monthly, claim_ages[survivor.label] * 12, survivor.fra_months)
    return max(own, surv) * 12


# ---------------------------------------------------------------------------
# Claim-age grid + Monte Carlo plan-success optimizer
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    claim_ages: Dict[str, int]
    prob_success: float
    median_terminal: float
    survivor_floor_annual: float
    lifetime_ss_real: float       # total real SS received to horizon (point death ages)


def claim_age_grid(people: List[Person]) -> List[Dict[str, int]]:
    """All claim-age combinations across 62..70 for each person."""
    ages = list(range(EARLIEST_CLAIM_AGE, LATEST_CLAIM_AGE + 1))
    if len(people) == 1:
        return [{people[0].label: a} for a in ages]
    p0, p1 = people[0].label, people[1].label
    return [{p0: a, p1: b} for a in ages for b in ages]


def optimize_claim_strategies(
    hh: HouseholdSSInputs,
    gross_spending_need_annual: float,
    initial_portfolio: float,
    eq_weight: float,
    return_assumptions,
    n_paths_grid: int = 2_000,
    seed: int = 20260501,
    tax_gross_up: float = 0.0,
) -> List[StrategyResult]:
    """
    Score every claiming strategy by Monte Carlo plan-success.

    Model: the household has a constant gross spending need (today's dollars).
    Social Security offsets it; the portfolio funds the remaining gap. Claiming
    later means larger SS checks but the portfolio alone funds the early "gap
    years," so the trade-off shows up directly in plan-success.

    `tax_gross_up` (0..1) inflates the portion of the spending need funded by
    the portfolio to approximate income tax on withdrawals + taxable SS (the
    "tax torpedo"); 0 disables it.

    Imports app.py lazily to avoid a circular import (app.py does not import
    this module).
    """
    from app import Scenario, SimInputs, InflowEvent, run_all_simulations

    results: List[StrategyResult] = []
    for claim_ages in claim_age_grid(hh.people):
        ss_real = household_ss_real_by_year(hh, claim_ages)

        # Net portfolio distribution need per year = gross need - SS, grossed up
        # for tax on the portfolio-funded portion. Build as a flat distribution
        # plus negative SS inflows so we can use the existing per-year inflow
        # machinery. We instead encode SS as positive InflowEvents and keep the
        # distribution at the gross need.
        inflows = _ss_inflow_events(ss_real, InflowEvent)

        grossed_need = gross_spending_need_annual * (1.0 + tax_gross_up)
        scen = Scenario(
            name="SS Strategy",
            eq_weight=eq_weight,
            fi_weight=1.0 - eq_weight,
            annual_distribution=grossed_need,
            contribution_years=0,
            annual_contribution=0.0,
        )
        sim = SimInputs(
            initial=initial_portfolio,
            horizon_years=hh.horizon_years,
            inflation=hh.inflation,
            distribution_frequency="Annual",
            return_assumptions=return_assumptions,
            scenarios=[scen],
            extra_inflows=inflows,
            n_paths=n_paths_grid,
            # Common random numbers: identical return paths across every
            # strategy so the success ranking reflects the claiming decision
            # alone, not Monte Carlo sampling noise.
            seed=seed,
        )
        r = run_all_simulations(sim)[0]
        results.append(StrategyResult(
            claim_ages=dict(claim_ages),
            prob_success=1.0 - r["p_ruin"],
            median_terminal=r["median_yfinal"],
            survivor_floor_annual=survivor_income_floor_annual(hh, claim_ages),
            lifetime_ss_real=float(np.sum(ss_real)),
        ))
    return results


def _ss_inflow_events(ss_real_by_year: np.ndarray, InflowEvent) -> list:
    """
    Convert a real (today's-dollars) annual SS array into per-year InflowEvents.

    InflowEvent.amount_in_year inflates `amount` by (1+inflation)^(year-1), which
    matches a COLA = inflation. So we must pass the DEFLATED base amount such
    that after re-inflation it equals the intended real benefit. Since the real
    array is already in today's dollars and constant in real terms once claimed,
    each year's base amount is exactly ss_real_by_year[year-1].
    """
    events = []
    for idx, amt in enumerate(ss_real_by_year):
        if amt <= 0:
            continue
        events.append(InflowEvent(
            amount=float(amt), start_year=idx + 1, years=1, label="Social Security"))
    return events


def best_strategy(results: List[StrategyResult]) -> StrategyResult:
    """Pick the recommended strategy: highest plan-success, tie-broken by
    median terminal wealth, then by higher survivor floor."""
    return max(
        results,
        key=lambda r: (round(r.prob_success, 4), r.median_terminal, r.survivor_floor_annual),
    )


# ---------------------------------------------------------------------------
# Charts (pure matplotlib → BytesIO PNG; shared by the page and the PDF)
# ---------------------------------------------------------------------------
def _brand():
    """Lazy brand-color fetch (avoids importing app at module load)."""
    from app import NAVY_HEX, GOLD_HEX, CANYON_HEX
    return NAVY_HEX, GOLD_HEX, CANYON_HEX


def chart_strategy_grid(results: List[StrategyResult], people: List[Person]):
    """
    Plan-success across the claim-age grid.

    Couples -> heatmap (higher-earner age × lower-earner age). Single -> bar
    chart of success by claim age. Returns a BytesIO PNG.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    NAVY_HEX, GOLD_HEX, CANYON_HEX = _brand()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.parse_math": False})
    ages = list(range(EARLIEST_CLAIM_AGE, LATEST_CLAIM_AGE + 1))
    best = best_strategy(results)
    buf = io.BytesIO()

    if len(people) == 1:
        lbl = people[0].label
        succ = {r.claim_ages[lbl]: r.prob_success for r in results}
        fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=180)
        vals = [succ[a] * 100 for a in ages]
        bars = ax.bar([str(a) for a in ages], vals, color=NAVY_HEX)
        best_age = best.claim_ages[lbl]
        bars[ages.index(best_age)].set_color(GOLD_HEX)
        ax.set_xlabel("Claim age"); ax.set_ylabel("Probability of success (%)")
        ax.set_title("Plan Success by Claiming Age", color=NAVY_HEX, fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}%",
                    ha="center", fontsize=8)
        fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf

    # Couples heatmap. Rows = higher-PIA earner, cols = lower-PIA earner.
    higher = max(people, key=lambda p: p.pia_monthly).label
    lower = min(people, key=lambda p: p.pia_monthly).label
    grid = np.full((len(ages), len(ages)), np.nan)
    lookup = {(r.claim_ages[higher], r.claim_ages[lower]): r.prob_success for r in results}
    for i, ha in enumerate(ages):
        for j, la in enumerate(ages):
            grid[i, j] = lookup[(ha, la)] * 100

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=180)
    im = ax.imshow(grid, cmap="YlGnBu", origin="lower", aspect="auto")
    ax.set_xticks(range(len(ages))); ax.set_xticklabels(ages)
    ax.set_yticks(range(len(ages))); ax.set_yticklabels(ages)
    ax.set_xlabel(f"{lower} claim age (lower earner)")
    ax.set_ylabel(f"{higher} claim age (higher earner)")
    ax.set_title("Probability of Plan Success by Claiming Combination",
                 color=NAVY_HEX, fontweight="bold", pad=12)
    # Annotate the recommended cell.
    bi, bj = ages.index(best.claim_ages[higher]), ages.index(best.claim_ages[lower])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor=CANYON_HEX, linewidth=3))
    span = np.nanmax(grid) - np.nanmin(grid)
    for i in range(len(ages)):
        for j in range(len(ages)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanmin(grid) + span * 0.55 else "#222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability of success (%)")
    fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf


def chart_ss_income_stream(hh: HouseholdSSInputs, claim_ages: Dict[str, int]):
    """Stacked annual household SS income (real $) over the horizon for the
    recommended strategy, with the survivor-transition marked. BytesIO PNG."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    NAVY_HEX, GOLD_HEX, CANYON_HEX = _brand()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.parse_math": False})
    ss = household_ss_real_by_year(hh, claim_ages)
    years = np.arange(1, hh.horizon_years + 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=180)
    ax.fill_between(years, ss / 1000, color=NAVY_HEX, alpha=0.85, step="pre")
    ax.set_xlabel("Simulation year"); ax.set_ylabel("Household SS income ($K, today's $)")
    ax.set_title("Social Security Income — Recommended Strategy",
                 color=NAVY_HEX, fontweight="bold")
    # Mark first death (survivor transition) if a couple.
    if len(hh.people) == 2 and hh.death_age:
        first_death_year = min(
            (hh.death_age[p.label] - p.current_age + 1)
            for p in hh.people if p.label in hh.death_age)
        if 1 <= first_death_year <= hh.horizon_years:
            ax.axvline(first_death_year, color=CANYON_HEX, linestyle="--",
                       linewidth=1.5, label="First death (survivor benefit begins)")
            ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf
