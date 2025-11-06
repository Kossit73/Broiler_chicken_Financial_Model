#!/usr/bin/env python3
"""Broiler chicken financial model without external dependencies.

This script codifies a 10-year broiler chicken production and financing model
using only the Python standard library.  It captures production assumptions,
per-cycle economics, annual rollups, debt service, free cash flow, and
valuation metrics (NPV/IRR).  Results can be exported as CSV and/or JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class Assumptions:
    farm_name: str = "Baseline Broiler Farm"
    cycles_per_year: int = 6
    birds_per_cycle: int = 20000
    mortality_rate: float = 0.05
    final_weight_kg: float = 2.5
    live_price_per_kg: float = 1.85
    chick_cost: float = 0.55
    feed_conversion_ratio: float = 1.65
    feed_cost_per_kg: float = 0.42
    processing_cost_per_bird: float = 0.18
    vaccination_cost_per_bird: float = 0.06
    litter_disposal_per_cycle: float = 3200.0
    propane_per_cycle: float = 4100.0
    electricity_per_cycle: float = 1800.0
    labor_per_cycle: float = 9500.0
    maintenance_per_cycle: float = 2400.0
    management_fee_per_cycle: float = 3500.0
    insurance_per_cycle: float = 1200.0
    overhead_per_cycle: float = 2700.0
    capex_housing: float = 950000.0
    capex_equipment: float = 280000.0
    working_capital: float = 60000.0
    discount_rate: float = 0.1
    price_growth: float = 0.02
    cost_inflation: float = 0.015
    tax_rate: float = 0.24
    debt_ratio: float = 0.55
    debt_interest_rate: float = 0.055
    debt_term_years: int = 7
    depreciation_years: int = 15
    maintenance_capex_annual: float = 25000.0


ASSUMPTION_SCHEDULE_LAYOUT = [
    ("Production", "Farm name", "farm_name"),
    ("Production", "Cycles per year", "cycles_per_year"),
    ("Production", "Birds placed per cycle", "birds_per_cycle"),
    ("Production", "Mortality rate", "mortality_rate"),
    ("Production", "Final weight (kg)", "final_weight_kg"),
    ("Production", "Feed conversion ratio", "feed_conversion_ratio"),
    ("Production", "Live price per kg", "live_price_per_kg"),
    ("Production", "Annual price growth", "price_growth"),
    ("Operating costs", "Feed cost per kg", "feed_cost_per_kg"),
    ("Operating costs", "Chick cost per bird", "chick_cost"),
    ("Operating costs", "Processing cost per bird", "processing_cost_per_bird"),
    ("Operating costs", "Vaccination cost per bird", "vaccination_cost_per_bird"),
    ("Operating costs", "Litter & disposal per cycle", "litter_disposal_per_cycle"),
    ("Operating costs", "Propane per cycle", "propane_per_cycle"),
    ("Operating costs", "Electricity per cycle", "electricity_per_cycle"),
    ("Operating costs", "Labor per cycle", "labor_per_cycle"),
    ("Operating costs", "Maintenance per cycle", "maintenance_per_cycle"),
    ("Operating costs", "Management fee per cycle", "management_fee_per_cycle"),
    ("Operating costs", "Insurance per cycle", "insurance_per_cycle"),
    ("Operating costs", "Overhead per cycle", "overhead_per_cycle"),
    ("Capital structure", "Housing capex", "capex_housing"),
    ("Capital structure", "Equipment capex", "capex_equipment"),
    ("Capital structure", "Maintenance capex (annual)", "maintenance_capex_annual"),
    ("Capital structure", "Working capital", "working_capital"),
    ("Capital structure", "Depreciation years", "depreciation_years"),
    ("Financing", "Debt ratio", "debt_ratio"),
    ("Financing", "Debt interest rate", "debt_interest_rate"),
    ("Financing", "Debt term (years)", "debt_term_years"),
    ("Financing", "Discount rate", "discount_rate"),
    ("Financing", "Cost inflation", "cost_inflation"),
    ("Financing", "Tax rate", "tax_rate"),
]

REVENUE_CATEGORIES = [
    "Broiler Revenue",
    "Eggs Revenue",
    "Poultry Manure Revenue",
    "Live Birds Revenue",
    "By-Product (feathers, offal, livers) Revenue",
]


@dataclass
class CycleResults:
    cycle: int
    survivors: int
    live_weight_kg: float
    revenue: float
    feed_cost: float
    chick_cost: float
    processing_cost: float
    health_cost: float
    energy_cost: float
    labor_cost: float
    overhead_cost: float
    total_cost: float
    gross_margin: float
    ebitda: float


@dataclass
class AnnualSummary:
    year: int
    revenue: float
    feed_cost: float
    chick_cost: float
    processing_cost: float
    health_cost: float
    energy_cost: float
    labor_cost: float
    overhead_cost: float
    total_cost: float
    ebitda: float
    depreciation: float
    ebit: float


@dataclass
class CashFlowRow:
    year: int
    revenue: float
    operating_expense: float
    ebitda: float
    depreciation: float
    interest_expense: float
    taxes: float
    operating_cash_flow: float
    maintenance_capex: float
    debt_service: float
    free_cash_flow: float
    discount_factor: float
    present_value: float


def amortization_schedule(principal: float, rate: float, term_years: int) -> List[Dict[str, float]]:
    if principal <= 0 or term_years <= 0:
        return []
    rate = float(rate)
    payment = _pmt(rate, term_years, principal)
    schedule = []
    balance = principal
    for year in range(1, term_years + 1):
        interest = balance * rate
        principal_paid = payment - interest
        balance = max(0.0, balance - principal_paid)
        schedule.append({
            "year": year,
            "payment": payment,
            "interest": interest,
            "principal": principal_paid,
            "balance": balance,
        })
    return schedule


def _pmt(rate: float, term_years: int, principal: float) -> float:
    if rate == 0:
        return principal / term_years
    factor = (1 + rate) ** term_years
    return principal * rate * factor / (factor - 1)


def compute_cycle(assumptions: Assumptions, cycle_number: int) -> CycleResults:
    survivors = round(assumptions.birds_per_cycle * (1 - assumptions.mortality_rate))
    live_weight = survivors * assumptions.final_weight_kg
    revenue = live_weight * assumptions.live_price_per_kg

    feed_required = survivors * assumptions.final_weight_kg * assumptions.feed_conversion_ratio
    feed_cost = feed_required * assumptions.feed_cost_per_kg
    chick_cost = assumptions.birds_per_cycle * assumptions.chick_cost
    processing_cost = survivors * assumptions.processing_cost_per_bird
    health_cost = survivors * assumptions.vaccination_cost_per_bird
    energy_cost = assumptions.propane_per_cycle + assumptions.electricity_per_cycle + assumptions.litter_disposal_per_cycle
    labor_cost = assumptions.labor_per_cycle
    overhead = (
        assumptions.maintenance_per_cycle
        + assumptions.management_fee_per_cycle
        + assumptions.insurance_per_cycle
        + assumptions.overhead_per_cycle
    )

    total_cost = feed_cost + chick_cost + processing_cost + health_cost + energy_cost + labor_cost + overhead
    gross_margin = revenue - (feed_cost + chick_cost + processing_cost + health_cost)
    ebitda = revenue - total_cost

    return CycleResults(
        cycle=cycle_number,
        survivors=survivors,
        live_weight_kg=live_weight,
        revenue=revenue,
        feed_cost=feed_cost,
        chick_cost=chick_cost,
        processing_cost=processing_cost,
        health_cost=health_cost,
        energy_cost=energy_cost,
        labor_cost=labor_cost,
        overhead_cost=overhead,
        total_cost=total_cost,
        gross_margin=gross_margin,
        ebitda=ebitda,
    )


def build_assumptions_schedule(assumptions: Assumptions) -> List[Dict[str, Any]]:
    """Return a tabular schedule summarising model assumptions grouped by schedule."""

    raw = asdict(assumptions)
    schedule_rows: List[Dict[str, Any]] = []
    for schedule, label, key in ASSUMPTION_SCHEDULE_LAYOUT:
        schedule_rows.append(
            {
                "schedule": schedule,
                "item": label,
                "value": raw.get(key),
            }
        )
    return schedule_rows


def build_revenue_schedules(
    assumptions: Assumptions, cycles: Iterable[CycleResults]
) -> Dict[str, List[Dict[str, Any]]]:
    """Return revenue schedules for each poultry revenue category."""

    schedules: Dict[str, List[Dict[str, Any]]] = {}

    # Broiler revenue is derived from modelled production cycles.
    unit_price = assumptions.final_weight_kg * assumptions.live_price_per_kg
    broiler_rows: List[Dict[str, Any]] = []
    for cycle in cycles:
        broiler_rows.append(
            {
                "Category": "Broiler Revenue",
                "Period": f"Cycle {cycle.cycle}",
                "Units": cycle.survivors,
                "Unit price": unit_price,
                "Revenue": cycle.revenue,
                "Notes": "Derived from production cycle results",
            }
        )
    schedules["Broiler Revenue"] = broiler_rows

    # Templates for other revenue categories so users can enter supplemental sales.
    template_periods = assumptions.cycles_per_year or 1
    for category in REVENUE_CATEGORIES[1:]:
        template_rows = []
        for period in range(1, template_periods + 1):
            template_rows.append(
                {
                    "Category": category,
                    "Period": f"Cycle {period}",
                    "Units": None,
                    "Unit price": None,
                    "Revenue": None,
                    "Notes": "Template (enter values)",
                }
            )
        schedules[category] = template_rows

    return schedules


def compute_cycles(assumptions: Assumptions) -> List[CycleResults]:
    return [compute_cycle(assumptions, cycle) for cycle in range(1, assumptions.cycles_per_year + 1)]


def annual_summary(assumptions: Assumptions, cycles: Iterable[CycleResults]) -> AnnualSummary:
    dep = (assumptions.capex_housing + assumptions.capex_equipment) / assumptions.depreciation_years
    totals = {
        "revenue": 0.0,
        "feed_cost": 0.0,
        "chick_cost": 0.0,
        "processing_cost": 0.0,
        "health_cost": 0.0,
        "energy_cost": 0.0,
        "labor_cost": 0.0,
        "overhead_cost": 0.0,
        "total_cost": 0.0,
        "ebitda": 0.0,
    }
    for cycle in cycles:
        totals["revenue"] += cycle.revenue
        totals["feed_cost"] += cycle.feed_cost
        totals["chick_cost"] += cycle.chick_cost
        totals["processing_cost"] += cycle.processing_cost
        totals["health_cost"] += cycle.health_cost
        totals["energy_cost"] += cycle.energy_cost
        totals["labor_cost"] += cycle.labor_cost
        totals["overhead_cost"] += cycle.overhead_cost
        totals["total_cost"] += cycle.total_cost
        totals["ebitda"] += cycle.ebitda

    ebit = totals["ebitda"] - dep
    return AnnualSummary(
        year=1,
        revenue=totals["revenue"],
        feed_cost=totals["feed_cost"],
        chick_cost=totals["chick_cost"],
        processing_cost=totals["processing_cost"],
        health_cost=totals["health_cost"],
        energy_cost=totals["energy_cost"],
        labor_cost=totals["labor_cost"],
        overhead_cost=totals["overhead_cost"],
        total_cost=totals["total_cost"],
        ebitda=totals["ebitda"],
        depreciation=dep,
        ebit=ebit,
    )


def discounted_cash_flow(assumptions: Assumptions, base_annual: AnnualSummary) -> List[CashFlowRow]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    equity = total_capex * (1 - assumptions.debt_ratio)
    debt = total_capex * assumptions.debt_ratio
    loan_schedule = amortization_schedule(debt, assumptions.debt_interest_rate, assumptions.debt_term_years)

    rows: List[CashFlowRow] = []
    depreciation = base_annual.depreciation
    revenue = base_annual.revenue
    operating_expense = base_annual.total_cost - base_annual.ebitda  # fixed portion already in EBITDA

    # Year 0 initial investment
    upfront_cash = -(equity + assumptions.working_capital)
    rows.append(
        CashFlowRow(
            year=0,
            revenue=0.0,
            operating_expense=0.0,
            ebitda=0.0,
            depreciation=0.0,
            interest_expense=0.0,
            taxes=0.0,
            operating_cash_flow=upfront_cash,
            maintenance_capex=0.0,
            debt_service=0.0,
            free_cash_flow=upfront_cash,
            discount_factor=1.0,
            present_value=upfront_cash,
        )
    )

    for year in range(1, 11):
        revenue *= (1 + assumptions.price_growth)
        variable_costs = (
            base_annual.feed_cost
            + base_annual.chick_cost
            + base_annual.processing_cost
            + base_annual.health_cost
        ) * ((1 + assumptions.cost_inflation) ** year)
        fixed_costs = (
            base_annual.energy_cost
            + base_annual.labor_cost
            + base_annual.overhead_cost
        ) * ((1 + assumptions.cost_inflation) ** year)

        ebitda = revenue - variable_costs - fixed_costs
        interest_expense = 0.0
        debt_service = 0.0
        if year <= len(loan_schedule):
            sched = loan_schedule[year - 1]
            interest_expense = sched["interest"]
            debt_service = sched["payment"]

        ebit = ebitda - depreciation - interest_expense
        taxable_income = max(0.0, ebit)
        taxes = taxable_income * assumptions.tax_rate
        operating_cash_flow = ebitda - taxes
        free_cash_flow = (
            operating_cash_flow
            - assumptions.maintenance_capex_annual
            - debt_service
        )
        discount_factor = (1 + assumptions.discount_rate) ** year
        present_value = free_cash_flow / discount_factor

        rows.append(
            CashFlowRow(
                year=year,
                revenue=revenue,
                operating_expense=variable_costs + fixed_costs,
                ebitda=ebitda,
                depreciation=depreciation,
                interest_expense=interest_expense,
                taxes=taxes,
                operating_cash_flow=operating_cash_flow,
                maintenance_capex=assumptions.maintenance_capex_annual,
                debt_service=debt_service,
                free_cash_flow=free_cash_flow,
                discount_factor=discount_factor,
                present_value=present_value,
            )
        )

    return rows


def npv(rate: float, cashflows: Iterable[float]) -> float:
    total = 0.0
    for year, cash in enumerate(cashflows):
        total += cash / ((1 + rate) ** year)
    return total


def irr(cashflows: Iterable[float], guess: float = 0.1) -> float:
    cashflows = list(cashflows)
    if not cashflows:
        return float("nan")

    def npv_at(rate: float) -> float:
        return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cashflows))

    lower, upper = -0.99, guess if guess > -0.99 else 0.1
    f_lower = npv_at(lower)
    f_upper = npv_at(upper)

    # Expand upper bound until sign change or max rate reached
    while f_lower * f_upper > 0 and upper < 10:
        upper += 0.5
        f_upper = npv_at(upper)

    if f_lower * f_upper > 0:
        return float("nan")

    for _ in range(200):
        mid = (lower + upper) / 2
        f_mid = npv_at(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_lower * f_mid < 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid
    return mid


def write_csv(path: Path, rows: Iterable[Dict[str, float]]):
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


def generate_model_outputs(assumptions: Assumptions) -> Dict[str, Any]:
    """Run the financial model and return structured outputs."""

    assumption_schedule = build_assumptions_schedule(assumptions)
    cycles = compute_cycles(assumptions)
    annual = annual_summary(assumptions, cycles)
    cashflows = discounted_cash_flow(assumptions, annual)
    revenue_schedules = build_revenue_schedules(assumptions, cycles)

    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    discount_rate = assumptions.discount_rate
    model_npv = npv(discount_rate, valuation_cashflows)
    model_irr = irr(valuation_cashflows)

    valuation = {
        "discount_rate": discount_rate,
        "npv": model_npv,
        "irr": model_irr,
        "initial_investment": cashflows[0].free_cash_flow,
        "terminal_year": cashflows[-1].year,
    }

    return {
        "assumptions": assumptions,
        "assumptions_schedule": assumption_schedule,
        "cycles": cycles,
        "annual": annual,
        "cashflows": cashflows,
        "revenue_schedules": revenue_schedules,
        "valuation": valuation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate broiler chicken financial model outputs.")
    parser.add_argument("--out", required=True, help="Directory where outputs will be written")
    parser.add_argument("--formats", nargs="+", default=["csv", "json"], choices=["csv", "json"], help="Output formats")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = generate_model_outputs(Assumptions())
    assumptions = results["assumptions"]
    assumption_schedule = results["assumptions_schedule"]
    cycles = results["cycles"]
    annual = results["annual"]
    cashflows = results["cashflows"]
    valuation = results["valuation"]
    revenue_schedules = results["revenue_schedules"]

    if "csv" in args.formats:
        write_csv(output_dir / "assumptions_summary.csv", assumption_schedule)
        write_csv(output_dir / "assumptions.csv", [{"name": k, "value": v} for k, v in asdict(assumptions).items()])
        write_csv(output_dir / "production_cycles.csv", [asdict(cycle) for cycle in cycles])
        write_csv(output_dir / "annual_summary.csv", [asdict(annual)])
        write_csv(output_dir / "cash_flow.csv", [asdict(row) for row in cashflows])
        for category, rows in revenue_schedules.items():
            safe = (
                category.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("-", "_")
            )
            write_csv(output_dir / f"{safe}_schedule.csv", rows)

    if "json" in args.formats:
        write_json(output_dir / "assumptions.json", asdict(assumptions))
        write_json(output_dir / "assumptions_summary.json", assumption_schedule)
        write_json(output_dir / "production_cycles.json", [asdict(cycle) for cycle in cycles])
        write_json(output_dir / "annual_summary.json", asdict(annual))
        write_json(output_dir / "cash_flow.json", [asdict(row) for row in cashflows])
        write_json(output_dir / "revenue_schedules.json", revenue_schedules)

    write_json(output_dir / "valuation.json", valuation)

    write_json(
        output_dir / "manifest.json",
        {
            "cycles_per_year": assumptions.cycles_per_year,
            "years": len(cashflows) - 1,
            "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        },
    )

    print("✅ Financial model generated. Outputs written to", output_dir)
    print(f"NPV @ {valuation['discount_rate']:.1%}: {valuation['npv']:,.0f}")
    print(f"IRR: {valuation['irr']:.2%}")


if __name__ == "__main__":
    main()
