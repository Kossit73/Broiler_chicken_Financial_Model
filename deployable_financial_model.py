#!/usr/bin/env python3
"""Broiler chicken financial model without external dependencies.

This script codifies a broiler chicken production and financing model using
only the Python standard library.  It captures production assumptions,
per-cycle economics, annual rollups, debt service, free cash flow, and
valuation metrics (NPV/IRR) across a configurable planning horizon.  Results
can be exported as CSV and/or JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Assumptions:
    farm_name: str = "Baseline Broiler Farm"
    cycles_per_year: int = 6
    production_horizon_years: int = 10
    birds_per_cycle: int = 20000
    mortality_rate: float = 0.05
    final_weight_kg: float = 2.5
    live_price_per_kg: float = 1.85
    eggs_price_per_dozen: float = 1.9
    manure_price_per_ton: float = 45.0
    live_bird_price_per_head: float = 1.5
    byproduct_price_per_kg: float = 0.35
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
    ("Production", "Production horizon (years)", "production_horizon_years"),
    ("Production", "Birds placed per cycle", "birds_per_cycle"),
    ("Production", "Mortality rate", "mortality_rate"),
    ("Production", "Final weight (kg)", "final_weight_kg"),
    ("Production", "Feed conversion ratio", "feed_conversion_ratio"),
    ("Production", "Live price per kg", "live_price_per_kg"),
    ("Production", "Eggs price per dozen", "eggs_price_per_dozen"),
    ("Production", "Manure price per ton", "manure_price_per_ton"),
    ("Production", "Live bird price per head", "live_bird_price_per_head"),
    ("Production", "By-product price per kg", "byproduct_price_per_kg"),
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


DEFAULT_CUSTOM_SIMULATION_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "Scenario": "Live price +5%",
        "Description": "Increase live bird price per kg by 5%",
        "Parameter": "live_price_per_kg",
        "Change type": "percent",
        "Change value": 5.0,
    },
    {
        "Scenario": "Feed cost -5%",
        "Description": "Reduce feed cost per kg by 5%",
        "Parameter": "feed_cost_per_kg",
        "Change type": "percent",
        "Change value": -5.0,
    },
    {
        "Scenario": "Mortality -1pp",
        "Description": "Lower mortality rate by 1 percentage point",
        "Parameter": "mortality_rate",
        "Change type": "absolute",
        "Change value": -0.01,
    },
    {
        "Scenario": "Price growth target 3%",
        "Description": "Set long-term price growth to 3%",
        "Parameter": "price_growth",
        "Change type": "target",
        "Change value": 0.03,
    },
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
    variable_costs: float
    fixed_costs: float
    operating_expense: float
    ebitda: float
    depreciation: float
    interest_expense: float
    taxes: float
    net_income: float
    operating_cash_flow: float
    maintenance_capex: float
    debt_service: float
    principal_payment: float
    free_cash_flow: float
    discount_factor: float
    present_value: float
    ending_debt: float
    cumulative_cash: float


@dataclass
class IncomeStatementRow:
    year: int
    revenue: float
    cogs: float
    gross_profit: float
    operating_expenses: float
    ebitda: float
    depreciation: float
    ebit: float
    interest: float
    taxes: float
    net_income: float
    ebitda_margin: float
    net_margin: float


@dataclass
class BalanceSheetRow:
    year: int
    cash: float
    working_capital: float
    net_ppe: float
    total_assets: float
    debt: float
    equity: float
    retained_earnings: float
    debt_to_equity: float | None


@dataclass
class CashFlowStatementRow:
    year: int
    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float
    net_change_in_cash: float
    ending_cash: float


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
    price_lookup = {
        "Eggs Revenue": assumptions.eggs_price_per_dozen,
        "Poultry Manure Revenue": assumptions.manure_price_per_ton,
        "Live Birds Revenue": assumptions.live_bird_price_per_head,
        "By-Product (feathers, offal, livers) Revenue": assumptions.byproduct_price_per_kg,
    }

    for category in REVENUE_CATEGORIES[1:]:
        template_rows = []
        for period in range(1, template_periods + 1):
            template_rows.append(
                {
                    "Category": category,
                    "Period": f"Cycle {period}",
                    "Units": None,
                    "Unit price": price_lookup.get(category),
                    "Revenue": None,
                    "Notes": "Template (enter values)",
                }
            )
        schedules[category] = template_rows

    return schedules


def summarise_revenue_totals(
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
    cycles_per_year: int,
    projection_years: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate revenue schedules into annual totals per category and overall.

    The helper re-computes per-row revenue using Units × Unit price when
    possible before summing, ensuring the annual views mirror the underlying
    cycle-level data even if a user has not explicitly edited the revenue
    column.
    """

    per_category: List[Dict[str, Any]] = []
    per_year_totals: Dict[int, float] = {}
    cycles = int(cycles_per_year) if cycles_per_year else 0
    cycles = max(cycles, 1)
    years = int(projection_years) if projection_years else 0
    years = max(years, 1)

    for category, rows in revenue_schedules.items():
        if not rows:
            continue

        category_totals: Dict[int, float] = {}
        for idx, row in enumerate(rows):
            units = _to_float(row.get("Units"))
            unit_price = _to_float(row.get("Unit price"))
            revenue_value = _to_float(row.get("Revenue"))

            value: Optional[float] = None
            if units is not None and unit_price is not None:
                value = units * unit_price
                row["Revenue"] = value
            elif revenue_value is not None:
                value = revenue_value
                row["Revenue"] = revenue_value

            if value is None:
                continue

            year = (idx // cycles) + 1
            category_totals[year] = category_totals.get(year, 0.0) + value

        for year, total in sorted(category_totals.items()):
            total_float = float(total)
            per_category.append(
                {"Category": category, "Year": int(year), "Revenue": total_float}
            )
            per_year_totals[year] = per_year_totals.get(year, 0.0) + total_float

    annual_totals: List[Dict[str, Any]] = []
    max_year = max(per_year_totals.keys(), default=0)
    horizon = max(years, max_year)
    for year in range(1, horizon + 1):
        total = per_year_totals.get(year, 0.0)
        annual_totals.append({"Year": year, "Revenue": float(total)})

    return {"by_category": per_category, "annual_totals": annual_totals}


def _to_float(value: Any) -> Optional[float]:
    """Return a float when ``value`` is numeric, otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def discounted_cash_flow(
    assumptions: Assumptions, base_annual: AnnualSummary
) -> Tuple[List[CashFlowRow], List[Dict[str, float]]]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    equity = total_capex * (1 - assumptions.debt_ratio)
    debt = total_capex * assumptions.debt_ratio
    loan_schedule = amortization_schedule(debt, assumptions.debt_interest_rate, assumptions.debt_term_years)

    rows: List[CashFlowRow] = []
    depreciation = base_annual.depreciation
    revenue = base_annual.revenue
    # Year 0 initial investment
    upfront_cash = -(equity + assumptions.working_capital)
    cumulative_cash = upfront_cash
    rows.append(
        CashFlowRow(
            year=0,
            revenue=0.0,
            variable_costs=0.0,
            fixed_costs=0.0,
            operating_expense=0.0,
            ebitda=0.0,
            depreciation=0.0,
            interest_expense=0.0,
            taxes=0.0,
            net_income=0.0,
            operating_cash_flow=upfront_cash,
            maintenance_capex=0.0,
            debt_service=0.0,
            principal_payment=0.0,
            free_cash_flow=upfront_cash,
            discount_factor=1.0,
            present_value=upfront_cash,
            ending_debt=debt,
            cumulative_cash=cumulative_cash,
        )
    )

    projection_years = int(assumptions.production_horizon_years)
    if projection_years <= 0:
        projection_years = 1

    for year in range(1, projection_years + 1):
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
        principal_payment = 0.0
        ending_balance = 0.0
        if year <= len(loan_schedule):
            sched = loan_schedule[year - 1]
            interest_expense = sched["interest"]
            debt_service = sched["payment"]
            principal_payment = sched["principal"]
            ending_balance = sched["balance"]

        ebit = ebitda - depreciation
        taxable_income = max(0.0, ebit - interest_expense)
        taxes = taxable_income * assumptions.tax_rate
        net_income = ebit - interest_expense - taxes
        operating_cash_flow = ebitda - taxes
        free_cash_flow = (
            operating_cash_flow
            - assumptions.maintenance_capex_annual
            - debt_service
        )
        discount_factor = (1 + assumptions.discount_rate) ** year
        present_value = free_cash_flow / discount_factor
        cumulative_cash += free_cash_flow

        rows.append(
            CashFlowRow(
                year=year,
                revenue=revenue,
                variable_costs=variable_costs,
                fixed_costs=fixed_costs,
                operating_expense=variable_costs + fixed_costs,
                ebitda=ebitda,
                depreciation=depreciation,
                interest_expense=interest_expense,
                taxes=taxes,
                net_income=net_income,
                operating_cash_flow=operating_cash_flow,
                maintenance_capex=assumptions.maintenance_capex_annual,
                debt_service=debt_service,
                principal_payment=principal_payment,
                free_cash_flow=free_cash_flow,
                discount_factor=discount_factor,
                present_value=present_value,
                ending_debt=ending_balance,
                cumulative_cash=cumulative_cash,
            )
        )

    return rows, loan_schedule


def build_financial_statements(
    assumptions: Assumptions,
    cashflows: List[CashFlowRow],
    loan_schedule: List[Dict[str, float]],
) -> Dict[str, List[Any]]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    depreciation = (assumptions.capex_housing + assumptions.capex_equipment) / assumptions.depreciation_years
    equity_base = total_capex * (1 - assumptions.debt_ratio)
    equity_total = equity_base + assumptions.working_capital

    income_rows: List[IncomeStatementRow] = []
    cash_statement: List[CashFlowStatementRow] = []
    balance_rows: List[BalanceSheetRow] = []

    # Cash flow statement year 0 (construction / funding)
    investing_cash = -(total_capex + assumptions.working_capital)
    financing_cash = equity_total + (total_capex * assumptions.debt_ratio)
    net_change = investing_cash + financing_cash
    cash_balance = net_change
    cash_statement.append(
        CashFlowStatementRow(
            year=0,
            operating_cash_flow=0.0,
            investing_cash_flow=investing_cash,
            financing_cash_flow=financing_cash,
            net_change_in_cash=net_change,
            ending_cash=cash_balance,
        )
    )

    for row in cashflows:
        if row.year == 0:
            balance_rows.append(
                BalanceSheetRow(
                    year=0,
                    cash=cash_balance,
                    working_capital=assumptions.working_capital,
                    net_ppe=total_capex,
                    total_assets=total_capex + assumptions.working_capital + cash_balance,
                    debt=total_capex * assumptions.debt_ratio,
                    equity=equity_total + cash_balance,
                    retained_earnings=0.0,
                    debt_to_equity=(total_capex * assumptions.debt_ratio) / equity_total if equity_total else None,
                )
            )
            continue

        # Income statement
        gross_profit = row.revenue - row.variable_costs
        ebit = row.ebitda - row.depreciation
        income_rows.append(
            IncomeStatementRow(
                year=row.year,
                revenue=row.revenue,
                cogs=row.variable_costs,
                gross_profit=gross_profit,
                operating_expenses=row.fixed_costs,
                ebitda=row.ebitda,
                depreciation=row.depreciation,
                ebit=ebit,
                interest=row.interest_expense,
                taxes=row.taxes,
                net_income=row.net_income,
                ebitda_margin=(row.ebitda / row.revenue) if row.revenue else 0.0,
                net_margin=(row.net_income / row.revenue) if row.revenue else 0.0,
            )
        )

        # Cash flow statement (operating/investing/financing for the year)
        operating_cash = row.net_income + row.depreciation
        investing_cash = -assumptions.maintenance_capex_annual
        financing_cash = -row.principal_payment
        net_change = operating_cash + investing_cash + financing_cash
        cash_balance += net_change
        cash_statement.append(
            CashFlowStatementRow(
                year=row.year,
                operating_cash_flow=operating_cash,
                investing_cash_flow=investing_cash,
                financing_cash_flow=financing_cash,
                net_change_in_cash=net_change,
                ending_cash=cash_balance,
            )
        )

        # Balance sheet
        accum_dep = min(row.year, assumptions.depreciation_years) * depreciation
        net_ppe = max(0.0, total_capex - accum_dep)
        debt_balance = row.ending_debt if row.ending_debt else 0.0
        total_assets = cash_balance + assumptions.working_capital + net_ppe
        equity = total_assets - debt_balance
        retained = equity - equity_total
        debt_to_equity = (debt_balance / equity) if equity else None
        balance_rows.append(
            BalanceSheetRow(
                year=row.year,
                cash=cash_balance,
                working_capital=assumptions.working_capital,
                net_ppe=net_ppe,
                total_assets=total_assets,
                debt=debt_balance,
                equity=equity,
                retained_earnings=retained,
                debt_to_equity=debt_to_equity,
            )
        )

    return {
        "income_statement": income_rows,
        "balance_sheet": balance_rows,
        "cash_flow_statement": cash_statement,
        "loan_schedule": loan_schedule,
    }


def _calculate_payback_period(cashflows: Iterable[CashFlowRow]) -> float:
    cumulative = 0.0
    previous = 0.0
    payback = float("nan")
    for row in cashflows:
        cumulative += row.free_cash_flow
        if row.year == 0:
            previous = cumulative
            continue
        if cumulative >= 0 and payback != payback:
            delta = cumulative - previous
            if delta != 0:
                fraction = (0 - previous) / delta
                payback = (row.year - 1) + max(0.0, min(1.0, fraction))
            else:
                payback = float(row.year)
            break
        previous = cumulative
    return payback


def _compute_dscr_series(
    cashflows: Iterable[CashFlowRow],
) -> Tuple[List[Dict[str, float]], float, float]:
    rows: List[Dict[str, float]] = []
    values: List[float] = []
    for row in cashflows:
        if row.year == 0 or not row.debt_service:
            continue
        cash_available = row.operating_cash_flow + row.interest_expense
        try:
            dscr = cash_available / row.debt_service
        except ZeroDivisionError:
            dscr = float("nan")
        rows.append({"year": row.year, "dscr": dscr})
        if dscr == dscr:
            values.append(dscr)
    average = sum(values) / len(values) if values else float("nan")
    minimum = min(values) if values else float("nan")
    return rows, average, minimum


def _percentile(data: List[float], percentile: float) -> float:
    if not data:
        return float("nan")
    if percentile <= 0:
        return float(sorted(data)[0])
    if percentile >= 1:
        return float(sorted(data)[-1])
    ordered = sorted(data)
    k = (len(ordered) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)


def _evaluate_case_metrics(
    assumptions: Assumptions,
    include_details: bool = False,
) -> Dict[str, Any]:
    cycles = compute_cycles(assumptions)
    annual = annual_summary(assumptions, cycles)
    cashflows, _ = discounted_cash_flow(assumptions, annual)
    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    dscr_rows, avg_dscr, min_dscr = _compute_dscr_series(cashflows)
    summary = {
        "npv": npv(assumptions.discount_rate, valuation_cashflows),
        "irr": irr(valuation_cashflows),
        "payback": _calculate_payback_period(cashflows),
        "avg_dscr": avg_dscr,
        "min_dscr": min_dscr,
        "terminal_cash": cashflows[-1].cumulative_cash if cashflows else float("nan"),
    }
    result: Dict[str, Any] = {"metrics": summary, "dscr_rows": dscr_rows}
    if include_details:
        result["cashflows"] = cashflows
        result["annual"] = annual
    return result


def perform_what_if_analysis(
    assumptions: Assumptions, base_metrics: Dict[str, float]
) -> List[Dict[str, Any]]:
    scenarios = [
        {
            "name": "Baseline",
            "description": "Current assumptions",
            "changes": {},
        },
        {
            "name": "Live price +10%",
            "description": "Increase live bird price per kg by 10%",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 1.10
            },
        },
        {
            "name": "Live price -10%",
            "description": "Reduce live bird price per kg by 10%",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 0.90
            },
        },
        {
            "name": "Feed cost +10%",
            "description": "Increase feed cost per kg by 10%",
            "changes": {
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 1.10
            },
        },
        {
            "name": "Feed cost -10%",
            "description": "Reduce feed cost per kg by 10%",
            "changes": {
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 0.90
            },
        },
        {
            "name": "Mortality +2pp",
            "description": "Increase mortality rate by 2 percentage points",
            "changes": {
                "mortality_rate": max(0.0, assumptions.mortality_rate + 0.02)
            },
        },
        {
            "name": "Mortality -2pp",
            "description": "Decrease mortality rate by 2 percentage points",
            "changes": {
                "mortality_rate": max(0.0, assumptions.mortality_rate - 0.02)
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        if scenario["name"] == "Baseline":
            metrics = base_metrics
        else:
            mutated = replace(assumptions, **scenario["changes"])
            metrics = _evaluate_case_metrics(mutated)["metrics"]
        entry = {
            "Scenario": scenario["name"],
            "Description": scenario["description"],
            "NPV": metrics["npv"],
            "NPV Δ": metrics["npv"] - base_metrics["npv"],
            "IRR": metrics["irr"],
            "Avg DSCR": metrics["avg_dscr"],
            "Min DSCR": metrics["min_dscr"],
            "Payback": metrics["payback"],
        }
        results.append(entry)
    return results


def run_custom_simulations(
    assumptions: Assumptions,
    base_metrics: Dict[str, float],
    definitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate custom simulation definitions against the base assumptions."""

    processed: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    base_entry = {
        "Scenario": "Baseline",
        "Description": "Unmodified assumptions",
        "Parameter": "-",
        "Change type": "-",
        "Change value": 0.0,
        "NPV": base_metrics.get("npv", float("nan")),
        "NPV Δ": 0.0,
        "IRR": base_metrics.get("irr", float("nan")),
        "Avg DSCR": base_metrics.get("avg_dscr", float("nan")),
        "Min DSCR": base_metrics.get("min_dscr", float("nan")),
        "Payback": base_metrics.get("payback", float("nan")),
    }
    results.append(base_entry)

    for definition in definitions:
        if not isinstance(definition, dict):
            continue

        name = str(definition.get("Scenario") or definition.get("name") or "").strip()
        parameter = str(
            definition.get("Parameter") or definition.get("parameter") or ""
        ).strip()
        change_type = str(
            definition.get("Change type")
            or definition.get("change_type")
            or "percent"
        ).strip().lower()
        change_value_raw = definition.get("Change value") or definition.get("change_value")
        description = definition.get("Description") or definition.get("description") or ""

        if not name or not parameter or not hasattr(assumptions, parameter):
            continue

        try:
            change_value = float(change_value_raw)
        except (TypeError, ValueError):
            continue

        current_value = getattr(assumptions, parameter)
        if current_value is None:
            continue

        if change_type in {"percent", "percentage", "%"}:
            mutated_value = current_value * (1.0 + change_value / 100.0)
            applied_type = "percent"
        elif change_type in {"absolute", "delta"}:
            mutated_value = current_value + change_value
            applied_type = "absolute"
        elif change_type in {"target", "value"}:
            mutated_value = change_value
            applied_type = "target"
        else:
            # Unknown change type – skip this definition
            continue

        mutated = replace(assumptions, **{parameter: mutated_value})
        metrics = _evaluate_case_metrics(mutated)["metrics"]
        entry = {
            "Scenario": name,
            "Description": description,
            "Parameter": parameter,
            "Change type": applied_type,
            "Change value": change_value,
            "NPV": metrics["npv"],
            "NPV Δ": metrics["npv"] - base_metrics.get("npv", 0.0),
            "IRR": metrics["irr"],
            "Avg DSCR": metrics["avg_dscr"],
            "Min DSCR": metrics["min_dscr"],
            "Payback": metrics["payback"],
        }
        results.append(entry)
        processed.append(
            {
                "Scenario": name,
                "Description": description,
                "Parameter": parameter,
                "Change type": applied_type,
                "Change value": change_value,
            }
        )

    return {"definitions": processed, "results": results}


def run_monte_carlo_analysis(
    assumptions: Assumptions, iterations: int = 200
) -> Dict[str, Any]:
    rng = random.Random(42)
    npv_results: List[float] = []
    irr_results: List[float] = []
    min_dscr_results: List[float] = []
    samples: List[Dict[str, float]] = []

    for idx in range(1, iterations + 1):
        varied = replace(
            assumptions,
            live_price_per_kg=assumptions.live_price_per_kg
            * rng.uniform(0.9, 1.1),
            feed_cost_per_kg=assumptions.feed_cost_per_kg * rng.uniform(0.9, 1.1),
            mortality_rate=max(
                0.0, min(0.25, assumptions.mortality_rate + rng.uniform(-0.02, 0.02))
            ),
            price_growth=max(
                -0.05, min(0.10, assumptions.price_growth + rng.uniform(-0.01, 0.03))
            ),
            cost_inflation=max(
                0.0, min(0.05, assumptions.cost_inflation + rng.uniform(-0.005, 0.02))
            ),
        )
        evaluation = _evaluate_case_metrics(varied)["metrics"]
        npv_results.append(evaluation["npv"])
        irr_results.append(evaluation["irr"])
        min_dscr_results.append(evaluation["min_dscr"])
        samples.append(
            {
                "iteration": idx,
                "live_price_per_kg": varied.live_price_per_kg,
                "feed_cost_per_kg": varied.feed_cost_per_kg,
                "mortality_rate": varied.mortality_rate,
                "price_growth": varied.price_growth,
                "cost_inflation": varied.cost_inflation,
                "npv": evaluation["npv"],
                "irr": evaluation["irr"],
                "min_dscr": evaluation["min_dscr"],
            }
        )

    summary = {
        "iterations": iterations,
        "mean_npv": float(sum(npv_results) / iterations) if iterations else float("nan"),
        "p5_npv": _percentile(npv_results, 0.05),
        "p95_npv": _percentile(npv_results, 0.95),
        "probability_negative_npv": float(
            sum(1 for value in npv_results if value < 0) / iterations
        )
        if iterations
        else float("nan"),
        "mean_irr": float(sum(irr_results) / iterations) if iterations else float("nan"),
        "mean_min_dscr": float(sum(min_dscr_results) / iterations)
        if iterations
        else float("nan"),
    }

    return {"summary": summary, "samples": samples}


def break_even_analysis(
    annual: AnnualSummary,
    revenue_summary: Dict[str, List[Dict[str, Any]]],
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = {}
    for row in revenue_summary.get("by_category", []):
        category = row.get("Category")
        revenue = _to_float(row.get("Revenue"))
        if category is None or revenue is None:
            continue
        totals[category] = totals.get(category, 0.0) + revenue

    total_revenue = sum(totals.values())
    total_cost = annual.total_cost

    results: List[Dict[str, Any]] = []
    for category in REVENUE_CATEGORIES:
        schedule = revenue_schedules.get(category, [])
        revenue = totals.get(category, 0.0)
        units = 0.0
        price_values: List[float] = []
        for row in schedule:
            unit_value = _to_float(row.get("Units"))
            unit_price = _to_float(row.get("Unit price"))
            if unit_value is not None:
                units += unit_value
            if unit_price is not None:
                price_values.append(unit_price)
        avg_price = (revenue / units) if units else (sum(price_values) / len(price_values) if price_values else float("nan"))
        allocated_cost = (
            total_cost * (revenue / total_revenue)
            if total_revenue > 0 and revenue > 0
            else float("nan")
        )
        break_even_units = (
            allocated_cost / avg_price if avg_price and avg_price == avg_price else float("nan")
        )
        break_even_price = (
            allocated_cost / units if units and allocated_cost == allocated_cost else float("nan")
        )
        results.append(
            {
                "Category": category,
                "Annual revenue": revenue,
                "Allocated cost": allocated_cost,
                "Average unit price": avg_price,
                "Total units": units if units else float("nan"),
                "Break-even units": break_even_units,
                "Break-even unit price": break_even_price,
            }
        )
    return results


def goal_seek_live_price(
    assumptions: Assumptions,
    target_npv: float = 0.0,
    tolerance: float = 1_000.0,
    max_iterations: int = 25,
) -> Dict[str, Any]:
    base_price = assumptions.live_price_per_kg
    lower = base_price * 0.5
    upper = base_price * 1.8
    best_price = base_price
    best_metrics = _evaluate_case_metrics(assumptions)["metrics"]
    for _ in range(max_iterations):
        candidate = (lower + upper) / 2
        trial_assumptions = replace(assumptions, live_price_per_kg=candidate)
        metrics = _evaluate_case_metrics(trial_assumptions)["metrics"]
        difference = metrics["npv"] - target_npv
        best_price = candidate
        best_metrics = metrics
        if abs(difference) <= tolerance:
            status = "converged"
            break
        if difference > 0:
            upper = candidate
        else:
            lower = candidate
    else:
        status = "max_iterations"
    return {
        "target": "NPV",
        "target_value": target_npv,
        "status": status,
        "required_live_price_per_kg": best_price,
        "resulting_npv": best_metrics.get("npv"),
        "resulting_irr": best_metrics.get("irr"),
        "resulting_avg_dscr": best_metrics.get("avg_dscr"),
    }


def _linear_regression(series: List[float]) -> Tuple[float, float]:
    n = len(series)
    if n < 2:
        return 0.0, series[0] if series else 0.0
    x = list(range(n))
    sum_x = sum(x)
    sum_y = sum(series)
    sum_xy = sum(i * y for i, y in zip(x, series))
    sum_x2 = sum(i * i for i in x)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        slope = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _forecast_linear(series: List[float], steps: int) -> List[float]:
    slope, intercept = _linear_regression(series)
    forecasts: List[float] = []
    start = len(series)
    for step in range(steps):
        idx = start + step
        forecasts.append(intercept + slope * idx)
    return forecasts


def _forecast_ar1(series: List[float], steps: int) -> List[float]:
    n = len(series)
    if n < 2:
        return [series[-1] if series else 0.0] * steps
    mean = sum(series) / n
    num = sum((series[i - 1] - mean) * (series[i] - mean) for i in range(1, n))
    den = sum((series[i - 1] - mean) ** 2 for i in range(1, n))
    phi = num / den if den else 0.0
    c = mean * (1 - phi)
    forecasts: List[float] = []
    prev = series[-1]
    for _ in range(steps):
        next_val = c + phi * prev
        forecasts.append(next_val)
        prev = next_val
    return forecasts


def build_predictive_analytics(
    cashflows: List[CashFlowRow],
    income_statement: List[IncomeStatementRow],
    horizon: int = 5,
) -> Dict[str, Any]:
    historical_years = [row.year for row in cashflows if row.year > 0]
    revenue_series = [row.revenue for row in cashflows if row.year > 0]
    ebitda_series = [row.ebitda for row in cashflows if row.year > 0]
    net_income_series = [row.net_income for row in cashflows if row.year > 0]

    forecasts: List[Dict[str, float]] = []
    if revenue_series:
        revenue_forecast = _forecast_linear(revenue_series, horizon)
        ebitda_forecast = _forecast_linear(ebitda_series, horizon)
        net_income_forecast = _forecast_linear(net_income_series, horizon)
        for idx in range(horizon):
            year = (historical_years[-1] if historical_years else 0) + idx + 1
            forecasts.append(
                {
                    "Year": year,
                    "Revenue forecast": revenue_forecast[idx],
                    "EBITDA forecast": ebitda_forecast[idx] if ebitda_series else float("nan"),
                    "Net income forecast": net_income_forecast[idx]
                    if net_income_series
                    else float("nan"),
                }
            )

    arima_forecast = _forecast_ar1(revenue_series, horizon) if revenue_series else []

    growth_rates: List[float] = []
    anomalies: List[Dict[str, Any]] = []
    for idx in range(1, len(revenue_series)):
        prev = revenue_series[idx - 1]
        current = revenue_series[idx]
        if prev:
            growth = (current - prev) / prev
            growth_rates.append(growth)
    if growth_rates:
        mean_growth = sum(growth_rates) / len(growth_rates)
        variance = sum((g - mean_growth) ** 2 for g in growth_rates) / len(growth_rates)
        std_dev = math.sqrt(variance)
        for idx, growth in enumerate(growth_rates, start=1):
            z = (growth - mean_growth) / std_dev if std_dev else 0.0
            anomalies.append(
                {
                    "Year": historical_years[idx] if idx < len(historical_years) else idx,
                    "Growth": growth,
                    "Z-score": z,
                    "Flag": abs(z) > 2.0,
                }
            )
    else:
        mean_growth = float("nan")
        std_dev = float("nan")

    ml_methods = []
    if revenue_series:
        slope, intercept = _linear_regression(revenue_series)
        ml_methods.append(
            {
                "Method": "Linear regression",
                "Slope": slope,
                "Intercept": intercept,
            }
        )
    if ebitda_series:
        slope, intercept = _linear_regression(ebitda_series)
        ml_methods.append(
            {
                "Method": "EBITDA trend regression",
                "Slope": slope,
                "Intercept": intercept,
            }
        )

    time_series_analysis = {
        "method": "AR(1)",
        "forecast": [
            {
                "Year": (historical_years[-1] if historical_years else 0) + idx + 1,
                "Revenue forecast": value,
            }
            for idx, value in enumerate(arima_forecast)
        ],
    }

    risk_detection = {
        "mean_growth": mean_growth,
        "std_growth": std_dev,
        "observations": anomalies,
    }

    return {
        "automated_forecast": forecasts,
        "time_series": time_series_analysis,
        "risk_anomalies": risk_detection,
        "ml_methods": ml_methods,
    }


def scenario_planning(
    assumptions: Assumptions,
    base_summary: Dict[str, float],
    base_revenue: float,
) -> List[Dict[str, Any]]:
    scenarios = [
        {
            "name": "Baseline",
            "changes": {},
            "description": "Current assumption set",
        },
        {
            "name": "Downside",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 0.92,
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 1.08,
                "mortality_rate": min(0.4, assumptions.mortality_rate + 0.03),
            },
            "description": "Price compression with higher feed and mortality",
        },
        {
            "name": "Upside",
            "changes": {
                "live_price_per_kg": assumptions.live_price_per_kg * 1.08,
                "feed_cost_per_kg": assumptions.feed_cost_per_kg * 0.95,
                "mortality_rate": max(0.0, assumptions.mortality_rate - 0.02),
            },
            "description": "Pricing tailwinds and efficiency gains",
        },
        {
            "name": "Expansion",
            "changes": {
                "cycles_per_year": assumptions.cycles_per_year + 1,
                "birds_per_cycle": int(assumptions.birds_per_cycle * 1.05),
            },
            "description": "Add capacity through an extra cycle and larger placements",
        },
    ]

    results: List[Dict[str, Any]] = [
        {
            "Scenario": "Baseline",
            "Description": "Current assumption set",
            "Revenue": base_revenue,
            "NPV": base_summary["npv"],
            "IRR": base_summary["irr"],
            "Avg DSCR": base_summary["avg_dscr"],
            "Payback": base_summary["payback"],
            "NPV Δ": 0.0,
        }
    ]

    for scenario in scenarios[1:]:
        mutated = replace(assumptions, **scenario["changes"])
        evaluation = _evaluate_case_metrics(mutated, include_details=True)
        metrics = evaluation["metrics"]
        annual = evaluation.get("annual")
        revenue = annual.revenue if annual else float("nan")
        results.append(
            {
                "Scenario": scenario["name"],
                "Description": scenario["description"],
                "Revenue": revenue,
                "NPV": metrics["npv"],
                "IRR": metrics["irr"],
                "Avg DSCR": metrics["avg_dscr"],
                "Payback": metrics["payback"],
                "NPV Δ": metrics["npv"] - base_summary["npv"],
            }
        )
    return results


def _safe_div(numerator: float, denominator: float) -> float:
    try:
        if denominator is None:
            return float("nan")
        denom = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if denom == 0 or math.isnan(denom):
        return float("nan")
    try:
        return float(numerator) / denom
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: Iterable[float]) -> float:
    data = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if not data:
        return float("nan")
    return sum(data) / len(data)


def compute_advanced_analytics(
    assumptions: Assumptions,
    cashflows: List[CashFlowRow],
    income_statement: List[IncomeStatementRow],
    balance_sheet: List[BalanceSheetRow],
    revenue_summary: Dict[str, List[Dict[str, Any]]],
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
    annual: AnnualSummary,
) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []

    if income_statement:
        avg_ebitda_margin = sum(row.ebitda_margin for row in income_statement) / len(income_statement)
        avg_net_margin = sum(row.net_margin for row in income_statement) / len(income_statement)
    else:
        avg_ebitda_margin = 0.0
        avg_net_margin = 0.0

    valuation_cashflows = [row.free_cash_flow for row in cashflows]
    dscr_rows, avg_dscr, min_dscr = _compute_dscr_series(cashflows)
    payback = _calculate_payback_period(cashflows)
    base_metrics = {
        "npv": npv(assumptions.discount_rate, valuation_cashflows),
        "irr": irr(valuation_cashflows),
        "avg_dscr": avg_dscr,
        "min_dscr": min_dscr,
        "payback": payback,
    }

    income_by_year = {row.year: row for row in income_statement}
    balance_by_year = {row.year: row for row in balance_sheet}

    returns_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    leverage_rows: List[Dict[str, float]] = []

    for row in cashflows:
        if row.year == 0:
            continue

        income_row = income_by_year.get(row.year)
        balance_row = balance_by_year.get(row.year)

        # Returns analysis
        if income_row and balance_row:
            roa = _safe_div(income_row.net_income, balance_row.total_assets)
            roe = _safe_div(income_row.net_income, balance_row.equity)
            invested_capital = (
                (balance_row.debt or 0.0)
                + (balance_row.equity or 0.0)
                - (balance_row.cash or 0.0)
            )
            nopat = (income_row.net_income or 0.0) + (income_row.interest or 0.0)
            roic = _safe_div(nopat, invested_capital)
            returns_rows.append(
                {
                    "year": row.year,
                    "return_on_assets": roa,
                    "return_on_equity": roe,
                    "return_on_invested_capital": roic,
                }
            )

            debt_to_equity = balance_row.debt_to_equity
            debt_ratio = _safe_div(balance_row.debt, balance_row.total_assets)
            leverage_rows.append(
                {
                    "year": row.year,
                    "debt_to_equity": debt_to_equity,
                    "debt_ratio": debt_ratio,
                    "ending_debt": row.ending_debt,
                }
            )

        interest_coverage = float("nan")
        if income_row:
            interest_coverage = _safe_div(income_row.ebit, income_row.interest)

        fcf_to_debt_service = _safe_div(row.free_cash_flow, row.debt_service)
        maintenance_cov = _safe_div(
            row.operating_cash_flow, row.maintenance_capex
        )
        opening_debt = (row.ending_debt or 0.0) + (row.principal_payment or 0.0)
        paydown_velocity = _safe_div(row.principal_payment, opening_debt)
        coverage_rows.append(
            {
                "year": row.year,
                "interest_coverage": interest_coverage,
                "fcf_to_debt_service": fcf_to_debt_service,
                "maintenance_capex_coverage": maintenance_cov,
                "debt_paydown_velocity": paydown_velocity,
            }
        )

    metrics.extend(
        [
            {"metric": "Average EBITDA margin", "value": avg_ebitda_margin},
            {"metric": "Average net margin", "value": avg_net_margin},
            {"metric": "Average DSCR", "value": avg_dscr},
            {"metric": "Minimum DSCR", "value": min_dscr},
            {"metric": "Payback period (years)", "value": payback},
            {
                "metric": "Average return on assets",
                "value": _nanmean(r["return_on_assets"] for r in returns_rows),
            },
            {
                "metric": "Average return on equity",
                "value": _nanmean(r["return_on_equity"] for r in returns_rows),
            },
            {
                "metric": "Average return on invested capital",
                "value": _nanmean(
                    r["return_on_invested_capital"] for r in returns_rows
                ),
            },
            {
                "metric": "Average interest coverage",
                "value": _nanmean(r["interest_coverage"] for r in coverage_rows),
            },
            {
                "metric": "Average free cash flow to debt service",
                "value": _nanmean(r["fcf_to_debt_service"] for r in coverage_rows),
            },
            {
                "metric": "Average maintenance capex coverage",
                "value": _nanmean(
                    r["maintenance_capex_coverage"] for r in coverage_rows
                ),
            },
            {
                "metric": "Average debt paydown velocity",
                "value": _nanmean(r["debt_paydown_velocity"] for r in coverage_rows),
            },
            {"metric": "Base case NPV", "value": base_metrics["npv"]},
            {"metric": "Base case IRR", "value": base_metrics["irr"]},
        ]
    )

    trend_rows = [
        {
            "year": row.year,
            "revenue": row.revenue,
            "ebitda": row.ebitda,
            "net_income": row.net_income,
            "free_cash_flow": row.free_cash_flow,
            "cumulative_cash": row.cumulative_cash,
        }
        for row in cashflows
        if row.year > 0
    ]

    return {
        "metrics": metrics,
        "dscr": dscr_rows,
        "trend": trend_rows,
        "returns": returns_rows,
        "coverage": coverage_rows,
        "leverage": leverage_rows,
        "what_if": perform_what_if_analysis(assumptions, base_metrics),
        "monte_carlo": run_monte_carlo_analysis(assumptions),
        "break_even": break_even_analysis(annual, revenue_summary, revenue_schedules),
        "goal_seek": goal_seek_live_price(assumptions),
        "predictive": build_predictive_analytics(cashflows, income_statement),
        "scenario_planning": scenario_planning(
            assumptions,
            base_metrics,
            annual.revenue,
        ),
        "custom_simulations": run_custom_simulations(
            assumptions, base_metrics, DEFAULT_CUSTOM_SIMULATION_DEFINITIONS
        ),
        "base_metrics": base_metrics,
    }


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
    cashflows, loan_schedule = discounted_cash_flow(assumptions, annual)
    revenue_schedules = build_revenue_schedules(assumptions, cycles)
    revenue_summary = summarise_revenue_totals(
        revenue_schedules,
        assumptions.cycles_per_year,
        assumptions.production_horizon_years,
    )
    financials = build_financial_statements(assumptions, cashflows, loan_schedule)
    advanced = compute_advanced_analytics(
        assumptions,
        cashflows,
        financials["income_statement"],
        financials["balance_sheet"],
        revenue_summary,
        revenue_schedules,
        annual,
    )

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
        "revenue_summary": revenue_summary,
        "valuation": valuation,
        "financial_statements": financials,
        "advanced_analytics": advanced,
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
    revenue_summary = results["revenue_summary"]
    financials = results["financial_statements"]
    advanced = results["advanced_analytics"]

    income_statement = financials["income_statement"]
    balance_sheet = financials["balance_sheet"]
    cash_flow_statement = financials["cash_flow_statement"]
    loan_schedule = financials["loan_schedule"]

    if "csv" in args.formats:
        write_csv(output_dir / "assumptions_summary.csv", assumption_schedule)
        write_csv(output_dir / "assumptions.csv", [{"name": k, "value": v} for k, v in asdict(assumptions).items()])
        write_csv(output_dir / "production_cycles.csv", [asdict(cycle) for cycle in cycles])
        write_csv(output_dir / "annual_summary.csv", [asdict(annual)])
        write_csv(output_dir / "cash_flow.csv", [asdict(row) for row in cashflows])
        write_csv(output_dir / "income_statement.csv", [asdict(row) for row in income_statement])
        write_csv(output_dir / "balance_sheet.csv", [asdict(row) for row in balance_sheet])
        write_csv(output_dir / "cash_flow_statement.csv", [asdict(row) for row in cash_flow_statement])
        write_csv(output_dir / "loan_schedule.csv", loan_schedule)
        write_csv(output_dir / "advanced_metrics.csv", advanced["metrics"])
        write_csv(output_dir / "dscr_summary.csv", advanced["dscr"])
        write_csv(output_dir / "trend_analysis.csv", advanced["trend"])
        write_csv(output_dir / "return_metrics.csv", advanced["returns"])
        write_csv(output_dir / "coverage_metrics.csv", advanced["coverage"])
        write_csv(output_dir / "leverage_metrics.csv", advanced["leverage"])
        write_csv(output_dir / "what_if_analysis.csv", advanced.get("what_if", []))
        monte_carlo = advanced.get("monte_carlo", {})
        if monte_carlo:
            summary_row = monte_carlo.get("summary")
            if summary_row:
                write_csv(
                    output_dir / "monte_carlo_summary.csv",
                    [summary_row],
                )
            write_csv(
                output_dir / "monte_carlo_samples.csv",
                monte_carlo.get("samples", []),
            )
        write_csv(
            output_dir / "break_even_analysis.csv", advanced.get("break_even", [])
        )
        goal_seek = advanced.get("goal_seek")
        if goal_seek:
            write_csv(output_dir / "goal_seek_results.csv", [goal_seek])
        predictive = advanced.get("predictive", {})
        write_csv(
            output_dir / "automated_forecast.csv",
            predictive.get("automated_forecast", []),
        )
        write_csv(
            output_dir / "time_series_forecast.csv",
            predictive.get("time_series", {}).get("forecast", []),
        )
        write_csv(
            output_dir / "risk_anomalies.csv",
            predictive.get("risk_anomalies", {}).get("observations", []),
        )
        write_csv(
            output_dir / "ml_methods_summary.csv",
            predictive.get("ml_methods", []),
        )
        write_csv(
            output_dir / "scenario_planning.csv",
            advanced.get("scenario_planning", []),
        )
        custom_simulations = advanced.get("custom_simulations", {})
        write_csv(
            output_dir / "custom_simulation_definitions.csv",
            custom_simulations.get("definitions", []),
        )
        write_csv(
            output_dir / "custom_simulation_results.csv",
            custom_simulations.get("results", []),
        )
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

        write_csv(
            output_dir / "revenue_summary_by_category.csv",
            revenue_summary.get("by_category", []),
        )
        write_csv(
            output_dir / "revenue_summary_annual.csv",
            revenue_summary.get("annual_totals", []),
        )

    if "json" in args.formats:
        write_json(output_dir / "assumptions.json", asdict(assumptions))
        write_json(output_dir / "assumptions_summary.json", assumption_schedule)
        write_json(output_dir / "production_cycles.json", [asdict(cycle) for cycle in cycles])
        write_json(output_dir / "annual_summary.json", asdict(annual))
        write_json(output_dir / "cash_flow.json", [asdict(row) for row in cashflows])
        write_json(output_dir / "income_statement.json", [asdict(row) for row in income_statement])
        write_json(output_dir / "balance_sheet.json", [asdict(row) for row in balance_sheet])
        write_json(output_dir / "cash_flow_statement.json", [asdict(row) for row in cash_flow_statement])
        write_json(output_dir / "loan_schedule.json", loan_schedule)
        write_json(output_dir / "advanced_analytics.json", advanced)
        write_json(output_dir / "revenue_schedules.json", revenue_schedules)
        write_json(output_dir / "revenue_summary.json", revenue_summary)

    write_json(output_dir / "valuation.json", valuation)

    write_json(
        output_dir / "manifest.json",
        {
            "cycles_per_year": assumptions.cycles_per_year,
            "production_horizon_years": assumptions.production_horizon_years,
            "years": len(cashflows) - 1,
            "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        },
    )

    print("✅ Financial model generated. Outputs written to", output_dir)
    print(f"NPV @ {valuation['discount_rate']:.1%}: {valuation['npv']:,.0f}")
    print(f"IRR: {valuation['irr']:.2%}")


if __name__ == "__main__":
    main()
