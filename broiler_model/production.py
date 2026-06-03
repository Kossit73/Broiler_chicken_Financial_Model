"""Production cycle calculations and revenue schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .assumptions import Assumptions, REVENUE_CATEGORIES


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


def compute_cycle(assumptions: Assumptions, cycle_number: int) -> CycleResults:
    survivors = round(assumptions.birds_per_cycle * (1 - assumptions.mortality_rate))
    live_weight = survivors * assumptions.final_weight_kg
    revenue = live_weight * assumptions.live_price_per_kg

    feed_required = (
        survivors * assumptions.final_weight_kg * assumptions.feed_conversion_ratio
    )
    feed_cost = feed_required * assumptions.feed_cost_per_kg
    chick_cost = assumptions.birds_per_cycle * assumptions.chick_cost
    processing_cost = survivors * assumptions.processing_cost_per_bird
    health_cost = survivors * assumptions.vaccination_cost_per_bird
    energy_cost = (
        assumptions.propane_per_cycle
        + assumptions.electricity_per_cycle
        + assumptions.litter_disposal_per_cycle
    )
    labor_cost = assumptions.labor_per_cycle
    overhead = (
        assumptions.maintenance_per_cycle
        + assumptions.management_fee_per_cycle
        + assumptions.insurance_per_cycle
        + assumptions.overhead_per_cycle
    )

    total_cost = (
        feed_cost
        + chick_cost
        + processing_cost
        + health_cost
        + energy_cost
        + labor_cost
        + overhead
    )
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


def compute_cycles(assumptions: Assumptions) -> List[CycleResults]:
    return [
        compute_cycle(assumptions, cycle)
        for cycle in range(1, assumptions.cycles_per_year + 1)
    ]


def annual_summary(
    assumptions: Assumptions,
    cycles: Iterable[CycleResults],
    *,
    annual_depreciation: float | None = None,
) -> AnnualSummary:
    depreciation = (
        float(annual_depreciation)
        if annual_depreciation is not None
        else (assumptions.capex_housing + assumptions.capex_equipment)
        / assumptions.depreciation_years
    )
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

    ebit = totals["ebitda"] - depreciation
    return AnnualSummary(
        year=int(assumptions.production_start_year),
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
        depreciation=depreciation,
        ebit=ebit,
    )


def build_revenue_schedules(
    assumptions: Assumptions, cycles: Iterable[CycleResults]
) -> Dict[str, List[Dict[str, Any]]]:
    """Return revenue schedules for each poultry revenue category."""

    schedules: Dict[str, List[Dict[str, Any]]] = {}
    cycle_list = list(cycles)
    cycle_by_number = {cycle.cycle: cycle for cycle in cycle_list}

    unit_price = assumptions.final_weight_kg * assumptions.live_price_per_kg
    unit_lookup = {
        "Broiler Revenue": "bird",
        "Eggs Revenue": "dozen",
        "Poultry Manure Revenue": "ton",
        "Live Birds Revenue": "head",
        "By-Product (feathers, offal, livers) Revenue": "kg",
    }
    cycles_per_year = max(int(assumptions.cycles_per_year or 1), 1)
    projection_years = max(int(assumptions.production_horizon_years or 1), 1)
    template_periods = cycles_per_year * projection_years

    broiler_rows: List[Dict[str, Any]] = []
    for period in range(1, template_periods + 1):
        cycle_number = ((period - 1) % cycles_per_year) + 1
        year_index = ((period - 1) // cycles_per_year) + 1
        cycle = cycle_by_number.get(cycle_number)
        survivors = cycle.survivors if cycle else round(
            assumptions.birds_per_cycle * (1 - assumptions.mortality_rate)
        )
        growth_multiplier = (1 + assumptions.price_growth) ** max(year_index - 1, 0)
        period_unit_price = unit_price * growth_multiplier
        broiler_rows.append(
            {
                "Category": "Broiler Revenue",
                "Period": f"Cycle {period}",
                "Unit": unit_lookup["Broiler Revenue"],
                "Units": survivors,
                "Unit price": period_unit_price,
                "Revenue": survivors * period_unit_price,
                "Notes": "Derived from production cycle results",
            }
        )
    schedules["Broiler Revenue"] = broiler_rows

    price_lookup = {
        "Eggs Revenue": assumptions.eggs_price_per_dozen,
        "Poultry Manure Revenue": assumptions.manure_price_per_ton,
        "Live Birds Revenue": assumptions.live_bird_price_per_head,
        "By-Product (feathers, offal, livers) Revenue": assumptions.byproduct_price_per_kg,
    }

    default_yield_map = {
        "Eggs Revenue": 0.06,  # dozens per surviving bird (proxy uplift assumption)
        "Poultry Manure Revenue": 0.003,  # tons per surviving bird per cycle
        "Live Birds Revenue": 1.0,  # heads per surviving bird
        "By-Product (feathers, offal, livers) Revenue": 0.08,  # by-product kg per kg live weight
    }
    fallback_survivors = round(
        assumptions.birds_per_cycle * (1 - assumptions.mortality_rate)
    )
    eggs_per_bird_per_cycle = max(
        float(getattr(assumptions, "eggs_per_bird_per_cycle", 0.0)), 0.0
    )
    total_eggs_per_cycle = (
        max(float(assumptions.birds_per_cycle), 0.0) * eggs_per_bird_per_cycle
    )
    eggs_default_dozens = total_eggs_per_cycle / 12.0
    fallback_live_weight_kg = fallback_survivors * assumptions.final_weight_kg

    for category in REVENUE_CATEGORIES[1:]:
        template_rows = []
        yield_value = default_yield_map.get(category, 0.0)
        for period in range(1, template_periods + 1):
            cycle_number = ((period - 1) % cycles_per_year) + 1
            year_index = ((period - 1) // cycles_per_year) + 1
            cycle = cycle_by_number.get(cycle_number)
            survivors = cycle.survivors if cycle else fallback_survivors
            live_weight_kg = cycle.live_weight_kg if cycle else fallback_live_weight_kg
            if category == "Eggs Revenue":
                units = eggs_default_dozens
            elif category == "By-Product (feathers, offal, livers) Revenue":
                units = live_weight_kg * yield_value
            else:
                units = survivors * yield_value
            unit_price_value = _to_float(price_lookup.get(category))
            if unit_price_value is not None:
                growth_multiplier = (1 + assumptions.price_growth) ** max(
                    year_index - 1, 0
                )
                unit_price_value *= growth_multiplier
            revenue = (
                units * unit_price_value if unit_price_value is not None else None
            )
            template_rows.append(
                {
                    "Category": category,
                    "Period": f"Cycle {period}",
                    "Unit": unit_lookup.get(category, ""),
                    "Units": units,
                    "Unit price": unit_price_value,
                    "Revenue": revenue,
                    "Notes": "Auto-estimated from assumptions (editable)",
                }
            )
        schedules[category] = template_rows

    return schedules


def summarise_revenue_totals(
    revenue_schedules: Dict[str, List[Dict[str, Any]]],
    cycles_per_year: int,
    projection_years: int,
    start_year: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate revenue schedules into annual totals per category and overall."""

    per_category: List[Dict[str, Any]] = []
    per_year_totals: Dict[int, float] = {}
    cycles = int(cycles_per_year) if cycles_per_year else 0
    cycles = max(cycles, 1)
    years = int(projection_years) if projection_years else 0
    years = max(years, 1)
    base_year = int(start_year) if start_year else 0

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
                value = 0.0
                row["Revenue"] = 0.0

            year = (idx // cycles) + 1
            category_totals[year] = category_totals.get(year, 0.0) + value

        for year_index, total in sorted(category_totals.items()):
            total_float = float(total)
            calendar_year = (
                base_year + year_index - 1 if base_year else year_index
            )
            per_category.append(
                {
                    "Category": category,
                    "Period": int(year_index),
                    "Year": int(calendar_year),
                    "Revenue": total_float,
                }
            )
            per_year_totals[year_index] = (
                per_year_totals.get(year_index, 0.0) + total_float
            )

    annual_totals: List[Dict[str, Any]] = []
    max_year = max(per_year_totals.keys(), default=0)
    horizon = max(years, max_year)
    for year_index in range(1, horizon + 1):
        total = per_year_totals.get(year_index, 0.0)
        calendar_year = base_year + year_index - 1 if base_year else year_index
        annual_totals.append(
            {
                "Period": int(year_index),
                "Year": int(calendar_year),
                "Revenue": float(total),
            }
        )

    timeline = {
        "start_year": base_year if base_year else 1,
        "end_year": (base_year + horizon - 1) if base_year else horizon,
        "projection_years": horizon,
    }

    return {
        "by_category": per_category,
        "annual_totals": annual_totals,
        "timeline": timeline,
    }
