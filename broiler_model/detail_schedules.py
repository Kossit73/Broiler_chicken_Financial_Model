"""Detailed editable schedules for capex, operating costs, and debt facilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .assumptions import Assumptions


DETAIL_SCHEDULE_KEYS = (
    "equipment_capex",
    "housing_capex",
    "labor",
    "maintenance",
    "management_fee",
    "debt_facilities",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_to_float(value, float(default))))
    except (TypeError, ValueError):
        return default


def _rate_value(value: Any) -> float:
    rate = _to_float(value, 0.0)
    if rate > 1.0:
        rate = rate / 100.0
    return max(rate, 0.0)


def _text_value(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def build_default_detail_schedules(assumptions: Assumptions) -> Dict[str, List[Dict[str, Any]]]:
    dep_rate = 1.0 / max(int(assumptions.depreciation_years or 1), 1)
    total_capex = float(assumptions.capex_housing) + float(assumptions.capex_equipment)
    debt_principal = total_capex * float(assumptions.debt_ratio or 0.0)

    return {
        "equipment_capex": [
            {
                "Item": "Core equipment",
                "Opening amount": float(assumptions.capex_equipment),
                "New additions": 0.0,
                "Depreciation rate": dep_rate,
                "Annual depreciation": float(assumptions.capex_equipment) * dep_rate,
                "Closing book value": max(
                    float(assumptions.capex_equipment) * (1.0 - dep_rate), 0.0
                ),
                "Notes": "Baseline equipment asset pool",
            }
        ],
        "housing_capex": [
            {
                "Item": "Farm housing",
                "Opening amount": float(assumptions.capex_housing),
                "New additions": 0.0,
                "Depreciation rate": dep_rate,
                "Annual depreciation": float(assumptions.capex_housing) * dep_rate,
                "Closing book value": max(
                    float(assumptions.capex_housing) * (1.0 - dep_rate), 0.0
                ),
                "Notes": "Baseline housing asset pool",
            }
        ],
        "labor": [
            {
                "Role": "Farm labour team",
                "Headcount": 1,
                "Cost per head per cycle": float(assumptions.labor_per_cycle),
                "Amount per cycle": float(assumptions.labor_per_cycle),
                "Notes": "Rolled from the baseline labour assumption",
            }
        ],
        "maintenance": [
            {
                "Item": "Routine maintenance",
                "Units": 1,
                "Unit cost per cycle": float(assumptions.maintenance_per_cycle),
                "Amount per cycle": float(assumptions.maintenance_per_cycle),
                "Notes": "Baseline recurring maintenance scope",
            }
        ],
        "management_fee": [
            {
                "Item": "Management contract",
                "Units": 1,
                "Unit cost per cycle": float(assumptions.management_fee_per_cycle),
                "Amount per cycle": float(assumptions.management_fee_per_cycle),
                "Notes": "Baseline management-fee assumption",
            }
        ],
        "debt_facilities": (
            [
                {
                    "Loan name": "Senior term loan",
                    "Facility type": "Term loan",
                    "Principal": debt_principal,
                    "Interest rate": float(assumptions.debt_interest_rate),
                    "Term (years)": int(assumptions.debt_term_years),
                    "Grace period (years)": 0,
                    "Start year": 1,
                    "Repayment type": "annuity",
                    "Notes": "Baseline pooled debt assumption",
                }
            ]
            if debt_principal > 0
            else []
        ),
    }


def _normalise_rows(
    rows: Iterable[Mapping[str, Any]] | None,
    defaults: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if rows is None:
        return [dict(row) for row in defaults]
    rows_list = list(rows)
    if not rows_list:
        return []
    normalised: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows_list):
        base = dict(defaults[min(idx, len(defaults) - 1)]) if defaults else {}
        base.update(dict(row))
        normalised.append(base)
    return normalised


def recalculate_capex_schedule(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    recalculated: List[Dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        item = _text_value(row.get("Item"), f"Asset {idx}")
        opening_amount = max(_to_float(row.get("Opening amount")), 0.0)
        new_additions = max(_to_float(row.get("New additions")), 0.0)
        depreciation_rate = _rate_value(row.get("Depreciation rate"))
        gross_amount = opening_amount + new_additions
        annual_depreciation = min(gross_amount, gross_amount * depreciation_rate)
        closing_book_value = max(gross_amount - annual_depreciation, 0.0)
        recalculated.append(
            {
                "Item": item,
                "Opening amount": opening_amount,
                "New additions": new_additions,
                "Gross amount": gross_amount,
                "Depreciation rate": depreciation_rate,
                "Annual depreciation": annual_depreciation,
                "Closing book value": closing_book_value,
                "Notes": _text_value(row.get("Notes"), ""),
            }
        )
    return recalculated


def recalculate_labor_schedule(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    recalculated: List[Dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        role = _text_value(row.get("Role"), f"Role {idx}")
        headcount = max(_to_int(row.get("Headcount"), 1), 0)
        unit_cost = max(_to_float(row.get("Cost per head per cycle")), 0.0)
        amount = headcount * unit_cost
        recalculated.append(
            {
                "Role": role,
                "Headcount": headcount,
                "Cost per head per cycle": unit_cost,
                "Amount per cycle": amount,
                "Notes": _text_value(row.get("Notes"), ""),
            }
        )
    return recalculated


def recalculate_cost_schedule(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    recalculated: List[Dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        item = _text_value(row.get("Item"), f"Cost item {idx}")
        units = max(_to_float(row.get("Units"), 1.0), 0.0)
        unit_cost = max(_to_float(row.get("Unit cost per cycle")), 0.0)
        amount = units * unit_cost
        recalculated.append(
            {
                "Item": item,
                "Units": units,
                "Unit cost per cycle": unit_cost,
                "Amount per cycle": amount,
                "Notes": _text_value(row.get("Notes"), ""),
            }
        )
    return recalculated


def recalculate_debt_facilities(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    recalculated: List[Dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        loan_name = _text_value(row.get("Loan name"), f"Loan {idx}")
        facility_type = _text_value(row.get("Facility type"), "Term loan")
        principal = max(_to_float(row.get("Principal")), 0.0)
        interest_rate = _rate_value(row.get("Interest rate"))
        term_years = max(_to_int(row.get("Term (years)"), 1), 1)
        grace_years = max(_to_int(row.get("Grace period (years)"), 0), 0)
        start_year = max(_to_int(row.get("Start year"), 1), 1)
        repayment_type = _text_value(row.get("Repayment type"), "annuity").lower()
        if repayment_type not in {"annuity", "straight_line", "interest_only", "bullet"}:
            repayment_type = "annuity"
        recalculated.append(
            {
                "Loan name": loan_name,
                "Facility type": facility_type,
                "Principal": principal,
                "Interest rate": interest_rate,
                "Term (years)": term_years,
                "Grace period (years)": min(grace_years, max(term_years - 1, 0)),
                "Start year": start_year,
                "Repayment type": repayment_type,
                "Notes": _text_value(row.get("Notes"), ""),
            }
        )
    return recalculated


def normalise_detail_schedules(
    assumptions: Assumptions,
    schedules: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    payload = schedules if isinstance(schedules, Mapping) else {}
    defaults = build_default_detail_schedules(assumptions)
    normalised: Dict[str, List[Dict[str, Any]]] = {}
    normalised["equipment_capex"] = recalculate_capex_schedule(
        _normalise_rows(payload.get("equipment_capex"), defaults["equipment_capex"])
    )
    normalised["housing_capex"] = recalculate_capex_schedule(
        _normalise_rows(payload.get("housing_capex"), defaults["housing_capex"])
    )
    normalised["labor"] = recalculate_labor_schedule(
        _normalise_rows(payload.get("labor"), defaults["labor"])
    )
    normalised["maintenance"] = recalculate_cost_schedule(
        _normalise_rows(payload.get("maintenance"), defaults["maintenance"])
    )
    normalised["management_fee"] = recalculate_cost_schedule(
        _normalise_rows(payload.get("management_fee"), defaults["management_fee"])
    )
    normalised["debt_facilities"] = recalculate_debt_facilities(
        _normalise_rows(payload.get("debt_facilities"), defaults["debt_facilities"])
    )
    return normalised


def _capex_total(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(_to_float(row.get("Gross amount")) for row in rows)


def _capex_annual_depreciation(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(_to_float(row.get("Annual depreciation")) for row in rows)


def _cost_total(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(_to_float(row.get("Amount per cycle")) for row in rows)


def build_asset_book_schedule(
    equipment_rows: Iterable[Mapping[str, Any]],
    housing_rows: Iterable[Mapping[str, Any]],
    projection_years: int,
    start_year: int,
) -> List[Dict[str, Any]]:
    horizon = max(int(projection_years or 1), 1)
    schedule: List[Dict[str, Any]] = []
    grouped = {
        "Equipment": list(equipment_rows),
        "Housing": list(housing_rows),
    }
    for category, rows in grouped.items():
        for row in rows:
            item = _text_value(row.get("Item"), category)
            gross_amount = max(_to_float(row.get("Gross amount")), 0.0)
            annual_depreciation = max(_to_float(row.get("Annual depreciation")), 0.0)
            depreciation_rate = _rate_value(row.get("Depreciation rate"))
            for year in range(1, horizon + 1):
                accumulated_depreciation = min(
                    gross_amount, annual_depreciation * year
                )
                closing_book_value = max(gross_amount - accumulated_depreciation, 0.0)
                schedule.append(
                    {
                        "Category": category,
                        "Item": item,
                        "Project year": year,
                        "Calendar year": start_year + year - 1 if start_year else None,
                        "Gross amount": gross_amount,
                        "Depreciation rate": depreciation_rate,
                        "Annual depreciation": annual_depreciation,
                        "Accumulated depreciation": accumulated_depreciation,
                        "Closing book value": closing_book_value,
                    }
                )
    return schedule


def build_asset_book_summary(book_schedule: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in book_schedule:
        year = _to_int(row.get("Project year"), 0)
        if year not in grouped:
            grouped[year] = {
                "Project year": year,
                "Calendar year": row.get("Calendar year"),
                "Gross amount": 0.0,
                "Annual depreciation": 0.0,
                "Accumulated depreciation": 0.0,
                "Closing book value": 0.0,
            }
        grouped[year]["Gross amount"] += _to_float(row.get("Gross amount"))
        grouped[year]["Annual depreciation"] += _to_float(
            row.get("Annual depreciation")
        )
        grouped[year]["Accumulated depreciation"] += _to_float(
            row.get("Accumulated depreciation")
        )
        grouped[year]["Closing book value"] += _to_float(
            row.get("Closing book value")
        )
    return [grouped[year] for year in sorted(grouped)]


def prepare_detail_context(
    assumptions: Assumptions,
    schedules: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> Tuple[Assumptions, Dict[str, Any]]:
    normalised = normalise_detail_schedules(assumptions, schedules)
    equipment_total = _capex_total(normalised["equipment_capex"])
    housing_total = _capex_total(normalised["housing_capex"])
    annual_depreciation = _capex_annual_depreciation(
        normalised["equipment_capex"]
    ) + _capex_annual_depreciation(normalised["housing_capex"])
    labor_total = _cost_total(normalised["labor"])
    maintenance_total = _cost_total(normalised["maintenance"])
    management_total = _cost_total(normalised["management_fee"])
    total_capex = equipment_total + housing_total

    debt_rows = normalised["debt_facilities"]
    total_debt = sum(_to_float(row.get("Principal")) for row in debt_rows)
    weighted_rate = (
        sum(
            _to_float(row.get("Principal")) * _rate_value(row.get("Interest rate"))
            for row in debt_rows
        )
        / total_debt
        if total_debt > 0
        else float(assumptions.debt_interest_rate)
    )
    max_term = max(
        (_to_int(row.get("Term (years)"), int(assumptions.debt_term_years)) for row in debt_rows),
        default=int(assumptions.debt_term_years),
    )
    debt_ratio = (total_debt / total_capex) if total_capex > 0 else 0.0
    weighted_depreciation_years = (
        max(int(round(total_capex / annual_depreciation)), 1)
        if annual_depreciation > 0 and total_capex > 0
        else int(assumptions.depreciation_years)
    )

    resolved_assumptions = replace(
        assumptions,
        capex_equipment=equipment_total,
        capex_housing=housing_total,
        labor_per_cycle=labor_total,
        maintenance_per_cycle=maintenance_total,
        management_fee_per_cycle=management_total,
        debt_ratio=debt_ratio,
        debt_interest_rate=weighted_rate,
        debt_term_years=max_term,
        depreciation_years=weighted_depreciation_years,
    )

    projection_years = max(int(assumptions.production_horizon_years or 1), 1)
    start_year = int(assumptions.production_start_year or 0)
    asset_book_schedule = build_asset_book_schedule(
        normalised["equipment_capex"],
        normalised["housing_capex"],
        projection_years,
        start_year,
    )
    asset_book_summary = build_asset_book_summary(asset_book_schedule)

    return resolved_assumptions, {
        "schedules": normalised,
        "annual_depreciation": annual_depreciation,
        "capex_totals": {
            "equipment": equipment_total,
            "housing": housing_total,
            "total": total_capex,
        },
        "operating_cost_totals": {
            "labor": labor_total,
            "maintenance": maintenance_total,
            "management_fee": management_total,
        },
        "total_debt": total_debt,
        "weighted_debt_rate": weighted_rate,
        "asset_book_schedule": asset_book_schedule,
        "asset_book_summary": asset_book_summary,
    }
