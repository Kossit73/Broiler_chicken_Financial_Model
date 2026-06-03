"""Financing calculations, cash flows, and financial statements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .assumptions import Assumptions
from .production import AnnualSummary


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
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    accounts_payable: float = 0.0
    change_in_working_capital: float = 0.0
    ending_working_capital: float = 0.0
    calendar_year: Optional[int] = None


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
    calendar_year: Optional[int] = None


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
    calendar_year: Optional[int] = None


@dataclass
class CashFlowStatementRow:
    year: int
    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float
    net_change_in_cash: float
    ending_cash: float
    calendar_year: Optional[int] = None


def _pmt(rate: float, term_years: int, principal: float) -> float:
    if rate == 0:
        return principal / term_years
    factor = (1 + rate) ** term_years
    return principal * rate * factor / (factor - 1)


def amortization_schedule(
    principal: float,
    rate: float,
    term_years: int,
    *,
    grace_years: int = 0,
    start_year: int = 1,
    repayment_type: str = "annuity",
    loan_name: str = "Loan",
    facility_type: str = "Term loan",
) -> List[Dict[str, float]]:
    if principal <= 0 or term_years <= 0:
        return []
    rate = float(rate)
    grace_years = max(0, min(int(grace_years), int(term_years) - 1))
    start_year = max(int(start_year), 1)
    repayment_type = str(repayment_type or "annuity").lower()
    if repayment_type not in {"annuity", "straight_line", "interest_only", "bullet"}:
        repayment_type = "annuity"

    amortizing_years = max(term_years - grace_years, 1)
    payment = _pmt(rate, amortizing_years, principal) if repayment_type == "annuity" else 0.0
    schedule = []
    balance = principal
    for loan_year in range(1, term_years + 1):
        interest = balance * rate
        principal_paid = 0.0
        payment_amount = 0.0
        if loan_year <= grace_years:
            payment_amount = interest
        elif repayment_type == "annuity":
            payment_amount = payment
            principal_paid = payment_amount - interest
        elif repayment_type == "straight_line":
            principal_paid = principal / amortizing_years
            payment_amount = interest + principal_paid
        elif repayment_type == "interest_only":
            if loan_year < term_years:
                payment_amount = interest
            else:
                principal_paid = balance
                payment_amount = interest + principal_paid
        elif repayment_type == "bullet":
            if loan_year < term_years:
                payment_amount = interest
            else:
                principal_paid = balance
                payment_amount = interest + principal_paid

        principal_paid = min(max(principal_paid, 0.0), balance)
        balance = max(0.0, balance - principal_paid)
        project_year = start_year + loan_year - 1
        schedule.append(
            {
                "year": project_year,
                "loan_year": loan_year,
                "loan_name": loan_name,
                "facility_type": facility_type,
                "repayment_type": repayment_type,
                "payment": payment_amount,
                "interest": interest,
                "principal": principal_paid,
                "balance": balance,
            }
        )
    return schedule


def _aggregate_loan_schedule(
    loan_rows: Iterable[Dict[str, float]], projection_years: int
) -> List[Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    for row in loan_rows:
        year = int(row.get("year", 0))
        bucket = buckets.setdefault(
            year,
            {
                "year": year,
                "payment": 0.0,
                "interest": 0.0,
                "principal": 0.0,
                "balance": 0.0,
                "loans_active": 0.0,
            },
        )
        bucket["payment"] += float(row.get("payment", 0.0))
        bucket["interest"] += float(row.get("interest", 0.0))
        bucket["principal"] += float(row.get("principal", 0.0))
        bucket["balance"] += float(row.get("balance", 0.0))
        bucket["loans_active"] += 1.0
    return [buckets[year] for year in range(1, max(int(projection_years), 1) + 1) if year in buckets]


def discounted_cash_flow(
    assumptions: Assumptions,
    base_annual: AnnualSummary,
    *,
    debt_facilities: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[CashFlowRow], List[Dict[str, float]]]:
    total_capex = assumptions.capex_housing + assumptions.capex_equipment
    if debt_facilities:
        detailed_loan_rows: List[Dict[str, float]] = []
        for facility in debt_facilities:
            principal = float(facility.get("Principal", 0.0) or 0.0)
            if principal <= 0:
                continue
            detailed_loan_rows.extend(
                amortization_schedule(
                    principal,
                    float(facility.get("Interest rate", assumptions.debt_interest_rate) or 0.0),
                    int(facility.get("Term (years)", assumptions.debt_term_years) or assumptions.debt_term_years),
                    grace_years=int(facility.get("Grace period (years)", 0) or 0),
                    start_year=int(facility.get("Start year", 1) or 1),
                    repayment_type=str(facility.get("Repayment type", "annuity") or "annuity"),
                    loan_name=str(facility.get("Loan name", "Loan") or "Loan"),
                    facility_type=str(facility.get("Facility type", "Term loan") or "Term loan"),
                )
            )
        projection_years = max(int(assumptions.production_horizon_years or 1), 1)
        loan_schedule = _aggregate_loan_schedule(detailed_loan_rows, projection_years)
        debt = sum(float(row.get("Principal", 0.0) or 0.0) for row in debt_facilities)
        equity = total_capex - debt
    else:
        equity = total_capex * (1 - assumptions.debt_ratio)
        debt = total_capex * assumptions.debt_ratio
        loan_schedule = amortization_schedule(
            debt, assumptions.debt_interest_rate, assumptions.debt_term_years
        )

    rows: List[CashFlowRow] = []
    depreciation = base_annual.depreciation
    revenue = base_annual.revenue
    initial_revenue = revenue * (1 + assumptions.price_growth)
    initial_variable_costs = (
        base_annual.feed_cost
        + base_annual.chick_cost
        + base_annual.processing_cost
        + base_annual.health_cost
    ) * (1 + assumptions.cost_inflation)
    base_ar = initial_revenue * (assumptions.ar_days / 365.0)
    base_inventory = initial_variable_costs * (assumptions.inventory_days / 365.0)
    base_ap = initial_variable_costs * (assumptions.ap_days / 365.0)
    base_working_capital = max(0.0, base_ar + base_inventory - base_ap)
    previous_working_capital = base_working_capital
    upfront_cash = -(equity + base_working_capital)
    cumulative_cash = upfront_cash
    start_year = int(assumptions.production_start_year) if assumptions.production_start_year else 0
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
            ending_working_capital=base_working_capital,
            calendar_year=start_year - 1 if start_year else None,
        )
    )

    projection_years = int(assumptions.production_horizon_years)
    if projection_years <= 0:
        projection_years = 1

    for year in range(1, projection_years + 1):
        revenue *= 1 + assumptions.price_growth
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
        accounts_receivable = revenue * (assumptions.ar_days / 365.0)
        inventory = variable_costs * (assumptions.inventory_days / 365.0)
        accounts_payable = variable_costs * (assumptions.ap_days / 365.0)
        target_working_capital = max(
            0.0, accounts_receivable + inventory - accounts_payable
        )
        change_in_working_capital = target_working_capital - previous_working_capital
        if year == projection_years:
            # Release working capital at project end.
            change_in_working_capital -= target_working_capital
            target_working_capital = 0.0
        free_cash_flow = (
            operating_cash_flow
            - assumptions.maintenance_capex_annual
            - debt_service
            - change_in_working_capital
        )
        discount_factor = (1 + assumptions.discount_rate) ** year
        present_value = free_cash_flow / discount_factor
        cumulative_cash += free_cash_flow
        previous_working_capital = target_working_capital

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
                accounts_receivable=accounts_receivable,
                inventory=inventory,
                accounts_payable=accounts_payable,
                change_in_working_capital=change_in_working_capital,
                ending_working_capital=target_working_capital,
                calendar_year=start_year + year - 1 if start_year else None,
            )
        )

    if start_year:
        for entry in loan_schedule:
            year_value = int(entry.get("year", 0))
            entry["calendar_year"] = start_year + year_value - 1 if year_value else start_year

    return rows, loan_schedule


def build_financial_statements(
    assumptions: Assumptions,
    cashflows: List[CashFlowRow],
    loan_schedule: List[Dict[str, float]],
    *,
    annual_depreciation: float | None = None,
    opening_total_capex: float | None = None,
    asset_book_summary: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Any]]:
    total_capex = (
        float(opening_total_capex)
        if opening_total_capex is not None
        else assumptions.capex_housing + assumptions.capex_equipment
    )
    depreciation = (
        float(annual_depreciation)
        if annual_depreciation is not None
        else (assumptions.capex_housing + assumptions.capex_equipment)
        / assumptions.depreciation_years
    )
    initial_working_capital = (
        cashflows[0].ending_working_capital if cashflows else 0.0
    )
    initial_debt = cashflows[0].ending_debt if cashflows else (total_capex * assumptions.debt_ratio)
    equity_base = total_capex - initial_debt
    equity_total = equity_base + initial_working_capital

    income_rows: List[IncomeStatementRow] = []
    cash_statement: List[CashFlowStatementRow] = []
    balance_rows: List[BalanceSheetRow] = []

    investing_cash = -(total_capex + initial_working_capital)
    financing_cash = equity_total + initial_debt
    net_change = investing_cash + financing_cash
    cash_balance = net_change
    start_year = int(assumptions.production_start_year) if assumptions.production_start_year else 0

    cash_statement.append(
        CashFlowStatementRow(
            year=0,
            operating_cash_flow=0.0,
            investing_cash_flow=investing_cash,
            financing_cash_flow=financing_cash,
            net_change_in_cash=net_change,
            ending_cash=cash_balance,
            calendar_year=start_year - 1 if start_year else None,
        )
    )

    for row in cashflows:
        if row.year == 0:
            balance_rows.append(
                BalanceSheetRow(
                    year=0,
                    cash=cash_balance,
                    working_capital=initial_working_capital,
                    net_ppe=total_capex,
                    total_assets=total_capex
                    + initial_working_capital
                    + cash_balance,
                    debt=initial_debt,
                    equity=equity_total + cash_balance,
                    retained_earnings=0.0,
                    debt_to_equity=(
                        (initial_debt / equity_total)
                        if equity_total
                        else None
                    ),
                    calendar_year=start_year - 1 if start_year else None,
                )
            )
            continue

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
                calendar_year=row.calendar_year,
            )
        )

        operating_cash = row.net_income + row.depreciation - row.change_in_working_capital
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
                calendar_year=row.calendar_year,
            )
        )

        summary_lookup = {
            int(entry.get("Project year", 0)): entry for entry in (asset_book_summary or [])
        }
        if row.year in summary_lookup:
            net_ppe = float(summary_lookup[row.year].get("Closing book value", 0.0))
        else:
            accum_dep = min(row.year, assumptions.depreciation_years) * depreciation
            net_ppe = max(0.0, total_capex - accum_dep)
        debt_balance = row.ending_debt if row.ending_debt else 0.0
        total_assets = cash_balance + row.ending_working_capital + net_ppe
        equity = total_assets - debt_balance
        retained = equity - equity_total
        debt_to_equity = (debt_balance / equity) if equity else None
        balance_rows.append(
            BalanceSheetRow(
                year=row.year,
                cash=cash_balance,
                working_capital=row.ending_working_capital,
                net_ppe=net_ppe,
                total_assets=total_assets,
                debt=debt_balance,
                equity=equity,
                retained_earnings=retained,
                debt_to_equity=debt_to_equity,
                calendar_year=row.calendar_year,
            )
        )

    return {
        "income_statement": income_rows,
        "balance_sheet": balance_rows,
        "cash_flow_statement": cash_statement,
        "loan_schedule": loan_schedule,
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
