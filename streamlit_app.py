"""Streamlit interface for the broiler chicken financial model."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO

import pandas as pd
import streamlit as st

from deployable_financial_model import Assumptions, generate_model_outputs


def to_csv(dataframe: pd.DataFrame) -> bytes:
    """Return the CSV representation of ``dataframe`` as UTF-8 bytes."""

    if dataframe is None or dataframe.empty:
        return b""

    buffer = BytesIO()
    buffer.write(dataframe.to_csv(index=False).encode("utf-8"))
    return buffer.getvalue()


def render_download_button(label: str, dataframe: pd.DataFrame, file_name: str):
    csv_bytes = to_csv(dataframe)
    disabled = len(csv_bytes) == 0
    st.download_button(
        label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        disabled=disabled,
        key=f"download-{file_name}",
        help="CSV download is unavailable" if disabled else None,
    )


def assumptions_form(defaults: Assumptions) -> Assumptions:
    st.header("Model assumptions")

    farm_name = st.text_input("Farm name", defaults.farm_name)

    production_tab, pricing_tab, costs_tab, capital_tab = st.tabs(
        ["Production", "Pricing", "Costs", "Capital & financing"]
    )

    with production_tab:
        col_a, col_b = st.columns(2)
        cycles_per_year = col_a.number_input(
            "Cycles per year", min_value=1, max_value=12, value=defaults.cycles_per_year
        )
        birds_per_cycle = col_b.number_input(
            "Birds per cycle",
            min_value=1000,
            max_value=100000,
            value=defaults.birds_per_cycle,
            step=1000,
        )
        mortality_rate = col_a.number_input(
            "Mortality rate",
            min_value=0.0,
            max_value=0.2,
            value=defaults.mortality_rate,
            step=0.005,
            format="%.3f",
        )
        final_weight_kg = col_b.number_input(
            "Final weight (kg)",
            min_value=1.0,
            max_value=4.0,
            value=defaults.final_weight_kg,
            step=0.1,
        )

    with pricing_tab:
        col_a, col_b = st.columns(2)
        live_price_per_kg = col_a.number_input(
            "Live price per kg",
            min_value=0.5,
            max_value=5.0,
            value=defaults.live_price_per_kg,
            step=0.05,
        )
        price_growth = col_b.number_input(
            "Annual price growth",
            min_value=-0.05,
            max_value=0.1,
            value=defaults.price_growth,
            step=0.005,
            format="%.3f",
        )

    with costs_tab:
        col_a, col_b = st.columns(2)
        feed_conversion_ratio = col_a.number_input(
            "Feed conversion ratio",
            min_value=1.0,
            max_value=2.5,
            value=defaults.feed_conversion_ratio,
            step=0.05,
        )
        feed_cost_per_kg = col_b.number_input(
            "Feed cost per kg",
            min_value=0.2,
            max_value=1.0,
            value=defaults.feed_cost_per_kg,
            step=0.01,
        )
        chick_cost = col_a.number_input(
            "Chick cost",
            min_value=0.3,
            max_value=2.0,
            value=defaults.chick_cost,
            step=0.05,
        )
        processing_cost_per_bird = col_b.number_input(
            "Processing cost per bird",
            min_value=0.05,
            max_value=1.0,
            value=defaults.processing_cost_per_bird,
            step=0.05,
        )
        vaccination_cost_per_bird = col_a.number_input(
            "Vaccination cost per bird",
            min_value=0.0,
            max_value=0.5,
            value=defaults.vaccination_cost_per_bird,
            step=0.01,
        )
        litter_disposal_per_cycle = col_b.number_input(
            "Litter & disposal per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.litter_disposal_per_cycle,
            step=100.0,
        )
        propane_per_cycle = col_a.number_input(
            "Propane per cycle",
            min_value=0.0,
            max_value=20000.0,
            value=defaults.propane_per_cycle,
            step=100.0,
        )
        electricity_per_cycle = col_b.number_input(
            "Electricity per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.electricity_per_cycle,
            step=100.0,
        )
        labor_per_cycle = col_a.number_input(
            "Labor per cycle",
            min_value=0.0,
            max_value=50000.0,
            value=defaults.labor_per_cycle,
            step=500.0,
        )
        maintenance_per_cycle = col_b.number_input(
            "Maintenance per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.maintenance_per_cycle,
            step=100.0,
        )
        management_fee_per_cycle = col_a.number_input(
            "Management fee per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.management_fee_per_cycle,
            step=100.0,
        )
        insurance_per_cycle = col_b.number_input(
            "Insurance per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.insurance_per_cycle,
            step=100.0,
        )
        overhead_per_cycle = col_a.number_input(
            "Overhead per cycle",
            min_value=0.0,
            max_value=10000.0,
            value=defaults.overhead_per_cycle,
            step=100.0,
        )
        cost_inflation = col_b.number_input(
            "Cost inflation",
            min_value=-0.05,
            max_value=0.1,
            value=defaults.cost_inflation,
            step=0.005,
            format="%.3f",
        )

    with capital_tab:
        col_a, col_b = st.columns(2)
        capex_housing = col_a.number_input(
            "Housing capex",
            min_value=0.0,
            max_value=5000000.0,
            value=defaults.capex_housing,
            step=10000.0,
        )
        capex_equipment = col_b.number_input(
            "Equipment capex",
            min_value=0.0,
            max_value=2000000.0,
            value=defaults.capex_equipment,
            step=5000.0,
        )
        working_capital = col_a.number_input(
            "Working capital",
            min_value=0.0,
            max_value=1000000.0,
            value=defaults.working_capital,
            step=5000.0,
        )
        depreciation_years = col_b.number_input(
            "Depreciation years",
            min_value=1,
            max_value=40,
            value=defaults.depreciation_years,
        )
        maintenance_capex_annual = col_a.number_input(
            "Maintenance capex (annual)",
            min_value=0.0,
            max_value=200000.0,
            value=defaults.maintenance_capex_annual,
            step=5000.0,
        )
        debt_ratio = col_b.number_input(
            "Debt ratio",
            min_value=0.0,
            max_value=1.0,
            value=defaults.debt_ratio,
            step=0.05,
        )
        debt_interest_rate = col_a.number_input(
            "Debt interest rate",
            min_value=0.0,
            max_value=0.2,
            value=defaults.debt_interest_rate,
            step=0.005,
            format="%.3f",
        )
        debt_term_years = col_b.number_input(
            "Debt term (years)",
            min_value=1,
            max_value=30,
            value=defaults.debt_term_years,
        )
        discount_rate = col_a.number_input(
            "Discount rate",
            min_value=0.0,
            max_value=0.5,
            value=defaults.discount_rate,
            step=0.01,
            format="%.3f",
        )
        tax_rate = col_b.number_input(
            "Tax rate",
            min_value=0.0,
            max_value=0.5,
            value=defaults.tax_rate,
            step=0.01,
            format="%.3f",
        )

    return Assumptions(
        farm_name=farm_name,
        cycles_per_year=int(cycles_per_year),
        birds_per_cycle=int(birds_per_cycle),
        mortality_rate=float(mortality_rate),
        final_weight_kg=float(final_weight_kg),
        live_price_per_kg=float(live_price_per_kg),
        chick_cost=float(chick_cost),
        feed_conversion_ratio=float(feed_conversion_ratio),
        feed_cost_per_kg=float(feed_cost_per_kg),
        processing_cost_per_bird=float(processing_cost_per_bird),
        vaccination_cost_per_bird=float(vaccination_cost_per_bird),
        litter_disposal_per_cycle=float(litter_disposal_per_cycle),
        propane_per_cycle=float(propane_per_cycle),
        electricity_per_cycle=float(electricity_per_cycle),
        labor_per_cycle=float(labor_per_cycle),
        maintenance_per_cycle=float(maintenance_per_cycle),
        management_fee_per_cycle=float(management_fee_per_cycle),
        insurance_per_cycle=float(insurance_per_cycle),
        overhead_per_cycle=float(overhead_per_cycle),
        capex_housing=float(capex_housing),
        capex_equipment=float(capex_equipment),
        working_capital=float(working_capital),
        discount_rate=float(discount_rate),
        price_growth=float(price_growth),
        cost_inflation=float(cost_inflation),
        tax_rate=float(tax_rate),
        debt_ratio=float(debt_ratio),
        debt_interest_rate=float(debt_interest_rate),
        debt_term_years=int(debt_term_years),
        depreciation_years=int(depreciation_years),
        maintenance_capex_annual=float(maintenance_capex_annual),
    )


def main() -> None:
    st.set_page_config(page_title="Broiler Chicken Financial Model", layout="wide")
    st.title("Broiler Chicken Financial Model")
    st.caption("Interactively explore production, operating, and financing assumptions for a broiler chicken farm.")

    defaults = Assumptions()
    assumptions = assumptions_form(defaults)

    results = generate_model_outputs(assumptions)
    valuation = results["valuation"]

    col1, col2, col3 = st.columns(3)
    col1.metric("NPV", f"${valuation['npv']:,.0f}")
    col2.metric("IRR", f"{valuation['irr']:.2%}")
    col3.metric("Discount rate", f"{valuation['discount_rate']:.2%}")

    st.subheader("Assumptions summary")
    st.json(asdict(assumptions))

    cycles_df = pd.DataFrame([asdict(cycle) for cycle in results["cycles"]])
    annual_df = pd.DataFrame([asdict(results["annual"])])
    cashflow_df = pd.DataFrame([asdict(row) for row in results["cashflows"]])

    tab1, tab2, tab3 = st.tabs(["Production cycles", "Annual summary", "Cash flows"])
    with tab1:
        st.dataframe(cycles_df, use_container_width=True)
        render_download_button("Download cycles CSV", cycles_df, "production_cycles.csv")
    with tab2:
        st.dataframe(annual_df, use_container_width=True)
        render_download_button("Download annual CSV", annual_df, "annual_summary.csv")
    with tab3:
        st.dataframe(cashflow_df, use_container_width=True)
        render_download_button("Download cash flow CSV", cashflow_df, "cash_flow.csv")

    st.markdown("---")
    st.caption("Use the assumption tabs above to adjust the operating model and financing structure.")


if __name__ == "__main__":
    main()

