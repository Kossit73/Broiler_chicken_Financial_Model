"""Streamlit interface for the broiler chicken financial model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from deployable_financial_model import Assumptions, generate_model_outputs


def _payload_to_ai_settings(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return AI settings merged with defaults."""

    base = (payload or {}).get("ai_settings", {})
    settings = DEFAULT_AI_SETTINGS.copy()
    settings.update({k: v for k, v in base.items() if v is not None})
    return settings


def _ai_settings_to_payload(settings: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Persist AI settings back to the payload."""

    payload["ai_settings"] = settings


def _rerun() -> None:
    """Trigger a Streamlit rerun."""

    st.experimental_rerun()


def _ensure_scenario_payload(
    selected_scenario: str, snapshot: Dict[str, Any]
) -> Tuple[ScenarioModel, Dict[str, Any]]:
    """Ensure the scenario model/results are cached for the provided snapshot."""

    cache = st.session_state.setdefault("scenario_payloads", {})
    existing = cache.get(selected_scenario)
    if existing and existing.get("snapshot") == snapshot:
        return existing["model"], existing["results"]

    assumptions_data = snapshot.get("assumptions", {})
    assumptions = Assumptions(**assumptions_data) if assumptions_data else Assumptions()
    model = ScenarioModel(scenario=selected_scenario, assumptions=assumptions)
    results = generate_model_outputs(assumptions)
    cache[selected_scenario] = {
        "snapshot": copy.deepcopy(snapshot),
        "model": model,
        "results": results,
    }
    return model, results


def _generate_excel_bytes(
    model: ScenarioModel, results: Dict[str, Any], scenario: str
) -> bytes:
    """Create an Excel workbook in memory for the supplied scenario results."""

    buffer = BytesIO()
    try:
        writer = pd.ExcelWriter(buffer, engine="xlsxwriter")
    except ValueError:
        writer = pd.ExcelWriter(buffer, engine="openpyxl")

    with writer:
        assumptions_df = pd.DataFrame(results["assumptions_schedule"])
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)

        input_df = pd.DataFrame([asdict(model.assumptions)])
        input_df.to_excel(writer, sheet_name="Input Values", index=False)

        cycles_df = pd.DataFrame([asdict(cycle) for cycle in results["cycles"]])
        cycles_df.to_excel(writer, sheet_name="Production Cycles", index=False)

        annual_df = pd.DataFrame([asdict(results["annual"])])
        annual_df.to_excel(writer, sheet_name="Annual Summary", index=False)

        cashflows_df = pd.DataFrame([asdict(row) for row in results["cashflows"]])
        cashflows_df.to_excel(writer, sheet_name="Cash Flows", index=False)

        valuation_df = pd.DataFrame([results["valuation"]])
        valuation_df.to_excel(writer, sheet_name="Valuation", index=False)

        financials = results["financial_statements"]
        pd.DataFrame([asdict(row) for row in financials["income_statement"]]).to_excel(
            writer, sheet_name="Income Statement", index=False
        )
        pd.DataFrame([asdict(row) for row in financials["balance_sheet"]]).to_excel(
            writer, sheet_name="Balance Sheet", index=False
        )
        pd.DataFrame([asdict(row) for row in financials["cash_flow_statement"]]).to_excel(
            writer, sheet_name="Cash Flow Statement", index=False
        )
        pd.DataFrame(financials["loan_schedule"]).to_excel(
            writer, sheet_name="Debt Schedule", index=False
        )

        advanced = results["advanced_analytics"]
        pd.DataFrame(advanced["metrics"]).to_excel(
            writer, sheet_name="Advanced Metrics", index=False
        )
        pd.DataFrame(advanced["dscr"]).to_excel(writer, sheet_name="DSCR", index=False)
        pd.DataFrame(advanced["trend"]).to_excel(
            writer, sheet_name="Trend Analysis", index=False
        )

        for category, rows in results["revenue_schedules"].items():
            safe_name = (
                category.replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("-", " ")
            )
            safe_name = " ".join(word.title() for word in safe_name.split())
            pd.DataFrame(rows).to_excel(
                writer, sheet_name=safe_name[:31] or "Revenue", index=False
            )

        ai_settings = st.session_state.get("ai_settings", DEFAULT_AI_SETTINGS)
        pd.DataFrame([ai_settings]).to_excel(writer, sheet_name="AI Settings", index=False)

        workbook = writer.book
        workbook.set_properties(
            {
                "title": f"Broiler Model - {scenario}",
                "subject": "Broiler chicken financial model",
                "comments": "Generated via Streamlit dashboard",
            }
        )

    buffer.seek(0)
    return buffer.getvalue()


AI_PROVIDER_OPTIONS = (
    "OpenAI",
    "Anthropic",
    "Azure OpenAI",
    "Vertex AI",
    "Custom",
)

ML_METHOD_LABELS: Dict[str, str] = {
    "linear_regression": "Linear Regression",
    "exponential_smoothing": "Exponential Smoothing",
    "random_forest": "Random Forest",
    "prophet": "Prophet Forecast",
}

ML_LABEL_TO_CODE = {label: code for code, label in ML_METHOD_LABELS.items()}

GEN_AI_FEATURE_LABELS: Dict[str, str] = {
    "summary": "Executive Summary",
    "risk_review": "Risk Review",
    "scenario_comparison": "Scenario Comparison",
    "cash_flow_focus": "Cash Flow Focus",
}

GEN_AI_LABEL_TO_CODE = {label: code for code, label in GEN_AI_FEATURE_LABELS.items()}

SCENARIO_OPTIONS = ["Baseline", "Expansion", "Downside"]

DEFAULT_AI_SETTINGS: Dict[str, Any] = {
    "enabled": False,
    "provider": "OpenAI",
    "model": "gpt-4o-mini",
    "forecast_horizon": 3,
    "ml_methods": ["linear_regression"],
    "generative_features": ["summary"],
    "api_key": "",
}


@dataclass
class ScenarioModel:
    """Lightweight wrapper storing the active scenario and assumptions."""

    scenario: str
    assumptions: Assumptions


def to_csv(dataframe: pd.DataFrame) -> bytes:
    """Return the CSV representation of ``dataframe`` as UTF-8 bytes."""

    if dataframe is None or dataframe.empty:
        return b""

    buffer = BytesIO()
    buffer.write(dataframe.to_csv(index=False).encode("utf-8"))
    return buffer.getvalue()


def render_download_button(
    label: str, dataframe: pd.DataFrame, file_name: str, key_suffix: str = ""
):
    csv_bytes = to_csv(dataframe)
    disabled = len(csv_bytes) == 0
    key = f"download-{file_name}"
    if key_suffix:
        key = f"{key}-{key_suffix}"
    st.download_button(
        label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        disabled=disabled,
        key=key,
        help="CSV download is unavailable" if disabled else None,
    )


def _render_ai_settings(payload: dict, container: Optional[DeltaGenerator] = None) -> None:
    """Render AI and machine-learning configuration controls."""

    target = container or st
    settings = st.session_state.setdefault("ai_settings", _payload_to_ai_settings(payload))
    st.session_state.setdefault("ai_api_key", settings.get("api_key", ""))

    provider_options = list(AI_PROVIDER_OPTIONS)
    if settings.get("provider") not in provider_options:
        provider_options.append(settings.get("provider"))

    current_provider = settings.get("provider", "OpenAI")
    try:
        provider_index = provider_options.index(current_provider)
    except ValueError:
        provider_index = 0

    ml_defaults = [
        ML_METHOD_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("ml_methods", ["linear_regression"])
    ]
    feature_defaults = [
        GEN_AI_FEATURE_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("generative_features", ["summary"])
    ]

    form = target.form("ai_settings_form")
    with form:
        enabled = form.checkbox(
            "Enable AI Enhancements",
            value=bool(settings.get("enabled", False)),
            help="Toggle machine-learning forecasts and generative commentary.",
        )
        provider = form.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            help="Select the API provider powering generative insights.",
        )
        model = form.text_input(
            "Model",
            value=settings.get("model", "gpt-4"),
            help="Name of the deployed model (for example `gpt-4o-mini`).",
        )
        horizon = form.number_input(
            "Forecast Horizon (years)",
            min_value=0,
            max_value=20,
            value=int(settings.get("forecast_horizon", 3)),
            step=1,
            help="Number of additional years used for machine-learning revenue forecasts.",
        )

        ml_selection = form.multiselect(
            "Machine Learning Methods",
            list(ML_METHOD_LABELS.values()),
            default=ml_defaults,
            help="Choose algorithms applied to projected net revenue.",
        )
        feature_selection = form.multiselect(
            "Generative Features",
            list(GEN_AI_FEATURE_LABELS.values()),
            default=feature_defaults,
            help="Pick the narrative focus areas generated by the AI summary.",
        )
        api_key = form.text_input(
            "API Key",
            value=st.session_state.get("ai_api_key", ""),
            type="password",
            help="Store your provider API key securely. Keys are retained only for the current session.",
        )

        submitted = form.form_submit_button("Save AI Configuration")

    if submitted:
        ml_codes = [
            ML_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower()) for label in ml_selection
        ]
        feature_codes = [
            GEN_AI_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower())
            for label in feature_selection
        ]

        settings.update(
            {
                "enabled": enabled,
                "provider": provider,
                "model": model.strip() or "gpt-4",
                "forecast_horizon": int(horizon),
                "ml_methods": ml_codes or ["linear_regression"],
                "generative_features": feature_codes or ["summary"],
                "api_key": api_key.strip(),
            }
        )
        st.session_state["ai_settings"] = settings
        st.session_state["ai_api_key"] = settings.get("api_key", "")
        _ai_settings_to_payload(settings, payload)
        st.success("AI configuration updated. Rerunning the model with the new settings.")
        _rerun()


def assumptions_form(defaults: Assumptions) -> Assumptions:
    st.header("Model assumptions")

    farm_name = st.text_input("Farm name", defaults.farm_name)

    st.subheader("Input Landing Page")

    st.markdown("**Production**")
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

    st.markdown("**Pricing**")
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

    st.markdown("**Costs**")
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

    st.markdown("**Capital & financing**")
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

    st.subheader("Scenario selection")
    selected_scenario = st.selectbox("Scenario", SCENARIO_OPTIONS, key="selected_scenario")

    scenario_store = st.session_state.setdefault("scenario_store", {})
    if selected_scenario not in scenario_store:
        scenario_store[selected_scenario] = {
            "assumptions": asdict(Assumptions()),
            "ai_settings": DEFAULT_AI_SETTINGS.copy(),
        }
    payload = scenario_store[selected_scenario]

    defaults = Assumptions(**payload.get("assumptions", {}))
    assumptions = assumptions_form(defaults)
    payload["assumptions"] = asdict(assumptions)

    ai_settings = _payload_to_ai_settings(payload)
    payload["ai_settings"] = ai_settings
    st.session_state["ai_settings"] = ai_settings

    input_page = copy.deepcopy(payload)
    snapshot = st.session_state.get("input_snapshot")
    if snapshot is None or st.session_state.get("snapshot_scenario") != selected_scenario:
        snapshot = copy.deepcopy(input_page)
        st.session_state.snapshot_scenario = selected_scenario
    elif snapshot != input_page:
        snapshot = copy.deepcopy(input_page)
    st.session_state.input_snapshot = snapshot

    model, results = _ensure_scenario_payload(selected_scenario, snapshot)
    model.scenario = selected_scenario
    st.session_state.model_results = (model, results)

    valuation = results["valuation"]
    assumption_schedule_df = pd.DataFrame(results["assumptions_schedule"])
    revenue_schedules = results["revenue_schedules"]
    financials = results["financial_statements"]
    advanced = results["advanced_analytics"]

    col1, col2, col3 = st.columns(3)
    col1.metric("NPV", f"${valuation['npv']:,.0f}")
    col2.metric("IRR", f"{valuation['irr']:.2%}")
    col3.metric("Discount rate", f"{valuation['discount_rate']:.2%}")

    cycles_df = pd.DataFrame([asdict(cycle) for cycle in results["cycles"]])
    annual_df = pd.DataFrame([asdict(results["annual"])])
    cashflow_df = pd.DataFrame([asdict(row) for row in results["cashflows"]])
    income_df = pd.DataFrame([asdict(row) for row in financials["income_statement"]])
    balance_df = pd.DataFrame([asdict(row) for row in financials["balance_sheet"]])
    cash_statement_df = pd.DataFrame([asdict(row) for row in financials["cash_flow_statement"]])
    loan_df = pd.DataFrame(financials["loan_schedule"])
    metrics_df = pd.DataFrame(advanced["metrics"])
    dscr_df = pd.DataFrame(advanced["dscr"])
    trend_df = pd.DataFrame(advanced["trend"])

    overview_tab, financials_tab, analytics_tab = st.tabs(
        ["Production & revenues", "Financial statements", "Advanced analytics"]
    )
    with overview_tab:
        download_container = st.container()
        excel_map: Dict[str, bytes] = st.session_state.setdefault("excel_bytes_map", {})
        excel_bytes = excel_map.get(selected_scenario)

        with download_container:
            st.markdown("### Excel export")
            if not excel_bytes:
                if st.button(
                    "Prepare Excel Model",
                    key=f"prepare_excel_{selected_scenario.lower()}",
                ):
                    with st.spinner("Preparing Excel workbook..."):
                        excel_bytes = _generate_excel_bytes(model, results, selected_scenario)
                    excel_map[selected_scenario] = excel_bytes
                    st.session_state.excel_bytes_map = excel_map
            if excel_bytes:
                download_name = f"Broiler_Financial_Model_{selected_scenario.replace(' ', '_')}.xlsx"
                st.download_button(
                    "Download Excel Model",
                    data=excel_bytes,
                    file_name=download_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{selected_scenario.lower()}",
                )
                if st.button(
                    "Clear Prepared Excel",
                    key=f"clear_excel_{selected_scenario.lower()}",
                ):
                    excel_map.pop(selected_scenario, None)
                    st.session_state.excel_bytes_map = excel_map
                    excel_bytes = None
            if not excel_bytes:
                st.info("Click 'Prepare Excel Model' to generate the workbook for download.")

        st.subheader("Assumptions summary")
        for schedule_name, group in assumption_schedule_df.groupby("schedule", sort=False):
            st.markdown(f"**{schedule_name}**")
            st.dataframe(
                group.drop(columns=["schedule"]),
                use_container_width=True,
                hide_index=True,
            )
        render_download_button(
            "Download assumptions CSV",
            assumption_schedule_df,
            "assumptions_summary.csv",
            key_suffix=selected_scenario,
        )

        st.subheader("Revenue schedules")
        for category, rows in revenue_schedules.items():
            st.markdown(f"**{category}**")
            schedule_df = pd.DataFrame(rows)
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)
            safe_name = (
                category.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("-", "_")
            )
            render_download_button(
                f"Download {category.lower()} CSV",
                schedule_df,
                f"{safe_name}_schedule.csv",
                key_suffix=selected_scenario,
            )

        st.subheader("Production cycles")
        st.dataframe(cycles_df, use_container_width=True)
        render_download_button(
            "Download cycles CSV",
            cycles_df,
            "production_cycles.csv",
            key_suffix=selected_scenario,
        )

        st.subheader("Annual summary")
        st.dataframe(annual_df, use_container_width=True)
        render_download_button(
            "Download annual CSV",
            annual_df,
            "annual_summary.csv",
            key_suffix=selected_scenario,
        )

        st.subheader("Discounted cash flows")
        st.dataframe(cashflow_df, use_container_width=True)
        render_download_button(
            "Download cash flow CSV",
            cashflow_df,
            "cash_flow.csv",
            key_suffix=selected_scenario,
        )

    with financials_tab:
        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(
            [
                "Income statement",
                "Balance sheet",
                "Cash flow statement",
                "Debt schedule",
            ]
        )
        with fin_tab1:
            st.dataframe(income_df, use_container_width=True, hide_index=True)
            render_download_button(
                "Download income statement CSV",
                income_df,
                "income_statement.csv",
                key_suffix=selected_scenario,
            )
        with fin_tab2:
            st.dataframe(balance_df, use_container_width=True, hide_index=True)
            render_download_button(
                "Download balance sheet CSV",
                balance_df,
                "balance_sheet.csv",
                key_suffix=selected_scenario,
            )
        with fin_tab3:
            st.dataframe(cash_statement_df, use_container_width=True, hide_index=True)
            render_download_button(
                "Download cash flow statement CSV",
                cash_statement_df,
                "cash_flow_statement.csv",
                key_suffix=selected_scenario,
            )
        with fin_tab4:
            st.dataframe(loan_df, use_container_width=True, hide_index=True)
            render_download_button(
                "Download debt schedule CSV",
                loan_df,
                "loan_schedule.csv",
                key_suffix=selected_scenario,
            )

    with analytics_tab:
        st.subheader("Advanced metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        render_download_button(
            "Download metrics CSV",
            metrics_df,
            "advanced_metrics.csv",
            key_suffix=selected_scenario,
        )

        if not dscr_df.empty:
            st.subheader("Debt service coverage ratio")
            dscr_chart = dscr_df.set_index("year")
            st.line_chart(dscr_chart)
            render_download_button(
                "Download DSCR CSV",
                dscr_df,
                "dscr_summary.csv",
                key_suffix=selected_scenario,
            )

        if not trend_df.empty:
            st.subheader("Performance trends")
            trend_chart = trend_df.set_index("year")[
                ["revenue", "ebitda", "net_income", "free_cash_flow"]
            ]
            st.line_chart(trend_chart)
            render_download_button(
                "Download trend CSV",
                trend_df,
                "trend_analysis.csv",
                key_suffix=selected_scenario,
            )

        ai_expander = st.expander("AI & Machine Learning Settings")
        _render_ai_settings(payload, ai_expander)

    st.session_state.input_snapshot = copy.deepcopy(payload)
    scenario_store[selected_scenario] = payload
    scenario_payloads = st.session_state.get("scenario_payloads")
    if scenario_payloads and selected_scenario in scenario_payloads:
        scenario_payloads[selected_scenario]["snapshot"] = copy.deepcopy(payload)

    st.markdown("---")
    st.caption("Use the Input Landing Page above to adjust the operating model and financing structure.")


if __name__ == "__main__":
    main()

