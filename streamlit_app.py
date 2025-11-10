"""Streamlit interface for the broiler chicken financial model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, get_type_hints

import pandas as pd
import streamlit as st
import altair as alt
from streamlit.delta_generator import DeltaGenerator

from broiler_model.assumptions import Assumptions, ASSUMPTION_SCHEDULE_LAYOUT
from broiler_model.analytics import (
    AnalyticsPlan,
    break_even_analysis,
    build_predictive_analytics,
    goal_seek_live_price,
    perform_what_if_analysis,
    run_custom_simulations,
    run_monte_carlo_analysis,
    scenario_planning,
)
from broiler_model.config import (
    load_custom_simulation_definitions,
    load_monte_carlo_distributions,
)
from broiler_model.model import generate_model_outputs
from broiler_model.production import summarise_revenue_totals


DEFAULT_CUSTOM_SIMULATION_DEFINITIONS = load_custom_simulation_definitions()
DEFAULT_MONTE_CARLO_DISTRIBUTIONS = load_monte_carlo_distributions()


ROW_REMOVAL_COLUMN = "Remove row"
ROW_EDIT_COLUMN = "Edit row"


_ASSUMPTION_LABEL_TO_KEY = {
    label: key for _, label, key in ASSUMPTION_SCHEDULE_LAYOUT
}
try:
    _ASSUMPTION_FIELD_TYPES = get_type_hints(Assumptions)
except Exception:  # pragma: no cover - defensive fallback
    _ASSUMPTION_FIELD_TYPES = {
        name: field.type for name, field in Assumptions.__dataclass_fields__.items()
    }


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series to numeric values where possible."""

    return pd.to_numeric(series, errors="coerce")


def _records_match(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> bool:
    """Return ``True`` when two iterable collections of mapping-like rows match."""

    try:
        left_serialised = json.dumps(list(left), sort_keys=True)
        right_serialised = json.dumps(list(right), sort_keys=True)
    except TypeError:
        return False
    return left_serialised == right_serialised


def _build_revenue_stack_chart(summary_df: pd.DataFrame) -> Optional[alt.Chart]:
    """Return a stacked area chart showing annual revenue by category."""

    if summary_df.empty:
        return None

    chart_data = summary_df.copy()
    if "Year" not in chart_data.columns or "Revenue" not in chart_data.columns:
        return None

    chart_data["Year"] = _to_numeric(chart_data["Year"])
    chart_data["Revenue"] = _to_numeric(chart_data["Revenue"])
    chart_data = chart_data.dropna(subset=["Year", "Revenue", "Category"])
    if chart_data.empty:
        return None

    chart_data["Year"] = chart_data["Year"].astype(int)

    return (
        alt.Chart(chart_data)
        .mark_area(opacity=0.75)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("sum(Revenue):Q", title="Revenue (USD)", stack="zero"),
            color=alt.Color("Category:N", title="Category"),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Revenue:Q", title="Revenue", format=",.0f"),
            ],
        )
    )


def _prepare_cashflow_bridge_frames(cashflow_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Build per-year cash-flow bridge frames for waterfall visualisations."""

    bridge_frames: Dict[int, pd.DataFrame] = {}
    if cashflow_df.empty or "year" not in cashflow_df.columns:
        return bridge_frames

    numeric_df = cashflow_df.apply(_to_numeric)

    for _, row in numeric_df.iterrows():
        year_value = row.get("year")
        if pd.isna(year_value):
            continue
        year = int(year_value)

        components: List[Tuple[str, float]] = []
        ocf = row.get("operating_cash_flow")
        maint = row.get("maintenance_capex")
        debt_service = row.get("debt_service")
        free_cash_flow = row.get("free_cash_flow")
        present_value = row.get("present_value")

        if not pd.isna(ocf):
            components.append(("Operating cash flow", float(ocf)))
        if not pd.isna(maint):
            components.append(("Maintenance capex", -abs(float(maint))))
        if not pd.isna(debt_service):
            components.append(("Debt service", -abs(float(debt_service))))

        records: List[Dict[str, Any]] = []
        running_total = 0.0

        for label, amount in components:
            start_value = running_total
            running_total += amount
            records.append(
                {
                    "Year": year,
                    "Component": label,
                    "Start": start_value,
                    "End": running_total,
                    "Amount": amount,
                    "Type": "Increase" if amount >= 0 else "Decrease",
                }
            )

        if not pd.isna(free_cash_flow):
            fcf_val = float(free_cash_flow)
            records.append(
                {
                    "Year": year,
                    "Component": "Free cash flow",
                    "Start": 0.0,
                    "End": fcf_val,
                    "Amount": fcf_val,
                    "Type": "Result",
                }
            )

        if not pd.isna(present_value):
            pv_val = float(present_value)
            records.append(
                {
                    "Year": year,
                    "Component": "Present value",
                    "Start": 0.0,
                    "End": pv_val,
                    "Amount": pv_val,
                    "Type": "Result",
                }
            )

        if not records:
            continue

        frame = pd.DataFrame(records)
        frame["Base"] = frame[["Start", "End"]].min(axis=1)
        frame["Cap"] = frame[["Start", "End"]].max(axis=1)
        frame["Mid"] = frame[["Base", "Cap"]].mean(axis=1)
        bridge_frames[year] = frame

    return bridge_frames


def _cashflow_waterfall_chart(frame: pd.DataFrame) -> Optional[alt.Chart]:
    """Return a waterfall chart for a single year's cash-flow bridge."""

    if frame.empty:
        return None

    component_order = [
        label
        for label in [
            "Operating cash flow",
            "Maintenance capex",
            "Debt service",
            "Free cash flow",
            "Present value",
        ]
        if label in frame["Component"].values
    ]

    color_scale = alt.Scale(
        domain=["Increase", "Decrease", "Result"],
        range=["#2ca02c", "#d62728", "#1f77b4"],
    )

    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Component:N", sort=component_order, title="Component"),
            y=alt.Y("Base:Q", title="Cash impact (USD)"),
            y2="Cap:Q",
            color=alt.Color("Type:N", scale=color_scale, title="Movement"),
            tooltip=[
                alt.Tooltip("Component:N"),
                alt.Tooltip("Amount:Q", format=",.0f"),
                alt.Tooltip("Start:Q", title="Start", format=",.0f"),
                alt.Tooltip("End:Q", title="End", format=",.0f"),
            ],
        )
    )

    labels = (
        alt.Chart(frame)
        .mark_text(color="black", dy=-6)
        .encode(
            x=alt.X("Component:N", sort=component_order),
            y=alt.Y("Cap:Q"),
            text=alt.Text("Amount:Q", format=",.0f"),
        )
    )

    return bars + labels


def _combined_leverage_chart(
    dscr_df: pd.DataFrame, coverage_df: pd.DataFrame, leverage_df: pd.DataFrame
) -> Optional[alt.Chart]:
    """Create a combined line chart for DSCR, interest coverage, and leverage metrics."""

    records: List[Dict[str, Any]] = []

    if not dscr_df.empty and {"year", "dscr"}.issubset(dscr_df.columns):
        temp = dscr_df.copy()
        temp["year"] = _to_numeric(temp["year"])
        temp["dscr"] = _to_numeric(temp["dscr"])
        for _, row in temp.dropna(subset=["year", "dscr"]).iterrows():
            records.append(
                {
                    "Year": int(row["year"]),
                    "Metric": "DSCR",
                    "Value": float(row["dscr"]),
                }
            )

    if not coverage_df.empty and "year" in coverage_df.columns:
        temp = coverage_df.copy()
        temp["year"] = _to_numeric(temp["year"])
        for column, label in [
            ("interest_coverage", "Interest coverage"),
        ]:
            if column in temp.columns:
                temp[column] = _to_numeric(temp[column])
                for _, row in temp.dropna(subset=["year", column]).iterrows():
                    records.append(
                        {
                            "Year": int(row["year"]),
                            "Metric": label,
                            "Value": float(row[column]),
                        }
                    )

    if not leverage_df.empty and "year" in leverage_df.columns:
        temp = leverage_df.copy()
        temp["year"] = _to_numeric(temp["year"])
        leverage_column = None
        leverage_label = ""
        if "debt_to_equity" in temp.columns:
            leverage_column = "debt_to_equity"
            leverage_label = "Debt-to-equity"
        elif "debt_ratio" in temp.columns:
            leverage_column = "debt_ratio"
            leverage_label = "Debt ratio"
        if leverage_column:
            temp[leverage_column] = _to_numeric(temp[leverage_column])
            for _, row in temp.dropna(subset=["year", leverage_column]).iterrows():
                records.append(
                    {
                        "Year": int(row["year"]),
                        "Metric": leverage_label,
                        "Value": float(row[leverage_column]),
                    }
                )

    if not records:
        return None

    data = pd.DataFrame(records)
    data = data.dropna(subset=["Year", "Value"])
    if data.empty:
        return None

    data["Year"] = data["Year"].astype(int)

    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Value:Q", title="Ratio"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Value:Q", format=",.2f"),
            ],
        )
    )


def _tornado_chart(
    what_if_df: pd.DataFrame, scenario_df: pd.DataFrame
) -> Optional[alt.Chart]:
    """Build a tornado chart comparing scenario deltas by NPV."""

    records: List[Dict[str, Any]] = []

    if not what_if_df.empty and {"Scenario", "NPV Δ"}.issubset(what_if_df.columns):
        for _, row in what_if_df.iterrows():
            scenario = row.get("Scenario")
            if scenario in {None, "Baseline"}:
                continue
            delta = row.get("NPV Δ")
            delta_val = float(delta) if pd.notna(delta) else None
            if delta_val is None:
                continue
            records.append(
                {
                    "Scenario": str(scenario),
                    "Impact": delta_val,
                    "Source": "What-if",
                }
            )

    if not scenario_df.empty and {"Scenario", "NPV Δ"}.issubset(scenario_df.columns):
        for _, row in scenario_df.iterrows():
            scenario = row.get("Scenario")
            if scenario in {None, "Baseline"}:
                continue
            delta = row.get("NPV Δ")
            delta_val = float(delta) if pd.notna(delta) else None
            if delta_val is None:
                continue
            records.append(
                {
                    "Scenario": str(scenario),
                    "Impact": delta_val,
                    "Source": "Scenario planning",
                }
            )

    if not records:
        return None

    data = pd.DataFrame(records)
    data = data.dropna(subset=["Impact"])
    if data.empty:
        return None

    data["Label"] = data["Source"] + " – " + data["Scenario"]
    data["abs_impact"] = data["Impact"].abs()
    order = data.sort_values("abs_impact", ascending=False)["Label"].tolist()

    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("Label:N", sort=order, title="Scenario"),
            x=alt.X("Impact:Q", title="NPV Δ (USD)"),
            color=alt.Color("Source:N", title="Source"),
            tooltip=[
                alt.Tooltip("Source:N"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Impact:Q", title="NPV Δ", format=",.0f"),
            ],
        )
    )

    zero_rule = alt.Chart(pd.DataFrame({"Impact": [0]})).mark_rule(color="black")

    return bars + zero_rule.encode(x="Impact:Q")


def _monte_carlo_distribution_charts(samples_df: pd.DataFrame) -> List[Tuple[str, alt.Chart]]:
    """Return histogram+density charts for Monte Carlo outputs."""

    charts: List[Tuple[str, alt.Chart]] = []
    if samples_df.empty:
        return charts

    numeric = samples_df.apply(_to_numeric)
    metric_map = {
        "npv": "NPV",
        "irr": "IRR",
        "min_dscr": "Minimum DSCR",
    }

    for column, label in metric_map.items():
        if column not in numeric.columns:
            continue
        series = numeric[column].dropna()
        if series.empty:
            continue
        base = pd.DataFrame({label: series})
        histogram = (
            alt.Chart(base)
            .mark_bar(opacity=0.45)
            .encode(
                x=alt.X(f"{label}:Q", bin=alt.Bin(maxbins=40), title=label),
                y=alt.Y("count():Q", title="Iterations"),
                tooltip=[alt.Tooltip("count():Q", title="Iterations")],
            )
        )
        density = (
            alt.Chart(base)
            .transform_density(label, as_=[label, "density"])
            .mark_line(color="#d62728")
            .encode(
                x=alt.X(f"{label}:Q", title=label),
                y=alt.Y("density:Q", title="Density"),
            )
        )
        charts.append((label, histogram + density))

    return charts


def _build_anomaly_points(risk_df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    """Align anomaly flags with historical revenue values for plotting."""

    if risk_df.empty or trend_df.empty:
        return pd.DataFrame()

    if "Year" not in risk_df.columns or {"year", "revenue"} - set(trend_df.columns):
        return pd.DataFrame()

    revenue_lookup = (
        trend_df[["year", "revenue"]]
        .assign(year=lambda df: pd.to_numeric(df["year"], errors="coerce"))
        .dropna(subset=["year", "revenue"])
        .set_index("year")["revenue"]
    )

    points: List[Dict[str, Any]] = []
    aligned = risk_df.copy()
    aligned["Year"] = _to_numeric(aligned["Year"])
    if "Observed growth" in aligned.columns:
        aligned["Observed growth"] = _to_numeric(aligned["Observed growth"])

    for _, row in aligned.dropna(subset=["Year"]).iterrows():
        year = float(row["Year"])
        if year not in revenue_lookup.index:
            continue
        points.append(
            {
                "Year": int(year),
                "Value": float(revenue_lookup.loc[year]),
                "Flag": row.get("Flag", "Anomaly"),
                "Observed growth": row.get("Observed growth"),
            }
        )

    return pd.DataFrame(points)


def _forecast_overlay_chart(
    trend_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    time_series_df: pd.DataFrame,
    anomaly_points: pd.DataFrame,
) -> Optional[alt.Chart]:
    """Render a combined forecast chart with anomaly highlights."""

    records: List[Dict[str, Any]] = []

    if not trend_df.empty and {"year", "revenue"}.issubset(trend_df.columns):
        temp = trend_df.copy()
        temp["year"] = _to_numeric(temp["year"])
        temp["revenue"] = _to_numeric(temp["revenue"])
        for _, row in temp.dropna(subset=["year", "revenue"]).iterrows():
            records.append(
                {
                    "Year": int(row["year"]),
                    "Series": "Historical revenue",
                    "Value": float(row["revenue"]),
                }
            )

    if not forecast_df.empty and {"Year", "Revenue forecast"}.issubset(forecast_df.columns):
        temp = forecast_df.copy()
        temp["Year"] = _to_numeric(temp["Year"])
        temp["Revenue forecast"] = _to_numeric(temp["Revenue forecast"])
        for _, row in temp.dropna(subset=["Year", "Revenue forecast"]).iterrows():
            records.append(
                {
                    "Year": int(row["Year"]),
                    "Series": "Automated forecast",
                    "Value": float(row["Revenue forecast"]),
                }
            )

    if not time_series_df.empty and {"Year", "Revenue forecast"}.issubset(time_series_df.columns):
        temp = time_series_df.copy()
        temp["Year"] = _to_numeric(temp["Year"])
        temp["Revenue forecast"] = _to_numeric(temp["Revenue forecast"])
        for _, row in temp.dropna(subset=["Year", "Revenue forecast"]).iterrows():
            records.append(
                {
                    "Year": int(row["Year"]),
                    "Series": "AR(1) forecast",
                    "Value": float(row["Revenue forecast"]),
                }
            )

    if not records:
        return None

    data = pd.DataFrame(records)
    data = data.dropna(subset=["Year", "Value"])
    if data.empty:
        return None

    data["Year"] = data["Year"].astype(int)

    base_chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Value:Q", title="Revenue (USD)"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[
                alt.Tooltip("Series:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Value:Q", title="Revenue", format=",.0f"),
            ],
        )
    )

    if anomaly_points.empty:
        return base_chart

    anomalies = anomaly_points.copy()
    anomalies["Year"] = _to_numeric(anomalies["Year"])
    anomalies["Value"] = _to_numeric(anomalies["Value"])
    anomalies = anomalies.dropna(subset=["Year", "Value"])
    anomalies["Year"] = anomalies["Year"].astype(int)

    points = (
        alt.Chart(anomalies)
        .mark_point(shape="triangle-up", size=120, color="#d62728")
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y("Value:Q"),
            tooltip=[
                alt.Tooltip("Flag:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Value:Q", title="Revenue", format=",.0f"),
                alt.Tooltip("Observed growth:Q", format=".2%"),
            ],
        )
    )

    return base_chart + points


def _break_even_heatmap_charts(
    break_even_df: pd.DataFrame,
) -> Tuple[Optional[alt.Chart], Optional[alt.Chart]]:
    """Build cost heatmap and break-even unit charts for the break-even analysis."""

    if break_even_df.empty or "Category" not in break_even_df.columns:
        return None, None

    cost_columns = [col for col in ["Direct cost", "Shared cost"] if col in break_even_df.columns]
    heatmap: Optional[alt.Chart] = None
    units_chart: Optional[alt.Chart] = None

    if cost_columns:
        cost_long = (
            break_even_df.melt(
                id_vars="Category",
                value_vars=cost_columns,
                var_name="Cost component",
                value_name="Cost",
            )
        )
        cost_long["Cost"] = _to_numeric(cost_long["Cost"])
        cost_long = cost_long.dropna(subset=["Cost"])
        if not cost_long.empty:
            heatmap_base = (
                alt.Chart(cost_long)
                .mark_rect()
                .encode(
                    x=alt.X("Category:N", title="Category"),
                    y=alt.Y("Cost component:N", title="Cost type"),
                    color=alt.Color(
                        "Cost:Q",
                        title="Annual cost (USD)",
                        scale=alt.Scale(scheme="yellowgreenblue"),
                    ),
                    tooltip=[
                        alt.Tooltip("Category:N"),
                        alt.Tooltip("Cost component:N"),
                        alt.Tooltip("Cost:Q", format=",.0f"),
                    ],
                )
            )
            heatmap_text = (
                alt.Chart(cost_long)
                .mark_text(color="black")
                .encode(
                    x=alt.X("Category:N"),
                    y=alt.Y("Cost component:N"),
                    text=alt.Text("Cost:Q", format=",.0f"),
                )
            )
            heatmap = heatmap_base + heatmap_text

    if "Break-even units" in break_even_df.columns:
        units_data = break_even_df[["Category", "Break-even units"]].copy()
        units_data["Break-even units"] = _to_numeric(units_data["Break-even units"])
        units_data = units_data.dropna(subset=["Break-even units"])
        if not units_data.empty:
            units_chart = (
                alt.Chart(units_data)
                .mark_bar()
                .encode(
                    x=alt.X("Category:N", title="Category"),
                    y=alt.Y("Break-even units:Q", title="Break-even units"),
                    color=alt.Color("Category:N", legend=None),
                    tooltip=[
                        alt.Tooltip("Category:N"),
                        alt.Tooltip("Break-even units:Q", format=",.0f"),
                    ],
                )
            )

    return heatmap, units_chart

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

    if hasattr(st, "rerun"):
        try:
            st.rerun()
            return
        except Exception:
            pass
    # Fallback for Streamlit versions that still expose experimental API.
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
    results = generate_model_outputs(
        assumptions,
        analytics_plan=AnalyticsPlan.summary(),
    )
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
        import xlsxwriter  # type: ignore  # noqa: F401

        engine = "xlsxwriter"
    except ImportError:
        try:
            import openpyxl  # type: ignore  # noqa: F401

            engine = "openpyxl"
        except ImportError as exc:  # pragma: no cover - surfaced in UI
            st.error(
                "Excel exports require either the `xlsxwriter` or `openpyxl` package. "
                "Install one of these dependencies and try again."
            )
            raise RuntimeError("Missing Excel writer dependency") from exc

    with pd.ExcelWriter(buffer, engine=engine) as writer:
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
        if engine == "xlsxwriter":
            workbook.set_properties(
                {
                    "title": f"Broiler Model - {scenario}",
                    "subject": "Broiler chicken financial model",
                    "comments": "Generated via Streamlit dashboard",
                }
            )
        else:
            props = workbook.properties
            props.title = f"Broiler Model - {scenario}"
            props.subject = "Broiler chicken financial model"
            props.comments = "Generated via Streamlit dashboard"

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


def _normalise_schedule_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    """Return a list of dictionaries for any supported row payload."""

    normalised: List[Dict[str, Any]] = []
    for row in rows:
        if row is None:
            continue
        if isinstance(row, dict):
            normalised.append(copy.deepcopy(row))
        elif is_dataclass(row):
            normalised.append(asdict(row))
        else:
            try:
                normalised.append(dict(row))
            except TypeError:
                raise TypeError(
                    "Schedule defaults must be dict-like or dataclass instances."
                ) from None
    return normalised


def _initialise_schedule_state(
    namespace: str,
    scenario: str,
    schedule_key: str,
    default_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the stored DataFrame for a schedule, initialising from defaults as needed."""

    store = st.session_state.setdefault(namespace, {})
    scenario_store = store.setdefault(scenario, {})
    default_records = _normalise_schedule_rows(default_rows)
    state = scenario_store.get(schedule_key)
    if state is None:
        state = {
            "data": copy.deepcopy(default_records),
            "default": copy.deepcopy(default_records),
        }
        scenario_store[schedule_key] = state
    else:
        stored_data = state.get("data", [])
        stored_default = state.get("default", [])
        if stored_data == stored_default and stored_default != default_records:
            state["data"] = copy.deepcopy(default_records)
            state["default"] = copy.deepcopy(default_records)
    df = pd.DataFrame(state.get("data", default_records))
    return df, state


def _coerce_assumption_value(key: str, raw_value: Any, current_value: Any) -> Any:
    """Convert schedule entries back to the dataclass field types."""

    if raw_value is None:
        return current_value
    if pd.isna(raw_value):  # type: ignore[arg-type]
        return current_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return current_value
        raw_value = stripped

    target_type = _ASSUMPTION_FIELD_TYPES.get(key, type(current_value))

    try:
        if target_type is int:
            return int(float(raw_value))
        if target_type is float:
            return float(raw_value)
        return str(raw_value)
    except (TypeError, ValueError):
        return current_value


def _assumptions_from_schedule(
    rows: List[Dict[str, Any]],
    current: Assumptions,
) -> Optional[Assumptions]:
    """Derive an updated ``Assumptions`` instance from schedule edits."""

    updated_values = asdict(current)
    changed = False

    for row in rows:
        item = row.get("item")
        if not item:
            continue
        key = _ASSUMPTION_LABEL_TO_KEY.get(str(item))
        if not key:
            continue

        existing = updated_values.get(key)
        new_value = _coerce_assumption_value(key, row.get("value"), existing)

        if isinstance(existing, float) and isinstance(new_value, float):
            if math.isnan(existing) and math.isnan(new_value):
                continue
            if math.isclose(existing, new_value, rel_tol=1e-9, abs_tol=1e-9):
                continue
        elif existing == new_value:
            continue

        updated_values[key] = new_value
        changed = True

    if not changed:
        return None

    return Assumptions(**updated_values)


def _next_period_label(df: pd.DataFrame) -> str:
    """Generate a reasonable period label for a newly added row."""

    if "Period" not in df.columns or df["Period"].dropna().empty:
        return f"Period {len(df) + 1}"
    last_value = str(df["Period"].dropna().iloc[-1]).strip()
    parts = last_value.split()
    if parts and parts[-1].isdigit():
        prefix = " ".join(parts[:-1]) or "Period"
        return f"{prefix} {int(parts[-1]) + 1}"
    match = re.search(r"(\d+)(?!.*\d)", last_value)
    if match:
        number = int(match.group(1)) + 1
        prefix = last_value[: match.start()].strip() or "Period"
        suffix = last_value[match.end() :].strip()
        if suffix:
            return f"{prefix} {number} {suffix}".strip()
        return f"{prefix} {number}".strip()
    return f"Period {len(df) + 1}"


def _ensure_fixed_columns(
    df: pd.DataFrame, fixed_columns: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    if not fixed_columns:
        return df
    new_df = df.copy()
    for column, value in fixed_columns.items():
        if column in new_df.columns:
            new_df[column] = value
    return new_df


def _add_schedule_row(
    df: pd.DataFrame,
    row_defaults: Optional[Dict[str, Any]] = None,
    fixed_columns: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Append a new row using defaults and fixed column values."""

    new_row = {col: None for col in df.columns}
    defaults = row_defaults or {}
    for column, value in defaults.items():
        if column in new_row:
            new_row[column] = copy.deepcopy(value)
    if "Period" in df.columns and "Period" not in defaults:
        new_row["Period"] = _next_period_label(df)
    if fixed_columns:
        for column, value in fixed_columns.items():
            if column in new_row:
                new_row[column] = value
    if ROW_REMOVAL_COLUMN in new_row:
        new_row[ROW_REMOVAL_COLUMN] = False
    if ROW_EDIT_COLUMN in new_row:
        new_row[ROW_EDIT_COLUMN] = False
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def _apply_yearly_increment(
    df: pd.DataFrame, columns: List[str], rate: float
) -> pd.DataFrame:
    """Apply a compound yearly increment across rows for selected columns."""

    if not columns or rate == 0.0 or df.empty:
        return df
    new_df = df.copy()
    for column in columns:
        if column not in new_df.columns:
            continue
        numeric_series = pd.to_numeric(new_df[column], errors="coerce")
        if numeric_series.notna().sum() == 0:
            # nothing numeric to increment; leave column as-is
            continue
        # ensure the column can hold floating point results from the increment
        new_df[column] = numeric_series.astype("Float64")
        prev_value: Optional[float] = None
        for idx in new_df.index:
            current = new_df.at[idx, column]
            if prev_value is None:
                if not pd.isna(current):
                    prev_value = float(current)
                continue
            prev_value = prev_value * (1 + rate)
            new_df.at[idx, column] = prev_value
    return new_df


def _auto_compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute revenue when units and price are available."""

    if not {"Units", "Unit price", "Revenue"}.issubset(df.columns):
        return df
    new_df = df.copy()
    for idx in range(len(new_df)):
        units = pd.to_numeric(new_df.at[idx, "Units"], errors="coerce")
        price = pd.to_numeric(new_df.at[idx, "Unit price"], errors="coerce")
        if pd.isna(units) or pd.isna(price):
            continue
        new_df.at[idx, "Revenue"] = float(units) * float(price)
    return new_df


def _sanitize_key(label: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", label.lower()).strip("_") or "schedule"


def _stringify_iterable_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)
    return value


def _stringify_iterable_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    new_df = df.copy()
    for column in columns:
        if column in new_df.columns:
            new_df[column] = new_df[column].apply(_stringify_iterable_value)
    return new_df


def _parse_iterable_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parts = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
            if not parts:
                return None
            parsed_parts: List[Any] = []
            for item in parts:
                try:
                    parsed_parts.append(float(item))
                except ValueError:
                    parsed_parts.append(item)
            parsed = parsed_parts
        return parsed
    return value


def _coerce_editor_input(value: str) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _render_row_edit_form(
    df: pd.DataFrame,
    idx: int,
    fixed_columns: Optional[Dict[str, Any]],
    namespace: str,
    schedule_key: str,
    scenario: str,
) -> Tuple[pd.DataFrame, bool]:
    fixed_columns = fixed_columns or {}
    editable_columns = [
        column
        for column in df.columns
        if column not in {ROW_REMOVAL_COLUMN, ROW_EDIT_COLUMN}
        and column not in fixed_columns
    ]
    if not editable_columns:
        return df, False

    form_key = f"edit_row_form_{namespace}_{schedule_key}_{scenario}_{idx}"
    with st.form(form_key):
        st.markdown(f"**Editing row {idx + 1}**")
        inputs: Dict[str, str] = {}
        for column in editable_columns:
            current_value = df.at[idx, column]
            default_text = "" if pd.isna(current_value) else str(current_value)
            inputs[column] = st.text_input(column, value=default_text)
        submitted = st.form_submit_button("Save row")

    if not submitted:
        return df, False

    new_df = df.copy()
    for column, raw_value in inputs.items():
        series = new_df[column]
        if pd.api.types.is_string_dtype(series):
            new_df[column] = series.astype(object)
        new_df.at[idx, column] = _coerce_editor_input(raw_value)
    st.success(f"Row {idx + 1} updated.")
    return new_df, True


def _parse_iterable_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    new_df = df.copy()
    for column in columns:
        if column in new_df.columns:
            new_df[column] = new_df[column].apply(_parse_iterable_value)
    return new_df


def _chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _build_column_config(df: pd.DataFrame, *, disabled: bool, fixed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create Streamlit column configuration based on series dtypes."""

    fixed = fixed or {}
    config: Dict[str, Any] = {}
    for column in df.columns:
        col_disabled = disabled or column in fixed
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            config[column] = st.column_config.CheckboxColumn(disabled=col_disabled)
        elif pd.api.types.is_numeric_dtype(series):
            config[column] = st.column_config.NumberColumn(disabled=col_disabled)
        elif pd.api.types.is_datetime64_any_dtype(series):
            config[column] = st.column_config.DatetimeColumn(disabled=col_disabled)
        else:
            config[column] = st.column_config.TextColumn(disabled=col_disabled)
    return config


def _render_input_section(
    title: str,
    fields: List[Dict[str, Any]],
    defaults: Assumptions,
    values: Dict[str, Any],
    *,
    columns: int = 3,
) -> None:
    """Render a consistently styled multi-column assumptions section."""

    st.markdown(f"### {title}")
    for group in _chunked(fields, columns):
        cols = st.columns(len(group))
        for widget, field in zip(cols, group):
            attr = field["attr"]
            label = field["label"]
            dtype = field.get("type", "float")
            min_value = field.get("min")
            max_value = field.get("max")
            step = field.get("step")
            fmt = field.get("format")
            help_text = field.get("help")

            default_val = getattr(defaults, attr)

            if min_value is not None:
                default_val = max(default_val, min_value)
            if max_value is not None:
                default_val = min(default_val, max_value)
            input_kwargs: Dict[str, Any] = {}
            if help_text:
                input_kwargs["help"] = help_text

            if dtype == "int":
                input_kwargs["value"] = int(round(default_val))
                input_kwargs["step"] = int(step) if step is not None else 1
                if min_value is not None:
                    input_kwargs["min_value"] = int(min_value)
                if max_value is not None:
                    input_kwargs["max_value"] = int(max_value)
                values[attr] = widget.number_input(label, **input_kwargs)
            else:
                input_kwargs["value"] = float(default_val)
                if min_value is not None:
                    input_kwargs["min_value"] = float(min_value)
                if max_value is not None:
                    input_kwargs["max_value"] = float(max_value)
                if step is not None:
                    input_kwargs["step"] = float(step)
                if fmt is not None:
                    input_kwargs["format"] = fmt
                values[attr] = widget.number_input(label, **input_kwargs)


def _render_schedule_editor(
    title: str,
    schedule_key: str,
    default_rows: List[Dict[str, Any]],
    scenario: str,
    namespace: str,
    *,
    fixed_columns: Optional[Dict[str, Any]] = None,
    row_defaults: Optional[Dict[str, Any]] = None,
    allow_yearly_increment: bool = True,
    auto_update_revenue: bool = False,
) -> pd.DataFrame:
    """Render an editable schedule with add/remove and yearly increment controls.

    Parameters
    ----------
    title, schedule_key
        Identifiers for the UI caption and the per-schedule state slot.
    default_rows
        List of default row dictionaries (or dataclasses) that seed the editor
        when no user overrides have been captured for the active scenario.
    scenario, namespace
        Keys used to scope the stored state inside ``st.session_state``.
    fixed_columns
        Optional mapping of column names to constant values. These columns are
        enforced after every edit and rendered as read-only in the table.
    row_defaults
        Optional template applied when users click “Add row”.
    allow_yearly_increment
        Enables the yearly growth controls when the schedule contains numeric
        columns.
    auto_update_revenue
        Recomputes ``Revenue`` from ``Units`` × ``Unit price`` after edits.
    """

    st.markdown(f"**{title}**")
    df, state = _initialise_schedule_state(namespace, scenario, schedule_key, default_rows)
    df = df.convert_dtypes()
    original_columns = [
        col
        for col in df.columns
        if col not in {ROW_REMOVAL_COLUMN, ROW_EDIT_COLUMN}
    ]

    fixed_columns = fixed_columns or {}
    row_defaults = row_defaults or {}

    edit_toggle_key = f"edit_enabled_{namespace}_{schedule_key}_{scenario}"
    edit_enabled = st.checkbox(
        "Enable editing",
        value=st.session_state.get(edit_toggle_key, True),
        key=edit_toggle_key,
        help="Toggle to edit this schedule. When disabled the schedule is read-only.",
    )

    operation_applied = False
    if edit_enabled:
        if ROW_REMOVAL_COLUMN not in df.columns:
            df[ROW_REMOVAL_COLUMN] = False
        if ROW_EDIT_COLUMN not in df.columns:
            df[ROW_EDIT_COLUMN] = False
        controls = st.columns(2)
        if controls[0].button(
            "Add row",
            key=f"add_{namespace}_{schedule_key}_{scenario}",
        ):
            df = _add_schedule_row(df, row_defaults=row_defaults, fixed_columns=fixed_columns)
            operation_applied = True
        if controls[1].button(
            "Remove row",
            key=f"remove_{namespace}_{schedule_key}_{scenario}",
        ):
            if not df.empty:
                df = df.iloc[:-1].reset_index(drop=True)
                operation_applied = True

        df = _ensure_fixed_columns(df, fixed_columns)

        numeric_columns = [
            column
            for column in df.columns
            if pd.to_numeric(df[column], errors="coerce").notna().sum() > 0
        ]

        if allow_yearly_increment and numeric_columns:
            inc_cols, inc_rate_col, inc_btn_col = st.columns([2, 1, 1])
            selected_columns = inc_cols.multiselect(
                "Yearly increment columns",
                options=numeric_columns,
                default=numeric_columns,
                key=f"inc_cols_{namespace}_{schedule_key}_{scenario}",
                help="Select numeric columns that should follow the yearly increment growth.",
            )
            increment_rate = inc_rate_col.number_input(
                "Yearly increment (%)",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"inc_rate_{namespace}_{schedule_key}_{scenario}",
            )
            if inc_btn_col.button(
                "Apply yearly increment",
                key=f"apply_inc_{namespace}_{schedule_key}_{scenario}",
            ):
                if selected_columns:
                    df = _apply_yearly_increment(df, selected_columns, increment_rate / 100.0)
                    operation_applied = True
                else:
                    st.warning("Select at least one column before applying an increment.")
        elif allow_yearly_increment:
            st.caption("No numeric columns available for yearly increment.")
    else:
        if ROW_REMOVAL_COLUMN in df.columns:
            df = df.drop(columns=[ROW_REMOVAL_COLUMN])
        if ROW_EDIT_COLUMN in df.columns:
            df = df.drop(columns=[ROW_EDIT_COLUMN])
        st.caption("Enable editing to modify this schedule. Current values remain read-only.")

    if (operation_applied and auto_update_revenue) or (auto_update_revenue and edit_enabled):
        df = _auto_compute_revenue(df)

    df = _ensure_fixed_columns(df, fixed_columns)

    column_config = _build_column_config(
        df,
        disabled=not edit_enabled,
        fixed=fixed_columns,
    )
    instructions: List[str] = []
    if edit_enabled and ROW_EDIT_COLUMN in df.columns:
        column_config[ROW_EDIT_COLUMN] = st.column_config.CheckboxColumn(
            "Edit",
            help="Tick to open the inline row editor before saving.",
            disabled=False,
        )
        instructions.append("Tick 'Edit' to open the row editor and save updates below.")
    if edit_enabled and ROW_REMOVAL_COLUMN in df.columns:
        column_config[ROW_REMOVAL_COLUMN] = st.column_config.CheckboxColumn(
            "Remove",
            help="Tick to remove this row when saving edits.",
            disabled=False,
        )
        instructions.append("Use the 'Remove' checkbox to delete the row before saving.")
    if instructions:
        st.caption(" ".join(instructions))

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        disabled=not edit_enabled,
        key=f"editor_{namespace}_{schedule_key}_{scenario}",
    )

    edited_df = pd.DataFrame(edited_df)
    row_form_applied = False
    if ROW_EDIT_COLUMN in edited_df.columns:
        edit_mask = edited_df[ROW_EDIT_COLUMN].fillna(False)
        edit_indices = [idx for idx, flag in enumerate(edit_mask) if flag]
        for row_idx in edit_indices:
            edited_df, submitted = _render_row_edit_form(
                edited_df,
                row_idx,
                fixed_columns,
                namespace,
                schedule_key,
                scenario,
            )
            row_form_applied = row_form_applied or submitted
        edited_df[ROW_EDIT_COLUMN] = False
    if ROW_REMOVAL_COLUMN in edited_df.columns:
        remove_mask = edited_df[ROW_REMOVAL_COLUMN].fillna(False)
        if remove_mask.any():
            edited_df = edited_df.loc[~remove_mask].reset_index(drop=True)
        edited_df = edited_df.drop(columns=[ROW_REMOVAL_COLUMN])
    if ROW_EDIT_COLUMN in edited_df.columns:
        edited_df = edited_df.drop(columns=[ROW_EDIT_COLUMN])
    if row_form_applied:
        operation_applied = True
    if auto_update_revenue:
        edited_df = _auto_compute_revenue(edited_df)
    edited_df = _ensure_fixed_columns(edited_df, fixed_columns)

    state["data"] = edited_df.replace({pd.NA: None}).to_dict("records")
    return edited_df.reindex(columns=original_columns, fill_value=None)


def _render_analytics_schedule(
    title: str,
    schedule_id: str,
    df: pd.DataFrame,
    scenario: str,
    *,
    namespace: str = "advanced_schedule_state",
    allow_yearly_increment: bool = False,
    row_defaults: Optional[Dict[str, Any]] = None,
    fixed_columns: Optional[Dict[str, Any]] = None,
    iterable_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper to expose schedule editing in advanced analytics."""

    if df is None or df.empty:
        return df

    clean_df = df.copy()
    iterable_columns = iterable_columns or []
    if iterable_columns:
        clean_df = _stringify_iterable_columns(clean_df, iterable_columns)

    defaults = clean_df.where(pd.notna(clean_df), None).to_dict("records")
    template = row_defaults or {column: None for column in clean_df.columns}
    if iterable_columns and template:
        template = template.copy()
        for column in iterable_columns:
            if column in template:
                template[column] = _stringify_iterable_value(template[column])

    return _render_schedule_editor(
        title,
        _sanitize_key(schedule_id or title),
        defaults,
        scenario,
        namespace=namespace,
        fixed_columns=fixed_columns,
        row_defaults=template,
        allow_yearly_increment=allow_yearly_increment,
    ).pipe(lambda edited: _parse_iterable_columns(edited, iterable_columns) if iterable_columns else edited)


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


def assumptions_form(defaults: Assumptions, payload: Dict[str, Any]) -> Assumptions:
    st.header("Model assumptions")

    farm_name = st.text_input("Farm name", defaults.farm_name)

    ai_container = st.expander("AI & Machine Learning Settings", expanded=False)
    _render_ai_settings(payload, ai_container)

    st.subheader("Input Landing Page")
    st.caption("Adjust the production, pricing, cost, and capital structure assumptions below.")

    values: Dict[str, Any] = {}

    _render_input_section(
        "Production",
        [
            {"attr": "cycles_per_year", "label": "Cycles per year", "min": 1, "max": 12, "type": "int"},
            {
                "attr": "production_start_year",
                "label": "Production start year",
                "min": 1900,
                "max": 2100,
                "type": "int",
            },
            {
                "attr": "production_horizon_years",
                "label": "Production horizon (years)",
                "min": 1,
                "max": 40,
                "type": "int",
            },
            {
                "attr": "birds_per_cycle",
                "label": "Birds per cycle",
                "min": 1000,
                "max": 100000,
                "step": 1000,
                "type": "int",
            },
            {
                "attr": "mortality_rate",
                "label": "Mortality rate",
                "min": 0.0,
                "max": 0.2,
                "step": 0.005,
                "format": "%.3f",
            },
            {
                "attr": "final_weight_kg",
                "label": "Final weight (kg)",
                "min": 1.0,
                "max": 4.0,
                "step": 0.1,
            },
        ],
        defaults,
        values,
        columns=4,
    )

    start_year = int(values.get("production_start_year", defaults.production_start_year))
    horizon_years = int(values.get("production_horizon_years", defaults.production_horizon_years))
    if horizon_years <= 0:
        horizon_years = 1
    end_year = start_year + horizon_years - 1
    timeline_cols = st.columns(2)
    timeline_cols[0].metric("Production start year", start_year)
    timeline_cols[1].metric("Production end year", end_year)

    _render_input_section(
        "Pricing",
        [
            {
                "attr": "live_price_per_kg",
                "label": "Live broiler price per kg",
                "min": 0.5,
                "max": 5.0,
                "step": 0.05,
            },
            {
                "attr": "eggs_price_per_dozen",
                "label": "Eggs price per dozen",
                "min": 0.5,
                "max": 8.0,
                "step": 0.1,
            },
            {
                "attr": "manure_price_per_ton",
                "label": "Poultry manure price per ton",
                "min": 0.0,
                "max": 150.0,
                "step": 1.0,
            },
            {
                "attr": "live_bird_price_per_head",
                "label": "Live bird price per head",
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
            },
            {
                "attr": "byproduct_price_per_kg",
                "label": "By-product price per kg",
                "min": 0.0,
                "max": 5.0,
                "step": 0.05,
            },
            {
                "attr": "price_growth",
                "label": "Annual price growth",
                "min": -0.05,
                "max": 0.1,
                "step": 0.005,
                "format": "%.3f",
            },
        ],
        defaults,
        values,
        columns=3,
    )

    _render_input_section(
        "Operating costs",
        [
            {
                "attr": "feed_conversion_ratio",
                "label": "Feed conversion ratio",
                "min": 1.0,
                "max": 2.5,
                "step": 0.05,
            },
            {
                "attr": "feed_cost_per_kg",
                "label": "Feed cost per kg",
                "min": 0.2,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "attr": "chick_cost",
                "label": "Chick cost",
                "min": 0.3,
                "max": 2.0,
                "step": 0.05,
            },
            {
                "attr": "processing_cost_per_bird",
                "label": "Processing cost per bird",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "attr": "vaccination_cost_per_bird",
                "label": "Vaccination cost per bird",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
            },
            {
                "attr": "litter_disposal_per_cycle",
                "label": "Litter & disposal per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "propane_per_cycle",
                "label": "Propane per cycle",
                "min": 0.0,
                "max": 20000.0,
                "step": 100.0,
            },
            {
                "attr": "electricity_per_cycle",
                "label": "Electricity per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "labor_per_cycle",
                "label": "Labor per cycle",
                "min": 0.0,
                "max": 50000.0,
                "step": 500.0,
            },
            {
                "attr": "maintenance_per_cycle",
                "label": "Maintenance per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "management_fee_per_cycle",
                "label": "Management fee per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "insurance_per_cycle",
                "label": "Insurance per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "overhead_per_cycle",
                "label": "Overhead per cycle",
                "min": 0.0,
                "max": 10000.0,
                "step": 100.0,
            },
            {
                "attr": "cost_inflation",
                "label": "Cost inflation",
                "min": -0.05,
                "max": 0.1,
                "step": 0.005,
                "format": "%.3f",
            },
        ],
        defaults,
        values,
        columns=4,
    )

    _render_input_section(
        "Capital & financing",
        [
            {
                "attr": "capex_housing",
                "label": "Housing capex",
                "min": 0.0,
                "max": 5000000.0,
                "step": 10000.0,
            },
            {
                "attr": "capex_equipment",
                "label": "Equipment capex",
                "min": 0.0,
                "max": 2000000.0,
                "step": 5000.0,
            },
            {
                "attr": "working_capital",
                "label": "Working capital",
                "min": 0.0,
                "max": 1000000.0,
                "step": 5000.0,
            },
            {
                "attr": "depreciation_years",
                "label": "Depreciation years",
                "min": 1,
                "max": 40,
                "type": "int",
            },
            {
                "attr": "maintenance_capex_annual",
                "label": "Maintenance capex (annual)",
                "min": 0.0,
                "max": 200000.0,
                "step": 5000.0,
            },
            {
                "attr": "debt_ratio",
                "label": "Debt ratio",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "attr": "debt_interest_rate",
                "label": "Debt interest rate",
                "min": 0.0,
                "max": 0.2,
                "step": 0.005,
                "format": "%.3f",
            },
            {
                "attr": "debt_term_years",
                "label": "Debt term (years)",
                "min": 1,
                "max": 30,
                "type": "int",
            },
            {
                "attr": "discount_rate",
                "label": "Discount rate",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
                "format": "%.3f",
            },
            {
                "attr": "tax_rate",
                "label": "Tax rate",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
                "format": "%.3f",
            },
        ],
        defaults,
        values,
        columns=4,
    )

    return Assumptions(
        farm_name=farm_name,
        cycles_per_year=int(values["cycles_per_year"]),
        production_start_year=int(values["production_start_year"]),
        production_horizon_years=int(values["production_horizon_years"]),
        birds_per_cycle=int(values["birds_per_cycle"]),
        mortality_rate=float(values["mortality_rate"]),
        final_weight_kg=float(values["final_weight_kg"]),
        live_price_per_kg=float(values["live_price_per_kg"]),
        chick_cost=float(values["chick_cost"]),
        feed_conversion_ratio=float(values["feed_conversion_ratio"]),
        feed_cost_per_kg=float(values["feed_cost_per_kg"]),
        processing_cost_per_bird=float(values["processing_cost_per_bird"]),
        vaccination_cost_per_bird=float(values["vaccination_cost_per_bird"]),
        litter_disposal_per_cycle=float(values["litter_disposal_per_cycle"]),
        propane_per_cycle=float(values["propane_per_cycle"]),
        electricity_per_cycle=float(values["electricity_per_cycle"]),
        labor_per_cycle=float(values["labor_per_cycle"]),
        maintenance_per_cycle=float(values["maintenance_per_cycle"]),
        management_fee_per_cycle=float(values["management_fee_per_cycle"]),
        insurance_per_cycle=float(values["insurance_per_cycle"]),
        overhead_per_cycle=float(values["overhead_per_cycle"]),
        capex_housing=float(values["capex_housing"]),
        capex_equipment=float(values["capex_equipment"]),
        working_capital=float(values["working_capital"]),
        discount_rate=float(values["discount_rate"]),
        price_growth=float(values["price_growth"]),
        eggs_price_per_dozen=float(values["eggs_price_per_dozen"]),
        manure_price_per_ton=float(values["manure_price_per_ton"]),
        live_bird_price_per_head=float(values["live_bird_price_per_head"]),
        byproduct_price_per_kg=float(values["byproduct_price_per_kg"]),
        cost_inflation=float(values["cost_inflation"]),
        tax_rate=float(values["tax_rate"]),
        debt_ratio=float(values["debt_ratio"]),
        debt_interest_rate=float(values["debt_interest_rate"]),
        debt_term_years=int(values["debt_term_years"]),
        depreciation_years=int(values["depreciation_years"]),
        maintenance_capex_annual=float(values["maintenance_capex_annual"]),
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

    (
        input_tab,
        production_tab,
        financials_tab,
        analytics_tab,
    ) = st.tabs(
        [
            "Input Landing Page",
            "Production & revenues",
            "Financial statements",
            "Advanced analytics",
        ]
    )

    with input_tab:
        assumptions = assumptions_form(defaults, payload)

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
    revenue_summary = results.get(
        "revenue_summary",
        summarise_revenue_totals(
            revenue_schedules,
            model.assumptions.cycles_per_year,
            model.assumptions.production_horizon_years,
            model.assumptions.production_start_year,
        ),
    )
    financials = results["financial_statements"]
    advanced = results["advanced_analytics"]

    col1, col2, col3 = st.columns(3)
    col1.metric("NPV", f"${valuation['npv']:,.0f}")
    col2.metric("IRR", f"{valuation['irr']:.2%}")
    col3.metric("Discount rate", f"{valuation['discount_rate']:.2%}")

    cycles_df = pd.DataFrame([asdict(cycle) for cycle in results["cycles"]])
    annual_summary_obj = results["annual"]
    annual_df = pd.DataFrame([asdict(annual_summary_obj)])
    cashflow_df = pd.DataFrame([asdict(row) for row in results["cashflows"]])
    cashflow_bridge_frames = _prepare_cashflow_bridge_frames(cashflow_df)
    income_df = pd.DataFrame([asdict(row) for row in financials["income_statement"]])
    balance_df = pd.DataFrame([asdict(row) for row in financials["balance_sheet"]])
    cash_statement_df = pd.DataFrame([asdict(row) for row in financials["cash_flow_statement"]])
    loan_df = pd.DataFrame(financials["loan_schedule"])
    asset_schedule_df = (
        balance_df[
            ["year", "cash", "working_capital", "net_ppe", "total_assets"]
        ]
        .copy()
        .rename(
            columns={
                "year": "Year",
                "cash": "Ending cash",
                "working_capital": "Working capital",
                "net_ppe": "Net PP&E",
                "total_assets": "Total assets",
            }
        )
    )
    metrics_df = pd.DataFrame(advanced["metrics"])
    dscr_df = pd.DataFrame(advanced["dscr"])
    trend_df = pd.DataFrame(advanced["trend"])
    returns_df = pd.DataFrame(advanced.get("returns", []))
    coverage_df = pd.DataFrame(advanced.get("coverage", []))
    leverage_df = pd.DataFrame(advanced.get("leverage", []))
    what_if_df = pd.DataFrame(advanced.get("what_if", []))
    monte_carlo = advanced.get("monte_carlo", {})
    monte_carlo_summary_df = pd.DataFrame([monte_carlo["summary"]]) if monte_carlo.get("summary") else pd.DataFrame()
    monte_carlo_samples_df = pd.DataFrame(monte_carlo.get("samples", []))
    break_even_df = pd.DataFrame(advanced.get("break_even", []))
    goal_seek = advanced.get("goal_seek", {})
    predictive = advanced.get("predictive", {})
    forecast_df = pd.DataFrame(predictive.get("automated_forecast", []))
    time_series_df = pd.DataFrame(predictive.get("time_series", {}).get("forecast", []))
    risk_metadata = predictive.get("risk_anomalies", {})
    risk_df = pd.DataFrame(risk_metadata.get("observations", []))
    ml_methods_df = pd.DataFrame(predictive.get("ml_methods", []))
    scenario_df = pd.DataFrame(advanced.get("scenario_planning", []))
    custom_simulations_data = advanced.get("custom_simulations", {})
    custom_definition_defaults = advanced.get("custom_simulation_definitions")
    if not custom_definition_defaults:
        custom_definition_defaults = copy.deepcopy(
            DEFAULT_CUSTOM_SIMULATION_DEFINITIONS
        )
    else:
        custom_definition_defaults = copy.deepcopy(custom_definition_defaults)

    metrics_lookup = metrics_df.set_index("metric")["value"] if not metrics_df.empty else pd.Series(dtype="float64")

    def _metric_value(name: str) -> float:
        try:
            return float(metrics_lookup.get(name, float("nan")))
        except Exception:
            return float("nan")

    base_metrics = advanced.get("base_metrics") or {
        "npv": valuation.get("npv"),
        "irr": valuation.get("irr"),
        "avg_dscr": _metric_value("Average DSCR"),
        "min_dscr": _metric_value("Minimum DSCR"),
        "payback": _metric_value("Payback period (years)"),
    }

    with production_tab:
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
                    try:
                        with st.spinner("Preparing Excel workbook..."):
                            excel_bytes = _generate_excel_bytes(
                                model, results, selected_scenario
                            )
                    except RuntimeError:
                        excel_bytes = None
                    else:
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
        updated_assumptions: List[Dict[str, Any]] = []
        for schedule_name, group in assumption_schedule_df.groupby("schedule", sort=False):
            schedule_key = _sanitize_key(schedule_name)
            defaults = group.drop(columns=["schedule"]).to_dict("records")
            edited_df = _render_schedule_editor(
                schedule_name,
                schedule_key,
                defaults,
                selected_scenario,
                namespace="assumption_schedule_state",
                allow_yearly_increment=True,
            )
            for record in edited_df.replace({pd.NA: None}).to_dict("records"):
                merged = {"schedule": schedule_name}
                merged.update(record)
                updated_assumptions.append(merged)
        if updated_assumptions:
            results["assumptions_schedule"] = updated_assumptions
            new_assumptions = _assumptions_from_schedule(
                updated_assumptions, model.assumptions
            )
            if new_assumptions is not None:
                payload["assumptions"] = asdict(new_assumptions)
                scenario_store[selected_scenario]["assumptions"] = asdict(
                    new_assumptions
                )
                st.session_state.snapshot_scenario = selected_scenario
                st.session_state.input_snapshot = copy.deepcopy(payload)

                updated_model = ScenarioModel(
                    scenario=selected_scenario, assumptions=new_assumptions
                )
                updated_results = generate_model_outputs(
                    new_assumptions,
                    analytics_plan=AnalyticsPlan.summary(),
                )

                scenario_payloads = st.session_state.setdefault(
                    "scenario_payloads", {}
                )
                scenario_payloads[selected_scenario] = {
                    "snapshot": copy.deepcopy(st.session_state.input_snapshot),
                    "model": updated_model,
                    "results": updated_results,
                }
                results = updated_results
                st.session_state.model_results = (updated_model, updated_results)
                st.success(
                    "Assumption schedule edits applied. Recalculating model outputs..."
                )
                _rerun()

        st.subheader("Revenue schedules")
        updated_revenue: Dict[str, List[Dict[str, Any]]] = {}
        for category, rows in revenue_schedules.items():
            schedule_key = _sanitize_key(category)
            defaults = copy.deepcopy(rows)
            row_defaults: Dict[str, Any] = {}
            if rows:
                row_defaults["Notes"] = rows[0].get("Notes")
            edited_df = _render_schedule_editor(
                category,
                schedule_key,
                defaults,
                selected_scenario,
                namespace="revenue_schedule_state",
                fixed_columns={"Category": category},
                row_defaults=row_defaults,
                allow_yearly_increment=True,
                auto_update_revenue=True,
            )
            updated_revenue[category] = (
                edited_df.replace({pd.NA: None}).to_dict("records")
            )
        if updated_revenue:
            revenue_schedules = updated_revenue
            results["revenue_schedules"] = updated_revenue
            revenue_summary = summarise_revenue_totals(
                revenue_schedules,
                model.assumptions.cycles_per_year,
                model.assumptions.production_horizon_years,
                model.assumptions.production_start_year,
            )
            results["revenue_summary"] = revenue_summary

        summary_by_category = pd.DataFrame(revenue_summary.get("by_category", []))
        if not summary_by_category.empty:
            st.markdown("##### Annual revenue by category")
            st.dataframe(
                summary_by_category,
                use_container_width=True,
                hide_index=True,
            )
            revenue_chart = _build_revenue_stack_chart(summary_by_category)
            if revenue_chart is not None:
                st.altair_chart(revenue_chart, use_container_width=True)

        annual_totals_df = pd.DataFrame(revenue_summary.get("annual_totals", []))
        if not annual_totals_df.empty:
            st.markdown("##### Total revenue by year")
            st.dataframe(
                annual_totals_df,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Production cycles")
        st.dataframe(cycles_df, use_container_width=True)

        st.subheader("Annual summary")
        st.dataframe(annual_df, use_container_width=True)

        st.subheader("Discounted cash flows")
        st.dataframe(cashflow_df, use_container_width=True)
        if cashflow_bridge_frames:
            waterfall_years = sorted(cashflow_bridge_frames)
            waterfall_year = st.selectbox(
                "Cash-flow bridge year",
                waterfall_years,
                key=f"waterfall_year_{selected_scenario}",
            )
            waterfall_chart = _cashflow_waterfall_chart(
                cashflow_bridge_frames.get(waterfall_year, pd.DataFrame())
            )
            if waterfall_chart is not None:
                st.altair_chart(waterfall_chart, use_container_width=True)

        st.subheader("Debt schedule")
        st.dataframe(loan_df, use_container_width=True, hide_index=True)

        st.subheader("Asset schedule")
        st.dataframe(asset_schedule_df, use_container_width=True, hide_index=True)

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
        with fin_tab2:
            st.dataframe(balance_df, use_container_width=True, hide_index=True)
        with fin_tab3:
            st.dataframe(cash_statement_df, use_container_width=True, hide_index=True)
        with fin_tab4:
            st.dataframe(loan_df, use_container_width=True, hide_index=True)

    analytics_namespace = "advanced_schedule_state"

    with analytics_tab:
        if not metrics_df.empty:
            st.subheader("Advanced metrics")
            metrics_df = _render_analytics_schedule(
                "Advanced metrics",
                "advanced_metrics",
                metrics_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["metrics"] = metrics_df.replace({pd.NA: None}).to_dict("records")

        st.subheader("Simulation builder")
        custom_defaults = copy.deepcopy(custom_definition_defaults)
        custom_editor_df = _render_schedule_editor(
            "Custom scenario definitions",
            "custom_simulations",
            custom_defaults,
            selected_scenario,
            namespace="custom_simulation_state",
            row_defaults={
                "Scenario": "New scenario",
                "Description": "",
                "Parameter": "",
                "Change type": "percent",
                "Change value": 0.0,
            },
            allow_yearly_increment=False,
        )
        custom_rows = (
            custom_editor_df.replace({pd.NA: None}).to_dict("records")
            if not custom_editor_df.empty
            else []
        )
        simulation_inputs = custom_rows or copy.deepcopy(custom_definition_defaults)
        stored_payload = custom_simulations_data or {}
        stored_definitions = stored_payload.get("definitions", [])
        run_custom_button = st.button(
            "Run custom simulations",
            key=f"run_custom_simulations_{selected_scenario}",
            help="Recalculate the scenarios above with the current definitions.",
        )

        simulation_payload = stored_payload
        if run_custom_button:
            with st.spinner("Running custom simulations..."):
                simulation_payload = run_custom_simulations(
                    model.assumptions,
                    base_metrics,
                    simulation_inputs,
                )
            custom_simulations_data = simulation_payload
            advanced["custom_simulations"] = simulation_payload
            advanced["custom_simulation_definitions"] = copy.deepcopy(simulation_inputs)
            results["advanced_analytics"]["custom_simulations"] = copy.deepcopy(
                simulation_payload
            )
            results["advanced_analytics"][
                "custom_simulation_definitions"
            ] = copy.deepcopy(simulation_inputs)
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "custom_simulations"
                ] = copy.deepcopy(simulation_payload)
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "custom_simulation_definitions"
                ] = copy.deepcopy(simulation_inputs)
        else:
            if not _records_match(simulation_inputs, stored_definitions):
                st.info(
                    "Definitions changed. Click **Run custom simulations** to refresh the "
                    "results below."
                )

        custom_state_store = st.session_state.setdefault("custom_simulation_state", {})
        scenario_store_custom = custom_state_store.setdefault(selected_scenario, {})
        schedule_state = scenario_store_custom.setdefault("custom_simulations", {})
        schedule_state["data"] = copy.deepcopy(simulation_inputs)
        schedule_state.setdefault("default", copy.deepcopy(custom_defaults))

        custom_results_df = pd.DataFrame(simulation_payload.get("results", []))
        if custom_rows:
            valid_count = len(simulation_payload.get("definitions", []))
            configured = sum(1 for row in custom_rows if row.get("Scenario"))
            if run_custom_button and valid_count < configured:
                st.warning(
                    "Some rows were skipped because the parameter name or change value "
                    "could not be applied. Check the entries above and try again."
                )
        invalid_rows = simulation_payload.get("invalid", [])
        if invalid_rows:
            st.info(
                "Rows with validation issues are retained above but excluded from the "
                "simulation results."
            )
            invalid_df = pd.DataFrame(invalid_rows)
            st.dataframe(
                invalid_df,
                use_container_width=True,
                hide_index=True,
            )
        if not custom_results_df.empty:
            st.dataframe(
                custom_results_df,
                use_container_width=True,
                hide_index=True,
            )
            delta_df = pd.DataFrame(simulation_payload.get("delta_summary", []))
            if not delta_df.empty:
                st.caption("NPV delta by custom scenario")
                try:
                    chart_series = delta_df.set_index("Scenario")["NPV Δ"].astype(float)
                    st.bar_chart(chart_series)
                except KeyError:
                    pass

        st.subheader("What-if analysis")
        run_what_if = st.button(
            "Run what-if analysis",
            key=f"run_what_if_{selected_scenario}",
            help="Recompute the what-if scenarios using the latest assumptions.",
        )
        if run_what_if:
            with st.spinner("Running what-if scenarios..."):
                what_if_rows = perform_what_if_analysis(
                    model.assumptions,
                    base_metrics,
                )
            what_if_df = pd.DataFrame(what_if_rows)
            advanced["what_if"] = what_if_rows
            results["advanced_analytics"]["what_if"] = copy.deepcopy(what_if_rows)
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "what_if"
                ] = copy.deepcopy(what_if_rows)
        if what_if_df.empty:
            st.info(
                "No what-if scenarios available yet. Click **Run what-if analysis** to "
                "generate the table."
            )
        else:
            what_if_df = _render_analytics_schedule(
                "What-if scenarios",
                "what_if",
                what_if_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["what_if"] = what_if_df.replace({pd.NA: None}).to_dict("records")

        st.subheader("Scenario planning")
        run_scenario_planning = st.button(
            "Run scenario planning",
            key=f"run_scenario_planning_{selected_scenario}",
            help="Recalculate Downside, Upside, and Expansion scenarios.",
        )
        if run_scenario_planning:
            with st.spinner("Evaluating planning scenarios..."):
                scenario_rows = scenario_planning(
                    model.assumptions,
                    base_metrics,
                    getattr(annual_summary_obj, "revenue", 0.0),
                )
            scenario_df = pd.DataFrame(scenario_rows)
            advanced["scenario_planning"] = scenario_rows
            results["advanced_analytics"]["scenario_planning"] = copy.deepcopy(
                scenario_rows
            )
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "scenario_planning"
                ] = copy.deepcopy(scenario_rows)
        if scenario_df.empty:
            st.info(
                "No scenario planning results yet. Click **Run scenario planning** to "
                "evaluate alternative cases."
            )
        else:
            scenario_df = _render_analytics_schedule(
                "Scenario planning",
                "scenario_planning",
                scenario_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["scenario_planning"] = (
                scenario_df.replace({pd.NA: None}).to_dict("records")
            )

        tornado_chart = _tornado_chart(what_if_df, scenario_df)
        if tornado_chart is not None:
            st.markdown("#### Sensitivity tornado (NPV deltas)")
            st.altair_chart(tornado_chart, use_container_width=True)

        monte_carlo_settings = (
            monte_carlo.get("settings", {}) if isinstance(monte_carlo, dict) else {}
        )
        default_iterations = int(
            monte_carlo_settings.get("iterations")
            or (
                int(monte_carlo_summary_df.iloc[0].get("iterations"))
                if not monte_carlo_summary_df.empty
                and pd.notna(monte_carlo_summary_df.iloc[0].get("iterations"))
                else len(monte_carlo_samples_df) or 200
            )
        )
        default_iterations = min(max(default_iterations, 1), 10000)

        distribution_defaults = monte_carlo_settings.get("distributions")
        if not distribution_defaults:
            distribution_defaults = copy.deepcopy(DEFAULT_MONTE_CARLO_DISTRIBUTIONS)
        else:
            distribution_defaults = copy.deepcopy(distribution_defaults)
        distributions_df = pd.DataFrame(distribution_defaults)

        st.subheader("Monte Carlo simulation")
        st.caption("Adjust parameter distributions before running simulations.")
        distributions_df = _render_analytics_schedule(
            "Monte Carlo distributions",
            "monte_carlo_distributions",
            distributions_df,
            selected_scenario,
            namespace=analytics_namespace,
            iterable_columns=["bounds"],
        )
        distribution_records = (
            distributions_df.replace({pd.NA: None}).to_dict("records")
            if not distributions_df.empty
            else []
        )
        if not distribution_records:
            distribution_records = copy.deepcopy(DEFAULT_MONTE_CARLO_DISTRIBUTIONS)
        monte_carlo_settings["distributions"] = copy.deepcopy(distribution_records)

        iterations_key = f"monte_carlo_iterations_{selected_scenario}"
        if iterations_key not in st.session_state:
            st.session_state[iterations_key] = default_iterations

        iterations_value = st.number_input(
            "Iterations",
            min_value=1,
            max_value=10000,
            value=st.session_state[iterations_key],
            step=50,
            key=iterations_key,
            help="Choose how many Monte Carlo iterations to run (1-10,000).",
        )

        run_button = st.button(
            "Run Monte Carlo",
            key=f"run_monte_carlo_{selected_scenario}",
        )

        if run_button:
            iterations = int(iterations_value)
            with st.spinner("Running Monte Carlo simulation..."):
                updated_mc = run_monte_carlo_analysis(
                    model.assumptions,
                    iterations=iterations,
                    distributions=distribution_records,
                )
            monte_carlo = updated_mc
            advanced["monte_carlo"] = updated_mc
            results["advanced_analytics"]["monte_carlo"] = copy.deepcopy(updated_mc)
            advanced["monte_carlo"].setdefault("settings", {})["distributions"] = (
                copy.deepcopy(distribution_records)
            )
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "monte_carlo"
                ] = copy.deepcopy(updated_mc)
            monte_carlo_summary_df = pd.DataFrame([updated_mc.get("summary", {})])
            monte_carlo_samples_df = pd.DataFrame(updated_mc.get("samples", []))
            monte_carlo_settings = updated_mc.get("settings", {})

        if not monte_carlo_summary_df.empty:
            st.markdown("#### Summary")
            st.dataframe(
                monte_carlo_summary_df,
                use_container_width=True,
                hide_index=True,
            )

        if not monte_carlo_samples_df.empty:
            st.caption("Monte Carlo outcome distributions")
            for label, chart in _monte_carlo_distribution_charts(
                monte_carlo_samples_df
            ):
                st.markdown(f"**{label} distribution**")
                st.altair_chart(chart, use_container_width=True)

        st.subheader("Break-even analysis by product")
        run_break_even = st.button(
            "Run break-even analysis",
            key=f"run_break_even_{selected_scenario}",
            help="Rebuild the break-even schedule using current revenue inputs.",
        )
        if run_break_even:
            with st.spinner("Calculating break-even metrics..."):
                break_even_rows = break_even_analysis(
                    annual_summary_obj,
                    revenue_summary,
                    revenue_schedules,
                )
            break_even_df = pd.DataFrame(break_even_rows)
            advanced["break_even"] = break_even_rows
            results["advanced_analytics"]["break_even"] = copy.deepcopy(break_even_rows)
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "break_even"
                ] = copy.deepcopy(break_even_rows)
        if break_even_df.empty:
            st.info(
                "No break-even schedule is available. Click **Run break-even analysis** to "
                "generate the table."
            )
        else:
            break_even_df = _render_analytics_schedule(
                "Break-even analysis",
                "break_even",
                break_even_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["break_even"] = (
                break_even_df.replace({pd.NA: None}).to_dict("records")
            )
            heatmap_chart, units_chart = _break_even_heatmap_charts(break_even_df)
            if heatmap_chart is not None:
                st.altair_chart(heatmap_chart, use_container_width=True)
            if units_chart is not None:
                st.altair_chart(units_chart, use_container_width=True)

        st.subheader("Goal seek (target NPV)")
        run_goal_seek = st.button(
            "Run goal seek",
            key=f"run_goal_seek_{selected_scenario}",
            help="Solve for the live price per kg that delivers zero NPV.",
        )
        if run_goal_seek:
            with st.spinner("Running goal seek..."):
                goal_seek = goal_seek_live_price(model.assumptions)
            advanced["goal_seek"] = goal_seek
            results["advanced_analytics"]["goal_seek"] = copy.deepcopy(goal_seek)
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "goal_seek"
                ] = copy.deepcopy(goal_seek)
        goal_df = pd.DataFrame([goal_seek]) if goal_seek else pd.DataFrame()
        if goal_df.empty:
            st.info(
                "No goal seek result yet. Click **Run goal seek** to calculate the target "
                "price."
            )
        else:
            goal_df = _render_analytics_schedule(
                "Goal seek",
                "goal_seek",
                goal_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            goal_records = goal_df.replace({pd.NA: None}).to_dict("records")
            advanced["goal_seek"] = goal_records[0] if goal_records else {}

        if not dscr_df.empty:
            st.subheader("Debt service coverage ratio")
            dscr_df = _render_analytics_schedule(
                "DSCR summary",
                "dscr",
                dscr_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["dscr"] = dscr_df.replace({pd.NA: None}).to_dict("records")

        if not returns_df.empty:
            st.subheader("Return diagnostics")
            returns_df = _render_analytics_schedule(
                "Return diagnostics",
                "return_diagnostics",
                returns_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["returns"] = returns_df.replace({pd.NA: None}).to_dict("records")
            try:
                returns_chart = returns_df.set_index("year")
                returns_chart = returns_chart.apply(pd.to_numeric, errors="coerce")
                subset_cols = [
                    col
                    for col in [
                        "return_on_assets",
                        "return_on_equity",
                        "return_on_invested_capital",
                    ]
                    if col in returns_chart.columns
                ]
                if subset_cols:
                    st.line_chart(returns_chart[subset_cols].dropna(how="all"))
            except KeyError:
                pass

        if not coverage_df.empty:
            st.subheader("Coverage & resilience")
            coverage_df = _render_analytics_schedule(
                "Coverage & resilience",
                "coverage",
                coverage_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["coverage"] = (
                coverage_df.replace({pd.NA: None}).to_dict("records")
            )

        if not leverage_df.empty:
            st.subheader("Leverage profile")
            leverage_df = _render_analytics_schedule(
                "Leverage profile",
                "leverage",
                leverage_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["leverage"] = (
                leverage_df.replace({pd.NA: None}).to_dict("records")
            )

        combined_leverage_chart = _combined_leverage_chart(
            dscr_df, coverage_df, leverage_df
        )
        if combined_leverage_chart is not None:
            st.markdown("#### Coverage vs leverage overview")
            st.altair_chart(combined_leverage_chart, use_container_width=True)

        if not trend_df.empty:
            st.subheader("Performance trends")
            trend_df = _render_analytics_schedule(
                "Performance trends",
                "performance_trends",
                trend_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            advanced["trend"] = trend_df.replace({pd.NA: None}).to_dict("records")
            try:
                trend_chart = trend_df.set_index("year")
                trend_chart = trend_chart.apply(pd.to_numeric, errors="coerce")
                subset_cols = [
                    col
                    for col in [
                        "revenue",
                        "ebitda",
                        "net_income",
                        "free_cash_flow",
                    ]
                    if col in trend_chart.columns
                ]
                if subset_cols:
                    st.line_chart(trend_chart[subset_cols].dropna(how="all"))
            except KeyError:
                pass

        st.subheader("Predictive analytics")
        run_predictive = st.button(
            "Run predictive analytics",
            key=f"run_predictive_{selected_scenario}",
            help="Refresh automated forecasts, time-series projections, and risk checks.",
        )
        if run_predictive:
            with st.spinner("Recomputing predictive analytics..."):
                predictive = build_predictive_analytics(
                    results["cashflows"],
                    financials["income_statement"],
                )
            advanced["predictive"] = predictive
            results["advanced_analytics"]["predictive"] = copy.deepcopy(predictive)
            payload_cache = st.session_state.get("scenario_payloads", {})
            if selected_scenario in payload_cache:
                payload_cache[selected_scenario]["results"]["advanced_analytics"][
                    "predictive"
                ] = copy.deepcopy(predictive)
        forecast_df = pd.DataFrame(predictive.get("automated_forecast", []))
        time_series_df = pd.DataFrame(
            predictive.get("time_series", {}).get("forecast", [])
        )
        risk_metadata = predictive.get("risk_anomalies", {})
        risk_df = pd.DataFrame(risk_metadata.get("observations", []))
        ml_methods_df = pd.DataFrame(predictive.get("ml_methods", []))

        if (
            forecast_df.empty
            and time_series_df.empty
            and risk_df.empty
            and ml_methods_df.empty
        ):
            st.info(
                "No predictive analytics available. Click **Run predictive analytics** to "
                "generate forecasts and anomaly checks."
            )

        if not forecast_df.empty:
            st.markdown("##### Automated forecasting")
            forecast_df = _render_analytics_schedule(
                "Automated forecast",
                "automated_forecast",
                forecast_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            predictive.setdefault("automated_forecast", [])
            predictive["automated_forecast"] = (
                forecast_df.replace({pd.NA: None}).to_dict("records")
            )
            try:
                forecast_chart = forecast_df.set_index("Year")
                forecast_chart = forecast_chart.apply(pd.to_numeric, errors="coerce")
                subset_cols = [
                    col
                    for col in ["Revenue forecast", "EBITDA forecast"]
                    if col in forecast_chart.columns
                ]
                if subset_cols:
                    st.line_chart(forecast_chart[subset_cols].dropna(how="all"))
            except KeyError:
                pass

        if not time_series_df.empty:
            st.markdown("##### Time series (AR(1)) outlook")
            time_series_df = _render_analytics_schedule(
                "Time series outlook",
                "time_series",
                time_series_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            predictive.setdefault("time_series", {}).setdefault("forecast", [])
            predictive["time_series"]["forecast"] = (
                time_series_df.replace({pd.NA: None}).to_dict("records")
            )
            try:
                time_series_chart = time_series_df.set_index("Year")
                time_series_chart = time_series_chart.apply(pd.to_numeric, errors="coerce")
                st.line_chart(time_series_chart.dropna(how="all"))
            except KeyError:
                pass

        if not risk_df.empty:
            st.markdown("##### Risk & anomaly detection")
            risk_df = _render_analytics_schedule(
                "Risk & anomalies",
                "risk_anomalies",
                risk_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            predictive.setdefault("risk_anomalies", {})
            predictive["risk_anomalies"]["observations"] = (
                risk_df.replace({pd.NA: None}).to_dict("records")
            )
            st.caption(
                f"Mean growth: {risk_metadata.get('mean_growth', float('nan')):.2%} — "
                f"Std dev: {risk_metadata.get('std_growth', float('nan')):.2%}"
            )

        anomaly_points = _build_anomaly_points(risk_df, trend_df)
        overlay_chart = _forecast_overlay_chart(
            trend_df, forecast_df, time_series_df, anomaly_points
        )
        if overlay_chart is not None:
            st.markdown("#### Revenue outlook (historical vs forecasts)")
            st.altair_chart(overlay_chart, use_container_width=True)

        if not ml_methods_df.empty:
            st.markdown("##### ML method diagnostics")
            ml_methods_df = _render_analytics_schedule(
                "ML methods",
                "ml_methods",
                ml_methods_df,
                selected_scenario,
                namespace=analytics_namespace,
            )
            predictive["ml_methods"] = (
                ml_methods_df.replace({pd.NA: None}).to_dict("records")
            )

        advanced["monte_carlo"] = monte_carlo
        advanced["predictive"] = predictive

    st.session_state.input_snapshot = copy.deepcopy(payload)
    scenario_store[selected_scenario] = payload
    scenario_payloads = st.session_state.get("scenario_payloads")
    if scenario_payloads and selected_scenario in scenario_payloads:
        scenario_payloads[selected_scenario]["snapshot"] = copy.deepcopy(payload)
        scenario_payloads[selected_scenario]["results"] = results

    st.markdown("---")
    st.caption("Use the Input Landing Page above to adjust the operating model and financing structure.")


if __name__ == "__main__":
    main()

