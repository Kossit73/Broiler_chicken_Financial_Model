"""Streamlit interface for the broiler chicken financial model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from io import BytesIO
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from deployable_financial_model import (
    Assumptions,
    generate_model_outputs,
    summarise_revenue_totals,
)


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
        import xlsxwriter  # type: ignore  # noqa: F401

        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"

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


def _row_label(df: pd.DataFrame, idx: int) -> str:
    if "Period" in df.columns:
        value = df.at[idx, "Period"]
        if pd.notna(value) and str(value).strip():
            return str(value)
    return f"Row {idx + 1}"


def _coerce_row_value(raw_value: Optional[str], series: pd.Series) -> Any:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text == "":
        return None

    # Detect boolean-like inputs first
    lowered = text.lower()
    if lowered in {"true", "false", "yes", "no", "y", "n", "1", "0"}:
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False

    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() > 0:
        parsed = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(parsed):
            if numeric_series.dropna().apply(lambda v: float(v).is_integer()).all():
                return int(round(float(parsed)))
            return float(parsed)

    return raw_value


def _render_row_editors(
    df: pd.DataFrame,
    *,
    original_columns: List[str],
    fixed_columns: Dict[str, Any],
    key_prefix: str,
) -> Tuple[pd.DataFrame, bool]:
    """Render per-row edit forms and return the updated DataFrame."""

    if df.empty:
        return df, False

    updated_df = df.copy()
    changed = False

    st.markdown("##### Edit rows")
    for idx in updated_df.index:
        label = _row_label(updated_df, idx)
        with st.expander(f"Edit {label}"):
            form_key = f"form_{key_prefix}_{idx}"
            with st.form(form_key):
                pending_updates: Dict[str, Tuple[str, str]] = {}
                for column in original_columns:
                    current_value = updated_df.at[idx, column] if column in updated_df.columns else None
                    display_value = "" if pd.isna(current_value) else str(current_value)
                    widget_key = f"{form_key}_{column}"
                    if column in fixed_columns:
                        st.text_input(
                            column,
                            value=display_value,
                            disabled=True,
                            key=widget_key,
                        )
                        continue
                    new_value = st.text_input(
                        column,
                        value=display_value,
                        key=widget_key,
                    )
                    pending_updates[column] = (widget_key, new_value)

                submitted = st.form_submit_button("Save row")

            if submitted:
                for column, (widget_key, raw) in pending_updates.items():
                    target_series = (
                        updated_df[column]
                        if column in updated_df.columns
                        else pd.Series(dtype="object")
                    )
                    coerced = _coerce_row_value(raw, target_series)
                    if column in updated_df.columns:
                        if (
                            pd.api.types.is_string_dtype(updated_df[column].dtype)
                            and not (coerced is None or isinstance(coerced, str))
                        ):
                            updated_df[column] = updated_df[column].astype("object")
                        updated_df.at[idx, column] = coerced
                for column, value in fixed_columns.items():
                    if column in updated_df.columns:
                        updated_df.at[idx, column] = value
                changed = True
                st.success(f"Saved changes for {label}.")

    return updated_df, changed


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
    original_columns = list(df.columns)

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
        st.caption("Enable editing to modify this schedule. Current values remain read-only.")

    if (operation_applied and auto_update_revenue) or (auto_update_revenue and edit_enabled):
        df = _auto_compute_revenue(df)

    df = _ensure_fixed_columns(df, fixed_columns)

    column_config: Dict[str, Any] = {}
    if fixed_columns:
        for column in fixed_columns:
            column_config[column] = st.column_config.TextColumn(disabled=True)

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
    if auto_update_revenue:
        edited_df = _auto_compute_revenue(edited_df)
    edited_df = _ensure_fixed_columns(edited_df, fixed_columns)

    if edit_enabled:
        editor_key_prefix = f"{namespace}_{schedule_key}_{scenario}"
        edited_df, row_changed = _render_row_editors(
            edited_df,
            original_columns=original_columns,
            fixed_columns=fixed_columns,
            key_prefix=editor_key_prefix,
        )
        if row_changed and auto_update_revenue:
            edited_df = _auto_compute_revenue(edited_df)
            edited_df = _ensure_fixed_columns(edited_df, fixed_columns)

    state["data"] = edited_df.replace({pd.NA: None}).to_dict("records")
    return edited_df.reindex(columns=original_columns, fill_value=None)
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

    def render_section(
        title: str,
        fields: List[Dict[str, Any]],
        columns: int = 3,
    ) -> None:
        """Render a grouped section with evenly spaced inputs."""

        with st.container():
            st.markdown(f"### {title}")
            cols = st.columns(columns)
            for idx, field in enumerate(fields):
                col = cols[idx % columns]
                attr = field["attr"]
                label = field["label"]
                dtype = field.get("type", "float")
                min_value = field.get("min")
                max_value = field.get("max")
                step = field.get("step")
                fmt = field.get("format")

                default_val = getattr(defaults, attr)

                input_kwargs: Dict[str, Any] = {}
                if dtype == "int":
                    input_kwargs["value"] = int(default_val)
                    input_kwargs["step"] = int(step) if step is not None else 1
                    if min_value is not None:
                        input_kwargs["min_value"] = int(min_value)
                    if max_value is not None:
                        input_kwargs["max_value"] = int(max_value)
                    values[attr] = col.number_input(label, **input_kwargs)
                else:
                    if min_value is not None:
                        input_kwargs["min_value"] = float(min_value)
                    if max_value is not None:
                        input_kwargs["max_value"] = float(max_value)
                    if step is not None:
                        input_kwargs["step"] = float(step)
                    if fmt is not None:
                        input_kwargs["format"] = fmt
                    input_kwargs["value"] = float(default_val)
                    values[attr] = col.number_input(label, **input_kwargs)

    render_section(
        "Production",
        [
            {"attr": "cycles_per_year", "label": "Cycles per year", "min": 1, "max": 12, "type": "int"},
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
        columns=4,
    )

    render_section(
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
        columns=3,
    )

    render_section(
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
        columns=4,
    )

    render_section(
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
        columns=4,
    )

    return Assumptions(
        farm_name=farm_name,
        cycles_per_year=int(values["cycles_per_year"]),
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
            revenue_schedules, model.assumptions.cycles_per_year
        ),
    )
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
                revenue_schedules, model.assumptions.cycles_per_year
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

    with analytics_tab:
        st.subheader("Advanced metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        if not what_if_df.empty:
            st.subheader("What-if analysis")
            st.dataframe(what_if_df, use_container_width=True, hide_index=True)

        if not scenario_df.empty:
            st.subheader("Scenario planning")
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        if not monte_carlo_summary_df.empty:
            st.subheader("Monte Carlo summary")
            st.dataframe(
                monte_carlo_summary_df,
                use_container_width=True,
                hide_index=True,
            )
        if not monte_carlo_samples_df.empty:
            st.caption("NPV distribution across Monte Carlo iterations")
            try:
                st.line_chart(
                    monte_carlo_samples_df.set_index("iteration")["npv"],
                )
            except KeyError:
                pass
            st.dataframe(
                monte_carlo_samples_df,
                use_container_width=True,
                hide_index=True,
            )

        if not break_even_df.empty:
            st.subheader("Break-even analysis by product")
            st.dataframe(break_even_df, use_container_width=True, hide_index=True)

        if goal_seek:
            st.subheader("Goal seek (target NPV)")
            goal_df = pd.DataFrame([goal_seek])
            st.dataframe(goal_df, use_container_width=True, hide_index=True)

        if not dscr_df.empty:
            st.subheader("Debt service coverage ratio")
            dscr_chart = dscr_df.set_index("year")
            st.line_chart(dscr_chart)

        if not returns_df.empty:
            st.subheader("Return diagnostics")
            returns_chart = returns_df.set_index("year")
            st.line_chart(
                returns_chart[[
                    "return_on_assets",
                    "return_on_equity",
                    "return_on_invested_capital",
                ]]
            )
            st.dataframe(returns_df, use_container_width=True, hide_index=True)

        if not coverage_df.empty:
            st.subheader("Coverage & resilience")
            coverage_chart = coverage_df.set_index("year")
            st.line_chart(
                coverage_chart[[
                    "interest_coverage",
                    "fcf_to_debt_service",
                    "maintenance_capex_coverage",
                ]]
            )
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)

        if not leverage_df.empty:
            st.subheader("Leverage profile")
            leverage_chart = leverage_df.set_index("year")
            st.line_chart(leverage_chart[["debt_to_equity", "debt_ratio"]])
            st.dataframe(leverage_df, use_container_width=True, hide_index=True)

        if not trend_df.empty:
            st.subheader("Performance trends")
            trend_chart = trend_df.set_index("year")[
                ["revenue", "ebitda", "net_income", "free_cash_flow"]
            ]
            st.line_chart(trend_chart)

        if not forecast_df.empty:
            st.subheader("Automated forecasting")
            try:
                st.line_chart(
                    forecast_df.set_index("Year")[
                        ["Revenue forecast", "EBITDA forecast"]
                    ]
                )
            except KeyError:
                pass
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        if not time_series_df.empty:
            st.subheader("Time series (AR(1)) outlook")
            try:
                st.line_chart(time_series_df.set_index("Year"))
            except KeyError:
                pass
            st.dataframe(time_series_df, use_container_width=True, hide_index=True)

        if not risk_df.empty:
            st.subheader("Risk & anomaly detection")
            st.caption(
                f"Mean growth: {risk_metadata.get('mean_growth', float('nan')):.2%} — "
                f"Std dev: {risk_metadata.get('std_growth', float('nan')):.2%}"
            )
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

        if not ml_methods_df.empty:
            st.subheader("ML method diagnostics")
            st.dataframe(ml_methods_df, use_container_width=True, hide_index=True)

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

