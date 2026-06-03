"""Workbook exporter for the broiler chicken financial model."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from .assumptions import Assumptions
from .production import summarise_revenue_totals


def _records_to_frame(rows: Any) -> pd.DataFrame:
    """Return *rows* as a dataframe, handling dataclasses and mappings."""

    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if is_dataclass(rows):
        return pd.DataFrame([asdict(rows)])
    if isinstance(rows, Mapping):
        return pd.DataFrame([dict(rows)])

    records: List[Dict[str, Any]] = []
    for row in rows:
        if is_dataclass(row):
            records.append(asdict(row))
        elif isinstance(row, Mapping):
            records.append(dict(row))
        else:
            records.append({"value": row})
    return pd.DataFrame(records)


def _sheet_name(name: str) -> str:
    cleaned = (
        str(name)
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", " ")
        .replace("\\", " ")
        .replace(":", " ")
        .replace("*", " ")
        .replace("?", " ")
    )
    cleaned = " ".join(cleaned.split()).strip()
    return (cleaned[:31] or "Sheet").strip()


def _excel_ready_df(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    if safe.empty:
        return safe
    for column in safe.columns:
        if safe[column].dtype == object:
            safe[column] = safe[column].apply(
                lambda value: (
                    str(value)
                    if not isinstance(value, (dict, list, tuple, set))
                    else str(value)
                )
            )
    return safe


def _write_frame_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    title: str,
    frame: pd.DataFrame,
    formats: Dict[str, Any],
    *,
    notes: str | None = None,
    chart_spec: Dict[str, Any] | None = None,
) -> None:
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(4, 0)
    worksheet.set_zoom(90)

    worksheet.merge_range("A1:H1", title, formats["sheet_title"])
    if notes:
        worksheet.merge_range("A2:H2", notes, formats["sheet_note"])

    display = _excel_ready_df(frame)
    start_row = 3

    if display.empty:
        worksheet.write(start_row, 0, "No data available for this sheet.", formats["empty"])
        worksheet.set_column(0, 0, 28)
        return

    display.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=0)
    rows, cols = display.shape
    worksheet.add_table(
        start_row,
        0,
        start_row + rows,
        cols - 1,
        {
            "style": "Table Style Medium 2",
            "columns": [{"header": str(col)} for col in display.columns],
        },
    )

    for idx, column in enumerate(display.columns):
        width = max(len(str(column)), 14)
        if not display.empty:
            width = max(width, min(30, display[column].astype(str).map(len).quantile(0.9) if rows > 1 else display[column].astype(str).map(len).max()))
        worksheet.set_column(idx, idx, min(float(width) + 2, 36))

        series = pd.to_numeric(display[column], errors="coerce")
        if series.notna().sum() == 0:
            continue

        label = str(column).lower()
        cell_format = None
        if any(token in label for token in ("margin", "rate", "irr", "tax", "ratio")):
            cell_format = formats["percent"]
        elif any(token in label for token in ("price", "revenue", "cost", "cash", "debt", "equity", "ebit", "npv", "income", "capex", "payment", "balance", "value")):
            cell_format = formats["currency"]
        elif any(token in label for token in ("year", "period", "cycle", "survivors", "birds", "units")):
            cell_format = formats["integer"]
        if cell_format is not None:
            worksheet.set_column(idx, idx, None, cell_format)

    if chart_spec and chart_spec.get("category") in display.columns:
        category_col = display.columns.get_loc(chart_spec["category"])
        series_columns = [col for col in chart_spec.get("series", []) if col in display.columns]
        if series_columns:
            chart = workbook.add_chart({"type": chart_spec.get("type", "line")})
            for column in series_columns:
                col_idx = display.columns.get_loc(column)
                chart.add_series(
                    {
                        "name": [sheet_name, start_row, col_idx],
                        "categories": [sheet_name, start_row + 1, category_col, start_row + rows, category_col],
                        "values": [sheet_name, start_row + 1, col_idx, start_row + rows, col_idx],
                    }
                )
            chart.set_title({"name": chart_spec.get("title", title)})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 720, "height": 360})
            worksheet.insert_chart(chart_spec.get("position", "J2"), chart)


def _build_export_frames(
    assumptions: Assumptions,
    results: Dict[str, Any],
    ai_settings: Mapping[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    financials = results["financial_statements"]
    advanced = results["advanced_analytics"]
    predictive = advanced.get("predictive", {})
    monte_carlo = advanced.get("monte_carlo", {})
    revenue_summary = results.get("revenue_summary") or summarise_revenue_totals(
        results["revenue_schedules"],
        assumptions.cycles_per_year,
        assumptions.production_horizon_years,
        assumptions.production_start_year,
    )

    return {
        "Assumptions": _records_to_frame(results["assumptions_schedule"]),
        "Input Values": _records_to_frame(asdict(assumptions)),
        "Production Cycles": _records_to_frame(results["cycles"]),
        "Annual Summary": _records_to_frame(results["annual"]),
        "Cash Flows": _records_to_frame(results["cashflows"]),
        "Valuation": _records_to_frame(results["valuation"]),
        "Income Statement": _records_to_frame(financials["income_statement"]),
        "Balance Sheet": _records_to_frame(financials["balance_sheet"]),
        "Cash Flow Statement": _records_to_frame(financials["cash_flow_statement"]),
        "Debt Schedule": _records_to_frame(financials["loan_schedule"]),
        "Advanced Metrics": _records_to_frame(advanced.get("metrics", [])),
        "DSCR": _records_to_frame(advanced.get("dscr", [])),
        "Trend Analysis": _records_to_frame(advanced.get("trend", [])),
        "Returns": _records_to_frame(advanced.get("returns", [])),
        "Coverage": _records_to_frame(advanced.get("coverage", [])),
        "Leverage": _records_to_frame(advanced.get("leverage", [])),
        "What If": _records_to_frame(advanced.get("what_if", [])),
        "Scenario Planning": _records_to_frame(advanced.get("scenario_planning", [])),
        "Break Even": _records_to_frame(advanced.get("break_even", [])),
        "Goal Seek": _records_to_frame(advanced.get("goal_seek", {})),
        "Forecast": _records_to_frame(predictive.get("automated_forecast", [])),
        "Time Series": _records_to_frame(predictive.get("time_series", {}).get("forecast", [])),
        "Risk Observations": _records_to_frame(predictive.get("risk_anomalies", {}).get("observations", [])),
        "ML Methods": _records_to_frame(predictive.get("ml_methods", [])),
        "Monte Carlo Summary": _records_to_frame(monte_carlo.get("summary", {})),
        "Monte Carlo Samples": _records_to_frame(monte_carlo.get("samples", [])),
        "Revenue Annual Totals": _records_to_frame(revenue_summary.get("annual_totals", [])),
        "Revenue By Category": _records_to_frame(revenue_summary.get("by_category", [])),
        "AI Settings": _records_to_frame(ai_settings or {}),
    }


def generate_excel_workbook(
    assumptions: Assumptions,
    results: Dict[str, Any],
    scenario: str,
    *,
    ai_settings: Mapping[str, Any] | None = None,
) -> bytes:
    """Create a professionally formatted Excel workbook for the broiler model."""

    try:
        import xlsxwriter  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing Excel writer dependency") from exc

    frames = _build_export_frames(assumptions, results, ai_settings=ai_settings)
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book
        workbook.set_properties(
            {
                "title": f"Broiler Chicken Financial Model - {scenario}",
                "subject": "Broiler chicken investment model",
                "author": "NumQuants",
                "company": "NumQuants",
                "comments": "Professionally formatted model workbook generated from the broiler Streamlit app.",
            }
        )

        palette = {
            "ink": "#183028",
            "green": "#254F3A",
            "green_dark": "#173626",
            "gold": "#C38B2F",
            "sand": "#F6F0E2",
            "mist": "#EDF3EE",
            "line": "#D8C7A4",
            "white": "#FFFFFF",
        }
        formats = {
            "hero": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 20,
                    "font_color": palette["white"],
                    "bg_color": palette["green_dark"],
                    "align": "left",
                    "valign": "vcenter",
                }
            ),
            "hero_sub": workbook.add_format(
                {
                    "font_size": 10,
                    "font_color": palette["white"],
                    "bg_color": palette["green_dark"],
                    "text_wrap": True,
                    "valign": "top",
                }
            ),
            "kpi_label": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 9,
                    "font_color": palette["green_dark"],
                    "bg_color": palette["sand"],
                    "border": 1,
                    "border_color": palette["line"],
                    "align": "center",
                }
            ),
            "kpi_value_currency": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 15,
                    "font_color": palette["ink"],
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "align": "center",
                    "num_format": '$#,##0;[Red]-$#,##0',
                }
            ),
            "kpi_value_percent": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 15,
                    "font_color": palette["ink"],
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "align": "center",
                    "num_format": '0.0%',
                }
            ),
            "kpi_value_number": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 15,
                    "font_color": palette["ink"],
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "align": "center",
                    "num_format": '#,##0.0',
                }
            ),
            "section": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 11,
                    "font_color": palette["white"],
                    "bg_color": palette["green"],
                    "align": "left",
                }
            ),
            "sheet_title": workbook.add_format(
                {
                    "bold": True,
                    "font_size": 16,
                    "font_color": palette["white"],
                    "bg_color": palette["green"],
                    "valign": "vcenter",
                }
            ),
            "sheet_note": workbook.add_format(
                {
                    "font_size": 10,
                    "font_color": palette["ink"],
                    "bg_color": palette["mist"],
                    "italic": True,
                    "text_wrap": True,
                    "valign": "vcenter",
                }
            ),
            "label": workbook.add_format(
                {
                    "bold": True,
                    "font_color": palette["green_dark"],
                    "bg_color": palette["sand"],
                    "border": 1,
                    "border_color": palette["line"],
                }
            ),
            "value": workbook.add_format(
                {
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                }
            ),
            "currency": workbook.add_format(
                {
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "num_format": '$#,##0;[Red]-$#,##0',
                }
            ),
            "percent": workbook.add_format(
                {
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "num_format": '0.0%',
                }
            ),
            "integer": workbook.add_format(
                {
                    "bg_color": palette["white"],
                    "border": 1,
                    "border_color": palette["line"],
                    "num_format": '#,##0',
                }
            ),
            "empty": workbook.add_format(
                {
                    "italic": True,
                    "font_color": "#6B7280",
                }
            ),
        }

        overview = workbook.add_worksheet("Overview")
        writer.sheets["Overview"] = overview
        overview.hide_gridlines(2)
        overview.set_zoom(90)
        overview.set_default_row(22)
        overview.set_column("A:A", 4)
        overview.set_column("B:B", 24)
        overview.set_column("C:H", 18)
        overview.set_column("J:N", 16)

        overview.merge_range("B2:H3", f"Broiler Chicken Financial Model  |  {scenario}", formats["hero"])
        overview.merge_range(
            "B4:H5",
            (
                f"{assumptions.farm_name} | Production window {assumptions.production_start_year}"
                f" to {assumptions.production_start_year + max(assumptions.production_horizon_years - 1, 0)} | "
                "Designed for lender, investor, and operations review."
            ),
            formats["hero_sub"],
        )

        valuation = results.get("valuation", {})
        annual_summary = frames["Annual Summary"]
        dscr = frames["DSCR"]
        avg_dscr = float(dscr["dscr"].mean()) if not dscr.empty and "dscr" in dscr.columns else 0.0
        annual_revenue = float(annual_summary.iloc[0]["revenue"]) if not annual_summary.empty and "revenue" in annual_summary.columns else 0.0
        annual_ebitda = float(annual_summary.iloc[0]["ebitda"]) if not annual_summary.empty and "ebitda" in annual_summary.columns else 0.0

        kpis = [
            ("NPV", valuation.get("npv", 0.0), formats["kpi_value_currency"]),
            ("IRR", valuation.get("irr", 0.0), formats["kpi_value_percent"]),
            ("Annual Revenue", annual_revenue, formats["kpi_value_currency"]),
            ("Annual EBITDA", annual_ebitda, formats["kpi_value_currency"]),
            ("Average DSCR", avg_dscr, formats["kpi_value_number"]),
        ]
        start_col = 1
        for idx, (label, value, value_format) in enumerate(kpis):
            col = start_col + idx
            overview.write(6, col, label, formats["kpi_label"])
            overview.write(7, col, value, value_format)

        overview.merge_range("B10:E10", "Project profile", formats["section"])
        profile_rows = [
            ("Farm", assumptions.farm_name),
            ("Scenario", scenario),
            ("Cycles per year", assumptions.cycles_per_year),
            ("Birds per cycle", assumptions.birds_per_cycle),
            ("Debt ratio", assumptions.debt_ratio),
            ("Discount rate", assumptions.discount_rate),
        ]
        row = 10
        for label, value in profile_rows:
            overview.write(row, 1, label, formats["label"])
            target_format = formats["value"]
            if isinstance(value, float) and label in {"Debt ratio", "Discount rate"}:
                target_format = formats["percent"]
            elif isinstance(value, int):
                target_format = formats["integer"]
            overview.write(row, 2, value, target_format)
            row += 1

        overview.merge_range("G10:J10", "Model guidance", formats["section"])
        guidance = [
            "Use Assumptions to audit the operating case.",
            "Review Revenue Annual Totals and Production Cycles first.",
            "Use DSCR, Cash Flows, and Debt Schedule for credit review.",
            "Advanced analytics sheets support downside and scenario analysis.",
        ]
        guide_row = 10
        for line in guidance:
            overview.merge_range(guide_row, 6, guide_row, 9, line, formats["value"])
            guide_row += 1

        revenue_totals = _excel_ready_df(frames["Revenue Annual Totals"])
        revenue_by_category = _excel_ready_df(frames["Revenue By Category"])
        if not revenue_totals.empty:
            revenue_totals.to_excel(writer, sheet_name="Overview", startrow=17, startcol=1, index=False)
            overview.add_table(
                17,
                1,
                17 + len(revenue_totals),
                revenue_totals.shape[1],
                {
                    "style": "Table Style Medium 6",
                    "columns": [{"header": str(col)} for col in revenue_totals.columns],
                },
            )
            chart = workbook.add_chart({"type": "line"})
            revenue_year_col = revenue_totals.columns.get_loc("Year") + 1
            revenue_value_col = revenue_totals.columns.get_loc("Revenue") + 1
            chart.add_series(
                {
                    "name": "Total revenue",
                    "categories": ["Overview", 18, revenue_year_col, 17 + len(revenue_totals), revenue_year_col],
                    "values": ["Overview", 18, revenue_value_col, 17 + len(revenue_totals), revenue_value_col],
                    "line": {"color": palette["green"]},
                }
            )
            chart.set_title({"name": "Annual revenue trend"})
            chart.set_legend({"none": True})
            chart.set_size({"width": 520, "height": 300})
            overview.insert_chart("J18", chart)

        if not revenue_by_category.empty:
            startrow = 31
            revenue_by_category.to_excel(writer, sheet_name="Overview", startrow=startrow, startcol=1, index=False)
            overview.add_table(
                startrow,
                1,
                startrow + len(revenue_by_category),
                revenue_by_category.shape[1],
                {
                    "style": "Table Style Medium 4",
                    "columns": [{"header": str(col)} for col in revenue_by_category.columns],
                },
            )
            if {"Category", "Year", "Revenue"}.issubset(revenue_by_category.columns):
                pivot = revenue_by_category.pivot_table(index="Year", columns="Category", values="Revenue", aggfunc="sum").reset_index()
                pivot_sheet = workbook.add_worksheet("Revenue Dashboard")
                writer.sheets["Revenue Dashboard"] = pivot_sheet
                pivot_sheet.hide_gridlines(2)
                pivot.to_excel(writer, sheet_name="Revenue Dashboard", startrow=3, startcol=1, index=False)
                pivot_sheet.merge_range("B2:H2", "Revenue Mix Dashboard", formats["sheet_title"])
                pivot_sheet.add_table(
                    3,
                    1,
                    3 + len(pivot),
                    pivot.shape[1],
                    {
                        "style": "Table Style Medium 3",
                        "columns": [{"header": str(col)} for col in pivot.columns],
                    },
                )
                area = workbook.add_chart({"type": "area", "subtype": "stacked"})
                for col_idx in range(1, pivot.shape[1]):
                    area.add_series(
                        {
                            "name": ["Revenue Dashboard", 3, col_idx + 1],
                            "categories": ["Revenue Dashboard", 4, 1, 3 + len(pivot), 1],
                            "values": ["Revenue Dashboard", 4, col_idx + 1, 3 + len(pivot), col_idx + 1],
                        }
                    )
                area.set_title({"name": "Revenue by category"})
                area.set_size({"width": 760, "height": 360})
                area.set_legend({"position": "bottom"})
                pivot_sheet.insert_chart("J4", area)

        chart_configs = {
            "Production Cycles": {
                "title": "Production cycle economics",
                "notes": "Cycle-level output, operating costs, and EBITDA.",
                "chart_spec": {"category": "cycle", "series": ["revenue", "total_cost", "ebitda"], "type": "column", "title": "Cycle revenue vs EBITDA", "position": "J2"},
            },
            "Cash Flows": {
                "title": "Project cash flows",
                "notes": "Multi-year free cash flow and cumulative liquidity bridge.",
                "chart_spec": {"category": "calendar_year", "series": ["free_cash_flow", "cumulative_cash"], "type": "line", "title": "Free cash flow and cumulative cash", "position": "J2"},
            },
            "Income Statement": {
                "title": "Income statement",
                "notes": "Revenue, EBITDA, EBIT, and net income by year.",
                "chart_spec": {"category": "calendar_year", "series": ["revenue", "ebitda", "net_income"], "type": "line", "title": "Profitability trend", "position": "J2"},
            },
            "Debt Schedule": {
                "title": "Debt schedule",
                "notes": "Scheduled payment, principal reduction, and closing balance.",
                "chart_spec": {"category": "calendar_year", "series": ["payment", "principal", "balance"], "type": "line", "title": "Debt service profile", "position": "J2"},
            },
            "DSCR": {
                "title": "Debt service coverage",
                "notes": "Coverage ratios for lender review.",
                "chart_spec": {"category": "calendar_year", "series": ["dscr"], "type": "line", "title": "DSCR trend", "position": "J2"},
            },
            "Break Even": {
                "title": "Break-even analysis",
                "notes": "Unit economics and break-even pricing outputs.",
                "chart_spec": {"category": "product", "series": ["break_even_price_per_kg"], "type": "column", "title": "Break-even price by product", "position": "J2"},
            },
        }

        for raw_name, frame in frames.items():
            if raw_name in {"Revenue By Category", "Revenue Annual Totals"}:
                title = raw_name
                notes = "Revenue summaries prepared from the current scenario schedules."
                chart_spec = None
            else:
                cfg = chart_configs.get(raw_name, {})
                title = cfg.get("title", raw_name)
                notes = cfg.get("notes", "Generated from the current model scenario.")
                chart_spec = cfg.get("chart_spec")
            _write_frame_sheet(
                writer,
                _sheet_name(raw_name),
                title,
                frame,
                formats,
                notes=notes,
                chart_spec=chart_spec,
            )

        for category, rows in results["revenue_schedules"].items():
            category_title = category.replace(",", "")
            _write_frame_sheet(
                writer,
                _sheet_name(category_title),
                category_title,
                _records_to_frame(rows),
                formats,
                notes="Editable revenue schedule generated from the current operating assumptions.",
                chart_spec={
                    "category": "Period",
                    "series": ["Revenue"],
                    "type": "line",
                    "title": f"{category_title} revenue trend",
                    "position": "J2",
                },
            )

    buffer.seek(0)
    return buffer.getvalue()
