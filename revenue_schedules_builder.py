#!/usr/bin/env python3
"""Revenue schedules builder for poultry financial models.

This module analyses an Excel-based poultry financial model and builds
category-specific revenue schedules for broiler meat, eggs, manure, live birds
sales, and by-product revenue streams.  It reproduces the behaviour described in
the provided utility script so that this repository bundles both the
assumption-driven Python model and an optional helper for Excel workbooks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PATTERNS: Dict[str, List[str]] = {
    "Broiler Revenue": [r"broiler", r"meat", r"harvest", r"sale.*broiler"],
    "Eggs Revenue": [r"egg", r"layer", r"table\\s*eggs"],
    "Poultry Manure Revenue": [r"manure", r"litter", r"fertilizer"],
    "Live Birds Revenue": [r"live\\s*birds?", r"culled", r"spent\\s*h(en|ens)", r"cull"],
    "By-Product Revenue (feathers, offal, livers)": [
        r"by[-\\s]*product",
        r"offal",
        r"feather",
        r"liver",
        r"giblet",
        r"(heads|feet)",
    ],
}

REVENUE_KEYWORDS = [r"revenue", r"sales?", r"income"]
PERIOD_KEYWORDS = [r"month", r"period", r"year", r"date", r"quarter", r"week"]
QTY_KEYWORDS = [r"qty", r"quantity", r"units", r"birds", r"eggs", r"tons?", r"kg", r"bags?"]
PRICE_KEYWORDS = [r"price", r"rate", r"unit\\s*price", r"per\\s*unit"]


def _find_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    cols = []
    for col in df.columns:
        col_l = str(col).lower()
        if any(re.search(p, col_l) for p in patterns):
            cols.append(col)
    return cols


def _choose_numeric_series(df: pd.DataFrame, cols: List[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.2 * len(s))):
            return s, c
    return None, None


def _build_schedule_from_df(df: pd.DataFrame, category: str) -> Optional[pd.DataFrame]:
    period_cols = _find_columns(df, PERIOD_KEYWORDS)
    rev_cols = _find_columns(df, REVENUE_KEYWORDS + PATTERNS[category])
    cat_cols = [
        c for c in rev_cols if any(re.search(p, str(c).lower()) for p in PATTERNS[category])
    ]
    if not cat_cols:
        cat_cols = [
            c for c in rev_cols if any(re.search(p, str(c).lower()) for p in REVENUE_KEYWORDS)
        ]

    revenue_series, revenue_col = _choose_numeric_series(df, cat_cols)

    schedule = pd.DataFrame()
    if period_cols:
        schedule["Period"] = df[period_cols[0]]
    else:
        schedule["Period"] = np.arange(1, len(df) + 1)

    qty_cols = _find_columns(df, QTY_KEYWORDS)
    price_cols = _find_columns(df, PRICE_KEYWORDS)

    if qty_cols:
        schedule["Units"] = pd.to_numeric(df[qty_cols[0]], errors="coerce")
    else:
        schedule["Units"] = np.nan

    if price_cols:
        schedule["Unit Price"] = pd.to_numeric(df[price_cols[0]], errors="coerce")
    else:
        schedule["Unit Price"] = np.nan

    if revenue_series is not None:
        schedule["Revenue"] = revenue_series
        schedule["Source Column"] = revenue_col
    else:
        schedule["Revenue"] = schedule["Units"] * schedule["Unit Price"]
        schedule["Source Column"] = "Units x Unit Price"

    mask_all_empty = schedule[["Revenue", "Units", "Unit Price"]].isna().all(axis=1)
    schedule = schedule[~mask_all_empty]

    if schedule.empty:
        return None

    schedule.insert(0, "Category", category)
    return schedule


def _template_schedule(category: str, start_year: int, periods: int, freq: str = "MS") -> pd.DataFrame:
    periods_index = pd.date_range(start=f"{start_year}-01-01", periods=periods, freq=freq)
    labels = periods_index.strftime("%b-%Y")
    df = pd.DataFrame(
        {
            "Category": category,
            "Period": labels,
            "Units": np.nan,
            "Unit Price": np.nan,
            "Revenue": np.nan,
            "Source Column": "Template (fill Units & Unit Price)",
            "__Source Sheet": "N/A",
        }
    )
    return df


def build_revenue_schedules(
    input_path: Path,
    output_path: Path,
    start_year: int = 2025,
    periods: int = 12,
) -> Path:
    xls = pd.ExcelFile(input_path)
    sheet_names = xls.sheet_names

    sheets: Dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=name, header=0)
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        except Exception:
            continue

    category_schedules: Dict[str, pd.DataFrame] = {}

    for category in PATTERNS:
        matches = []
        for name, df in sheets.items():
            try:
                name_l = name.lower()
                if any(re.search(p, name_l) for p in PATTERNS[category] + REVENUE_KEYWORDS) or any(
                    any(re.search(p, str(c).lower()) for p in PATTERNS[category] + REVENUE_KEYWORDS)
                    for c in df.columns
                ):
                    sch = _build_schedule_from_df(df, category)
                    if sch is not None and (
                        sch["Revenue"].notna().any() or sch["Units"].notna().any()
                    ):
                        sch["__Source Sheet"] = name
                        matches.append(sch)
            except Exception:
                continue

        if matches:
            cat_df = pd.concat(matches, ignore_index=True)
        else:
            cat_df = _template_schedule(category, start_year=start_year, periods=periods)

        category_schedules[category] = cat_df

    all_rev = pd.concat(category_schedules.values(), ignore_index=True)

    totals_by_period = all_rev.groupby("Period", dropna=False, as_index=False)["Revenue"].sum(min_count=1)
    totals_by_category = all_rev.groupby("Category", as_index=False)["Revenue"].sum(min_count=1)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        readme = pd.DataFrame(
            {
                "How to use": [
                    "Each revenue schedule pulls (best-effort) from your template’s sheets based on column names.",
                    "If a schedule is empty or marked as Template, enter Units and Unit Price; Revenue will compute.",
                    "You can paste these sheets back into your model or link cells to them.",
                    "Columns: Period (month/period label), Units (quantity), Unit Price, Revenue (Units x Unit Price if not sourced), Source Column (origin), __Source Sheet (origin sheet).",
                ]
            }
        )
        readme.to_excel(writer, sheet_name="README", index=False)

        for cat, df in category_schedules.items():
            safe_name = re.sub(r"[^A-Za-z0-9 ]+", "", cat)[:28]
            df.to_excel(writer, sheet_name=safe_name, index=False)

        all_rev.to_excel(writer, sheet_name="All Revenues (Detail)", index=False)
        totals_by_period.to_excel(writer, sheet_name="Totals by Period", index=False)
        totals_by_category.to_excel(writer, sheet_name="Totals by Category", index=False)

    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build poultry revenue schedules from an Excel financial model."
    )
    p.add_argument("--input", required=True, type=Path, help="Path to source Excel model (.xlsx)")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the revenue schedules workbook (.xlsx)",
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=2025,
        help="Start year for template schedules (default: 2025)",
    )
    p.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Number of periods (months) for template schedules (default: 12)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = build_revenue_schedules(
        input_path=args.input,
        output_path=args.output,
        start_year=args.start_year,
        periods=args.periods,
    )
    print(f"Revenue schedules written to: {out}")


if __name__ == "__main__":
    main()
