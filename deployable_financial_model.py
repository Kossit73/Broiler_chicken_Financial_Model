#!/usr/bin/env python3
"""CLI entry point for the broiler chicken financial model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

from broiler_model import (
    Assumptions,
    apply_overrides,
    generate_model_outputs,
    load_assumptions_from_file,
    parse_overrides,
    write_csv,
    write_json,
)


def _export_csv(
    output_dir: Path,
    formats: Iterable[str],
    assumptions: Assumptions,
    results: Dict[str, object],
) -> None:
    if "csv" not in formats:
        return

    assumption_schedule = results["assumptions_schedule"]
    cycles = results["cycles"]
    annual = results["annual"]
    cashflows = results["cashflows"]
    revenue_schedules = results["revenue_schedules"]
    revenue_summary = results["revenue_summary"]
    valuation = results["valuation"]
    financials = results["financial_statements"]
    advanced = results["advanced_analytics"]

    income_statement = financials["income_statement"]
    balance_sheet = financials["balance_sheet"]
    cash_flow_statement = financials["cash_flow_statement"]
    loan_schedule = financials["loan_schedule"]

    write_csv(output_dir / "assumptions_summary.csv", assumption_schedule)
    write_csv(
        output_dir / "assumptions.csv",
        [{"name": k, "value": v} for k, v in asdict(assumptions).items()],
    )
    write_csv(output_dir / "production_cycles.csv", [asdict(cycle) for cycle in cycles])
    write_csv(output_dir / "annual_summary.csv", [asdict(annual)])
    write_csv(output_dir / "cash_flow.csv", [asdict(row) for row in cashflows])
    write_csv(output_dir / "income_statement.csv", [asdict(row) for row in income_statement])
    write_csv(output_dir / "balance_sheet.csv", [asdict(row) for row in balance_sheet])
    write_csv(
        output_dir / "cash_flow_statement.csv",
        [asdict(row) for row in cash_flow_statement],
    )
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
            write_csv(output_dir / "monte_carlo_summary.csv", [summary_row])
        write_csv(
            output_dir / "monte_carlo_samples.csv",
            monte_carlo.get("samples", []),
        )
        distributions = monte_carlo.get("settings", {}).get("distributions")
        if isinstance(distributions, list) and distributions:
            write_csv(output_dir / "monte_carlo_distributions.csv", distributions)

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
    write_csv(
        output_dir / "custom_simulation_invalid.csv",
        custom_simulations.get("invalid", []),
    )
    write_csv(
        output_dir / "custom_simulation_delta_summary.csv",
        custom_simulations.get("delta_summary", []),
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

    write_csv(output_dir / "valuation.csv", [valuation])


def _export_json(
    output_dir: Path,
    formats: Iterable[str],
    assumptions: Assumptions,
    results: Dict[str, object],
) -> None:
    if "json" not in formats:
        return

    write_json(output_dir / "assumptions.json", asdict(assumptions))
    write_json(output_dir / "assumptions_summary.json", results["assumptions_schedule"])
    write_json(
        output_dir / "production_cycles.json",
        [asdict(cycle) for cycle in results["cycles"]],
    )
    write_json(output_dir / "annual_summary.json", asdict(results["annual"]))
    write_json(
        output_dir / "cash_flow.json",
        [asdict(row) for row in results["cashflows"]],
    )
    financials = results["financial_statements"]
    write_json(
        output_dir / "income_statement.json",
        [asdict(row) for row in financials["income_statement"]],
    )
    write_json(
        output_dir / "balance_sheet.json",
        [asdict(row) for row in financials["balance_sheet"]],
    )
    write_json(
        output_dir / "cash_flow_statement.json",
        [asdict(row) for row in financials["cash_flow_statement"]],
    )
    write_json(output_dir / "loan_schedule.json", financials["loan_schedule"])
    write_json(output_dir / "advanced_analytics.json", results["advanced_analytics"])
    write_json(output_dir / "revenue_schedules.json", results["revenue_schedules"])
    write_json(output_dir / "revenue_summary.json", results["revenue_summary"])
    write_json(output_dir / "valuation.json", results["valuation"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate broiler chicken financial model outputs."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["csv", "json"],
        choices=["csv", "json"],
        help="Output formats to generate",
    )
    parser.add_argument(
        "--assumptions-file",
        type=Path,
        help="Optional JSON/YAML file containing assumption overrides",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="*",
        default=[],
        help="Override individual assumptions (repeat as needed)",
    )
    parser.add_argument(
        "--custom-simulations",
        type=Path,
        help="Optional JSON file defining custom simulation scenarios",
    )
    parser.add_argument(
        "--monte-carlo-config",
        type=Path,
        help="Optional JSON file defining Monte Carlo distributions",
    )
    args = parser.parse_args()

    if args.assumptions_file:
        assumptions = load_assumptions_from_file(args.assumptions_file)
    else:
        assumptions = Assumptions()

    if args.set:
        overrides = parse_overrides(args.set)
        assumptions = apply_overrides(assumptions, overrides)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = generate_model_outputs(
        assumptions,
        custom_simulation_path=args.custom_simulations,
        monte_carlo_config_path=args.monte_carlo_config,
    )
    _export_csv(output_dir, args.formats, assumptions, results)
    _export_json(output_dir, args.formats, assumptions, results)

    write_json(
        output_dir / "manifest.json",
        {
            "cycles_per_year": assumptions.cycles_per_year,
            "production_horizon_years": assumptions.production_horizon_years,
            "years": len(results["cashflows"]) - 1,
            "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        },
    )

    valuation = results["valuation"]
    print("✅ Financial model generated. Outputs written to", output_dir)
    print(f"NPV @ {valuation['discount_rate']:.1%}: {valuation['npv']:,.0f}")
    print(f"IRR: {valuation['irr']:.2%}")


if __name__ == "__main__":
    main()
