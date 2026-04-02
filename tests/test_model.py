import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from broiler_model.assumptions import Assumptions
from broiler_model.model import generate_model_outputs
from broiler_model.production import (
    build_revenue_schedules,
    compute_cycles,
    annual_summary,
    summarise_revenue_totals,
)
from broiler_model.financing import discounted_cash_flow, build_financial_statements
from broiler_model.analytics import AnalyticsPlan, compute_advanced_analytics


class GenerateModelOutputsTests(unittest.TestCase):
    def test_generate_outputs_contains_expected_sections(self) -> None:
        assumptions = Assumptions()
        results = generate_model_outputs(assumptions)

        self.assertIs(results["assumptions"], assumptions)
        self.assertIn("valuation", results)
        valuation = results["valuation"]
        self.assertIn("npv", valuation)
        self.assertTrue(math.isfinite(valuation["npv"]))
        self.assertGreater(len(results["cashflows"]), 0)
        self.assertIn("advanced_analytics", results)


class AdvancedAnalyticsTests(unittest.TestCase):
    def test_compute_advanced_analytics_returns_metrics(self) -> None:
        assumptions = Assumptions()
        cycles = compute_cycles(assumptions)
        annual = annual_summary(assumptions, cycles)
        cashflows, loan_schedule = discounted_cash_flow(assumptions, annual)
        revenue_schedules = build_revenue_schedules(assumptions, cycles)
        revenue_summary = summarise_revenue_totals(
            revenue_schedules,
            assumptions.cycles_per_year,
            assumptions.production_horizon_years,
            assumptions.production_start_year,
        )
        financials = build_financial_statements(assumptions, cashflows, loan_schedule)

        analytics = compute_advanced_analytics(
            assumptions,
            cashflows,
            financials["income_statement"],
            financials["balance_sheet"],
            revenue_summary,
            revenue_schedules,
            annual,
        )

        metrics = analytics["metrics"]
        metric_names = {entry["metric"] for entry in metrics}
        self.assertIn("Base case NPV", metric_names)
        self.assertIn("Average DSCR", metric_names)

        monte_carlo = analytics["monte_carlo"]
        self.assertIn("summary", monte_carlo)
        self.assertIn("settings", monte_carlo)
        self.assertEqual(
            monte_carlo["summary"].get("iterations"),
            monte_carlo["settings"].get("iterations"),
        )
        self.assertIn("distributions", monte_carlo["settings"])

        custom_definitions = analytics.get("custom_simulation_definitions", [])
        self.assertTrue(custom_definitions)

        custom_payload = analytics.get("custom_simulations", {})
        self.assertIn("delta_summary", custom_payload)

        break_even_rows = analytics.get("break_even", [])
        if break_even_rows:
            self.assertIn("Direct cost", break_even_rows[0])

    def test_summary_plan_skips_heavy_sections(self) -> None:
        assumptions = Assumptions()
        cycles = compute_cycles(assumptions)
        annual = annual_summary(assumptions, cycles)
        cashflows, loan_schedule = discounted_cash_flow(assumptions, annual)
        revenue_schedules = build_revenue_schedules(assumptions, cycles)
        revenue_summary = summarise_revenue_totals(
            revenue_schedules,
            assumptions.cycles_per_year,
            assumptions.production_horizon_years,
            assumptions.production_start_year,
        )
        financials = build_financial_statements(assumptions, cashflows, loan_schedule)

        analytics = compute_advanced_analytics(
            assumptions,
            cashflows,
            financials["income_statement"],
            financials["balance_sheet"],
            revenue_summary,
            revenue_schedules,
            annual,
            plan=AnalyticsPlan.summary(),
        )

        self.assertEqual(analytics["monte_carlo"]["summary"].get("iterations"), 0)
        self.assertFalse(analytics["custom_simulations"].get("results"))
        self.assertFalse(analytics["scenario_planning"])


class CLIBaselineRegressionTests(unittest.TestCase):
    def test_cli_outputs_within_expected_ranges(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            cmd = [
                sys.executable,
                "deployable_financial_model.py",
                "--out",
                str(out_dir),
                "--formats",
                "csv",
                "json",
            ]
            subprocess.run(cmd, check=True, cwd=repo_root)

            valuation_path = out_dir / "valuation.json"
            self.assertTrue(valuation_path.exists())
            with valuation_path.open() as fh:
                valuation = json.load(fh)

            self.assertAlmostEqual(valuation["npv"], -813371.45, delta=1_500)
            self.assertAlmostEqual(valuation["irr"], -0.1663, delta=0.01)

            dscr_path = out_dir / "dscr_summary.csv"
            self.assertTrue(dscr_path.exists())
            with dscr_path.open() as fh:
                reader = csv.DictReader(fh)
                first_row = next(reader)

            self.assertAlmostEqual(float(first_row["dscr"]), 0.8973, delta=0.02)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
