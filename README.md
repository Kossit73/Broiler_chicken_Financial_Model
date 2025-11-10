# Broiler Chicken Financial Model

This repository contains a fully scripted broiler chicken financial model that
runs without Excel or third-party Python dependencies.  The model codifies key
production assumptions, per-cycle economics, annual rollups, debt service, and
discounted cash flow valuation metrics across a configurable production
horizon.  Results can be exported to CSV and/or JSON for further analysis.  An
optional Streamlit dashboard lets you interactively explore the assumptions and
outputs.

## Contents

- `deployable_financial_model.py` – CLI that generates the model outputs.
- `broiler_model/` – Python package containing the assumptions, production,
  financing, analytics, and orchestration modules used by both the CLI and the
  Streamlit app.
- `streamlit_app.py` – Streamlit UI that wraps the same model logic.
- `revenue_schedules_builder.py` – optional Excel helper that extracts or
  templates detailed revenue schedules by category.
- `outputs/` *(created when you run the CLI)* – contains CSV/JSON tables,
  valuation summary, and a manifest of generated files.

## Running the model

1. Ensure you have Python 3.9+ installed.  The script uses only the standard
   library—no additional packages are required.
2. Execute the CLI and point it to an output directory:

```bash
python deployable_financial_model.py --out outputs --formats csv json
```

You can also feed custom assumptions either from a JSON/YAML file or by passing
individual overrides on the command line:

```bash
python deployable_financial_model.py \
  --out outputs \
  --assumptions-file my_assumptions.json \
  --set price_growth=0.03 cost_inflation=0.02
```

You can also point the CLI to alternative analytics configurations.  Provide a
JSON file of custom scenario definitions via `--custom-simulations` and a JSON
file of Monte Carlo distributions via `--monte-carlo-config`:

```
python deployable_financial_model.py \
  --out outputs \
  --custom-simulations configs/my_simulations.json \
  --monte-carlo-config configs/monte_carlo_overrides.json
```

By default the CLI and Streamlit app load the reference configurations stored in
`broiler_model/config/custom_simulations.json` and
`broiler_model/config/monte_carlo_distributions.json`.  Editing those files (or
providing alternates on the command line) lets you persist bespoke analytics
setups outside the codebase.

When you need faster runs (for example in CI or quick iteration loops) you can
skip the heaviest analytics modules by adding `--analytics-mode summary`.  The
summary plan keeps the core metrics, DSCR traces, trend tables, break-even
analysis, and valuation outputs intact while deferring Monte Carlo, goal seek,
predictive ML, scenario planning, and custom simulations until you explicitly
request them.

### Troubleshooting slow dashboard refreshes

The Streamlit app automatically evaluates the model whenever you change
assumptions or edit a schedule.  A full pass recomputes production cycles,
annual rollups, cash flows, financial statements, and (optionally) the advanced
analytics stack, so complex simulations can make edits feel sluggish.  If the UI
lags:

- Leave the app in the default *summary* analytics mode while iterating.  Heavy
  modules (Monte Carlo, goal seek, predictive ML, scenario planning, and custom
  simulations) stay disabled until you press their dedicated **Run** buttons.
- Reduce the Monte Carlo iteration count when exploring the stochastic results.
  Each iteration rebuilds the full model, so trimming runs from hundreds to a
  few dozen dramatically improves turnaround time.
- Batch multiple edits before triggering a refresh.  The model cache stores the
  latest scenario snapshot, so applying several tweaks at once avoids a full
  recompute after every single change.

The command creates the `outputs/` directory (if needed) and writes:

- `assumptions_summary.(csv|json)` – four-schedule table (Production,
  Operating costs, Capital structure, Financing) that captures every model
  assumption.
- `assumptions.(csv|json)` – raw key/value assumptions used by the model.
- `production_cycles.(csv|json)` – production and cost metrics for each flock
  cycle within a year.
- `annual_summary.(csv|json)` – aggregated annual revenue, cost, and EBITDA
  metrics.
- `cash_flow.(csv|json)` – cash flow projections for the configured horizon with
  operating cash flow, maintenance capex, debt service, and discounted values.
- `income_statement.(csv|json)` – modeled multi-year income statement with
  revenue, COGS, EBITDA, taxes, and net income matching the selected horizon.
- `balance_sheet.(csv|json)` – simplified balance sheet highlighting cash,
  working capital, net PP&E, debt, and equity movements across the horizon.
- `cash_flow_statement.(csv|json)` – three-section cash flow statement (operating,
  investing, financing) plus ending cash balances.
- `loan_schedule.(csv|json)` – annual amortization table for the project debt.
- Advanced analytics exports – `advanced_metrics.csv`, `dscr_summary.csv`,
  `trend_analysis.csv`, `return_metrics.csv`, `coverage_metrics.csv`,
  `leverage_metrics.csv`, `what_if_analysis.csv`, `monte_carlo_summary.csv`,
  `monte_carlo_samples.csv`, `monte_carlo_distributions.csv`,
  `break_even_analysis.csv`, `goal_seek_results.csv`, `automated_forecast.csv`,
  `time_series_forecast.csv`, `risk_anomalies.csv`, `ml_methods_summary.csv`,
  `scenario_planning.csv`, `custom_simulation_definitions.csv`,
  `custom_simulation_results.csv`, `custom_simulation_invalid.csv`, and
  `custom_simulation_delta_summary.csv`
  (plus consolidated `advanced_analytics.json`) so you can reuse the richer
  analytics outside the dashboard.
- Revenue schedule CSVs for each category (`broiler_revenue_schedule.csv`,
  `eggs_revenue_schedule.csv`, etc.) plus `revenue_schedules.json` capturing all
  five categories in one structure for downstream tooling.
- `revenue_summary_by_category.csv`, `revenue_summary_annual.csv`, and
  `revenue_summary.json` – Year-by-year revenue totals (with both period index
  and calendar year columns) ensuring annual figures tie directly to the summed
  cycle-level schedules.
- `valuation.json` – NPV/IRR results using the modeled cash flows. The payload
  now includes both the relative terminal year and the corresponding calendar
  year based on the configured production start.
- `manifest.json` – convenience listing of the generated files plus the cycles
  per year, production start year, horizon, and computed end year captured in
  the run.

All values are expressed in US dollars except where otherwise noted.  Adjust the
`Assumptions` dataclass in `broiler_model/assumptions.py` (or load a customised
JSON/YAML file via the CLI) to model different operations—number of birds,
pricing, debt structure, inflation, and more.

## Testing & regression checks

Run the automated suite to exercise both the unit-level calculators and an
end-to-end CLI regression:

```bash
python -m unittest discover tests
```

The regression test launches `deployable_financial_model.py`, parses the
generated outputs, and validates a set of baseline headline metrics. When using
the shipped assumptions you should see results within the following ranges:

- Net present value (`valuation.json`): approximately **-834k USD** (±1.5k).
- Internal rate of return (`valuation.json`): approximately **-20.5%** (±1%).
- Year 1 DSCR (`dscr_summary.csv`): approximately **0.90** (±0.02).

If scenario changes drive materially different economics, update the documented
thresholds alongside the regression test so future runs remain auditable.

## Interactive dashboard (optional)

To experiment with the model in a browser-based UI, install the required
packages and launch Streamlit:

```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```

The dashboard now presents all inputs on a single **Input Landing Page** that
groups controls into Production, Pricing, Costs, and Capital & financing
sections. The Production block captures the production start year in addition to
the horizon and surfaces the computed end year so the projection timeline stays
front-and-centre. The refreshed Pricing block now exposes dedicated inputs for
broiler, egg, manure, live-bird, and by-product prices alongside the annual
price growth assumption so each revenue stream can be tuned individually. A
scenario selector at the top lets you maintain alternative
assumption sets (Baseline, Expansion, Downside) without losing previous
configurations. As you tweak values, the NPV/IRR metrics and detailed tables
refresh instantly. Below the metrics you will find three workspaces:

1. **Production & revenues** – retains the assumptions summary (grouped into the
   four schedules), detailed revenue schedules with annual rollups, production
   cycle results, annual summary, discounted cash flows, and the derived debt and
   asset schedules. The page now layers in a stacked area chart of annual
   revenue by category plus a selectable cash-flow waterfall bridge so you can
   visualise how operating cash flow, maintenance capex, and debt service roll
   into free cash flow and discounted value each year.
2. **Financial statements** – exposes the modeled income statement, balance
   sheet, cash flow statement, and debt amortization tables.
3. **Advanced analytics** – surfaces KPIs (average margins, DSCR, payback,
   return-on-asset/equity/invested-capital averages, coverage ratios, and
   leverage velocity) alongside interactive what-if tables, scenario planning,
   Monte Carlo stress tests, product break-even and goal-seek diagnostics, plus
   automated/predictive forecasting (linear and AR(1) time-series), risk &
   anomaly detection, and ML method summaries. New visuals include a combined
   DSCR/interest-coverage/leverage line chart, tornado-style sensitivity bars,
   histogram+density plots for Monte Carlo outcomes, a layered revenue forecast
   overlay (historical vs. automated vs. AR(1) projections with anomaly flags),
   and a break-even cost heatmap with companion unit bars. Monte Carlo
   distributions remain editable from the UI and the engine draws vectorised
   samples from the configured normal/lognormal/triangular definitions,
   exporting the applied distributions alongside the summary. Break-even
   schedules continue to separate direct versus shared cost allocations to avoid
   revenue-share distortions when unit economics differ across categories. A
   **Simulation builder** validates per-scenario rows (parameter, change type,
   magnitude), highlights invalid entries, and charts NPV deltas so analysts can
   compare bespoke cases before exporting the resulting tables via the CLI.

The Production & revenues tab includes an **Excel export** card—click *Prepare
Excel Model* to generate a multi-sheet workbook and download it directly from
the browser.

An **AI & Machine Learning Settings** expander appears at the top of the Input
Landing Page so you can configure optional forecasting and narrative preferences
before editing assumptions. The settings panel stores values per scenario and
supports multiple providers, model names, and narrative focus areas.

To keep the interface responsive, the dashboard bootstraps each scenario using a
summary analytics plan (the same mode exposed via the CLI). Monte Carlo, custom
simulations, scenario planning, goal seek, and predictive ML blocks only run
after you press their respective **Run** buttons, ensuring edits propagate
quickly while still allowing deeper analysis on demand.

### Configuring editable schedules

The Streamlit app exposes every assumption and revenue table through a shared
editor so users can tweak defaults, add rows, and apply yearly increments.
When adding a new schedule or adjusting the defaults of an existing one:

1. **Provide default rows** – pass a list of dictionaries (or dataclass
   instances) to `_render_schedule_editor`. Values are normalised via
   `_normalise_schedule_rows`, so the editor always starts with the supplied
   defaults for the active scenario.
2. **Optionally lock columns** – use the `fixed_columns` argument to enforce
   read-only headers such as the category name while allowing other fields to be
   edited.
3. **Seed “Add row” templates** – supply `row_defaults` to pre-populate new
   rows with sensible values (for example, a default note or the next period
   label).
4. **Enable yearly growth** – leave `allow_yearly_increment=True` (the default)
   so users can apply compound growth to any numeric column. The helper safely
   coerces the selected columns to floats before applying the increment.
5. **Persist the edited data** – the helper writes the current rows back to
   `st.session_state` under the provided namespace and scenario key. Callers can
   consume the returned `DataFrame` immediately to refresh downstream
   calculations.

Because the default rows are stored alongside their original snapshot, updating
the source data automatically resets the table for new scenarios while keeping
existing edits intact. See `_render_schedule_editor` and
`_initialise_schedule_state` in `streamlit_app.py` for reference usage.

## Revenue schedules helper (optional)

If you maintain a separate Excel-based poultry model, you can generate
category-by-category revenue schedules using the included
`revenue_schedules_builder.py` script. The helper requires a few lightweight
packages:

```bash
pip install pandas numpy openpyxl
```

Run the builder against your workbook and choose an output path for the
generated schedules workbook:

```bash
python revenue_schedules_builder.py \
  --input /path/to/source_model.xlsx \
  --output /path/to/revenue_schedules.xlsx
```

The resulting Excel file contains:

- One sheet per revenue category (broiler meat, eggs, manure, live birds, and
  by-products) either populated from detected columns or ready-to-fill
  templates if no data exists.
- An "All Revenues (Detail)" consolidation tab.
- Summary sheets for totals by period and by category.
