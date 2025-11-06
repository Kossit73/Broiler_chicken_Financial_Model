# Broiler Chicken Financial Model

This repository contains a fully scripted broiler chicken financial model that
runs without Excel or third-party Python dependencies.  The model codifies key
production assumptions, per-cycle economics, annual rollups, debt service, and
10-year discounted cash flow valuation metrics.  Results can be exported to CSV
and/or JSON for further analysis.  An optional Streamlit dashboard lets you
interactively explore the assumptions and outputs.

## Contents

- `deployable_financial_model.py` – CLI that generates the model outputs.
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

The command creates the `outputs/` directory (if needed) and writes:

- `assumptions_summary.(csv|json)` – four-schedule table (Production,
  Operating costs, Capital structure, Financing) that captures every model
  assumption.
- `assumptions.(csv|json)` – raw key/value assumptions used by the model.
- `production_cycles.(csv|json)` – production and cost metrics for each flock
  cycle within a year.
- `annual_summary.(csv|json)` – aggregated annual revenue, cost, and EBITDA
  metrics.
- `cash_flow.(csv|json)` – 10-year cash flow projections with operating cash
  flow, maintenance capex, debt service, and discounted values.
- `income_statement.(csv|json)` – modeled 10-year income statement with revenue,
  COGS, EBITDA, taxes, and net income.
- `balance_sheet.(csv|json)` – simplified balance sheet highlighting cash,
  working capital, net PP&E, debt, and equity movements.
- `cash_flow_statement.(csv|json)` – three-section cash flow statement (operating,
  investing, financing) plus ending cash balances.
- `loan_schedule.(csv|json)` – annual amortization table for the project debt.
- `advanced_metrics.csv`, `dscr_summary.csv`, `trend_analysis.csv` (and
  consolidated `advanced_analytics.json`) – KPI pack covering DSCR, payback, and
  performance trend data points used in the dashboard analytics views.
- Revenue schedule CSVs for each category (`broiler_revenue_schedule.csv`,
  `eggs_revenue_schedule.csv`, etc.) plus `revenue_schedules.json` capturing all
  five categories in one structure for downstream tooling.
- `valuation.json` – NPV/IRR results using the modeled cash flows.
- `manifest.json` – convenience listing of the generated files.

All values are expressed in US dollars except where otherwise noted.  Adjust the
`Assumptions` dataclass in `deployable_financial_model.py` to model different
operations (e.g., number of birds, pricing, debt structure, or cost inflation).

## Interactive dashboard (optional)

To experiment with the model in a browser-based UI, install the required
packages and launch Streamlit:

```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```

The dashboard now presents all inputs on a single **Input Landing Page** that
groups controls into Production, Pricing, Costs, and Capital & financing
sections. A scenario selector at the top lets you maintain alternative
assumption sets (Baseline, Expansion, Downside) without losing previous
configurations. As you tweak values, the NPV/IRR metrics and detailed tables
refresh instantly. Below the metrics you will find three workspaces:

1. **Production & revenues** – retains the assumptions summary (grouped into the
   four schedules), detailed revenue schedules, production cycle results, annual
   summary, and discounted cash flows.
2. **Financial statements** – exposes the modeled income statement, balance
   sheet, cash flow statement, and debt amortization tables.
3. **Advanced analytics** – surfaces KPIs (average margins, DSCR, payback) and
   line charts for DSCR and long-range revenue/EBITDA/net-income/free-cash-flow
   trends.

The Production & revenues tab includes an **Excel export** card—click *Prepare
Excel Model* to generate a multi-sheet workbook and download it directly from
the browser.

Open the **AI & Machine Learning Settings** expander in the Advanced analytics
tab to configure optional forecasting and narrative preferences. The settings
panel stores values per scenario and supports multiple providers, model names,
and narrative focus areas.

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
