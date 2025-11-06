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
- `cash_flow.(csv|json)` – 10-year cash flow statement with operating cash flow,
  maintenance capex, debt service, and discounted values.
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

The dashboard organises the inputs into horizontal tabs that mirror the CLI
assumption schedule. As you tweak values, the NPV/IRR metrics and detailed
tables refresh instantly. The Production tab now houses the four-part
assumptions summary as well as dedicated revenue schedules for broilers, eggs,
manure, live birds, and by-products. Download buttons on each table let you
export the current view as CSV files.

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
