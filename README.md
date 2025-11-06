# Broiler Chicken Financial Model

This repository contains a fully scripted broiler chicken financial model that
runs without Excel or third-party Python dependencies.  The model codifies key
production assumptions, per-cycle economics, annual rollups, debt service, and
10-year discounted cash flow valuation metrics.  Results can be exported to CSV
and/or JSON for further analysis.

## Contents

- `deployable_financial_model.py` – CLI that generates the model outputs.
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

- `assumptions.(csv|json)` – input assumptions used by the model.
- `production_cycles.(csv|json)` – production and cost metrics for each flock
  cycle within a year.
- `annual_summary.(csv|json)` – aggregated annual revenue, cost, and EBITDA
  metrics.
- `cash_flow.(csv|json)` – 10-year cash flow statement with operating cash flow,
  maintenance capex, debt service, and discounted values.
- `valuation.json` – NPV/IRR results using the modeled cash flows.
- `manifest.json` – convenience listing of the generated files.

All values are expressed in US dollars except where otherwise noted.  Adjust the
`Assumptions` dataclass in `deployable_financial_model.py` to model different
operations (e.g., number of birds, pricing, debt structure, or cost inflation).
