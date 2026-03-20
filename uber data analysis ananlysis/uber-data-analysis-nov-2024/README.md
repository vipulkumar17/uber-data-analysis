# Uber Data Analysis (Nov 2024) - Rookie B.Tech Project

**Project goal (as in your resume style):**
- Conducted in-depth EDA on an Uber ride dataset using **Python, Pandas, NumPy, Matplotlib, and Seaborn**
- Identified **demand trends**, **peak hours**, **surge/pricing patterns**, and **high-traffic geo-locations**
- Derived actionable business insights through **groupby aggregations**, **correlation heatmaps**, and **time-series analysis** with compelling visual storytelling

## What you will do
1. Load the Uber dataset from `data/`
2. Clean it (parse date-time, handle missing values)
3. Create visuals:
   - Demand by hour + peak hours
   - (If available) surge/pricing patterns by hour/day
   - Geo hotspots (top latitude/longitude bins)
   - Correlation heatmap of numeric features
   - Time-series demand plot (daily trend)

## Dataset you need
Put your CSV in `data/` and rename it to one of these:
- `uber_dataset.csv` (recommended)

This project is written to work with the common Uber datasets that include a datetime column like:
- `Date/Time` or `datetime` or `pickup_datetime`

Latitude/longitude columns are expected like:
- `Lat` / `Lon` (or similarly named)

If your dataset also contains a surge-like column (examples):
- `Surge`, `surge_multiplier`, `fare_amount`, `price`
the project will automatically plot surge/pricing patterns.

## Install dependencies
Run in PowerShell:
```powershell
pip install -r requirements.txt
```

## Run the full analysis (recommended)
```powershell
python .\src\run_analysis.py --data .\data\uber_dataset.csv --output-dir .\outputs
```

Plots will be saved into `outputs/` and a short summary report will be written to `reports/`.

## Notes (very important)
- If your dataset has different column names, the code tries to auto-detect them, but you may need to adjust column names inside `src/uber_eda.py`.
- If the dataset does not have a surge/pricing column, the project will still produce demand/geo/time-series/correlation visuals.

