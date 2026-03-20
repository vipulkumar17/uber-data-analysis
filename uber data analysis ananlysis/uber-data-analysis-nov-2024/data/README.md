# Data Folder

Place your Uber dataset here.

## Recommended file name
- `uber_dataset.csv`

## Expected columns (common cases)
- Date/time column: `Date/Time`, `datetime`, `pickup_datetime`, etc.
- Latitude/longitude: `Lat`, `Lon` (or similarly named)
- Optional surge/pricing signal: `Surge`, `surge_multiplier`, `fare_amount`, `price`, etc.

If your dataset uses different column names, the project tries to auto-detect them, but you may need to tweak detection in `src/uber_eda.py`.

