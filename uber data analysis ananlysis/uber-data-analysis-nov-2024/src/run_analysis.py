from __future__ import annotations

import argparse
import os

from uber_eda import analyze, load_data


def write_markdown_report(report_path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines: list[str] = []
    lines.append("# Uber Data Analysis (Nov 2024) - Summary Report\n")
    lines.append("## Dataset snapshot\n")
    lines.append(f"- Rows: {summary.get('rows')}\n")
    lines.append(f"- Columns: {summary.get('columns')}\n")
    lines.append(f"- Datetime column: `{summary.get('datetime_col')}`\n")
    lines.append(f"- Lat/Lon columns: `{summary.get('lat_col')}` / `{summary.get('lon_col')}`\n")
    lines.append(f"- Surge/Pricing signal column: `{summary.get('surge_col')}`\n")
    lines.append(f"- Date range: {summary.get('date_min')} to {summary.get('date_max')}\n")

    lines.append("\n## Peak hours (demand)\n")
    lines.append(f"- Peak hour: {summary.get('peak_hour')} (rides: {summary.get('peak_hour_rides')})\n")

    lines.append("\n## Surge/Pricing patterns (if available)\n")
    if summary.get("surge_pattern_summary"):
        sp = summary["surge_pattern_summary"]
        lines.append(f"- Used column: `{sp.get('surge_col')}`\n")
        lines.append(
            f"- Highest average day/hour: {sp.get('highest_avg_day')} at hour {sp.get('highest_avg_hour')} "
            f"(avg value: {sp.get('highest_avg_value')})\n"
        )
    else:
        lines.append("- No surge/pricing column detected; plotted demand heatmap instead.\n")

    if summary.get("top_geo_hotspots"):
        lines.append("\n## High-traffic geo hotspots (top bins)\n")
        lines.append("- " + "\n- ".join(
            [
                f"{g['lat_bin']},{g['lon_bin']} -> {g['rides']} rides"
                for g in summary.get("top_geo_hotspots", [])[:10]
            ]
        ) + "\n")

    lines.append("\n## Output plots\n")
    lines.append("- `outputs/01_peak_hours.png`\n")
    lines.append("- `outputs/02_time_series_daily_demand.png`\n")
    lines.append("- `outputs/03_geo_hotspots.png` (if lat/lon exists)\n")
    lines.append("- `outputs/04_correlation_heatmap.png`\n")
    lines.append("- `outputs/05_surging_or_pricing_patterns.png`\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Uber EDA and generate plots.")
    parser.add_argument("--data", required=True, help="Path to your Uber dataset CSV/Parquet file.")
    parser.add_argument("--output-dir", default="./outputs", help="Where to save plots.")
    parser.add_argument("--geo-precision", type=int, default=2, help="Decimal rounding for geo bins.")
    parser.add_argument("--top-n-geo", type=int, default=10, help="How many geo bins to show.")
    args = parser.parse_args()

    df = load_data(args.data)
    summary = analyze(
        df,
        output_dir=args.output_dir,
        geo_precision=args.geo_precision,
        top_n_geo=args.top_n_geo,
    )

    # Write report into ./reports relative to project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_path = os.path.join(project_root, "reports", "analysis_summary.md")
    write_markdown_report(report_path, summary)

    # A small console summary (rookie friendly).
    print("\nUber EDA finished.")
    print(f"Peak hour: {summary.get('peak_hour')} (rides: {summary.get('peak_hour_rides')})")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()

