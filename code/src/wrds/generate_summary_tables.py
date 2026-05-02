from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "wrds"
OUTPUT_DIR = REPO_ROOT / "output" / "tables" / "wrds"

SALES_PATH = DATA_DIR / "wrds_sales_agg_by_segmentxyear.csv"
COUNT_PATH = DATA_DIR / "wrds_naics_code_count_by_segmentxyear.csv"


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def panel_stats(df: pd.DataFrame, value_col: str, extra_rows: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    rows = [
        ("Observations", fmt_int(len(df))),
        ("Unique firms", fmt_int(df["gvkey"].nunique())),
        ("Unique firm segments", fmt_int(df["firm_segment_name"].nunique())),
        ("Years covered", f'{int(df["year"].min())}--{int(df["year"].max())}'),
        (f"Mean {value_col}", fmt_float(df[value_col].mean())),
        (f"Std. dev. {value_col}", fmt_float(df[value_col].std())),
        (f"P25 {value_col}", fmt_float(df[value_col].quantile(0.25))),
        (f"Median {value_col}", fmt_float(df[value_col].median())),
        (f"P75 {value_col}", fmt_float(df[value_col].quantile(0.75))),
        (f"Min {value_col}", fmt_float(df[value_col].min())),
        (f"Max {value_col}", fmt_float(df[value_col].max())),
    ]
    if extra_rows:
        rows[4:4] = extra_rows
    return rows


def make_table(
    caption: str,
    label: str,
    col_labels: list[str],
    body_rows: list[tuple[str, list[str]]],
    note: str,
) -> str:
    cols = "l" + "r" * len(col_labels)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{threeparttable}",
        "\\small",
        f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        "\\toprule",
        "Statistic & " + " & ".join(f"\\textbf{{{c}}}" for c in col_labels) + " \\\\",
        "\\midrule",
    ]
    for row_label, values in body_rows:
        if row_label == "__MIDRULE__":
            lines.append("\\midrule")
            continue
        lines.append(row_label + " & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]",
            "\\footnotesize",
            f"\\item \\textit{{Notes:}} {note}",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_sales_table(sales: pd.DataFrame) -> str:
    samples = {
        "All firms": sales.copy(),
        "Current S\\&P 500 firms": sales[sales["curr_sp500_flag"] == 1].copy(),
        "Non-S\\&P 500 firms": sales[sales["curr_sp500_flag"] != 1].copy(),
    }
    body_rows = []
    stats_by_sample = {
        name: panel_stats(
            df,
            "sales",
            extra_rows=[("Total sales", fmt_float(df["sales"].sum()))],
        )
        for name, df in samples.items()
    }
    row_names = [row[0] for row in next(iter(stats_by_sample.values()))]
    for idx, row_name in enumerate(row_names):
        body_rows.append((row_name, [stats_by_sample[name][idx][1] for name in samples]))
    note = (
        "This table reports summary statistics for the segment-year sales panel in "
        "\\texttt{wrds\\_sales\\_agg\\_by\\_segmentxyear.csv}. "
        "The S\\&P 500 indicator is the current-membership flag carried in the WRDS extract "
        "rather than a historical index-membership series."
    )
    return make_table(
        caption="Summary Statistics for WRDS Segment Sales",
        label="tab:wrds_sales_summary",
        col_labels=list(samples.keys()),
        body_rows=body_rows,
        note=note,
    )


def build_count_table(counts: pd.DataFrame) -> str:
    samples = {
        "All firms": counts.copy(),
        "Current S\\&P 500 firms": counts[counts["curr_sp500_flag"] == 1].copy(),
        "Non-S\\&P 500 firms": counts[counts["curr_sp500_flag"] != 1].copy(),
    }
    body_rows = []
    stats_by_sample = {
        name: panel_stats(
            df,
            "count",
            extra_rows=[
                ("Total count mass", fmt_float(df["count"].sum())),
                ("Unique NAICS codes", fmt_int(df["naics_code"].nunique())),
            ],
        )
        for name, df in samples.items()
    }
    row_names = [row[0] for row in next(iter(stats_by_sample.values()))]
    for idx, row_name in enumerate(row_names):
        body_rows.append((row_name, [stats_by_sample[name][idx][1] for name in samples]))
    note = (
        "This table reports summary statistics for the segment-year-industry count panel in "
        "\\texttt{wrds\\_naics\\_code\\_count\\_by\\_segmentxyear.csv}. "
        "The count variable is the count-like industry-appearance measure used in the notebook analysis."
    )
    return make_table(
        caption="Summary Statistics for WRDS Segment-Industry Counts",
        label="tab:wrds_count_summary",
        col_labels=list(samples.keys()),
        body_rows=body_rows,
        note=note,
    )


def build_coverage_table(sales: pd.DataFrame, counts: pd.DataFrame) -> str:
    yearly_sales = (
        sales.groupby("year", as_index=False)
        .agg(
            firms=("gvkey", "nunique"),
            segment_obs=("gvkey", "size"),
            total_sales=("sales", "sum"),
        )
    )
    yearly_sp500 = (
        sales.loc[sales["curr_sp500_flag"] == 1]
        .groupby("year", as_index=False)
        .agg(sp500_firms=("gvkey", "nunique"))
    )
    yearly = (
        yearly_sales
        .merge(yearly_sp500, on="year", how="left")
        .merge(
            counts.groupby("year", as_index=False).agg(
                count_rows=("naics_code", "size"),
                total_count_mass=("count", "sum"),
            ),
            on="year",
            how="left",
        )
    )
    yearly["sp500_firms"] = yearly["sp500_firms"].fillna(0).astype(int)

    body_rows: list[tuple[str, list[str]]] = []
    for _, row in yearly.iterrows():
        body_rows.append(
            (
                str(int(row["year"])),
                [
                    fmt_int(row["firms"]),
                    fmt_int(row["sp500_firms"]),
                    fmt_int(row["segment_obs"]),
                    fmt_int(row["count_rows"]),
                    fmt_float(row["total_sales"]),
                    fmt_float(row["total_count_mass"]),
                ],
            )
        )

    overall = (
        "All years",
        [
            fmt_int(sales["gvkey"].nunique()),
            fmt_int(sales.loc[sales["curr_sp500_flag"] == 1, "gvkey"].nunique()),
            fmt_int(len(sales)),
            fmt_int(len(counts)),
            fmt_float(sales["sales"].sum()),
            fmt_float(counts["count"].sum()),
        ],
    )
    body_rows.append(("__MIDRULE__", []))
    body_rows.append(overall)

    note = (
        "Yearly firm counts are unique \\texttt{gvkey}s in the sales panel. "
        "``Current S\\&P 500 firms'' counts firms with \\texttt{curr\\_sp500\\_flag=1} in the exported WRDS files. "
        "Sales totals and count mass are reported in the raw units carried by the processed input files."
    )
    return make_table(
        caption="Coverage of the WRDS Segment Panels by Year",
        label="tab:wrds_coverage_by_year",
        col_labels=[
            "Firms",
            "Current S\\&P 500 firms",
            "Segment-year observations",
            "Segment-industry rows",
            "Total sales",
            "Total count mass",
        ],
        body_rows=body_rows,
        note=note,
    )


def build_master_file() -> str:
    return "\n".join(
        [
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{threeparttable}",
            "\\begin{document}",
            "\\input{wrds_sales_summary.tex}",
            "\\clearpage",
            "\\input{wrds_count_summary.tex}",
            "\\clearpage",
            "\\input{wrds_coverage_by_year.tex}",
            "\\end{document}",
            "",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(SALES_PATH)
    counts = pd.read_csv(COUNT_PATH)

    (OUTPUT_DIR / "wrds_sales_summary.tex").write_text(build_sales_table(sales))
    (OUTPUT_DIR / "wrds_count_summary.tex").write_text(build_count_table(counts))
    (OUTPUT_DIR / "wrds_coverage_by_year.tex").write_text(build_coverage_table(sales, counts))
    (OUTPUT_DIR / "wrds_summary_tables.tex").write_text(build_master_file())


if __name__ == "__main__":
    main()
