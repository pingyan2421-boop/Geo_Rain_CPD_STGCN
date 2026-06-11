import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data
from data_loader.rainfall_loader import forecast_context_for_starts, load_forecast_features


def split_rolling_cpd(starts, change_points, n_his, n_pred, val_len=18, test_len=18, gap=0):
    folds = []
    for cp in change_points:
        test_start = max(0, cp - n_his)
        test_end = test_start + test_len
        val_end = max(0, test_start - gap)
        val_start = max(0, val_end - val_len)
        train_end = val_start

        train = starts[starts < train_end]
        val = starts[(starts >= val_start) & (starts < val_end)]
        test = starts[(starts >= test_start) & (starts < test_end)]
        if len(train) >= 20 and len(val) >= 4 and len(test) >= 4:
            folds.append((f"cp_{cp}", train, val, test))
    return folds


def summarize_context(rows, fold_name, split_name, context_df):
    if context_df.empty:
        return
    available = context_df["forecast_available"] > 0.0
    row = {
        "fold": fold_name,
        "split": split_name,
        "samples": int(len(context_df)),
        "origin_start": str(context_df["origin_date"].min()),
        "origin_end": str(context_df["origin_date"].max()),
        "available_samples": int(available.sum()),
        "available_pct": float(available.mean() * 100.0),
        "missing_samples": int((~available).sum()),
        "issue_age_mean_days": float(context_df.loc[available, "forecast_issue_age_days"].mean())
        if available.any()
        else np.nan,
        "issue_age_max_days": float(context_df.loc[available, "forecast_issue_age_days"].max())
        if available.any()
        else np.nan,
    }
    if "forecast_is_fallback" in context_df.columns:
        fallback = context_df["forecast_is_fallback"] > 0.0
        row["fallback_samples"] = int(fallback.sum())
        row["fallback_pct"] = float(fallback.mean() * 100.0)
        row["official_samples"] = int((~fallback).sum())
        row["official_pct"] = float((~fallback).mean() * 100.0)
    for col in [c for c in context_df.columns if c.startswith("forecast_rain_")]:
        row[f"{col}_nonzero_samples"] = int((context_df[col] > 0.0).sum())
        row[f"{col}_nonzero_pct"] = float((context_df[col] > 0.0).mean() * 100.0)
        row[f"{col}_mean"] = float(context_df[col].mean())
    for col in [c for c in context_df.columns if c.startswith("forecast_event_")]:
        row[f"{col}_samples"] = int((context_df[col] > 0.0).sum())
    rows.append(row)


def forecast_file_summary(forecast_csv):
    forecast = load_forecast_features(forecast_csv)
    lead_days = (forecast["valid_date"] - forecast["issue_date"]).dt.days
    per_issue = forecast.groupby("issue_date").size()
    return {
        "forecast_rows": int(len(forecast)),
        "issue_dates": int(forecast["issue_date"].nunique()),
        "valid_dates": int(forecast["valid_date"].nunique()),
        "lead_days_min": int(lead_days.min()),
        "lead_days_max": int(lead_days.max()),
        "lead_days_unique": ",".join(map(str, sorted(lead_days.dropna().astype(int).unique()))),
        "rows_per_issue_min": int(per_issue.min()),
        "rows_per_issue_max": int(per_issue.max()),
        "rows_per_issue_mean": float(per_issue.mean()),
        "issue_start": forecast["issue_date"].min().date().isoformat(),
        "issue_end": forecast["issue_date"].max().date().isoformat(),
    }


def missing_summary(missing_csv):
    if not missing_csv:
        return pd.DataFrame()
    path = Path(missing_csv)
    if not path.exists():
        return pd.DataFrame()
    missing = pd.read_csv(path)
    if missing.empty or "issue_date" not in missing.columns:
        return pd.DataFrame()
    issue = pd.to_datetime(missing["issue_date"], errors="coerce")
    return (
        missing.assign(issue_year=issue.dt.year)
        .groupby("issue_year", dropna=False)
        .size()
        .reset_index(name="missing_issue_dates")
    )


def split_gap_summary(summary_df):
    rows = []
    metrics = [col for col in ("available_pct", "fallback_pct", "official_pct") if col in summary_df.columns]
    for fold_name, fold_df in summary_df.groupby("fold"):
        by_split = fold_df.set_index("split")
        for left, right in (("train", "val"), ("val", "test"), ("train", "test")):
            if left not in by_split.index or right not in by_split.index:
                continue
            row = {"fold": fold_name, "left_split": left, "right_split": right}
            for metric in metrics:
                row[f"{metric}_gap"] = float(abs(by_split.loc[left, metric] - by_split.loc[right, metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(report_path, file_info, summary_df, missing_df, gap_df):
    lines = [
        "# Forecast Context Diagnostics",
        "",
        "## Forecast CSV",
        "",
    ]
    for key, value in file_info.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Split Coverage", ""])
    display_cols = [
        "fold",
        "split",
        "samples",
        "origin_start",
        "origin_end",
        "available_samples",
        "available_pct",
        "fallback_samples",
        "fallback_pct",
        "official_samples",
        "official_pct",
        "issue_age_max_days",
        "forecast_rain_15d_nonzero_samples",
        "forecast_rain_15d_nonzero_pct",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]
    lines.append(markdown_table(summary_df[existing]))
    if not gap_df.empty:
        lines.extend(["", "## Split Source Gaps", ""])
        lines.append(markdown_table(gap_df))
    if not missing_df.empty:
        lines.extend(["", "## Missing Issue Dates", ""])
        lines.append(markdown_table(missing_df))
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This diagnostic only checks historical forecast availability and feature sparsity. "
            "It does not prove forecast skill or landslide hydrologic causality.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose forecast_context coverage for CPD validation folds.")
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--forecast_features", required=True)
    parser.add_argument("--missing_csv", default=None)
    parser.add_argument("--cpd_method", default="binseg")
    parser.add_argument("--fold_filter", default="cp_180")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--output_dir", default="output/forecast_context_diagnostics")
    args = parser.parse_args()

    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    _, change_points = detect_change_points_professional(raw_seq, method=args.cpd_method)
    starts = np.arange(0, raw_seq.shape[0] - args.n_his - args.n_pred + 1, dtype=np.int32)
    folds = split_rolling_cpd(starts, change_points, args.n_his, args.n_pred)
    if args.fold_filter:
        requested = {name.strip() for name in args.fold_filter.split(",") if name.strip()}
        folds = [fold for fold in folds if fold[0] in requested]
    if not folds:
        raise ValueError(f"No rolling CPD folds matched fold_filter={args.fold_filter!r}.")

    full_context = forecast_context_for_starts(args.forecast_features, time_cols, starts, args.n_his)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    summary_rows = []
    for fold_name, train_idx, val_idx, test_idx in folds:
        for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            context = full_context.iloc[np.asarray(indices, dtype=np.int32)].copy()
            context.insert(0, "split", split_name)
            context.insert(0, "fold", fold_name)
            context.insert(2, "sample_start", np.asarray(indices, dtype=np.int32))
            sample_rows.append(context)
            summarize_context(summary_rows, fold_name, split_name, context)

    sample_df = pd.concat(sample_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    file_info = forecast_file_summary(args.forecast_features)
    file_info_df = pd.DataFrame([file_info])
    missing_df = missing_summary(args.missing_csv)
    gap_df = split_gap_summary(summary_df)

    sample_df.to_csv(output_dir / "forecast_context_by_sample.csv", index=False)
    summary_df.to_csv(output_dir / "forecast_context_summary.csv", index=False)
    gap_df.to_csv(output_dir / "forecast_context_split_gaps.csv", index=False)
    file_info_df.to_csv(output_dir / "forecast_file_summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(output_dir / "forecast_missing_summary.csv", index=False)
    write_report(output_dir / "forecast_context_diagnostics.md", file_info, summary_df, missing_df, gap_df)

    print(f"Wrote forecast diagnostics to {output_dir}")
    preview_cols = [
        col
        for col in ["fold", "split", "samples", "available_samples", "available_pct", "fallback_samples", "fallback_pct"]
        if col in summary_df.columns
    ]
    print(summary_df[preview_cols].to_string(index=False))
    if not gap_df.empty:
        print(gap_df.to_string(index=False))


if __name__ == "__main__":
    main()
