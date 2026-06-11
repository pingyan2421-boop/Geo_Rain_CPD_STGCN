import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_date(df, col="date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def add_rain_background(rain):
    rain = ensure_date(rain).sort_values("date").reset_index(drop=True)
    rain["rainy_day"] = (rain["rain_mm"] >= 1.0).astype(int)
    positive = rain.loc[rain["rain_mm"] > 0.0, "rain_mm"]
    p50 = float(positive.quantile(0.50))
    p75 = float(positive.quantile(0.75))
    p90 = float(positive.quantile(0.90))
    p95 = float(positive.quantile(0.95))
    rain["p75_day"] = (rain["rain_mm"] >= p75).astype(int)
    rain["p90_day"] = (rain["rain_mm"] >= p90).astype(int)
    rain["p95_day"] = (rain["rain_mm"] >= p95).astype(int)

    shifted_rain = rain["rain_mm"].shift(1).fillna(0.0)
    shifted_rainy = rain["rainy_day"].shift(1).fillna(0)
    shifted_p75 = rain["p75_day"].shift(1).fillna(0)
    shifted_p90 = rain["p90_day"].shift(1).fillna(0)
    shifted_p95 = rain["p95_day"].shift(1).fillna(0)
    for days in (3, 7, 15, 30, 45, 60):
        rain[f"pre{days}_rain_sum"] = shifted_rain.rolling(days, min_periods=1).sum()
        rain[f"pre{days}_rainy_days"] = shifted_rainy.rolling(days, min_periods=1).sum()
    for days in (30, 60):
        rain[f"pre{days}_p75_days"] = shifted_p75.rolling(days, min_periods=1).sum()
        rain[f"pre{days}_p90_days"] = shifted_p90.rolling(days, min_periods=1).sum()
        rain[f"pre{days}_p95_days"] = shifted_p95.rolling(days, min_periods=1).sum()
    for days in (7, 15, 30):
        rain[f"max_roll{days}_before"] = shifted_rain.rolling(days, min_periods=1).sum().rolling(30, min_periods=1).max()

    for days in (3, 7, 15, 30):
        rain[f"post{days}_rain_sum"] = rain["rain_mm"].shift(-1).fillna(0.0).rolling(days, min_periods=1).sum().shift(-(days - 1)).fillna(0.0)
    return rain, {"P50": p50, "P75": p75, "P90": p90, "P95": p95}


def robust_scale(series):
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    denom = max(q75 - q25, 1e-6)
    return ((series - q25) / denom).clip(lower=0.0)


def select_event_spells(rain, thresholds, min_gap_days=7):
    wet60_q80 = float(rain["pre60_rain_sum"].quantile(0.80))
    wet30_q80 = float(rain["pre30_rain_sum"].quantile(0.80))
    candidates = rain[
        (rain["rain_mm"] >= thresholds["P75"])
        | ((rain["rain_mm"] >= thresholds["P50"]) & ((rain["pre60_rain_sum"] >= wet60_q80) | (rain["pre30_rain_sum"] >= wet30_q80)))
    ].copy()
    if candidates.empty:
        return candidates

    rain_score = candidates["rain_mm"] / max(thresholds["P95"], 1e-6)
    wet_score = 0.5 * robust_scale(candidates["pre30_rain_sum"]) + 0.5 * robust_scale(candidates["pre60_rain_sum"])
    candidates["event_selection_score"] = rain_score + wet_score
    selected_rows = []
    selected_dates = []
    ranked = candidates.sort_values(["event_selection_score", "rain_mm"], ascending=False)
    for row in ranked.itertuples(index=False):
        if any(abs((row.date - date).days) <= min_gap_days for date in selected_dates):
            continue
        selected_rows.append(row._asdict())
        selected_dates.append(row.date)

    rows = []
    for selected in selected_rows:
        selected = pd.Series(selected).copy()
        start = selected["date"] - pd.Timedelta(days=int(min_gap_days))
        end = selected["date"] + pd.Timedelta(days=int(min_gap_days))
        spell_df = rain[(rain["date"] >= start) & (rain["date"] <= end)].copy()
        wet_spell = spell_df[spell_df["rain_mm"] > 0.0]
        if wet_spell.empty:
            wet_spell = spell_df
        selected["spell_start"] = wet_spell["date"].min()
        selected["spell_end"] = wet_spell["date"].max()
        selected["spell_days"] = int((selected["spell_end"] - selected["spell_start"]).days + 1)
        selected["spell_rain_sum"] = float(wet_spell["rain_mm"].sum())
        selected["spell_peak_rain"] = float(wet_spell["rain_mm"].max())
        rows.append(selected)
    events = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    events["event_id"] = [f"E{i:03d}" for i in range(1, len(events) + 1)]
    return events


def load_rain_sensitive_nodes(path, active_quantile):
    susceptibility = pd.read_csv(path)
    if "rain_susceptibility" in susceptibility.columns and susceptibility["rain_susceptibility"].gt(0).any():
        active = susceptibility[susceptibility["rain_susceptibility"] > 0].copy()
    else:
        threshold = float(susceptibility["rain_susceptibility_dense"].quantile(active_quantile))
        active = susceptibility[susceptibility["rain_susceptibility_dense"] >= threshold].copy()
    return active, set(active["node_index"].astype(int).tolist())


def add_response_labels(events, cpd_path, active_nodes, lags, profiles, min_response_nodes):
    cpd = pd.read_csv(cpd_path)
    cpd["date"] = pd.to_datetime(cpd["date"])
    cpd["node_index"] = cpd["node_index"].astype(int)
    cpd = cpd[cpd["node_index"].isin(active_nodes)]
    if profiles:
        cpd = cpd[cpd["profile"].isin(profiles)]
    denom = max(len(active_nodes), 1)

    events = events.copy()
    for lag in lags:
        counts = []
        densities = []
        profile_hits = []
        for row in events.itertuples(index=False):
            end = row.date + pd.Timedelta(days=int(lag))
            in_window = cpd[(cpd["date"] >= row.date) & (cpd["date"] <= end)]
            counts.append(int(in_window["node_index"].nunique()))
            densities.append(float(in_window["node_index"].nunique() / denom))
            profile_hits.append(int(len(in_window)))
        events[f"response_nodes_{lag}d"] = counts
        events[f"trigger_probability_{lag}d"] = densities
        events[f"cp_records_{lag}d"] = profile_hits
    prob_cols = [f"trigger_probability_{lag}d" for lag in lags]
    count_cols = [f"response_nodes_{lag}d" for lag in lags]
    events["trigger_probability"] = events[prob_cols].max(axis=1)
    events["response_nodes_max"] = events[count_cols].max(axis=1)
    events["dominant_response_window"] = events[prob_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)[0]
    for lag in lags:
        events[f"is_response_event_{lag}d"] = events[f"response_nodes_{lag}d"] >= int(min_response_nodes)
    events["is_response_event"] = events[[f"is_response_event_{lag}d" for lag in lags]].any(axis=1)
    response_lag_cols = [f"is_response_event_{lag}d" for lag in lags]
    first_lags = []
    for row in events[response_lag_cols].itertuples(index=False):
        first_lag = 0
        for lag, flag in zip(lags, row):
            if bool(flag):
                first_lag = int(lag)
                break
        first_lags.append(first_lag)
    events["first_response_lag_days"] = first_lags
    events["delayed_response_event"] = (
        events.get("is_response_event_15d", False)
        & ~events.get("is_response_event_3d", False)
        & ~events.get("is_response_event_7d", False)
    )
    return events


def assign_levels(events, thresholds, min_response_nodes):
    events = events.copy()
    wet_raw = (
        robust_scale(events["pre15_rain_sum"])
        + robust_scale(events["pre30_rain_sum"])
        + robust_scale(events["pre60_rain_sum"])
        + robust_scale(events["pre60_p90_days"])
        + robust_scale(events["max_roll30_before"])
    ) / 5.0
    events["antecedent_wetness_index"] = wet_raw
    wet_q = events["antecedent_wetness_index"].quantile([0.33, 0.66, 0.85]).to_dict()

    def wet_level(x):
        if x >= wet_q[0.85]:
            return "W3_very_wet"
        if x >= wet_q[0.66]:
            return "W2_wet"
        if x >= wet_q[0.33]:
            return "W1_moderate"
        return "W0_dry"

    p_min = float(min_response_nodes) / max(float(events.attrs.get("active_node_count", 1)), 1.0)

    def response_level(row):
        p = row.trigger_probability
        if row.response_nodes_max < min_response_nodes:
            return "R0_below_event_threshold"
        if p >= 0.30:
            return "R3_strong"
        if p >= 0.10:
            return "R2_moderate"
        if p >= p_min:
            return "R1_weak"
        return "R0_below_event_threshold"

    events["wetness_level"] = events["antecedent_wetness_index"].map(wet_level)
    events["response_level"] = events.apply(response_level, axis=1)
    events["rain_level"] = np.select(
        [
            events["rain_mm"] >= thresholds["P95"],
            events["rain_mm"] >= thresholds["P90"],
            events["rain_mm"] >= thresholds["P75"],
        ],
        ["P95_extreme", "P90_heavy", "P75_moderate"],
        default="below_P75_wet_background",
    )

    def event_class(row):
        high_wet = row.wetness_level in {"W2_wet", "W3_very_wet"}
        strong = row.response_level == "R3_strong"
        moderate = row.response_level == "R2_moderate"
        dry = row.wetness_level == "W0_dry"
        heavy = row.rain_mm >= thresholds["P90"]
        extreme = row.rain_mm >= thresholds["P95"]
        delayed = row.dominant_response_window >= 15
        if row.delayed_response_event and row.response_level in {"R1_weak", "R2_moderate", "R3_strong"}:
            return "delayed_trigger"
        if high_wet and strong and not heavy:
            return "wet_background_trigger"
        if high_wet and strong:
            return "saturated_high_trigger"
        if extreme and dry and delayed:
            return "dry_extreme_delayed"
        if heavy and row.response_level in {"R0_none", "R1_weak"}:
            return "strong_rain_weak_response"
        if moderate or strong:
            return "moderate_trigger"
        return "weak_or_no_response"

    events["event_class"] = events.apply(event_class, axis=1)
    return events


def summarize_classes(events):
    rows = []
    for cls, group in events.groupby("event_class"):
        rows.append({
            "event_class": cls,
            "events": int(len(group)),
            "rain_mm_mean": float(group["rain_mm"].mean()),
            "pre30_rain_sum_mean": float(group["pre30_rain_sum"].mean()),
            "pre60_rain_sum_mean": float(group["pre60_rain_sum"].mean()),
            "trigger_probability_mean": float(group["trigger_probability"].mean()),
            "response_nodes_max_mean": float(group["response_nodes_max"].mean()),
            "response_events": int(group["is_response_event"].sum()),
            "delayed_response_events": int(group["delayed_response_event"].sum()),
            "dominant_window_mode": int(group["dominant_response_window"].mode().iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["trigger_probability_mean", "events"], ascending=[False, False])


def write_report(path, events, summary, thresholds, lags, profiles):
    lines = [
        "# Rainfall Event Catalog",
        "",
        "## Definition",
        "",
        "- Candidate event: a rainy spell with rain >= P75, plus moderate rain days >= P50 when antecedent wetness is high.",
        "- Nearby candidate days are merged with a 7-day gap rule; the event date is selected by rainfall perturbation plus antecedent wetness score.",
            "- Trigger label: rain-sensitive node CPD density after the event in configured lag windows.",
            "- Valid response event: response node count reaches the configured minimum count in any lag window.",
            "- Delayed response event: 3d and 7d stay below the threshold, while 15d reaches it.",
            "- Prediction-facing columns use only event-day and antecedent features; post-event rain columns are diagnostic only.",
        "",
        "## Rain Thresholds",
        "",
    ]
    for key, value in thresholds.items():
        lines.append(f"- {key}: {value:.3f} mm/day")
    lines.extend(
        [
            "",
            "## Label Setup",
            "",
            f"- lag windows: {', '.join(str(x) + 'd' for x in lags)}",
            f"- CPD profiles: {', '.join(profiles)}",
            f"- minimum response nodes: {int(events.attrs.get('min_response_nodes', 0))}",
            "",
            "## Event Class Summary",
            "",
            markdown_table(summary),
            "",
            "## Top Trigger Events",
            "",
            markdown_table(events.sort_values("trigger_probability", ascending=False).head(15)[[
                "event_id",
                "date",
                "rain_mm",
                "pre30_rain_sum",
                "pre60_rain_sum",
                "response_nodes_max",
                "is_response_event",
                "first_response_lag_days",
                "delayed_response_event",
                "trigger_probability",
                "dominant_response_window",
                "wetness_level",
                "rain_level",
                "event_class",
            ]]),
            "",
            "## Interpretation Boundary",
            "",
            "This catalog does not prove rainfall causality. It is a prediction-facing event-state target that separates antecedent wetness, rainfall perturbation, and rain-sensitive CPD response.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
            if isinstance(value, pd.Timestamp):
                values.append(value.date().isoformat())
            elif isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_figures(events, rain, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "weak_or_no_response": "#8a8f98",
        "strong_rain_weak_response": "#d49b35",
        "dry_extreme_delayed": "#b85c38",
        "moderate_trigger": "#4b8f8c",
        "wet_background_trigger": "#4f73b7",
        "saturated_high_trigger": "#9b4d8b",
    }

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(rain["date"], rain["rain_mm"], width=1.0, color="#b9d2e6", edgecolor="none", alpha=0.75)
    ax2 = ax.twinx()
    ax2.plot(rain["date"], rain["pre30_rain_sum"], color="#6b705c", linewidth=1.3, alpha=0.85, label="pre30 rain")
    ax2.plot(rain["date"], rain["pre60_rain_sum"], color="#3d5a80", linewidth=1.3, alpha=0.8, label="pre60 rain")
    for cls, group in events.groupby("event_class"):
        ax.scatter(
            group["date"],
            group["rain_mm"] + 1.0,
            s=40 + 260 * group["trigger_probability"],
            color=colors.get(cls, "#333333"),
            edgecolor="white",
            linewidth=0.5,
            label=cls,
            zorder=4,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily rainfall (mm)")
    ax2.set_ylabel("Antecedent rainfall sum (mm)")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "event_timeline_wetness_response.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for cls, group in events.groupby("event_class"):
        ax.scatter(
            group["antecedent_wetness_index"],
            group["rain_mm"],
            s=35 + 520 * group["trigger_probability"],
            color=colors.get(cls, "#333333"),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.6,
            label=cls,
        )
    ax.set_xlabel("Antecedent wetness index")
    ax.set_ylabel("Event-day rainfall (mm)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "event_wetness_rain_response_scatter.png", dpi=240)
    plt.close(fig)

    heat = events.sort_values(["trigger_probability", "date"], ascending=[False, True]).head(40).copy()
    prob_cols = [c for c in events.columns if c.startswith("trigger_probability_") and c.endswith("d")]
    matrix = heat[prob_cols].to_numpy(dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.6, max(5.0, 0.22 * len(heat))))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(0.30, float(np.nanmax(matrix)) if matrix.size else 0.30), aspect="auto")
    ax.set_xticks(np.arange(len(prob_cols)))
    ax.set_xticklabels([c.replace("trigger_probability_", "+") for c in prob_cols])
    labels = [f"{row.event_id} {row.date.date()} {row.event_class}" for row in heat.itertuples(index=False)]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="Rain-sensitive CPD density")
    fig.tight_layout()
    fig.savefig(output_dir / "event_trigger_probability_heatmap.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build a prediction-facing rainfall event catalog with rain-sensitive CPD response labels.")
    parser.add_argument("--rainfall_csv", default="dataset/rainfall/chirps_daily.csv")
    parser.add_argument("--node_change_points_csv", default="output/node_rainfall_hotspots_chirps/node_change_points.csv")
    parser.add_argument("--rain_susceptibility_csv", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--lags", default="3,7,15")
    parser.add_argument("--profiles", default="strict,medium")
    parser.add_argument("--min_gap_days", type=int, default=7)
    parser.add_argument("--min_response_nodes", type=int, default=100)
    parser.add_argument("--active_quantile", type=float, default=0.75)
    parser.add_argument("--output_dir", default="output/rainfall_event_catalog")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]

    rain = pd.read_csv(args.rainfall_csv)
    rain, thresholds = add_rain_background(rain)
    active_nodes_df, active_nodes = load_rain_sensitive_nodes(args.rain_susceptibility_csv, args.active_quantile)
    events = select_event_spells(rain, thresholds, min_gap_days=args.min_gap_days)
    events = add_response_labels(events, args.node_change_points_csv, active_nodes, lags, profiles, args.min_response_nodes)
    events.attrs["active_node_count"] = len(active_nodes)
    events.attrs["min_response_nodes"] = args.min_response_nodes
    events = assign_levels(events, thresholds, args.min_response_nodes)
    events.attrs["active_node_count"] = len(active_nodes)
    events.attrs["min_response_nodes"] = args.min_response_nodes
    summary = summarize_classes(events)

    events.to_csv(output_dir / "rainfall_event_catalog.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "event_class_summary.csv", index=False, encoding="utf-8")
    active_nodes_df.to_csv(output_dir / "event_catalog_active_rain_sensitive_nodes.csv", index=False, encoding="utf-8")
    write_report(output_dir / "rainfall_event_catalog_report.md", events, summary, thresholds, lags, profiles)
    save_figures(events, rain, output_dir)

    print(f">> Built rainfall event catalog: {output_dir}")
    print(f">> events={len(events)} active_rain_sensitive_nodes={len(active_nodes)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
