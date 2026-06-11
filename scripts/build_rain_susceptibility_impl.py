import argparse
import os
import sys

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import load_and_clean_data  # noqa: E402


def minmax(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def sparsify_hotspots(values, quantile=0.75):
    values = np.asarray(values, dtype=np.float32)
    positive = values[values > 0]
    if positive.size == 0:
        return values
    cutoff = float(np.quantile(positive, float(quantile)))
    hi = float(np.max(values))
    if hi - cutoff < 1e-6:
        return values
    return np.clip((values - cutoff) / (hi - cutoff), 0.0, 1.0).astype(np.float32)


def build_susceptibility(response_csv, file_path, output_csv, hotspot_quantile=0.75):
    _, coords, _, _, node_ids = load_and_clean_data(file_path)
    n_nodes = len(node_ids)
    response = pd.read_csv(response_csv)
    required = {"node_index", "threshold_level", "lag_days", "profile", "lift", "p_value", "sensitivity_hits"}
    missing = required - set(response.columns)
    if missing:
        raise ValueError(f"Node response CSV is missing required columns: {sorted(missing)}")

    subset = response[
        response["threshold_level"].isin(["P90", "P95"])
        & response["lag_days"].isin([3, 7, 15])
        & response["profile"].isin(["strict", "medium"])
    ].copy()
    if subset.empty:
        raise ValueError("No P90/P95, 3/7/15-day, strict/medium rows found in node response CSV.")

    subset["lift_score"] = np.log1p(np.maximum(pd.to_numeric(subset["lift"], errors="coerce").fillna(0.0), 0.0))
    subset["p_score"] = -np.log10(np.maximum(pd.to_numeric(subset["p_value"], errors="coerce").fillna(1.0), 1e-6))
    subset["hit_score"] = pd.to_numeric(subset["sensitivity_hits"], errors="coerce").fillna(0.0)

    grouped = subset.groupby("node_index").agg(
        lift_score=("lift_score", "mean"),
        p_score=("p_score", "mean"),
        sensitivity_hits=("hit_score", "max"),
        evidence_rows=("lift", "size"),
    )
    grouped["lift_score"] = minmax(grouped["lift_score"].to_numpy(dtype=np.float32))
    grouped["p_score"] = minmax(grouped["p_score"].to_numpy(dtype=np.float32))
    grouped["hit_score"] = minmax(grouped["sensitivity_hits"].to_numpy(dtype=np.float32))
    grouped["rain_susceptibility"] = (
        0.45 * grouped["lift_score"]
        + 0.35 * grouped["p_score"]
        + 0.20 * grouped["hit_score"]
    )

    dense_values = np.zeros(n_nodes, dtype=np.float32)
    for node_index in range(n_nodes):
        if node_index in grouped.index:
            dense_values[node_index] = float(grouped.loc[node_index, "rain_susceptibility"])
    sparse_values = sparsify_hotspots(dense_values, quantile=hotspot_quantile)

    rows = []
    for node_index in range(n_nodes):
        if node_index in grouped.index:
            item = grouped.loc[node_index]
            dense_susceptibility = float(item["rain_susceptibility"])
            rain_susceptibility = float(sparse_values[node_index])
            evidence_rows = int(item["evidence_rows"])
            sensitivity_hits = float(item["sensitivity_hits"])
        else:
            dense_susceptibility = 0.0
            rain_susceptibility = 0.0
            evidence_rows = 0
            sensitivity_hits = 0.0
        rows.append({
            "node_index": node_index,
            "node_id": node_ids[node_index],
            "longitude": float(coords[node_index, 0]),
            "latitude": float(coords[node_index, 1]),
            "rain_susceptibility": rain_susceptibility,
            "rain_susceptibility_dense": dense_susceptibility,
            "sensitivity_hits": sensitivity_hits,
            "evidence_rows": evidence_rows,
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f">> Saved rain susceptibility to: {output_csv}")
    print(
        f">> nodes={len(out)}, hotspot_quantile={float(hotspot_quantile):.2f}, "
        f"range=({out['rain_susceptibility'].min():.3f}, {out['rain_susceptibility'].max():.3f}), "
        f"mean={out['rain_susceptibility'].mean():.3f}, active={(out['rain_susceptibility'] > 0).sum()}"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build continuous node rain susceptibility from node-level rainfall CPD response.")
    parser.add_argument("--node_response_csv", default=os.path.join(project_root, "output", "node_rainfall_hotspots_chirps", "node_rain_response.csv"))
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--output_csv", default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"))
    parser.add_argument("--hotspot_quantile", type=float, default=0.75)
    args = parser.parse_args()
    build_susceptibility(args.node_response_csv, args.file_path, args.output_csv, hotspot_quantile=args.hotspot_quantile)


if __name__ == "__main__":
    main()
