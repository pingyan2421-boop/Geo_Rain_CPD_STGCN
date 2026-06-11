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

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data  # noqa: E402
from data_loader.cpd_methods import CPD_METHODS  # noqa: E402


def summarize_stages(raw_seq, time_cols, change_points, method):
    rows = []
    start = 0
    for stage_id, end in enumerate(change_points + [raw_seq.shape[0]]):
        segment = raw_seq[start:end]
        if len(segment) >= 2:
            delta = segment[-1] - segment[0]
            step_velocity = np.diff(segment, axis=0)
            mean_delta = float(np.mean(delta))
            mean_step_velocity = float(np.mean(step_velocity))
        else:
            mean_delta = float("nan")
            mean_step_velocity = float("nan")
        rows.append({
            "method": method,
            "stage_id": stage_id,
            "start_index": start,
            "end_index": end - 1,
            "start_date": time_cols[start],
            "end_date": time_cols[end - 1],
            "length": end - start,
            "mean_displacement_delta": mean_delta,
            "mean_step_velocity": mean_step_velocity,
        })
        start = end
    return rows


def save_timeline(path, raw_seq, time_cols, method_cps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(time_cols, errors="coerce")
    x_axis = dates if not pd.isna(dates).any() else np.arange(len(time_cols))
    signal = np.mean(raw_seq, axis=1)
    colors = {"binseg": "#d62728", "pelt": "#2ca02c", "bfast": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, signal, color="#333333", linewidth=1.8, label="Mean displacement")
    for method, cps in method_cps.items():
        for cp in cps:
            ax.axvline(x_axis[cp], color=colors.get(method, "#777777"), alpha=0.55, linewidth=1.4)
        if cps:
            ax.plot([], [], color=colors.get(method, "#777777"), label=method)
    ax.set_title("CPD method sensitivity")
    ax.set_ylabel("Mean displacement (mm)")
    ax.legend(loc="best")
    if not pd.isna(dates).any():
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Binseg, PELT and BFAST stage boundaries.")
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["binseg", "pelt", "bfast", "kernelcpd", "dynp", "bottomup", "window", "bocpd", "mdl_multivariate"],
        choices=list(CPD_METHODS),
    )
    parser.add_argument("--cpd_mode", choices=["mean", "std", "top", "mix"], default="mix")
    parser.add_argument("--cpd_cost", default="l2")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_top_ratio", type=float, default=0.10)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--pelt_penalty", type=float, default=None)
    parser.add_argument("--bfast_frequency", type=int, default=23)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "cpd_method_sensitivity"))
    args = parser.parse_args()

    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    os.makedirs(args.output_dir, exist_ok=True)

    cp_rows = []
    stage_rows = []
    method_cps = {}
    for method in args.methods:
        _, cps = detect_change_points_professional(
            raw_seq,
            penalty=args.cpd_penalty,
            mode=args.cpd_mode,
            top_ratio=args.cpd_top_ratio,
            min_size=args.cpd_min_size,
            method=method,
            model=args.cpd_cost,
            pelt_penalty=args.pelt_penalty,
            bfast_frequency=args.bfast_frequency,
        )
        method_cps[method] = cps
        for cp in cps:
            cp_rows.append({"method": method, "change_point_index": cp, "date": time_cols[cp]})
        stage_rows.extend(summarize_stages(raw_seq, time_cols, cps, method))

    cp_df = pd.DataFrame(cp_rows)
    stage_df = pd.DataFrame(stage_rows)
    cp_df.to_csv(os.path.join(args.output_dir, "cpd_method_change_points.csv"), index=False, encoding="utf-8")
    stage_df.to_csv(os.path.join(args.output_dir, "cpd_method_stage_summary.csv"), index=False, encoding="utf-8")
    save_timeline(os.path.join(args.output_dir, "cpd_method_timeline.png"), raw_seq, time_cols, method_cps)

    print(f">> Saved CPD method sensitivity outputs to: {args.output_dir}")
    if not cp_df.empty:
        print(cp_df.to_string(index=False))


if __name__ == "__main__":
    main()
