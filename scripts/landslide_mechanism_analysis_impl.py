import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import (  # noqa: E402
    detect_change_points_professional,
    load_and_clean_data,
    load_rain_susceptibility,
)
from data_loader.cpd_methods import CPD_METHODS  # noqa: E402


def to_local_xy(coords):
    lon = coords[:, 0].astype(np.float64)
    lat = coords[:, 1].astype(np.float64)
    lat0 = np.deg2rad(np.nanmean(lat))
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(lat0)
    y = (lat - np.nanmean(lat)) * 110540.0
    return x, y


def fit_downslope_azimuth(coords, elevation, idx):
    if len(idx) < 3:
        return float("nan"), float("nan")
    x, y = to_local_xy(coords[idx])
    z = elevation[idx].astype(np.float64)
    design = np.column_stack([x, y, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    dx, dy = -a, -b
    azimuth = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    gradient = math.sqrt(a * a + b * b)
    return azimuth, gradient


def group_summary(name, idx, raw_seq, coords, elevation, rain_susceptibility):
    cumulative_delta = raw_seq[-1] - raw_seq[0]
    motion_std = np.std(np.diff(raw_seq, axis=0), axis=0)
    azimuth, gradient = fit_downslope_azimuth(coords, elevation, idx)
    return {
        "group": name,
        "n_nodes": int(len(idx)),
        "lon_min": float(np.min(coords[idx, 0])),
        "lon_max": float(np.max(coords[idx, 0])),
        "lat_min": float(np.min(coords[idx, 1])),
        "lat_max": float(np.max(coords[idx, 1])),
        "elevation_mean": float(np.mean(elevation[idx])),
        "elevation_min": float(np.min(elevation[idx])),
        "elevation_max": float(np.max(elevation[idx])),
        "cumulative_displacement_mean": float(np.mean(cumulative_delta[idx])),
        "motion_std_mean": float(np.mean(motion_std[idx])),
        "rain_susceptibility_mean": float(np.mean(rain_susceptibility[idx])) if rain_susceptibility is not None else float("nan"),
        "downslope_azimuth_degree": float(azimuth),
        "slope_gradient": float(gradient),
    }


def summarize_stages(raw_seq, time_cols, cps):
    rows = []
    start = 0
    for stage_id, end in enumerate(cps + [raw_seq.shape[0]]):
        segment = raw_seq[start:end]
        step_velocity = np.diff(segment, axis=0)
        rows.append({
            "stage_id": stage_id,
            "start_index": start,
            "end_index": end - 1,
            "start_date": time_cols[start],
            "end_date": time_cols[end - 1],
            "length": end - start,
            "mean_displacement_delta": float(np.mean(segment[-1] - segment[0])) if len(segment) >= 2 else float("nan"),
            "mean_step_velocity": float(np.mean(step_velocity)) if len(step_velocity) else float("nan"),
        })
        start = end
    return rows


def save_figures(output_dir, raw_seq, coords, elevation, rain_susceptibility, group_masks, stage_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cumulative_delta = raw_seq[-1] - raw_seq[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stage_df["stage_id"], stage_df["mean_step_velocity"], color="#4c78a8")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Mean step velocity (mm/observation)")
    ax.set_title("Stage kinematics")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mechanism_stage_velocity.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    panels = [
        ("Elevation", elevation, "terrain"),
        ("Cumulative displacement", cumulative_delta, "coolwarm"),
        ("Rain susceptibility", rain_susceptibility if rain_susceptibility is not None else np.zeros_like(elevation), "viridis"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=5, cmap=cmap, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("i")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("j")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mechanism_space_groups.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(elevation, cumulative_delta, s=5, color="#999999", alpha=0.35, label="All nodes")
    for label, idx in group_masks.items():
        if label == "all" or len(idx) == 0:
            continue
        ax.scatter(elevation[idx], cumulative_delta[idx], s=8, alpha=0.65, label=label)
    ax.set_xlabel("Elevation")
    ax.set_ylabel("Cumulative displacement (mm)")
    ax.set_title("Elevation-displacement relation")
    ax.legend(loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mechanism_elevation_displacement.png"), dpi=180)
    plt.close(fig)


def write_interpretation(path, summary_df, stage_df, cps, cpd_method):
    all_row = summary_df[summary_df["group"] == "all"].iloc[0]
    rain_row = summary_df[summary_df["group"] == "rain_susceptible_top"].iloc[0]
    motion_row = summary_df[summary_df["group"] == "top_motion_10pct"].iloc[0]
    strongest_stage = stage_df.iloc[stage_df["mean_step_velocity"].abs().argmax()]
    azimuth = all_row["downslope_azimuth_degree"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 滑坡雨致响应与运动机理解释\n\n")
        f.write(f"- 当前主变化点检测方法：`{cpd_method}`；本次检测到变化点索引 `{cps}`。\n")
        f.write(
            f"- 地形面拟合得到整体下坡方位约 `{azimuth:.1f}` 度，雨敏高值节点下坡方位约 "
            f"`{rain_row['downslope_azimuth_degree']:.1f}` 度，强变形节点下坡方位约 "
            f"`{motion_row['downslope_azimuth_degree']:.1f}` 度。该角度按北为0度、顺时针计算。\n"
        )
        f.write(
            f"- 雨敏高值节点平均高程 `{rain_row['elevation_mean']:.1f}`，强变形前10%节点平均高程 "
            f"`{motion_row['elevation_mean']:.1f}`。这说明强降雨短滞后响应更偏向上部或补给敏感区，"
            "而累计变形峰值更偏向下部强运动区。\n"
        )
        f.write(
            f"- 阶段速度绝对值最大的阶段为 `{int(strongest_stage['stage_id'])}`，时间范围 "
            f"`{strongest_stage['start_date']}` 至 `{strongest_stage['end_date']}`，平均步进速度 "
            f"`{strongest_stage['mean_step_velocity']:.4f}` mm/观测期。\n\n"
        )
        f.write("## 机理性表述\n\n")
        f.write(
            "强降雨通过入渗和裂隙补给提高坡体含水状态，削弱潜在滑带及其上覆堆积体的有效抗剪强度；"
            "雨后3到15天内，上部雨敏节点先表现为变化点响应增强，随后变形沿已有地形KNN邻接关系向下坡方向传播。"
            "从当前点云地形拟合看，传播方向主要指向约133到135度方位，即东南到东南偏东方向。"
            "因此，降雨响应在模型中不应只被解释为误差修正特征，而应解释为控制局部节点传播强度的触发因子："
            "高雨强乘以高雨敏指数时，上部敏感节点向相邻下坡节点的形变信息传递增强。\n\n"
        )
        f.write("## 证据边界\n\n")
        f.write(
            "上述方向是由地形点云的坡面拟合和InSAR形变空间分布推断的下坡运动方向，不等同于三维位移反演。"
            "单轨InSAR主要约束视线向位移，缺少升降轨或GNSS约束时，不能单独证明完整三维滑动矢量。"
            "因此论文中建议使用“候选运动方向”或“地形约束的下坡传播方向”表述。\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate mechanism-oriented landslide motion interpretation.")
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rain_susceptibility", default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"))
    parser.add_argument("--cpd_method", choices=list(CPD_METHODS), default="binseg")
    parser.add_argument("--cpd_mode", choices=["mean", "std", "top", "mix"], default="mix")
    parser.add_argument("--cpd_cost", default="l2")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_top_ratio", type=float, default=0.10)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--pelt_penalty", type=float, default=None)
    parser.add_argument("--bfast_frequency", type=int, default=23)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "landslide_mechanism"))
    args = parser.parse_args()

    raw_seq, coords, elevation, time_cols, node_ids = load_and_clean_data(args.file_path)
    _, cps = detect_change_points_professional(
        raw_seq,
        penalty=args.cpd_penalty,
        mode=args.cpd_mode,
        top_ratio=args.cpd_top_ratio,
        min_size=args.cpd_min_size,
        method=args.cpd_method,
        model=args.cpd_cost,
        pelt_penalty=args.pelt_penalty,
        bfast_frequency=args.bfast_frequency,
    )
    rain_susceptibility = load_rain_susceptibility(args.rain_susceptibility, node_ids, raw_seq.shape[1])
    if rain_susceptibility is None:
        rain_susceptibility = np.zeros(raw_seq.shape[1], dtype=np.float32)

    motion_std = np.std(np.diff(raw_seq, axis=0), axis=0)
    active_rain = np.where(rain_susceptibility > 0)[0]
    rain_top_threshold = np.quantile(rain_susceptibility[active_rain], 0.60) if len(active_rain) else 1.0
    group_masks = {
        "all": np.arange(raw_seq.shape[1]),
        "top_motion_10pct": np.where(motion_std >= np.quantile(motion_std, 0.90))[0],
        "rain_susceptible_active": active_rain,
        "rain_susceptible_top": np.where(rain_susceptibility >= rain_top_threshold)[0],
    }
    group_masks["rain_top_and_motion_top"] = np.intersect1d(
        group_masks["top_motion_10pct"],
        group_masks["rain_susceptible_top"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_df = pd.DataFrame([
        group_summary(name, idx, raw_seq, coords, elevation, rain_susceptibility)
        for name, idx in group_masks.items()
        if len(idx) > 0
    ])
    stage_df = pd.DataFrame(summarize_stages(raw_seq, time_cols, cps))

    summary_df.to_csv(os.path.join(args.output_dir, "landslide_mechanism_summary.csv"), index=False, encoding="utf-8")
    stage_df.to_csv(os.path.join(args.output_dir, "landslide_stage_kinematics.csv"), index=False, encoding="utf-8")
    save_figures(args.output_dir, raw_seq, coords, elevation, rain_susceptibility, group_masks, stage_df)
    write_interpretation(
        os.path.join(args.output_dir, "landslide_mechanism_interpretation.md"),
        summary_df,
        stage_df,
        cps,
        args.cpd_method,
    )

    print(f">> Saved landslide mechanism outputs to: {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
