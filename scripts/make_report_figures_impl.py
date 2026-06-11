from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(24)
F_AXIS = font(22)
F_SMALL = font(18)
F_TINY = font(16)


def text(draw, xy, s, fill=(30, 35, 45), fnt=F_SMALL, anchor=None):
    draw.text(xy, str(s), fill=fill, font=fnt, anchor=anchor)


def draw_polyline(draw, pts, color, width=4):
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")


def nice_ticks(vmin, vmax, n=6):
    if vmin == vmax:
        return [vmin]
    raw = (vmax - vmin) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(abs(raw)))
    step = raw / mag
    if step <= 1:
        step = 1
    elif step <= 2:
        step = 2
    elif step <= 5:
        step = 5
    else:
        step = 10
    step *= mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    vals = []
    x = start
    while x <= end + step * 0.5:
        vals.append(x)
        x += step
    return vals


def fig1():
    data_path = ROOT / "dataset" / "inter228_5241.csv"
    df = pd.read_csv(data_path)
    date_cols = list(df.columns[4:])
    arr = df.iloc[:, 4:].to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    active_idx = np.argsort(arr.std(axis=1))[-max(1, int(arr.shape[0] * 0.2)) :]
    active_mean = arr[active_idx].mean(axis=0)
    cpd = [35, 80, 135, 160, 180]

    w, h = 1700, 960
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    margin_l, margin_r, margin_t, margin_b = 125, 80, 150, 150
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = w - margin_r, h - margin_b
    plot_w, plot_h = plot_r - plot_l, plot_b - plot_t

    text(d, (w // 2, 42), "Sela 滑坡 InSAR 平均位移序列与 CPD 阶段划分", fnt=F_TITLE, anchor="ma")
    text(
        d,
        (w // 2, 96),
        "变化点用于识别滑坡变形阶段转换，为 CPD-guided STGCN 提供阶段先验",
        fill=(80, 86, 96),
        fnt=F_SUB,
        anchor="ma",
    )

    # Stage shading.
    bounds = [0] + cpd + [len(date_cols) - 1]
    stage_colors = [(246, 248, 252), (238, 244, 255), (247, 244, 235), (238, 250, 244), (252, 242, 242), (244, 241, 252)]
    for i in range(len(bounds) - 1):
        x0 = plot_l + bounds[i] / (len(date_cols) - 1) * plot_w
        x1 = plot_l + bounds[i + 1] / (len(date_cols) - 1) * plot_w
        d.rectangle([x0, plot_t, x1, plot_b], fill=stage_colors[i % len(stage_colors)])
        text(d, ((x0 + x1) / 2, plot_t + 18), f"Stage {i}", fill=(95, 100, 112), fnt=F_TINY, anchor="ma")

    ymin = min(float(mean.min()), float(active_mean.min()))
    ymax = max(float(mean.max()), float(active_mean.max()))
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    def xmap(i):
        return plot_l + i / (len(date_cols) - 1) * plot_w

    def ymap(v):
        return plot_b - (v - ymin) / (ymax - ymin) * plot_h

    # Grid and y labels.
    for yv in nice_ticks(ymin, ymax, 7):
        y = ymap(yv)
        d.line([(plot_l, y), (plot_r, y)], fill=(220, 224, 230), width=1)
        text(d, (plot_l - 18, y), f"{yv:.0f}", fill=(80, 86, 96), fnt=F_SMALL, anchor="rm")

    # Axes.
    d.line([(plot_l, plot_t), (plot_l, plot_b), (plot_r, plot_b)], fill=(70, 76, 88), width=2)
    text(d, (plot_l, plot_t - 42), "位移 / mm", fill=(70, 76, 88), fnt=F_AXIS, anchor="la")
    text(d, ((plot_l + plot_r) / 2, h - 65), "时间观测序列", fill=(70, 76, 88), fnt=F_AXIS, anchor="ma")

    # X year labels.
    years = {}
    for i, ds in enumerate(date_cols):
        year = ds.split("/")[0]
        years.setdefault(year, i)
    for year, i in years.items():
        if int(year) % 2 == 1:
            continue
        x = xmap(i)
        d.line([(x, plot_b), (x, plot_b + 8)], fill=(70, 76, 88), width=2)
        text(d, (x, plot_b + 18), year, fill=(80, 86, 96), fnt=F_TINY, anchor="ma")

    # CPD lines.
    for cp in cpd:
        x = xmap(cp)
        d.line([(x, plot_t), (x, plot_b)], fill=(202, 70, 56), width=3)
        text(d, (x + 6, plot_t + 50), f"CP {cp}", fill=(165, 45, 35), fnt=F_TINY)

    mean_pts = [(xmap(i), ymap(v)) for i, v in enumerate(mean)]
    active_pts = [(xmap(i), ymap(v)) for i, v in enumerate(active_mean)]
    draw_polyline(d, active_pts, (36, 105, 196), 5)
    draw_polyline(d, mean_pts, (32, 145, 95), 5)

    # Legend.
    lx, ly = plot_l + 30, plot_b - 105
    d.rounded_rectangle([lx - 18, ly - 18, lx + 560, ly + 78], radius=12, fill=(255, 255, 255), outline=(218, 223, 232), width=1)
    d.line([(lx, ly), (lx + 55, ly)], fill=(32, 145, 95), width=5)
    text(d, (lx + 70, ly - 12), "全体节点平均位移", fnt=F_SMALL)
    d.line([(lx, ly + 42), (lx + 55, ly + 42)], fill=(36, 105, 196), width=5)
    text(d, (lx + 70, ly + 30), "高活动节点均值（前 20%）", fnt=F_SMALL)
    d.line([(lx + 330, ly), (lx + 385, ly)], fill=(202, 70, 56), width=4)
    text(d, (lx + 400, ly - 12), "CPD 变化点", fnt=F_SMALL)

    text(d, (plot_r, h - 32), "Data: inter228_5241.csv; CPD points: [35, 80, 135, 160, 180]", fill=(112, 118, 128), fnt=F_TINY, anchor="ra")
    img.save(OUT / "fig1_cpd_displacement_stages.png")


def fig2():
    result_path = ROOT / "output" / "cpd_split_validation_full" / "cpd_split_results.csv"
    df = pd.read_csv(result_path)
    folds = df["fold"].tolist()
    model_rmse = df["model_rmse"].astype(float).to_numpy()
    base_rmse = df["persistence_rmse"].astype(float).to_numpy()
    model_mae = df["model_mae"].astype(float).to_numpy()
    base_mae = df["persistence_mae"].astype(float).to_numpy()
    rmse_gain = df["rmse_gain_pct"].astype(float).to_numpy()
    mae_gain = df["mae_gain_pct"].astype(float).to_numpy()

    w, h = 1700, 980
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    text(d, (w // 2, 42), "CPD-aware 验证下模型误差对比", fnt=F_TITLE, anchor="ma")
    text(d, (w // 2, 96), "模型在多阶段划分与变化点后滚动验证中均优于 persistence 基线", fill=(80, 86, 96), fnt=F_SUB, anchor="ma")

    def panel(x0, y0, x1, y1, title, model, base, gains, ylabel):
        d.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(250, 251, 253), outline=(224, 228, 236), width=1)
        text(d, ((x0 + x1) / 2, y0 + 26), title, fnt=F_AXIS, anchor="ma")
        pl, pt, pr, pb = x0 + 80, y0 + 90, x1 - 45, y1 - 105
        ymax = max(float(base.max()), float(model.max())) * 1.22
        for yv in nice_ticks(0, ymax, 5):
            y = pb - yv / ymax * (pb - pt)
            d.line([(pl, y), (pr, y)], fill=(224, 228, 236), width=1)
            text(d, (pl - 16, y), f"{yv:.1f}", fill=(80, 86, 96), fnt=F_TINY, anchor="rm")
        d.line([(pl, pt), (pl, pb), (pr, pb)], fill=(70, 76, 88), width=2)
        text(d, (x0 + 28, (pt + pb) / 2), ylabel, fill=(70, 76, 88), fnt=F_SMALL, anchor="mm")

        n = len(folds)
        group_w = (pr - pl) / n
        bar_w = group_w * 0.26
        for i, fold in enumerate(folds):
            cx = pl + group_w * (i + 0.5)
            b1 = base[i] / ymax * (pb - pt)
            b2 = model[i] / ymax * (pb - pt)
            x_base0, x_base1 = cx - bar_w - 5, cx - 5
            x_mod0, x_mod1 = cx + 5, cx + bar_w + 5
            d.rectangle([x_base0, pb - b1, x_base1, pb], fill=(175, 181, 192))
            d.rectangle([x_mod0, pb - b2, x_mod1, pb], fill=(36, 105, 196))
            text(d, ((x_base0 + x_base1) / 2, pb - b1 - 24), f"{base[i]:.2f}", fill=(90, 96, 108), fnt=F_TINY, anchor="ma")
            text(d, ((x_mod0 + x_mod1) / 2, pb - b2 - 24), f"{model[i]:.2f}", fill=(36, 105, 196), fnt=F_TINY, anchor="ma")
            text(d, (cx, pb + 20), fold.replace("stratified_stage", "stratified"), fill=(55, 61, 72), fnt=F_TINY, anchor="ma")
            text(d, (cx, pb + 48), f"提升 {gains[i]:.1f}%", fill=(32, 145, 95), fnt=F_TINY, anchor="ma")

        lx, ly = pr - 390, y0 + 38
        d.rectangle([lx, ly, lx + 24, ly + 16], fill=(175, 181, 192))
        text(d, (lx + 34, ly - 4), "Persistence", fnt=F_TINY)
        d.rectangle([lx + 170, ly, lx + 194, ly + 16], fill=(36, 105, 196))
        text(d, (lx + 204, ly - 4), "CPD-STGCN", fnt=F_TINY)

    panel(70, 145, 1630, 530, "RMSE 对比", model_rmse, base_rmse, rmse_gain, "RMSE / mm")
    panel(70, 575, 1630, 915, "MAE 对比", model_mae, base_mae, mae_gain, "MAE / mm")
    text(d, (w - 80, h - 30), "Source: output/cpd_split_validation_full/cpd_split_results.csv", fill=(112, 118, 128), fnt=F_TINY, anchor="ra")
    img.save(OUT / "fig2_cpd_validation_performance.png")


if __name__ == "__main__":
    fig1()
    fig2()
    print(OUT / "fig1_cpd_displacement_stages.png")
    print(OUT / "fig2_cpd_validation_performance.png")
