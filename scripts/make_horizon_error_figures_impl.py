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
F_AXIS = font(20)
F_SMALL = font(17)
F_TINY = font(14)

INK = (32, 39, 52)
MUTED = (86, 96, 112)
GRID = (222, 227, 235)
BLUE = (39, 103, 195)
GREEN = (32, 145, 95)
RED = (206, 70, 57)
ORANGE = (224, 128, 48)
GREY = (160, 170, 184)


def interp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def text(draw, xy, s, fnt=F_SMALL, fill=INK, anchor=None):
    draw.text(xy, str(s), font=fnt, fill=fill, anchor=anchor)


def nice_ticks(vmin, vmax, n=5):
    raw = (vmax - vmin) / max(1, n - 1)
    if raw <= 0:
        return [vmin]
    mag = 10 ** math.floor(math.log10(raw))
    step = raw / mag
    step = 1 if step <= 1 else 2 if step <= 2 else 5 if step <= 5 else 10
    step *= mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    vals = []
    x = start
    while x <= end + step * 0.5:
        vals.append(x)
        x += step
    return vals


def draw_line_chart(draw, box, title, y_label, series):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=(248, 250, 253), outline=(218, 224, 235), width=1)
    text(draw, ((x0 + x1) / 2, y0 + 24), title, fnt=F_SUB, anchor="ma")
    pl, pt, pr, pb = x0 + 85, y0 + 82, x1 - 45, y1 - 82
    all_vals = np.concatenate([np.asarray(v["values"], float) for v in series])
    ymin = 0.0
    ymax = float(all_vals.max()) * 1.18

    for tick in nice_ticks(ymin, ymax, 6):
        y = pb - (tick - ymin) / (ymax - ymin) * (pb - pt)
        draw.line([(pl, y), (pr, y)], fill=GRID, width=1)
        text(draw, (pl - 14, y), f"{tick:.1f}", fnt=F_TINY, fill=MUTED, anchor="rm")
    draw.line([(pl, pt), (pl, pb), (pr, pb)], fill=(92, 108, 132), width=2)
    text(draw, (pl, pt - 28), y_label, fnt=F_SMALL, fill=MUTED, anchor="la")

    xs = [pl + i / 4 * (pr - pl) for i in range(5)]
    labels = [f"H+{i}" for i in range(1, 6)]
    for x, lab in zip(xs, labels):
        draw.line([(x, pb), (x, pb + 8)], fill=(92, 108, 132), width=1)
        text(draw, (x, pb + 18), lab, fnt=F_SMALL, fill=MUTED, anchor="ma")

    for sidx, item in enumerate(series):
        vals = np.asarray(item["values"], float)
        pts = [(xs[i], pb - (vals[i] - ymin) / (ymax - ymin) * (pb - pt)) for i in range(5)]
        draw.line(pts, fill=item["color"], width=4)
        for i, (x, y) in enumerate(pts):
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=item["color"], outline="white", width=2)
            offset = -28 if sidx == 0 else 18
            text(draw, (x, y + offset), f"{vals[i]:.2f}", fnt=F_TINY, fill=item["color"], anchor="ma")

    lx, ly = pr - 330, y0 + 38
    for i, item in enumerate(series):
        yy = ly + i * 28
        draw.line([(lx, yy), (lx + 44, yy)], fill=item["color"], width=5)
        text(draw, (lx + 58, yy - 11), item["name"], fnt=F_TINY, fill=INK)


def make_horizon_trend():
    base = ROOT / "output"
    pred = np.load(base / "y_test_pred_real.npy")[:, :, :, 0]
    true = np.load(base / "y_test_true_real.npy")[:, :, :, 0]
    persist = np.load(base / "y_test_persistence_real.npy")[:, :, :, 0]

    model_rmse, model_mae, per_rmse, per_mae = [], [], [], []
    for h in range(pred.shape[1]):
        e = pred[:, h] - true[:, h]
        pe = persist[:, h] - true[:, h]
        model_rmse.append(float(np.sqrt(np.mean(e * e))))
        model_mae.append(float(np.mean(np.abs(e))))
        per_rmse.append(float(np.sqrt(np.mean(pe * pe))))
        per_mae.append(float(np.mean(np.abs(pe))))

    w, h = 1800, 960
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    text(d, (w / 2, 42), "多步预测误差随预测步长变化", fnt=F_TITLE, anchor="ma")
    text(d, (w / 2, 96), "短期预测误差较低，随着步长增加误差逐步累积", fnt=F_SUB, fill=MUTED, anchor="ma")

    draw_line_chart(
        d,
        (70, 145, 1730, 500),
        "RMSE 趋势",
        "RMSE / mm",
        [
            {"name": "CPD-STGCN", "values": model_rmse, "color": BLUE},
            {"name": "Persistence", "values": per_rmse, "color": GREY},
        ],
    )
    draw_line_chart(
        d,
        (70, 560, 1730, 900),
        "MAE 趋势",
        "MAE / mm",
        [
            {"name": "CPD-STGCN", "values": model_mae, "color": GREEN},
            {"name": "Persistence", "values": per_mae, "color": GREY},
        ],
    )

    h1_gain = (per_mae[0] - model_mae[0]) / per_mae[0] * 100
    h5_gain = (per_mae[-1] - model_mae[-1]) / per_mae[-1] * 100
    text(d, (w / 2, 928), f"MAE 相对 persistence 提升：H+1 = {h1_gain:.1f}%，H+5 = {h5_gain:.1f}%；远期预测仍存在误差累积。", fnt=F_SMALL, fill=INK, anchor="ma")
    out = OUT / "fig6_horizon_error_trend.png"
    img.save(out)
    print(out)


def heat_color(v, vmax):
    t = max(0.0, min(1.0, v / vmax if vmax else 0.0))
    stops = [
        (0.00, (248, 250, 252)),
        (0.35, (255, 219, 140)),
        (0.70, (242, 132, 75)),
        (1.00, (174, 42, 54)),
    ]
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if p0 <= t <= p1:
            return interp(c0, c1, (t - p0) / (p1 - p0))
    return stops[-1][1]


def delta_color(v, negmax, posmax):
    if v >= 0:
        return heat_color(v, posmax)
    t = max(0.0, min(1.0, abs(v) / negmax if negmax else 0.0))
    return interp((248, 250, 252), (62, 150, 112), t)


def draw_colorbar(draw, box, vmin, vmax, label, mode="heat"):
    x0, y0, x1, y1 = box
    for i in range(x1 - x0):
        v = vmin + (vmax - vmin) * i / max(1, x1 - x0 - 1)
        if mode == "delta":
            col = delta_color(v, abs(vmin), vmax)
        else:
            col = heat_color(v, vmax)
        draw.line([(x0 + i, y0), (x0 + i, y1)], fill=col)
    draw.rectangle(box, outline=(120, 134, 154), width=1)
    text(draw, (x0, y1 + 8), f"{vmin:.1f}", fnt=F_TINY, fill=MUTED, anchor="la")
    text(draw, (x1, y1 + 8), f"{vmax:.1f}", fnt=F_TINY, fill=MUTED, anchor="ra")
    text(draw, ((x0 + x1) / 2, y0 - 24), label, fnt=F_TINY, fill=MUTED, anchor="ma")


def map_points(coords_x, coords_y, values, panel):
    x0, y0, x1, y1 = panel
    xmin, xmax = float(coords_x.min()), float(coords_x.max())
    ymin, ymax = float(coords_y.min()), float(coords_y.max())
    pad_x = (xmax - xmin) * 0.06
    pad_y = (ymax - ymin) * 0.06
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    sx = (x1 - x0) / (xmax - xmin)
    sy = (y1 - y0) / (ymax - ymin)
    return [(x0 + (x - xmin) * sx, y1 - (y - ymin) * sy, v) for x, y, v in zip(coords_x, coords_y, values)]


def make_error_amplification_map():
    base = ROOT / "output"
    pred = np.load(base / "y_test_pred_real.npy")[:, :, :, 0]
    true = np.load(base / "y_test_true_real.npy")[:, :, :, 0]
    df = pd.read_csv(ROOT / "dataset" / "inter228_5241.csv", usecols=["i", "j"])
    x = df["i"].to_numpy(float)
    y = df["j"].to_numpy(float)

    # Average over all test windows to make the map stable instead of cherry-picking one window.
    err_h1 = np.mean(np.abs(pred[:, 0] - true[:, 0]), axis=0)
    err_h5 = np.mean(np.abs(pred[:, 4] - true[:, 4]), axis=0)
    delta = err_h5 - err_h1
    emax = float(np.percentile(np.r_[err_h1, err_h5], 98))
    dmin = float(np.percentile(delta, 2))
    dmax = float(np.percentile(delta, 98))

    w, h = 1900, 1020
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    text(d, (w / 2, 42), "预测步长增加导致的空间误差放大", fnt=F_TITLE, anchor="ma")
    text(d, (w / 2, 100), "对全部测试窗口取平均：H+1 误差、H+5 误差及误差增量", fnt=F_SUB, fill=MUTED, anchor="ma")

    panels = [
        (90, 205, 610, 775, "H+1 平均绝对误差", err_h1, "heat"),
        (690, 205, 1210, 775, "H+5 平均绝对误差", err_h5, "heat"),
        (1290, 205, 1810, 775, "误差增量 H+5 - H+1", delta, "delta"),
    ]
    for x0, y0, x1, y1, title, vals, mode in panels:
        d.rounded_rectangle([x0 - 16, y0 - 48, x1 + 16, y1 + 22], radius=10, fill=(248, 250, 253), outline=(218, 224, 235), width=1)
        text(d, ((x0 + x1) / 2, y0 - 34), title, fnt=F_SUB, anchor="ma")
        d.rectangle([x0, y0, x1, y1], fill=(252, 253, 255), outline=(130, 146, 166), width=2)
        pts = map_points(x, y, vals, (x0, y0, x1, y1))
        for px, py, v in pts:
            if mode == "delta":
                col = delta_color(float(v), abs(dmin), dmax)
            else:
                col = heat_color(float(min(emax, v)), emax)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)

    draw_colorbar(d, (260, 845, 1040, 868), 0, emax, "平均绝对误差 / mm", "heat")
    draw_colorbar(d, (1325, 845, 1715, 868), dmin, dmax, "误差增量 / mm", "delta")
    text(d, (w / 2, 925), f"平均 MAE：H+1 = {float(err_h1.mean()):.3f} mm，H+5 = {float(err_h5.mean()):.3f} mm，增量 = {float(delta.mean()):.3f} mm", fnt=F_AXIS, fill=INK, anchor="ma")
    text(d, (w - 92, 980), "橙红色表示 H+5 相对 H+1 的误差放大区域", fnt=F_TINY, fill=MUTED, anchor="ra")

    out = OUT / "fig7_h1_h5_error_amplification_map.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    make_horizon_trend()
    make_error_amplification_map()
