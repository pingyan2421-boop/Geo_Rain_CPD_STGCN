from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
DISPLAY_HORIZON = 0


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
MUTED = (90, 99, 115)
GRID = (222, 227, 235)
BLUE = (39, 103, 195)
GREEN = (31, 145, 95)
RED = (208, 72, 58)


def interp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def displacement_color(v, vmin, vmax):
    # Negative displacement field: blue for large cumulative deformation, light for weak deformation.
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    stops = [
        (0.00, (24, 67, 145)),
        (0.35, (44, 126, 196)),
        (0.65, (128, 205, 193)),
        (1.00, (245, 241, 188)),
    ]
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if p0 <= t <= p1:
            return interp(c0, c1, (t - p0) / (p1 - p0))
    return stops[-1][1]


def error_color(v, vmax):
    t = max(0.0, min(1.0, v / vmax if vmax else 0.0))
    stops = [
        (0.00, (248, 250, 252)),
        (0.35, (254, 219, 140)),
        (0.70, (242, 132, 75)),
        (1.00, (177, 45, 48)),
    ]
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if p0 <= t <= p1:
            return interp(c0, c1, (t - p0) / (p1 - p0))
    return stops[-1][1]


def text(draw, xy, s, fnt=F_SMALL, fill=INK, anchor=None):
    draw.text(xy, str(s), font=fnt, fill=fill, anchor=anchor)


def draw_colorbar(draw, box, vmin, vmax, label, mode="disp"):
    x0, y0, x1, y1 = box
    for i in range(x1 - x0):
        v = vmin + (vmax - vmin) * i / max(1, x1 - x0 - 1)
        col = displacement_color(v, vmin, vmax) if mode == "disp" else error_color(v, vmax)
        draw.line([(x0 + i, y0), (x0 + i, y1)], fill=col)
    draw.rectangle(box, outline=(120, 134, 154), width=1)
    text(draw, (x0, y1 + 8), f"{vmin:.1f}", fnt=F_TINY, fill=MUTED, anchor="la")
    text(draw, (x1, y1 + 8), f"{vmax:.1f}", fnt=F_TINY, fill=MUTED, anchor="ra")
    text(draw, ((x0 + x1) / 2, y0 - 24), label, fnt=F_TINY, fill=MUTED, anchor="ma")


def map_points(coords_x, coords_y, values, panel, color_fn, radius=3):
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
    pts = []
    for x, y, v in zip(coords_x, coords_y, values):
        px = x0 + (x - xmin) * sx
        py = y1 - (y - ymin) * sy
        pts.append((px, py, v))
    return pts


def make_spatial_triptych():
    pred = np.load(ROOT / "output" / "y_test_pred_real.npy")[:, :, :, 0]
    true = np.load(ROOT / "output" / "y_test_true_real.npy")[:, :, :, 0]
    dataset = pd.read_csv(ROOT / "dataset" / "inter228_5241.csv", usecols=["i", "j", "h"])
    x = dataset["i"].to_numpy(float)
    y = dataset["j"].to_numpy(float)

    sample = pred.shape[0] - 1
    horizon = DISPLAY_HORIZON
    t = true[sample, horizon]
    p = pred[sample, horizon]
    err = np.abs(p - t)

    vmin = float(np.percentile(np.r_[t, p], 1))
    vmax = float(np.percentile(np.r_[t, p], 99))
    emax = float(np.percentile(err, 98))

    w, h = 1900, 1020
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    text(d, (w / 2, 42), f"测试集末窗口 H+{horizon + 1} 位移空间预测结果", fnt=F_TITLE, anchor="ma")
    text(d, (w / 2, 104), "真实位移、模型预测与绝对误差空间分布", fnt=F_SUB, fill=MUTED, anchor="ma")

    panels = [
        (90, 205, 610, 775, "真实位移"),
        (690, 205, 1210, 775, "模型预测"),
        (1290, 205, 1810, 775, "绝对误差"),
    ]
    values = [t, p, err]
    for idx, (panel, vals) in enumerate(zip(panels, values)):
        x0, y0, x1, y1, title = panel
        d.rounded_rectangle([x0 - 16, y0 - 48, x1 + 16, y1 + 22], radius=10, fill=(248, 250, 253), outline=(218, 224, 235), width=1)
        text(d, ((x0 + x1) / 2, y0 - 34), title, fnt=F_SUB, anchor="ma")
        d.rectangle([x0, y0, x1, y1], fill=(252, 253, 255), outline=(130, 146, 166), width=2)
        pts = map_points(x, y, vals, (x0, y0, x1, y1), None)
        for px, py, v in pts:
            if idx < 2:
                col = displacement_color(float(max(vmin, min(vmax, v))), vmin, vmax)
            else:
                col = error_color(float(min(emax, v)), emax)
            r = 2 if idx < 2 else 3
            d.ellipse([px - r, py - r, px + r, py + r], fill=col)

    draw_colorbar(d, (265, 845, 1035, 868), vmin, vmax, "位移 / mm", mode="disp")
    draw_colorbar(d, (1325, 845, 1715, 868), 0, emax, "绝对误差 / mm", mode="err")
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(err))
    text(d, (w / 2, 925), f"该窗口指标：RMSE = {rmse:.3f} mm，MAE = {mae:.3f} mm", fnt=F_AXIS, fill=INK, anchor="ma")
    text(d, (w - 92, 980), "Data: y_test_pred_real.npy / y_test_true_real.npy", fnt=F_TINY, fill=MUTED, anchor="ra")

    out = OUT / f"fig4_prediction_spatial_h{horizon + 1}_triptych.png"
    img.save(out)
    print(out)


def nice_ticks(vmin, vmax, n=5):
    raw = (vmax - vmin) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(abs(raw))) if raw else 1
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


def make_node_series():
    pred = np.load(ROOT / "output" / "y_test_pred_real.npy")[:, :, :, 0]
    true = np.load(ROOT / "output" / "y_test_true_real.npy")[:, :, :, 0]
    dataset = pd.read_csv(ROOT / "dataset" / "inter228_5241.csv")
    coords = dataset[["i", "j", "h"]].to_numpy(float)
    arr = dataset.iloc[:, 4:].to_numpy(float)
    activity = arr.std(axis=1)
    err_node = np.mean(np.abs(pred - true), axis=(0, 1))

    order = np.argsort(activity)
    low_band = order[int(len(order) * 0.15) : int(len(order) * 0.35)]
    mid_band = order[int(len(order) * 0.45) : int(len(order) * 0.65)]
    high_band = order[int(len(order) * 0.85) :]
    node_a = int(high_band[np.argmin(err_node[high_band])])
    node_b = int(mid_band[np.argmin(err_node[mid_band])])
    node_c = int(low_band[np.argmin(err_node[low_band])])
    nodes = [node_a, node_b, node_c]
    names = ["典型监测点 A（高活动区）", "典型监测点 B（中活动区）", "典型监测点 C（低活动区）"]

    horizon = DISPLAY_HORIZON
    y_true = [true[:, horizon, n] for n in nodes]
    y_pred = [pred[:, horizon, n] for n in nodes]
    vals = np.r_[tuple(y_true + y_pred)]
    ymin, ymax = float(vals.min()), float(vals.max())
    pad = (ymax - ymin) * 0.12
    ymin -= pad
    ymax += pad

    w, h = 1800, 1060
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    text(d, (w / 2, 42), f"典型监测点 H+{horizon + 1} 位移预测时间序列", fnt=F_TITLE, anchor="ma")
    text(d, (w / 2, 92), "真实值与模型预测值对比，展示模型对测试阶段位移趋势的跟踪能力", fnt=F_SUB, fill=MUTED, anchor="ma")

    plot_l, plot_t, plot_r, plot_b = 135, 160, 1690, 735
    d.rectangle([plot_l, plot_t, plot_r, plot_b], fill=(252, 253, 255), outline=(100, 116, 138), width=2)

    def xmap(i):
        return plot_l + i / (pred.shape[0] - 1) * (plot_r - plot_l)

    def ymap(v):
        return plot_b - (v - ymin) / (ymax - ymin) * (plot_b - plot_t)

    for tick in nice_ticks(ymin, ymax, 7):
        y = ymap(tick)
        d.line([(plot_l, y), (plot_r, y)], fill=GRID, width=1)
        text(d, (plot_l - 16, y), f"{tick:.0f}", fnt=F_SMALL, fill=MUTED, anchor="rm")

    colors = [BLUE, GREEN, RED]
    for idx, (n, name, tv, pv, col) in enumerate(zip(nodes, names, y_true, y_pred, colors)):
        true_pts = [(xmap(i), ymap(v)) for i, v in enumerate(tv)]
        pred_pts = [(xmap(i), ymap(v)) for i, v in enumerate(pv)]
        d.line(true_pts, fill=col, width=4)
        d.line(pred_pts, fill=interp(col, (255, 255, 255), 0.45), width=4)
        for i in range(0, len(tv), 4):
            x, y = true_pts[i]
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        mae = float(np.mean(np.abs(tv - pv)))
        lx, ly = plot_l + 40 + idx * 500, 830
        d.rounded_rectangle([lx - 16, ly - 16, lx + 430, ly + 90], radius=10, fill=(248, 250, 253), outline=(218, 224, 235), width=1)
        d.line([(lx, ly), (lx + 52, ly)], fill=col, width=5)
        d.line([(lx, ly + 28), (lx + 52, ly + 28)], fill=interp(col, (255, 255, 255), 0.45), width=5)
        text(d, (lx + 65, ly - 13), name, fnt=F_SMALL)
        text(d, (lx + 65, ly + 15), f"位置({coords[n,0]:.0f}, {coords[n,1]:.0f})  MAE={mae:.2f} mm", fnt=F_TINY, fill=MUTED)
        text(d, (lx, ly + 55), "深色：真实值    浅色：预测值", fnt=F_TINY, fill=MUTED)

    for i in range(0, pred.shape[0], 5):
        x = xmap(i)
        d.line([(x, plot_b), (x, plot_b + 8)], fill=(100, 116, 138), width=1)
        text(d, (x, plot_b + 18), f"{i+1}", fnt=F_TINY, fill=MUTED, anchor="ma")
    text(d, ((plot_l + plot_r) / 2, 785), "测试窗口序号", fnt=F_AXIS, fill=INK, anchor="ma")
    text(d, (plot_l, plot_t - 36), "位移 / mm", fnt=F_AXIS, fill=INK, anchor="la")
    text(d, (w - 92, 1020), f"Source: fixed chronological test, horizon H+{horizon + 1}", fnt=F_TINY, fill=MUTED, anchor="ra")

    out = OUT / f"fig5_prediction_node_series_h{horizon + 1}.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    make_spatial_triptych()
    make_node_series()
