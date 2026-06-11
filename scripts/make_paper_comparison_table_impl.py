from pathlib import Path

import numpy as np
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
F_SUB = font(23)
F_HEAD = font(20, True)
F_TEXT = font(18)
F_SMALL = font(15)

INK = (31, 39, 52)
MUTED = (84, 96, 114)
GRID = (218, 225, 236)
BLUE = (42, 103, 196)
GREEN = (33, 145, 95)
RED = (204, 70, 58)
ORANGE = (220, 130, 46)
BG = (247, 250, 253)


def text(d, xy, s, fnt=F_TEXT, fill=INK, anchor=None):
    d.text(xy, str(s), font=fnt, fill=fill, anchor=anchor)


def metric_arrays():
    pred = np.load(ROOT / "output" / "y_test_pred_real.npy")[:, :, :, 0]
    true = np.load(ROOT / "output" / "y_test_true_real.npy")[:, :, :, 0]
    rows = []
    for h in range(pred.shape[1]):
        e = pred[:, h] - true[:, h]
        rows.append(
            {
                "h": h + 1,
                "days": (h + 1) * 12,
                "rmse": float(np.sqrt(np.mean(e * e))),
                "mae": float(np.mean(np.abs(e))),
            }
        )
    e = pred - true
    overall = {
        "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))),
    }
    return rows, overall


def main():
    rows, overall = metric_arrays()
    paper_rmse = 5.05
    paper_mae = 2.34

    w, h = 1700, 980
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    text(d, (w / 2, 42), "不同预测步长与原 STGCN 均值指标对比", F_TITLE, INK, "ma")
    text(d, (w / 2, 96), "原模型公开指标作为均值基准；当前模型按 H+1 至 H+5 分步对比", F_SUB, MUTED, "ma")

    # Table.
    table_x, table_y = 75, 165
    col_w = [110, 130, 145, 155, 145, 145, 155, 145, 250]
    row_h = 68
    table_w = sum(col_w)
    header_h = 86
    group_h = 38
    sub_h = header_h - group_h

    d.rounded_rectangle([table_x, table_y, table_x + table_w, table_y + header_h + row_h * len(rows)], radius=14, fill="white", outline=GRID, width=2)
    d.rectangle([table_x, table_y, table_x + table_w, table_y + header_h], fill=(232, 238, 248), outline=GRID)

    # Two-level header: fixed columns span both rows, metric groups span three subcolumns.
    x_edges = [table_x]
    for cw in col_w:
        x_edges.append(x_edges[-1] + cw)

    merged_cols = [(0, "步长"), (1, "时间"), (8, "判断")]
    for c, label in merged_cols:
        x0, x1 = x_edges[c], x_edges[c + 1]
        d.rectangle([x0, table_y, x1, table_y + header_h], fill=(232, 238, 248), outline=GRID)
        text(d, ((x0 + x1) / 2, table_y + header_h / 2 - 12), label, F_HEAD, INK, "ma")

    span_cols = [
        (2, 5, "RMSE / mm"),
        (5, 8, "MAE / mm"),
    ]
    for c0, c1, label in span_cols:
        x0, x1 = x_edges[c0], x_edges[c1]
        d.rectangle([x0, table_y, x1, table_y + group_h], fill=(232, 238, 248), outline=GRID)
        text(d, ((x0 + x1) / 2, table_y + 10), label, F_HEAD, INK, "ma")

    subheaders = {2: "本模型", 3: "原模型均值", 4: "差值", 5: "本模型", 6: "原模型均值", 7: "差值"}
    for i, label in subheaders.items():
        x0, x1 = x_edges[i], x_edges[i + 1]
        d.rectangle([x0, table_y + group_h, x1, table_y + header_h], fill=(241, 245, 251), outline=GRID)
        text(d, ((x0 + x1) / 2, table_y + group_h + 14), label, F_TEXT, INK, "ma")

    y = table_y + header_h
    for idx, r in enumerate(rows):
        fill = (255, 255, 255) if idx % 2 == 0 else (250, 252, 255)
        d.rectangle([table_x, y, table_x + table_w, y + row_h], fill=fill, outline=GRID)
        rmse_better = r["rmse"] < paper_rmse
        mae_better = r["mae"] < paper_mae
        judgment = "双指标优于原论文" if rmse_better and mae_better else "RMSE 更优，MAE 变差"
        rmse_delta = r["rmse"] - paper_rmse
        mae_delta = r["mae"] - paper_mae
        vals = [
            f"H+{r['h']}",
            f"约 {r['days']} 天",
            f"{r['rmse']:.2f}",
            f"{paper_rmse:.2f}",
            f"{rmse_delta:+.2f}",
            f"{r['mae']:.2f}",
            f"{paper_mae:.2f}",
            f"{mae_delta:+.2f}",
            judgment,
        ]
        colors = [
            INK,
            INK,
            INK,
            MUTED,
            GREEN if rmse_better else RED,
            INK,
            MUTED,
            GREEN if mae_better else RED,
            GREEN if mae_better else ORANGE,
        ]
        x = table_x
        for val, cw, col in zip(vals, col_w, colors):
            text(d, (x + cw / 2, y + 22), val, F_TEXT, col, "ma")
            x += cw
        y += row_h

    # Vertical separator lines.
    x = table_x
    for cw in col_w[:-1]:
        x += cw
        d.line([(x, table_y), (x, table_y + header_h + row_h * len(rows))], fill=GRID, width=2)

    # Bottom note.
    note_y = 770
    d.rounded_rectangle([150, note_y, 1550, note_y + 112], radius=18, fill=(255, 250, 238), outline=(238, 196, 120), width=2)
    text(d, (w / 2, note_y + 18), "汇报表述建议", F_HEAD, ORANGE, "ma")
    text(d, (w / 2, note_y + 52), "当前模型在约 36 天以内的短中期预测中，RMSE 与 MAE 均优于原论文；", F_TEXT, INK, "ma")
    text(d, (w / 2, note_y + 82), "当预测步长达到约 48 天及以上时，MAE 开始劣于原论文，说明长期多步预测仍有误差累积。", F_TEXT, INK, "ma")

    text(d, (w - 88, 940), "注：原论文 RMSE 由 MSE=25.51 近似换算，MAE=2.34；两者并非完全同口径复现。", F_SMALL, MUTED, "ra")

    out = OUT / "fig9_paper_horizon_comparison_table.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
