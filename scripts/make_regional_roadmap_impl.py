from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(24)
F_HEAD = font(23, True)
F_TEXT = font(18)
F_SMALL = font(15)

INK = (31, 39, 52)
MUTED = (84, 96, 114)
LINE = (138, 154, 176)
BG = (247, 250, 253)
BLUE = (42, 103, 196)
GREEN = (34, 145, 95)
RED = (204, 70, 58)
ORANGE = (220, 130, 46)
PURPLE = (108, 86, 176)
CYAN = (65, 170, 185)


def wrap_lines(draw, text, fnt, max_w):
    lines = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            test = line + ch
            if draw.textlength(test, font=fnt) <= max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return lines


def draw_multiline(draw, box, text, fnt=F_TEXT, fill=INK, anchor_center=True, spacing=5):
    x0, y0, x1, y1 = box
    lines = wrap_lines(draw, text, fnt, x1 - x0 - 24)
    line_h = fnt.size + spacing
    total = len(lines) * line_h - spacing
    y = y0 + (y1 - y0 - total) / 2
    for line in lines:
        if anchor_center:
            draw.text(((x0 + x1) / 2, y), line, font=fnt, fill=fill, anchor="ma")
        else:
            draw.text((x0 + 14, y), line, font=fnt, fill=fill, anchor="la")
        y += line_h


def round_box(draw, box, title, body=None, color=BLUE, fill=(255, 255, 255), title_fill=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=color, width=2)
    draw.rounded_rectangle([x0, y0, x1, y0 + 42], radius=14, fill=color, outline=color, width=2)
    draw.rectangle([x0, y0 + 22, x1, y0 + 42], fill=color)
    draw.text(((x0 + x1) / 2, y0 + 10), title, font=F_HEAD, fill=title_fill or "white", anchor="ma")
    if body:
        draw_multiline(draw, [x0 + 8, y0 + 55, x1 - 8, y1 - 12], body, F_TEXT, INK)


def problem_box(draw, box, title, items, color):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=14, fill=(255, 255, 255), outline=color, width=2)
    draw.text((x0 + 18, y0 + 16), title, font=F_HEAD, fill=color, anchor="la")
    y = y0 + 58
    for item in items:
        draw.ellipse([x0 + 20, y + 5, x0 + 30, y + 15], fill=color)
        lines = wrap_lines(draw, item, F_TEXT, x1 - x0 - 58)
        for line in lines:
            draw.text((x0 + 42, y), line, font=F_TEXT, fill=INK, anchor="la")
            y += F_TEXT.size + 4
        y += 8


def arrow(draw, start, end, color=INK, width=4, head=15):
    x0, y0 = start
    x1, y1 = end
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    pts = [
        (x1, y1),
        (x1 - head * math.cos(ang - math.pi / 7), y1 - head * math.sin(ang - math.pi / 7)),
        (x1 - head * math.cos(ang + math.pi / 7), y1 - head * math.sin(ang + math.pi / 7)),
    ]
    draw.polygon(pts, fill=color)


def draw_region_icon(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(252, 253, 255), outline=LINE, width=2)
    # slope polygons
    polys = [
        [(x0+30,y1-45),(x0+135,y0+40),(x0+245,y1-50)],
        [(x0+190,y1-35),(x0+330,y0+65),(x0+455,y1-60)],
        [(x0+420,y1-40),(x0+535,y0+80),(x1-25,y1-55)],
    ]
    colors = [(212,232,222),(222,232,246),(238,226,214)]
    for p,c in zip(polys, colors):
        draw.polygon(p, fill=c, outline=(150,165,180))
    # points
    for i in range(70):
        px = x0 + 35 + (i * 71 % int(x1-x0-70))
        py = y0 + 38 + (i * 43 % int(y1-y0-76))
        col = BLUE if i % 5 else RED if i % 7 else GREEN
        draw.ellipse([px-3, py-3, px+3, py+3], fill=col)
    # blocks
    for bx in [x0+155, x0+360]:
        draw.line([(bx, y0+20), (bx, y1-20)], fill=(120,135,156), width=2)
    draw.text((x0 + 16, y0 + 14), "区域 InSAR 点云", font=F_SMALL, fill=MUTED, anchor="la")


def main():
    w, h = 1900, 1240
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((w / 2, 42), "区域尺度 CPD-STGCN 与降雨响应分析技术路线", font=F_TITLE, fill=INK, anchor="ma")
    d.text((w / 2, 98), "从单体滑坡预测扩展到区域滑坡群：分块处理、变化点响应、降雨滞后与区域预测", font=F_SUB, fill=MUTED, anchor="ma")

    # Top problem strip.
    problem_box(
        d,
        (70, 145, 1830, 345),
        "拟解决的关键问题",
        [
            "区域 InSAR 点数量大、时间序列长，直接建全图训练计算量过高",
            "不同滑坡单元变形不同步，全局变化点会掩盖局部响应",
            "降雨触发存在滞后效应，需要建立变化点与降雨过程的时间关系",
            "区域尺度模型不仅要预测位移，还要解释哪些区域对降雨更敏感",
        ],
        RED,
    )

    # Region icon and block strategy.
    draw_region_icon(d, (85, 390, 610, 620))
    round_box(
        d,
        (690, 390, 1035, 620),
        "分块与分区",
        "按滑坡边界、坡向、地貌单元\n或空间聚类划分子区域\n\n大区域 → 多个可训练子图",
        BLUE,
    )
    round_box(
        d,
        (1115, 390, 1460, 620),
        "局部图构建",
        "每个子区域独立构建\n地形感知 KNN 图\n\n距离 + 高程 + 坡向约束",
        GREEN,
    )
    round_box(
        d,
        (1540, 390, 1815, 620),
        "批量训练",
        "子图小批量训练\n区域级参数共享\n局部统计归一化",
        PURPLE,
    )
    arrow(d, (625, 505), (675, 505), CYAN)
    arrow(d, (1050, 505), (1100, 505), CYAN)
    arrow(d, (1475, 505), (1525, 505), CYAN)

    # Middle modeling pipeline.
    y0, y1 = 680, 910
    boxes = [
        ((85, y0, 360, y1), "区域位移序列", "InSAR 时间序列\n高活动点比例\n区域平均速率", BLUE),
        ((430, y0, 705, y1), "分区 CPD", "每个子区域独立检测\n变化点与阶段序列\n识别局部变形转换", RED),
        ((775, y0, 1050, y1), "降雨特征", "当日雨量\n前 3/7/15 日累计雨量\n最大连续降雨", ORANGE),
        ((1120, y0, 1395, y1), "响应关系", "变化点 - 降雨峰值匹配\n响应滞后时间\n降雨阈值估计", GREEN),
        ((1465, y0, 1815, y1), "降雨增强模型", "位移 + 图结构 + CPD 阶段\n+ 降雨滞后特征\n区域 CPD-STGCN", PURPLE),
    ]
    for box, title, body, color in boxes:
        round_box(d, box, title, body, color)
    for i in range(len(boxes) - 1):
        arrow(d, (boxes[i][0][2] + 15, (y0 + y1) / 2), (boxes[i + 1][0][0] - 15, (y0 + y1) / 2), INK, 3)

    # Bottom outputs.
    d.rounded_rectangle((70, 975, 1830, 1175), radius=18, fill=BG, outline=(210, 220, 235), width=2)
    d.text((95, 1000), "预期输出", font=F_HEAD, fill=INK, anchor="la")
    output_boxes = [
        ((120, 1045, 460, 1145), "区域变形阶段图", "不同滑坡单元的阶段转换时空分布", BLUE),
        ((555, 1045, 895, 1145), "降雨响应图谱", "响应滞后、敏感区域与可能阈值", ORANGE),
        ((990, 1045, 1330, 1145), "区域位移预测", "多步位移预测与误差放大区域识别", GREEN),
        ((1425, 1045, 1765, 1145), "风险识别与解释", "高敏感区、转折响应区和重点监测区", RED),
    ]
    for box, title, body, color in output_boxes:
        x0, yy0, x1, yy1 = box
        d.rounded_rectangle(box, radius=12, fill="white", outline=color, width=2)
        d.text(((x0+x1)/2, yy0+15), title, font=F_TEXT, fill=color, anchor="ma")
        draw_multiline(d, [x0+10, yy0+43, x1-10, yy1-8], body, F_SMALL, INK)

    # Small method note.
    d.text(
        (w / 2, 1200),
        "核心思想：用分块降低区域建模复杂度，用分区 CPD 捕捉局部变形阶段，用降雨滞后特征解释变化点响应。",
        font=F_TEXT,
        fill=INK,
        anchor="ma",
    )

    out = OUT / "fig8_regional_cpd_rainfall_roadmap.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
