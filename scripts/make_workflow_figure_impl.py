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
F_SECTION = font(25, True)
F_LABEL = font(22, True)
F_TEXT = font(19)
F_SMALL = font(16)
F_TINY = font(14)


COLORS = {
    "ink": (32, 39, 52),
    "muted": (86, 96, 112),
    "line": (130, 146, 166),
    "box": (255, 255, 255),
    "soft": (246, 249, 252),
    "blue": (42, 105, 196),
    "green": (36, 145, 96),
    "orange": (218, 132, 46),
    "red": (206, 70, 57),
    "purple": (112, 88, 176),
    "cyan": (80, 178, 190),
}


def multiline(draw, box, text, fnt=F_TEXT, fill=None, align="center", spacing=5):
    fill = fill or COLORS["ink"]
    x0, y0, x1, y1 = box
    max_w = x1 - x0 - 18
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
    line_h = fnt.size + spacing
    total_h = line_h * len(lines) - spacing
    y = y0 + (y1 - y0 - total_h) / 2
    for line in lines:
        if align == "center":
            x = (x0 + x1) / 2
            anchor = "ma"
        else:
            x = x0 + 12
            anchor = "la"
        draw.text((x, y), line, font=fnt, fill=fill, anchor=anchor)
        y += line_h


def round_box(draw, box, label, fill=(255, 255, 255), outline=None, width=2, fnt=F_TEXT, text_fill=None, radius=10):
    outline = outline or COLORS["line"]
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    multiline(draw, box, label, fnt=fnt, fill=text_fill or COLORS["ink"])


def arrow(draw, start, end, color=None, width=3, head=14):
    color = color or COLORS["ink"]
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


def vertical_label(draw, box, text, fill=(236, 241, 247), outline=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline or COLORS["line"], width=2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    chars = list(text)
    h = F_LABEL.size + 2
    start = cy - h * len(chars) / 2
    for i, ch in enumerate(chars):
        draw.text((cx, start + i * h), ch, font=F_LABEL, fill=COLORS["ink"], anchor="ma")


def draw_point_cloud(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(252, 253, 255), outline=(150, 164, 184), width=2)
    pts = [
        (0.15, 0.72), (0.25, 0.28), (0.42, 0.58), (0.55, 0.20), (0.72, 0.42),
        (0.80, 0.74), (0.34, 0.83), (0.63, 0.67), (0.18, 0.46),
    ]
    for i, (px, py) in enumerate(pts):
        r = 9 if i % 3 else 11
        cx = x0 + px * (x1 - x0)
        cy = y0 + py * (y1 - y0)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=COLORS["blue"])
    draw.line([(x0+18, y1-18), (x1-18, y0+18)], fill=(180, 190, 205), width=1)
    draw.text((x1-20, y0+20), "Time", font=F_TINY, fill=COLORS["muted"], anchor="ra")


def draw_matrix(draw, box):
    x0, y0, x1, y1 = box
    n = 12
    cell_w = (x1 - x0) / n
    cell_h = (y1 - y0) / n
    for i in range(n):
        for j in range(n):
            val = 230 - int(105 * math.exp(-abs(i-j)/2.3))
            fill = (70, min(190, val), 220) if i != j else (245, 247, 250)
            draw.rectangle([x0+i*cell_w, y0+j*cell_h, x0+(i+1)*cell_w, y0+(j+1)*cell_h], fill=fill)
    draw.rectangle(box, outline=(150, 164, 184), width=2)


def draw_graph(draw, box):
    x0, y0, x1, y1 = box
    nodes = [
        (0.18, 0.68), (0.30, 0.30), (0.50, 0.20), (0.72, 0.36),
        (0.80, 0.70), (0.55, 0.78), (0.35, 0.60)
    ]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,6),(6,3),(6,5),(2,5)]
    pts = [(x0+px*(x1-x0), y0+py*(y1-y0)) for px,py in nodes]
    for a, b in edges:
        draw.line([pts[a], pts[b]], fill=COLORS["orange"], width=3)
    for idx, p in enumerate(pts):
        r = 9
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=COLORS["blue"], outline="white", width=2)
    draw.text((x0+18, y0+18), "Wij", font=F_SMALL, fill=COLORS["ink"])
    draw.rectangle(box, outline=(150, 164, 184), width=2)


def draw_cube(draw, box):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    front = [x0 + w*0.22, y0 + h*0.26, x0 + w*0.78, y0 + h*0.82]
    dx, dy = w*0.16, -h*0.14
    draw.rectangle(front, fill=(250, 252, 255), outline=(120, 135, 156), width=2)
    draw.polygon([(front[0], front[1]), (front[0]+dx, front[1]+dy), (front[2]+dx, front[1]+dy), (front[2], front[1])], fill=(239, 244, 250), outline=(120, 135, 156))
    draw.polygon([(front[2], front[1]), (front[2]+dx, front[1]+dy), (front[2]+dx, front[3]+dy), (front[2], front[3])], fill=(232, 238, 247), outline=(120, 135, 156))
    for k in [1/3, 2/3]:
        x = front[0] + (front[2]-front[0]) * k
        y = front[1] + (front[3]-front[1]) * k
        draw.line([(x, front[1]), (x, front[3])], fill=(140, 154, 174), width=1)
        draw.line([(front[0], y), (front[2], y)], fill=(140, 154, 174), width=1)
    draw.text((x1-16, y0+18), "Time", font=F_TINY, fill=COLORS["muted"], anchor="ra")


def draw_model_block(draw, box):
    x0, y0, x1, y1 = box
    inner = [
        ("时间卷积 GLU", COLORS["blue"]),
        ("空间图卷积", COLORS["green"]),
        ("时间卷积", COLORS["blue"]),
        ("归一化", COLORS["purple"]),
        ("CPD 阶段调制", COLORS["red"]),
    ]
    gap = 10
    h = (y1 - y0 - gap * (len(inner) + 1)) / len(inner)
    y = y0 + gap
    for label, col in inner:
        round_box(draw, [x0+18, y, x1-18, y+h], label, fill=(255,255,255), outline=col, width=2, fnt=F_TINY, text_fill=COLORS["ink"], radius=6)
        y += h + gap


def draw_prediction(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(252, 253, 255), outline=(150, 164, 184), width=2)
    pts = [(0.18,0.74),(0.32,0.45),(0.52,0.24),(0.70,0.48),(0.82,0.70),(0.42,0.80)]
    for i, (px, py) in enumerate(pts):
        r = 10
        cx = x0 + px*(x1-x0)
        cy = y0 + py*(y1-y0)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=COLORS["red"])
    draw.text((x1-16, y0+20), "Future", font=F_TINY, fill=COLORS["muted"], anchor="ra")


def main():
    w, h = 1800, 1380
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((w/2, 36), "变化点引导的滑坡 InSAR 位移时空预测流程", font=F_TITLE, fill=COLORS["ink"], anchor="ma")
    d.text((w/2, 92), "CPD-guided Stage-history Residual STGCN", font=F_SECTION, fill=COLORS["muted"], anchor="ma")

    sections = [
        (70, 145, 1730, 390, "InSAR 数据与位移序列构建"),
        (70, 465, 1730, 745, "CPD-aware 数据预处理与图结构构建"),
        (70, 825, 1730, 1195, "CPD-STGCN 残差预测与验证"),
    ]
    labels = ["数据获取", "数据预处理", "模型预测"]
    for sec, lab in zip(sections, labels):
        x0, y0, x1, y1, title = sec
        d.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=COLORS["soft"], outline=(188, 200, 216), width=2)
        vertical_label(d, [x0+14, y0+20, x0+70, y1-20], lab)
        d.text((x0+92, y0+18), title, font=F_SECTION, fill=COLORS["ink"], anchor="la")

    # Section 1.
    y = 205
    boxes1 = [
        (145, y, 345, y+120, "Sentinel-1\nSAR 影像"),
        (405, y, 605, y+120, "DEM / 轨道\n地形校正"),
        (665, y, 865, y+120, "配准与\n干涉处理"),
        (925, y, 1125, y+120, "滤波 / 解缠\n大气校正"),
        (1185, y, 1385, y+120, "SVD 与\n时序分析"),
        (1445, y, 1660, y+120, "InSAR 位移\n时间序列"),
    ]
    for b in boxes1:
        round_box(d, b[:4], b[4], fill=(255,255,255), outline=(165, 178, 196), fnt=F_TEXT)
    for i in range(len(boxes1)-1):
        arrow(d, (boxes1[i][2]+18, y+60), (boxes1[i+1][0]-18, y+60), COLORS["ink"], 3)

    # Section 2.
    y2 = 535
    pc_box = (150, y2+50, 330, y2+190)
    mat_box = (455, y2+50, 615, y2+190)
    graph_box = (740, y2+50, 930, y2+190)
    cube_box = (1055, y2+50, 1235, y2+190)
    stage_box = (1355, y2+50, 1660, y2+190)
    d.text((240, y2+24), "插值相干点云", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_point_cloud(d, pc_box)
    d.text((535, y2+24), "地形感知权重矩阵", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_matrix(d, mat_box)
    d.text((835, y2+24), "KNN 图结构", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_graph(d, graph_box)
    d.text((1145, y2+24), "数据张量加载", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_cube(d, cube_box)
    round_box(d, stage_box, "CPD 变化点检测\n阶段划分\n阶段历史序列构建", fill=(255,255,255), outline=COLORS["red"], fnt=F_SMALL)
    for start, end in [
        ((350, y2+120), (430, y2+120)),
        ((635, y2+120), (715, y2+120)),
        ((950, y2+120), (1030, y2+120)),
        ((1255, y2+120), (1330, y2+120)),
    ]:
        arrow(d, start, end, COLORS["ink"], 3)
    d.text((390, y2+112), "+", font=font(36, True), fill=COLORS["cyan"], anchor="mm")

    # Section 3.
    y3 = 895
    model_box = (150, y3+55, 380, y3+275)
    prior_box = (470, y3+55, 700, y3+275)
    residual_box = (790, y3+55, 1020, y3+275)
    output_box = (1110, y3+55, 1340, y3+275)
    eval_box = (1430, y3+55, 1660, y3+275)

    d.text((265, y3+24), "时空特征提取", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_model_block(d, model_box)
    d.text((585, y3+24), "物理先验", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    round_box(d, prior_box, "Persistence prior\n上一时刻位移\n+\n保守残差校正", fill=(255,255,255), outline=COLORS["green"], fnt=F_SMALL)
    d.text((905, y3+24), "残差学习", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    round_box(d, residual_box, "CPD 加权训练\n跨变化点样本增强\n节点可靠性门控", fill=(255,255,255), outline=COLORS["orange"], fnt=F_SMALL)
    d.text((1225, y3+24), "预测输出", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    draw_prediction(d, output_box)
    d.text((1545, y3+24), "验证评价", font=F_SMALL, fill=COLORS["ink"], anchor="ma")
    round_box(d, eval_box, "固定时序测试\n多阶段分层验证\n变化点滚动验证", fill=(255,255,255), outline=COLORS["purple"], fnt=F_SMALL)
    for start, end in [
        ((400, y3+162), (445, y3+162)),
        ((720, y3+162), (765, y3+162)),
        ((1040, y3+162), (1085, y3+162)),
        ((1360, y3+162), (1405, y3+162)),
    ]:
        arrow(d, start, end, COLORS["ink"], 3)

    # Cross-section flow arrows.
    arrow(d, (900, 405), (900, 455), COLORS["cyan"], 5, head=18)
    arrow(d, (900, 760), (900, 815), COLORS["cyan"], 5, head=18)

    d.text((w/2, 1265), "图  方法流程：由 InSAR 位移序列出发，通过 CPD 阶段识别、地形感知图构建与残差 STGCN 实现滑坡位移预测。", font=F_TEXT, fill=COLORS["ink"], anchor="ma")
    d.text((w-78, 1320), "注：该图为本文方法示意图，非原论文图片复刻。", font=F_TINY, fill=COLORS["muted"], anchor="ra")

    out = OUT / "fig3_cpd_stgcn_workflow_cn.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
