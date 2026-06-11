from pathlib import Path

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


F_TITLE = font(46, True)
F_SUB = font(24)
F_HEAD = font(25, True)
F_TEXT = font(19)
F_SMALL = font(16)

INK = (31, 39, 52)
MUTED = (82, 94, 112)
BG = (247, 250, 253)
LINE = (220, 226, 236)
BLUE = (45, 104, 196)
GREEN = (34, 145, 95)
ORANGE = (221, 130, 47)
RED = (204, 70, 58)
PURPLE = (110, 88, 176)


def wrap(draw, text, fnt, max_w):
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


def draw_lines(draw, xy, lines, fnt=F_TEXT, fill=INK, leading=8):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill, anchor="la")
        y += fnt.size + leading
    return y


def card(draw, box, num, title, challenge, approach, color):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="white", outline=LINE, width=2)
    draw.rounded_rectangle([x0 + 26, y0 + 26, x0 + 78, y0 + 78], radius=14, fill=color)
    draw.text((x0 + 52, y0 + 37), str(num), font=F_HEAD, fill="white", anchor="ma")
    draw.text((x0 + 96, y0 + 28), title, font=F_HEAD, fill=INK, anchor="la")
    draw.line([(x0 + 30, y0 + 102), (x1 - 30, y0 + 102)], fill=LINE, width=2)

    y = y0 + 126
    draw.text((x0 + 32, y), "主要难点", font=F_SMALL, fill=color, anchor="la")
    y += 28
    y = draw_lines(draw, (x0 + 32, y), wrap(draw, challenge, F_TEXT, x1 - x0 - 64), F_TEXT, INK, 6)
    y += 16
    draw.text((x0 + 32, y), "拟解决思路", font=F_SMALL, fill=color, anchor="la")
    y += 28
    draw_lines(draw, (x0 + 32, y), wrap(draw, approach, F_TEXT, x1 - x0 - 64), F_TEXT, INK, 6)


def main():
    w, h = 1800, 1050
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((w / 2, 48), "未来研究方向与关键难点", font=F_TITLE, fill=INK, anchor="ma")
    d.text((w / 2, 106), "从单体滑坡 CPD-STGCN 扩展到区域滑坡群与降雨响应机制分析", font=F_SUB, fill=MUTED, anchor="ma")

    d.rounded_rectangle([70, 155, 1730, 905], radius=24, fill=BG, outline=LINE, width=2)

    cards = [
        (
            (115, 205, 865, 515),
            1,
            "区域尺度建模",
            "区域 InSAR 点数量大，直接构建全图会带来显著计算压力；不同滑坡单元的空间关系也不完全一致。",
            "按滑坡边界、坡向、地貌单元或空间聚类进行分块，构建多个局部子图，再进行区域级汇总。",
            BLUE,
        ),
        (
            (935, 205, 1685, 515),
            2,
            "分区变化点识别",
            "区域内不同滑坡单元变形并不同步，单一全局 CPD 容易掩盖局部阶段转换。",
            "对不同子区域或节点群分别进行 CPD，形成区域变形阶段图，识别转折响应区。",
            RED,
        ),
        (
            (115, 560, 865, 870),
            3,
            "降雨响应机制",
            "降雨对位移变化的影响存在滞后，不同区域响应强度和响应时间可能不同。",
            "构建当日雨量、前期累计雨量和连续降雨特征，分析变化点与降雨峰值之间的滞后关系。",
            ORANGE,
        ),
        (
            (935, 560, 1685, 870),
            4,
            "预测与解释统一",
            "区域模型不能只追求误差降低，还需要解释哪些区域对降雨更敏感、哪些区域更容易发生阶段转换。",
            "在 CPD-STGCN 中加入降雨滞后特征，输出位移预测、误差放大区域和降雨敏感分区。",
            GREEN,
        ),
    ]
    for args in cards:
        card(d, *args)

    d.rounded_rectangle([180, 940, 1620, 1000], radius=18, fill="white", outline=(205, 215, 230), width=2)
    d.text(
        (w / 2, 958),
        "核心目标：从“预测是否准确”进一步转向“变化点为何发生、何处响应降雨、响应滞后多久”。",
        font=F_TEXT,
        fill=PURPLE,
        anchor="ma",
    )

    out = OUT / "fig8_future_directions_challenges.png"
    img.save(out)
    print(out)


if __name__ == "__main__":
    main()
