"""从 results_baseline.json / results_optimized.json 复算指标，并生成对比图。

指标定义（全部可从仓库文件复算，不依赖人工判断）：

  hit        问题里的目标材料名出现在检索片段中          —— 检索是否覆盖到该问题
  hit_all    问题涉及的**全部**材料名都被检索到          —— 对比类问题的完整召回
  refusal    回答里出现「未提及」                        —— 模型是否明确承认无依据
  overclaim  目标材料没被检索到、回答里也没说未提及      —— 「无依据仍作答」
  unsupported 回答提到的材料名不在检索片段里             —— 更宽口径的编造代理
  avg_len    回答平均字符数                              —— 啰嗦度 / 成本代理

用法：
    python docs/metrics.py            # 打印指标
    python docs/metrics.py --chart    # 打印指标并重新生成 docs/comparison.png
"""

import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

# 15 种材料（来自 SKILL.md 材料库）。长名优先，避免「酷布7环」抢占「酷布弹簧20高9环」。
MATERIALS = sorted(
    [
        "纯棉梭织布",
        "羊毛棉花混纺",
        "白色马尾毛",
        "天然新疆棉",
        "驼绒",
        "黑色马尾毛",
        "羊毛q弹棉",
        "雪绒棉",
        "酷布网",
        "mini簧",
        "十环科技酷布",
        "麻布袋",
        "四叶草科技酷布",
        "酷布弹簧20高9环",
        "酷布7环",
    ],
    key=len,
    reverse=True,
)

REFUSAL_PAT = re.compile(r"未提及|没有提及|材料库中未")


def find_materials(text):
    """返回文本中出现的材料名集合（长名优先，已匹配区间不再复用）。"""
    found = set()
    spans = []
    for name in MATERIALS:
        start = 0
        while True:
            i = text.find(name, start)
            if i < 0:
                break
            if any(a <= i < b for a, b in spans):
                start = i + 1
                continue
            found.add(name)
            spans.append((i, i + len(name)))
            start = i + len(name)
    return found


def analyse(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    out = {
        "n": len(rows),
        "hit": 0,
        "hit_all": 0,
        "multi": 0,
        "multi_all": 0,
        "refusal": 0,
        "overclaim": 0,
        "unsupported": 0,
        "avg_len": 0,
        "overclaim_examples": [],
    }
    total_len = 0
    for r in rows:
        question, answer, docs = r["question"], r["answer"], r["docs"]
        total_len += len(answer)

        want = find_materials(question)
        got = find_materials(" ".join(docs))
        refusal = bool(REFUSAL_PAT.search(answer))
        unsupported = find_materials(answer) - got

        if want & got:
            out["hit"] += 1
        if want and want <= got:
            out["hit_all"] += 1
        if len(want) >= 2:
            out["multi"] += 1
            if want <= got:
                out["multi_all"] += 1
        if refusal:
            out["refusal"] += 1
        if want and not (want & got) and not refusal:
            out["overclaim"] += 1
            if len(out["overclaim_examples"]) < 5:
                out["overclaim_examples"].append(question)
        if unsupported:
            out["unsupported"] += 1

    out["avg_len"] = round(total_len / out["n"])
    return out


# ── 画图 ─────────────────────────────────────────────────────────────────────
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

INK = (23, 26, 33)
MUTED = (110, 118, 130)
GRID = (226, 230, 236)
BASE_C = (176, 182, 194)
OPT_C = (37, 99, 235)
BG = (255, 255, 255)


def _font(path, size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def draw_chart(base, opt, out_path):
    from PIL import Image, ImageDraw

    W, H = 1400, 760
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(FONT_BOLD, 38)
    f_sub = _font(FONT_REG, 19)
    f_ptitle = _font(FONT_BOLD, 21)
    f_value = _font(FONT_BOLD, 28)
    f_axis = _font(FONT_REG, 19)
    f_note = _font(FONT_REG, 17)

    d.text((70, 42), "RAG 幻觉治理：两种 Prompt 策略的对照", font=f_title, fill=INK)
    d.text(
        (72, 96),
        "50 个问题 · deepseek-chat T=0.7 · 两版检索结果逐题相同（26/50 命中目标材料，14 个对比题仅 2 题召全）",
        font=f_sub,
        fill=MUTED,
    )

    # 图例（右下角，避免与标题抢位置）
    d.rounded_rectangle([1096, 100, 1120, 124], 5, fill=BASE_C)
    d.text((1130, 98), "基线版 · 裸 Prompt", font=f_axis, fill=INK)
    d.rounded_rectangle([1096, 138, 1120, 162], 5, fill=OPT_C)
    d.text((1130, 136), "优化版 · few-shot + 事实约束", font=f_axis, fill=INK)

    panels = [
        ("无依据仍作答", base["overclaim"], opt["overclaim"], "题", "越少越好"),
        ("明确拒答", base["refusal"], opt["refusal"], "题", "越多越好"),
        ("回答含未检索材料", base["unsupported"], opt["unsupported"], "题", "越少越好"),
        ("平均回答长度", base["avg_len"], opt["avg_len"], "字", "越短越好"),
    ]

    top, bottom = 240, 630
    pw = 300
    for idx, (title, vb, vo, unit, note) in enumerate(panels):
        x0 = 70 + idx * 330
        x1 = x0 + pw

        d.text((x0, 176), title, font=f_ptitle, fill=INK)
        d.text((x0, 206), note, font=f_note, fill=MUTED)

        d.line([x0, bottom, x1, bottom], fill=GRID, width=2)

        vmax = max(vb, vo) * 1.35 or 1
        cx = x0 + pw / 2
        bar_w = 52
        for j, (val, color) in enumerate(((vb, BASE_C), (vo, OPT_C))):
            x = cx - bar_w - 12 + j * (bar_w + 24)
            h = (val / vmax) * (bottom - top)
            if h < 3:  # 0 值也画一条细底线，否则看起来像缺数据
                d.rectangle([x, bottom - 3, x + bar_w, bottom], fill=color)
            else:
                d.rounded_rectangle([x, bottom - h, x + bar_w, bottom], 7, fill=color)
            txt = str(val)
            tw = d.textlength(txt, font=f_value)
            d.text((x + bar_w / 2 - tw / 2, bottom - h - 40), txt, font=f_value, fill=INK)

        unit_w = d.textlength("（%s）" % unit, font=f_note)
        d.text((cx - unit_w / 2, bottom + 14), "（%s）" % unit, font=f_note, fill=MUTED)

    d.text(
        (70, H - 52),
        "注：指标由 docs/metrics.py 从 results_*.json 复算，属字符串口径的代理指标；本轮未改变检索策略，差异只来自 Prompt。",
        font=f_note,
        fill=MUTED,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    base = analyse(os.path.join(ROOT, "results_baseline.json"))
    opt = analyse(os.path.join(ROOT, "results_optimized.json"))

    rows = [
        ("检索命中（题）", base["hit"], opt["hit"], "越高越好"),
        ("目标材料全部召回（题）", base["hit_all"], opt["hit_all"], "越高越好"),
        ("明确拒答（题）", base["refusal"], opt["refusal"], "越高越好：承认没依据"),
        ("无依据仍作答（题）", base["overclaim"], opt["overclaim"], "越低越好"),
        ("回答含未检索材料（题）", base["unsupported"], opt["unsupported"], "越低越好"),
        ("平均答案字数", base["avg_len"], opt["avg_len"], "-"),
    ]
    print("%-26s %8s %8s   %s" % ("指标", "基线", "优化", "方向"))
    for label, vb, vo, note in rows:
        print("%-26s %8s %8s   %s" % (label, vb, vo, note))
    print("\n对比题：%d 个，两版都只把 %d 个问全" % (base["multi"], base["multi_all"]))

    print("\n基线「无依据仍作答」示例:")
    for q in base["overclaim_examples"]:
        print("  -", q)

    if "--chart" in sys.argv:
        out = draw_chart(base, opt, os.path.join(BASE, "comparison.png"))
        print("\n图已生成:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
