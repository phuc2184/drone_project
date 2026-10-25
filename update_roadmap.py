#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đọc roadmap_data.json -> ghi ROADMAP.md (tiếng Việt) và outputs/progress_roadmap.png.

Chỉ đọc roadmap_data.json. Chỉ ghi ROADMAP.md và outputs/progress_roadmap.png.
Chạy: python3 update_roadmap.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "roadmap_data.json")
MD_PATH = os.path.join(BASE_DIR, "ROADMAP.md")
PNG_PATH = os.path.join(BASE_DIR, "outputs", "progress_roadmap.png")

STATUS_VI = {
    "done": "XONG",
    "in_progress": "ĐANG LÀM",
    "partial": "MỘT PHẦN",
    "not_started": "CHƯA BẮT ĐẦU",
}
COLORS = {
    "done": "#2E7D32",
    "in_progress": "#1565C0",
    "partial": "#EF6C00",
    "not_started": "#616161",
}
TINTS = {
    "done": "#E8F5E9",
    "in_progress": "#E3F2FD",
    "partial": "#FFF3E0",
    "not_started": "#F5F5F5",
}
STATE_SYMBOL = {"x": "[x]", "~": "[~]", " ": "[ ]"}
STATE_COLOR = {"x": "#2E7D32", "~": "#EF6C00", " ": "#9E9E9E"}
FOOTER = (
    "Phần trăm là ước lượng. Chỉ kiểm tra trên 1 video (5.mp4); "
    "chưa được kết luận 'không có khóa nhầm'."
)
LINE_GAP = 6.0


def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("updated", "stages", "stage2_percent", "stage2_percent_note",
                "pending_approval", "verified_facts", "log"):
        if key not in data:
            raise SystemExit(f"roadmap_data.json thiếu khóa: {key}")
    for i, stage in enumerate(data["stages"], 1):
        if stage.get("status") not in STATUS_VI:
            raise SystemExit(f"Giai đoạn {i}: status không hợp lệ: {stage.get('status')!r}")
        for item in stage.get("checklist", []):
            if item.get("state") not in STATE_SYMBOL:
                raise SystemExit(f"Giai đoạn {i}: state không hợp lệ: {item.get('state')!r}")
    return data


def parse_percent(text):
    s = str(text).replace("%", "").strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        value = (float(lo.strip()) + float(hi.strip())) / 2.0
    else:
        value = float(s)
    return max(0.0, min(100.0, value))


def build_markdown(data):
    stages = data["stages"]
    lines = []
    lines.append("# Lộ trình 5 giai đoạn — UAV bám người (PX4)")
    lines.append("")
    lines.append(f"**Cập nhật:** {data['updated']}")
    lines.append("")
    lines.append("## Tổng quan 5 giai đoạn")
    lines.append("")
    lines.append("| # | Giai đoạn | Trạng thái | Người phụ trách |")
    lines.append("|---|---|---|---|")
    for i, stage in enumerate(stages, 1):
        label = STATUS_VI[stage["status"]]
        lines.append(f"| {i} | {stage['name']} | **{label}** | {stage.get('owner', '')} |")
    lines.append("")

    current = None
    for i, stage in enumerate(stages, 1):
        if stage["status"] in ("in_progress", "partial"):
            current = (i, stage)
            if stage["status"] == "in_progress":
                break
    lines.append("## BẠN ĐANG Ở ĐÂY")
    lines.append("")
    if current:
        i, stage = current
        lines.append(
            f"**BẠN ĐANG Ở ĐÂY: Giai đoạn {i} — {stage['name']}** "
            f"({STATUS_VI[stage['status']]})"
        )
    else:
        lines.append("**BẠN ĐANG Ở ĐÂY:** không có giai đoạn nào đang làm.")
    lines.append("")

    lines.append("## Chi tiết từng giai đoạn")
    lines.append("")
    for i, stage in enumerate(stages, 1):
        lines.append(f"### Giai đoạn {i} — {stage['name']}")
        lines.append("")
        lines.append(f"- **Trạng thái:** {STATUS_VI[stage['status']]}")
        lines.append(f"- **Người phụ trách:** {stage.get('owner', '')}")
        if i == 2 and "stage2_percent" in data:
            note = data.get("stage2_percent_note", "")
            lines.append(
                f"- **Ước lượng hoàn thành Giai đoạn 2: {data['stage2_percent']}%** — "
                f"đây là ước lượng theo tiêu chí hoàn thành đề xuất, không phải phép đo. "
                f"_({note})_"
            )
        lines.append("- **Checklist:**")
        if stage.get("checklist"):
            for item in stage["checklist"]:
                symbol = STATE_SYMBOL[item["state"]]
                lines.append(f"  - {symbol} {item['text']}")
        else:
            lines.append("  - (Chưa có mục nào — chờ bổ sung)")
        lines.append("")

    lines.append("## Chờ duyệt")
    lines.append("")
    for item in data["pending_approval"]:
        lines.append(f"- [ ] {item}")
    lines.append("")

    lines.append("## Sự thật đã kiểm chứng")
    lines.append("")
    for item in data["verified_facts"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Nhật ký thay đổi (mới nhất trước)")
    lines.append("")
    for entry in sorted(data["log"], key=lambda e: e["date"], reverse=True):
        lines.append(f"### {entry['date']}")
        lines.append("")
        lines.append(f"- **Đã làm:** {entry['done']}")
        lines.append(f"- **Bằng chứng:** {entry['evidence']}")
        lines.append(f"- **Chờ duyệt:** {entry['pending']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_{FOOTER}_")
    lines.append("")
    return "\n".join(lines)


class Canvas:
    def __init__(self):
        self.W, self.H, self.DPI = 1920, 1080, 100
        self.fig = plt.figure(
            figsize=(self.W / self.DPI, self.H / self.DPI),
            dpi=self.DPI,
            facecolor="white",
        )
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.texts = []

    def prop(self, size, weight="normal", style="normal"):
        return FontProperties(family="DejaVu Sans", size=size, weight=weight, style=style)

    def measure(self, text, fp):
        return self.renderer.get_text_width_height_descent(text, fp, False)[:2]

    def wrap(self, text, fp, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            trial = (current + " " + word).strip()
            wide = self.measure(trial, fp)[0] > max_width
            if wide and current:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        out = []
        for line in lines:
            if self.measure(line, fp)[0] <= max_width:
                out.append(line)
                continue
            chunk = ""
            for ch in line:
                if self.measure(chunk + ch, fp)[0] > max_width and chunk:
                    out.append(chunk)
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                out.append(chunk)
        return out or [""]

    def text(self, x, y, s, fp, color="#212121", ha="left", va="top", card=None):
        artist = self.ax.text(x, y, s, fontproperties=fp, color=color, ha=ha, va=va)
        self.texts.append((artist, card))
        return artist


def build_blocks(c, stage, index, inner_w, fs, data):
    blocks = []
    name_fp = c.prop(fs + 3, "bold")
    name_lines = c.wrap(stage["name"], name_fp, inner_w)
    blocks.append({"kind": "lines", "lines": name_lines, "fp": name_fp,
                   "color": "#1A237E", "after": 7})

    owner_fp = c.prop(max(fs - 2.0, 8.5), "normal", "italic")
    owner_lines = c.wrap(f"Phụ trách: {stage.get('owner', '')}", owner_fp, inner_w)
    blocks.append({"kind": "lines", "lines": owner_lines, "fp": owner_fp,
                   "color": "#607D8B", "after": 7})

    if index == 2 and "stage2_percent" in data:
        pct_fp = c.prop(11.5, "bold")
        pct_text = f"Ước lượng Giai đoạn 2: {data['stage2_percent']}% — không phải phép đo"
        pct_lines = c.wrap(pct_text, pct_fp, inner_w)
        blocks.append({"kind": "lines", "lines": pct_lines, "fp": pct_fp,
                       "color": COLORS["in_progress"], "after": 5})
        blocks.append({"kind": "bar", "frac": parse_percent(data["stage2_percent"]) / 100.0,
                       "after": 10})

    if stage.get("checklist"):
        chk_fp = c.prop(fs)
        for item in stage["checklist"]:
            symbol = STATE_SYMBOL[item["state"]]
            prefix_w = c.measure(symbol + " ", chk_fp)[0]
            text_lines = c.wrap(item["text"], chk_fp, inner_w - prefix_w)
            blocks.append({
                "kind": "check",
                "symbol": symbol,
                "prefix_w": prefix_w,
                "lines": text_lines,
                "fp": chk_fp,
                "symbol_color": STATE_COLOR[item["state"]],
                "after": 6,
            })
    else:
        empty_fp = c.prop(max(fs - 1.0, 8.5), "normal", "italic")
        empty_lines = c.wrap("Chưa có mục nào — chờ bổ sung", empty_fp, inner_w)
        blocks.append({"kind": "lines", "lines": empty_lines, "fp": empty_fp,
                       "color": "#78909C", "after": 6})
    return blocks


def blocks_height(c, blocks):
    total = 0.0
    for block in blocks:
        if block["kind"] in ("lines", "check"):
            heights = [c.measure(line, block["fp"])[1] for line in block["lines"]]
            total += sum(heights) + LINE_GAP * (len(heights) - 1)
        elif block["kind"] == "bar":
            total += 18.0
        total += block["after"]
    return total


def draw_blocks(c, blocks, x, y, inner_w, card):
    for block in blocks:
        if block["kind"] == "lines":
            heights = [c.measure(line, block["fp"])[1] for line in block["lines"]]
            for i, line in enumerate(block["lines"]):
                c.text(x, y, line, block["fp"], block["color"], card=card)
                y += heights[i] + (LINE_GAP if i < len(heights) - 1 else 0.0)
        elif block["kind"] == "check":
            heights = [c.measure(line, block["fp"])[1] for line in block["lines"]]
            c.text(x, y, block["symbol"], block["fp"], block["symbol_color"], card=card)
            for i, line in enumerate(block["lines"]):
                c.text(x + block["prefix_w"], y, line, block["fp"], "#263238", card=card)
                y += heights[i] + (LINE_GAP if i < len(heights) - 1 else 0.0)
        elif block["kind"] == "bar":
            c.ax.add_patch(FancyBboxPatch(
                (x, y), inner_w, 18,
                boxstyle="round,pad=0,rounding_size=8",
                fc="#E0E0E0", ec="#BDBDBD", lw=1,
            ))
            c.ax.add_patch(FancyBboxPatch(
                (x, y), max(inner_w * block["frac"], 12.0), 18,
                boxstyle="round,pad=0,rounding_size=8",
                fc=COLORS["in_progress"], ec="none",
            ))
            y += 18.0
        y += block["after"]
    return y


def build_png(data):
    c = Canvas()
    stages = data["stages"]
    n = len(stages)
    margin = 28.0
    arrow_gap = 26.0
    box_w = (c.W - 2 * margin - (n - 1) * arrow_gap) / n
    box_top = 138.0
    box_h = 748.0
    box_bottom = box_top + box_h
    inner_w = box_w - 28.0

    c.text(margin, 20, "LỘ TRÌNH 5 GIAI ĐOẠN — UAV BÁM NGƯỜI (PX4)",
           c.prop(29, "bold"), "#0D47A1")
    c.text(c.W - margin, 30, f"Cập nhật: {data['updated']}",
           c.prop(16, "bold"), "#455A64", ha="right")

    current = None
    for i, stage in enumerate(stages, 1):
        if stage["status"] == "in_progress":
            current = (i, stage)
            break
    if current is None:
        for i, stage in enumerate(stages, 1):
            if stage["status"] == "partial":
                current = (i, stage)
    if current:
        i, stage = current
        where = (f"BẠN ĐANG Ở ĐÂY: Giai đoạn {i} — {stage['name']} "
                 f"({STATUS_VI[stage['status']]})")
    else:
        where = "BẠN ĐANG Ở ĐÂY: không có giai đoạn nào đang làm"
    c.text(margin, 76, where, c.prop(19, "bold"), COLORS["in_progress"])

    c.text(margin, 114,
           "Màu và nhãn chữ cùng thể hiện trạng thái.   "
           "[x] xong    [~] một phần    [ ] chưa làm",
           c.prop(13), "#546E7A")

    overflow = []
    for index, stage in enumerate(stages, 1):
        left = margin + (index - 1) * (box_w + arrow_gap)
        right = left + box_w
        color = COLORS[stage["status"]]
        tint = TINTS[stage["status"]]
        card = (left, box_top, right, box_bottom)

        c.ax.add_patch(FancyBboxPatch(
            (left, box_top), box_w, box_h,
            boxstyle="round,pad=0,rounding_size=16",
            fc=tint, ec=color, lw=3,
        ))
        c.ax.add_patch(FancyBboxPatch(
            (left + 8, box_top + 8), box_w - 16, 44,
            boxstyle="round,pad=0,rounding_size=10",
            fc=color, ec=color, lw=0,
        ))
        c.text(left + 20, box_top + 30, f"GIAI ĐOẠN {index}",
               c.prop(14, "bold"), "white", va="center", card=card)
        c.text(right - 20, box_top + 30, STATUS_VI[stage["status"]],
               c.prop(14, "bold"), "white", ha="right", va="center", card=card)

        content_top = box_top + 64
        avail = box_bottom - 14 - content_top
        chosen = None
        chosen_blocks = None
        fs = 16.0
        while fs >= 8.0:
            blocks = build_blocks(c, stage, index, inner_w, fs, data)
            if blocks_height(c, blocks) <= avail:
                chosen, chosen_blocks = fs, blocks
                break
            fs -= 0.5
        if chosen is None:
            chosen = 8.0
            chosen_blocks = build_blocks(c, stage, index, inner_w, chosen, data)
            overflow.append(index)

        end_y = draw_blocks(c, chosen_blocks, left + 14, content_top, inner_w, card)
        if end_y > box_bottom - 8:
            overflow.append(index)

        if index < n:
            mid = box_top + box_h / 2.0
            c.ax.add_patch(FancyArrowPatch(
                (right + 4, mid), (right + arrow_gap - 4, mid),
                arrowstyle="-|>", mutation_scale=22, lw=2.5,
                color="#90A4AE", shrinkA=0, shrinkB=0,
            ))

    legend_fp = c.prop(13, "bold")
    legend_x = margin
    legend_y = 902.0
    legend_entries = [
        ("done", "XONG — hoàn thành"),
        ("in_progress", "ĐANG LÀM — đang thực hiện"),
        ("partial", "MỘT PHẦN — làm một phần"),
        ("not_started", "CHƯA BẮT ĐẦU — chưa làm"),
    ]
    for status, label in legend_entries:
        c.ax.add_patch(Rectangle((legend_x, legend_y), 20, 20,
                                 fc=COLORS[status], ec=COLORS[status]))
        c.text(legend_x + 28, legend_y + 10, label, legend_fp, "#37474F", va="center")
        legend_x += 28 + c.measure(label, legend_fp)[0] + 34

    c.text(legend_x, legend_y + 10, "|  [x] xong    [~] một phần    [ ] chưa",
           legend_fp, "#37474F", va="center")

    c.ax.plot([margin, c.W - margin], [968, 968], color="#E0E0E0", lw=1.5)
    c.text(c.W / 2.0, 1012, FOOTER, c.prop(15, "bold"), "#37474F", ha="center")
    c.text(margin, 1046, "Nguồn: roadmap_data.json → update_roadmap.py",
           c.prop(10.5), "#90A4AE")

    return c, overflow


def verify(c):
    c.fig.canvas.draw()
    renderer = c.fig.canvas.get_renderer()
    outside = []
    outside_card = []
    overlaps = []
    boxes = []
    for artist, card in c.texts:
        bb = artist.get_window_extent(renderer)
        x0, x1 = bb.x0, bb.x1
        y_top = c.H - bb.y1
        y_bottom = c.H - bb.y0
        if x0 < 0 or x1 > c.W or y_top < 0 or y_bottom > c.H:
            outside.append((artist.get_text()[:40], round(x0), round(y_top),
                            round(x1), round(y_bottom)))
        if card is not None:
            cx0, cy0, cx1, cy1 = card
            if (x0 < cx0 + 3 or x1 > cx1 - 3 or y_top < cy0 + 2
                    or y_bottom > cy1 - 2):
                outside_card.append((artist.get_text()[:40], round(x0), round(y_top),
                                     round(x1), round(y_bottom)))
        boxes.append((artist.get_text(), bb))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.5 and oy > 0.5:
                overlaps.append((boxes[i][0][:26], boxes[j][0][:26],
                                 round(ox, 1), round(oy, 1)))
    return outside, outside_card, overlaps


def main():
    print("Đọc:", JSON_PATH)
    data = load_data()
    stages = data["stages"]

    md = build_markdown(data)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Ghi: {MD_PATH} ({md.count(chr(10)) + 1} dòng, "
          f"{os.path.getsize(MD_PATH)} byte)")

    os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
    canvas, overflow = build_png(data)
    canvas.fig.savefig(PNG_PATH, dpi=canvas.DPI, facecolor="white")
    print(f"Ghi: {PNG_PATH} ({os.path.getsize(PNG_PATH)} byte)")

    outside, outside_card, overlaps = verify(canvas)
    plt.close(canvas.fig)
    print(f"Kiểm tra chữ: {len(canvas.texts)} đối tượng văn bản")
    print(f"  - ngoài khung ảnh: {len(outside)}")
    for item in outside:
        print("    ", item)
    print(f"  - tràn khỏi ô giai đoạn: {len(outside_card)}")
    for item in outside_card:
        print("    ", item)
    print(f"  - chồng lên nhau: {len(overlaps)}")
    for item in overlaps:
        print("    ", item)
    if overflow:
        print("  - ô phải thu nhỏ quá cỡ cho phép:", overflow)

    try:
        from PIL import Image
        with Image.open(PNG_PATH) as img:
            print(f"Kích thước ảnh: {img.size[0]}x{img.size[1]}")
    except Exception as exc:  # noqa: BLE001
        print(f"Không đọc được kích thước ảnh bằng PIL: {exc}")

    print("Trạng thái:", ", ".join(
        f"{i}={STATUS_VI[s['status']]}" for i, s in enumerate(stages, 1)))
    print(f"Ước lượng Giai đoạn 2: {data['stage2_percent']} "
          f"({data['stage2_percent_note']})")
    print(f"Chờ duyệt: {len(data['pending_approval'])} mục; "
          f"Sự thật đã kiểm chứng: {len(data['verified_facts'])} mục; "
          f"Nhật ký: {len(data['log'])} mục")

    if outside or outside_card or overlaps or overflow:
        print("KẾT QUẢ: FAIL - có chữ bị cắt hoặc chồng lấn, cần sửa.")
        return 1
    print("KẾT QUẢ: PASS - không có chữ bị cắt, không có chữ chồng lấn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
