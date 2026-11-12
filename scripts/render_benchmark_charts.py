# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Render the accuracy-vs-latency benchmark charts from the tables in ``docs/learn/benchmarks.md``.

Writes ``docs/assets/benchmark-{detection,segmentation,keypoints}.svg``. Every chart has the same canvas width and
panel size, so pages that stretch them to one width show them at one scale. AP50:95 (the primary COCO metric) is a
solid line and AP50 a dashed one; colour identifies the model family, and a family with a single model is a star.
Panels shown side by side share one y range. Rerun after editing the tables::

    python scripts/render_benchmark_charts.py
"""

import argparse
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

Point = tuple[float, float, str]
Layer = tuple[str, bool, list[Point]]
LegendEntry = tuple[str, str, str]

FONT = "DejaVu Sans, Arial, sans-serif"
INK = "#262626"
GRID = "#cccccc"
METRIC_INK = "#8c8c8c"
DASH = "9 8"
FS_TITLE, FS_TICK, FS_AXIS, FS_POINT, FS_LEGEND = 30, 23, 22, 20, 23
CANVAS_W, PANEL_W, PANEL_H, TOP = 2000, 870, 830, 70
LEFT_X, RIGHT_X, CENTER_X = 115, 1115, 565
LEGEND_ROW = 44
LATENCY_LABEL = "Latency [ms] (T4, TensorRT FP16, batch=1)"

DET_COLORS = {"RF-DETR": "#9467bd", "YOLO11": "#f28e2b", "YOLO26": "#000000", "LW-DETR": "#4c78a8", "D-FINE": "#9c7660"}
SEG_COLORS = {"RF-DETR-Seg": "#9467bd", "YOLOv11-Seg": "#f28e2b", "YOLOv8-Seg": "#edc948", "YOLO26-Seg": "#000000"}
SEG_NAMES = {"YOLOv11-Seg": "YOLOv11", "YOLOv8-Seg": "YOLOv8", "YOLO26-Seg": "YOLO26"}
KP_COLORS = {"RF-DETR Keypoint": "#9467bd", "YOLO26-pose": "#000000", "YOLO11-pose": "#f28e2b"}
METRIC_ROW: list[LegendEntry] = [(METRIC_INK, "solid", "mAP@50:95"), (METRIC_INK, "dash", "mAP@50")]


@dataclass(frozen=True)
class PanelSpec:
    """Title, axis label, data ranges and tick positions of one chart panel."""

    title: str
    ylabel: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_ticks: Sequence[float]
    y_ticks: Sequence[float]


def read_table(markdown: str, heading: str) -> list[dict[str, str]]:
    """Return the rows of the first Markdown table after the ``## heading`` line.

    Header cells lose inline HTML, so ``COCO AP<sub>50</sub>`` becomes ``COCO AP50``.

    Examples:
        >>> md = "## Demo\\n\\n| Architecture | AP<sub>50</sub> |\\n| :-: | :-: |\\n| RF-DETR | 1.0 |\\n"
        >>> read_table(md, "Demo")
        [{'Architecture': 'RF-DETR', 'AP50': '1.0'}]
    """
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    _, found, section = f"\n{markdown}".partition(f"\n## {heading}\n")
    if not found:
        raise ValueError(f"Heading '## {heading}' not found.")
    for line in section.splitlines():
        if not line.startswith("|"):
            if header:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if header is None:
            header = [re.sub(r"<[^>]+>", "", cell) for cell in cells]
        elif not set(line) <= set("|:- "):
            rows.append(dict(zip(header, cells)))
    if not rows:
        raise ValueError(f"No table rows found under '## {heading}'.")
    return rows


def series(rows: list[dict[str, str]], families: dict[str, str], column: str) -> dict[str, list[Point]]:
    """Group the (latency, metric, size) points of each family in ``families``, sorted by latency.

    Rows of other families and rows with a dash in ``column`` are skipped.

    Examples:
        >>> rows = [{"Architecture": "A", "Size": "S", "Latency (ms)": "2.0", "AP": "50.0"},
        ...         {"Architecture": "A", "Size": "N", "Latency (ms)": "1.0", "AP": "40.0"}]
        >>> series(rows, {"A": "#000000"}, "AP")
        {'A': [(1.0, 40.0, 'N'), (2.0, 50.0, 'S')]}
    """
    out: dict[str, list[Point]] = {family: [] for family in families}
    for row in rows:
        if row["Architecture"] in families and row[column] not in ("—", ""):
            out[row["Architecture"]].append((float(row["Latency (ms)"]), float(row[column]), row["Size"]))
    return {family: sorted(points) for family, points in out.items()}


def star(cx: float, cy: float, radius: float) -> str:
    """Return the SVG ``points`` of a five-point star centred on (cx, cy).

    Examples:
        >>> len(star(0, 0, 10).split())
        10
        >>> star(0, 0, 10).split()[0]
        '0.0,-10.0'
    """
    points = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * 0.45
        angle = -math.pi / 2 + i * math.pi / 5
        points.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
    return " ".join(points)


def _text(
    x: float, y: float, label: str, size: int, *, anchor: str = "middle", bold: bool = False, extra: str = ""
) -> str:
    """One SVG ``<text>`` element in ink colour; ``extra`` holds further attributes, each with a leading space.

    Examples:
        >>> _text(10, 20, "N", 18, bold=True)
        '<text x="10.0" y="20.0" text-anchor="middle" font-size="18" font-weight="700" fill="#262626">N</text>'
    """
    weight = ' font-weight="700"' if bold else ""
    fill = "" if ' fill="' in extra else f' fill="{INK}"'
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{weight}{fill}{extra}>{label}</text>'
    )


def _render_frame(x0: float, spec: PanelSpec) -> list[str]:
    """Title, border, grid, tick labels and axis titles of one panel."""
    y0, w, h = TOP, PANEL_W, PANEL_H
    (x_lo, x_hi), (y_lo, y_hi) = spec.x_range, spec.y_range
    out = [
        _text(x0 + w / 2, y0 - 22, spec.title, FS_TITLE, bold=True),
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fff" stroke="{INK}" stroke-width="1.5"/>',
    ]
    for tick in spec.x_ticks:
        x = x0 + (tick - x_lo) / (x_hi - x_lo) * w
        out.append(f'<path d="M{x:.1f} {y0}V{y0 + h}" stroke="{GRID}" stroke-width="1.5"/>')
        out.append(_text(x, y0 + h + 32, f"{tick:g}", FS_TICK))
    for tick in spec.y_ticks:
        y = y0 + h - (tick - y_lo) / (y_hi - y_lo) * h
        out.append(f'<path d="M{x0} {y:.1f}H{x0 + w}" stroke="{GRID}" stroke-width="1.5"/>')
        out.append(_text(x0 - 10, y + 8, f"{tick:g}", FS_TICK, anchor="end"))
    mid_y = y0 + h / 2
    out.append(_text(x0 + w / 2, y0 + h + 70, LATENCY_LABEL, FS_AXIS, bold=True))
    rotate = f' transform="rotate(-90 {x0 - 72} {mid_y:.1f})"'
    out.append(_text(x0 - 72, mid_y, spec.ylabel, FS_AXIS, bold=True, extra=rotate))
    return out


def render_panel(x0: float, spec: PanelSpec, layers: list[Layer]) -> list[str]:
    """Render one panel at canvas x ``x0``; ``layers`` are (colour, dashed, points), drawn in order."""
    (x_lo, x_hi), (y_lo, y_hi) = spec.x_range, spec.y_range

    def to_xy(point: Point) -> tuple[float, float]:
        x = x0 + (point[0] - x_lo) / (x_hi - x_lo) * PANEL_W
        return x, TOP + PANEL_H - (point[1] - y_lo) / (y_hi - y_lo) * PANEL_H

    clip_id = f"c{x0:g}"
    out = [f'<g font-family="{FONT}">', *_render_frame(x0, spec)]
    clip_rect = f'<rect x="{x0 - 14}" y="{TOP - 14}" width="{PANEL_W + 28}" height="{PANEL_H + 28}"/>'
    out.append(f'<clipPath id="{clip_id}">{clip_rect}</clipPath>')
    out.append(f'<g clip-path="url(#{clip_id})">')
    for color, dashed, points in layers:
        if len(points) == 1:
            out.append(
                f'<polygon points="{star(*to_xy(points[0]), 20)}" fill="{color}" stroke="#fff" stroke-width="2"/>'
            )
            continue
        dash = f' stroke-dasharray="{DASH}"' if dashed else ""
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(to_xy, points))
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3.5" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
    for color, _, points in layers:
        for point in points if len(points) > 1 else []:
            x, y = to_xy(point)
            halo = f' fill="{color}" stroke="#fff" stroke-width="4" paint-order="stroke"'
            out.append(_text(x, y + 7, point[2], FS_POINT, bold=True, extra=halo))
    out.append("</g></g>")
    return out


def render_legend(rows: list[list[LegendEntry]]) -> tuple[list[str], int]:
    """Render centred legend rows under the panels; return the SVG lines and the canvas height they need.

    Each entry is (colour, kind, label) with kind ``"solid"``, ``"dash"`` or ``"star"``.

    Examples:
        >>> lines, height = render_legend([[("#000000", "solid", "A")], [("#000000", "dash", "B")]])
        >>> height
        1096
    """
    y = TOP + PANEL_H + 122
    out = [f'<g font-family="{FONT}">']
    for entries in rows:
        widths = [46 + 12 + 13 * len(label) + 40 for _, _, label in entries]
        x = CANVAS_W / 2 - sum(widths) / 2
        for (color, kind, label), width in zip(entries, widths):
            if kind == "star":
                out.append(f'<polygon points="{star(x + 23, y, 14)}" fill="{color}"/>')
            else:
                dash = f' stroke-dasharray="{DASH}"' if kind == "dash" else ""
                stroke = f'stroke="{color}" stroke-width="3.5" stroke-linecap="round"{dash}'
                out.append(f'<path d="M{x:.0f} {y}H{x + 46:.0f}" {stroke}/>')
            out.append(_text(x + 58, y + 8, label, FS_LEGEND, anchor="start"))
            x += width
        y += LEGEND_ROW
    out.append("</g>")
    return out, y - LEGEND_ROW + 30


def two_metric_layers(
    families: dict[str, str], solid: dict[str, list[Point]], dashed: dict[str, list[Point]]
) -> list[Layer]:
    """Dashed layers under solid ones; the first family is drawn last so it sits on top."""
    order = list(families)[::-1]
    return [(families[f], True, dashed[f]) for f in order] + [(families[f], False, solid[f]) for f in order]


def svg_document(height: int, title: str, description: str, body: list[str]) -> str:
    """Wrap ``body`` in an accessible SVG document on a white canvas."""
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{height}" '
        f'viewBox="0 0 {CANVAS_W} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{title}</title>',
        f'<desc id="description">{description}</desc>',
        f'<rect width="{CANVAS_W}" height="{height}" fill="#ffffff"/>',
    ]
    return "\n".join([*head, *body, "</svg>", ""])


def _render_detection(markdown: str) -> str:
    rows = read_table(markdown, "Detection")
    x_range, x_ticks = (0.5, 18.5), range(2, 19, 2)
    y_range, y_ticks = (27, 92), range(30, 91, 5)  # shared by both panels so they compare at a glance
    coco = two_metric_layers(
        DET_COLORS, series(rows, DET_COLORS, "COCO AP50:95"), series(rows, DET_COLORS, "COCO AP50")
    )
    rf100 = two_metric_layers(
        DET_COLORS, series(rows, DET_COLORS, "RF100VL AP50:95"), series(rows, DET_COLORS, "RF100VL AP50")
    )
    coco_spec = PanelSpec("Object Detection (MS COCO)", "MS COCO mAP", x_range, y_range, x_ticks, y_ticks)
    rf100_spec = PanelSpec("Object Detection (RF100-VL)", "RF100-VL average mAP", x_range, y_range, x_ticks, y_ticks)
    legend, height = render_legend([[(color, "solid", family) for family, color in DET_COLORS.items()], METRIC_ROW])
    return svg_document(
        height,
        "Object detection accuracy vs latency: RF-DETR, YOLO11, YOLO26, LW-DETR, D-FINE",
        "Two panels, MS COCO and RF100-VL average, each plotting mAP@50:95 (solid) and mAP@50 (dashed) against "
        "T4 TensorRT FP16 batch-1 latency in milliseconds. Values are listed in the benchmark tables.",
        render_panel(LEFT_X, coco_spec, coco) + render_panel(RIGHT_X, rf100_spec, rf100) + legend,
    )


def _render_segmentation(markdown: str) -> str:
    rows = read_table(markdown, "Segmentation")
    layers = two_metric_layers(
        SEG_COLORS, series(rows, SEG_COLORS, "COCO AP50:95"), series(rows, SEG_COLORS, "COCO AP50")
    )
    spec = PanelSpec(
        "Instance Segmentation (MS COCO)", "MS COCO mask mAP", (1, 23), (25, 76), range(2, 23, 2), range(25, 76, 5)
    )
    legend, height = render_legend(
        [[(color, "solid", SEG_NAMES.get(family, family)) for family, color in SEG_COLORS.items()], METRIC_ROW]
    )
    return svg_document(
        height,
        "Instance segmentation accuracy vs latency: RF-DETR-Seg, YOLO26, YOLOv8, YOLOv11",
        "COCO mask mAP@50:95 (solid) and mAP@50 (dashed) against T4 TensorRT FP16 batch-1 latency in milliseconds. "
        "Values are listed in the benchmark tables.",
        render_panel(CENTER_X, spec, layers) + legend,
    )


def _render_keypoints(markdown: str) -> str:
    points = series(read_table(markdown, "Keypoints"), KP_COLORS, "COCO AP50:95")
    layers: list[Layer] = [(KP_COLORS[family], False, points[family]) for family in list(KP_COLORS)[::-1]]
    spec = PanelSpec(
        "Keypoint Detection (MS COCO)",
        "MS COCO keypoint mAP@50:95",
        (1, 12),
        (45, 75),
        range(2, 13, 2),
        range(45, 76, 5),
    )
    legend, height = render_legend(
        [[(color, "star" if len(points[family]) == 1 else "solid", family) for family, color in KP_COLORS.items()]]
    )
    return svg_document(
        height,
        "Keypoint detection accuracy vs latency: RF-DETR Keypoint, YOLO26-pose, YOLO11-pose",
        "COCO person keypoint mAP@50:95 against T4 TensorRT FP16 batch-1 latency in milliseconds. "
        "Values are listed in the benchmark tables.",
        render_panel(CENTER_X, spec, layers) + legend,
    )


def render_charts(markdown: str) -> dict[str, str]:
    """Render every benchmark chart from the benchmarks page text; keys are the file names under ``docs/assets``."""
    return {
        "benchmark-detection.svg": _render_detection(markdown),
        "benchmark-segmentation.svg": _render_segmentation(markdown),
        "benchmark-keypoints.svg": _render_keypoints(markdown),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write the charts next to the other docs assets."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--benchmarks", type=Path, default=root / "docs/learn/benchmarks.md", help="Benchmarks page to read."
    )
    parser.add_argument("--output-dir", type=Path, default=root / "docs/assets", help="Directory to write the SVGs to.")
    args = parser.parse_args(argv)
    for name, svg in render_charts(args.benchmarks.read_text(encoding="utf-8")).items():
        (args.output_dir / name).write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
