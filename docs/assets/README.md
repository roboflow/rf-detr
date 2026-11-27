# Benchmark charts

`benchmark-detection.svg`, `benchmark-segmentation.svg` and `benchmark-keypoints.svg` plot accuracy against latency. The README and the docs pages (`docs/index.md`, `docs/learn/benchmarks.md`) embed them. Every value comes from the Detection, Segmentation and Keypoints tables in `docs/learn/benchmarks.md`, so redraw the charts whenever a table changes, and never plot a value the tables do not list.

## Data

- x: latency in ms from the `Latency (ms)` column (T4, TensorRT FP16, batch 1).
- y: the AP columns in percent.
    - Detection: `COCO AP50`, `COCO AP50:95`, `RF100VL AP50`, `RF100VL AP50:95`.
    - Segmentation: `COCO AP50`, `COCO AP50:95`.
    - Keypoints: `COCO AP50:95` only.
- Plot every row of a plotted family, the Plus sizes included. Skip rows with a dash in the plotted column, such as the author-reported SAM 3 row.

## Layout

All three charts use the same canvas width and panel size. The pages stretch them to one display width, so equal panels keep text and lines at one scale.

| Element        | Value                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| Canvas         | 2000 wide, white background; height fits the legend (1096 with two legend rows, 1052 with one)                |
| Panel          | 870 × 830, top edge at y = 70                                                                                 |
| Two panels     | left edges at x = 115 and x = 1115 (detection: MS COCO left, RF100-VL right)                                  |
| One panel      | left edge at x = 565, centred (segmentation, keypoints)                                                       |
| Side-by-side y | panels in one chart share one y range and tick set, so values compare at a glance                             |
| Fonts          | DejaVu Sans / Arial; panel title 30 bold, tick labels 23, axis titles 22 bold, size labels 20 bold, legend 23 |
| Grid and frame | grid `#e8d2ff` 1.5 px; frame and text `#1d1d1f`                                                               |
| x-axis title   | `Latency [ms] (T4, TensorRT FP16, batch=1)`                                                                   |

## Encoding

- **One panel per dataset.** AP50:95 (the primary COCO metric) is a solid line and AP50 a dashed one (`stroke-dasharray="9 8"`), both 3.5 px with round joins and caps.

- **Colour identifies the model family** and stays the same in every chart.

    | Family                           | Colour    | Why                                                                                     |
    | -------------------------------- | --------- | --------------------------------------------------------------------------------------- |
    | RF-DETR, RF-DETR-Seg, Keypoint   | `#8315f9` | brand purple, the docs theme primary; the hero series                                   |
    | YOLO11, YOLOv11-Seg, YOLO11-pose | `#d97706` | amber, the one warm accent; lighter than RF-DETR, 3.2:1 on white for its bold labels    |
    | YOLO26, YOLO26-Seg, YOLO26-pose  | `#1d1d1f` | ink; the only family darker than RF-DETR                                                |
    | LW-DETR                          | `#1e9e89` | teal; lighter than RF-DETR and clearly darker than D-FINE, whose lines it nearly covers |
    | D-FINE                           | `#b9b4c6` | light grey; the lightest line in the detection chart                                    |
    | YOLOv8-Seg                       | `#4fd6bf` | mint; lighter than YOLO11 and the opposite hue, since their lines nearly coincide       |

- **Palette.** The colours follow the docs theme: `#8315f9` is its primary and the grid `#e8d2ff` its light purple (both in `docs/stylesheets/rf.css`). Ink `#1d1d1f` and grey `#6f6c79` come from the palette the docs diagrams share. Amber `#d97706`, teal `#1e9e89`, light grey `#b9b4c6` and mint `#4fd6bf` are derived for these charts, not brand colours. Hue alone does not separate the families: every family differs from RF-DETR in lightness under CIE L\*, CSS `grayscale()` and Rec. 601 greyscale, so RF-DETR stays apart in greyscale and for colour-blind readers. The families whose lines nearly coincide (LW-DETR and D-FINE, YOLO11 and YOLOv8-Seg) differ in both hue and lightness. YOLO11 and LW-DETR share a lightness, but their hues are opposite and their lines do not cross. Mint and light grey are faint on white (about 2:1), so their size labels read less strongly than the others. The grid is drawn over the frame, so a grid line that falls on a panel edge takes the frame colour `#1d1d1f`; in the light grid colour it would hide that edge.

- **Points are size labels, not markers.** The table's `Size` value (A, F, P, N, S, M, L, XL, 2XL, T, X) is drawn in the family colour on a 4 px white halo. A family with a single model, such as RF-DETR Keypoint, is a filled star instead of a line.

- **RF-DETR is drawn last** so its lines sit on top; dashed lines are drawn before solid ones.

## Legend

- Centred under the panels, outside the plot area.
- First row: the model families, as a solid line in the family colour.
- Second row, only when a chart shows both metrics: `mAP@50:95` as a solid line and `mAP@50` as a dashed line, both mid grey `#6f6c79`, far darker than the D-FINE grey, so they do not read as a family.

## Embedding

- Each SVG carries a `<title>` and a `<desc>` that name the plotted metrics.
- `docs/index.md` sets `width="840"` and a `height` matching the canvas ratio (460 for a 1096-high canvas, 442 for 1052). Update both when the canvas changes.
- Keep the alt text in `README.md`, `docs/index.md` and `docs/learn/benchmarks.md` true to the chart, for example the values it quotes.

## Diagram style

Every SVG in `docs/` is an opaque rounded card, so it reads on both the light and the dark docs theme. Light diagrams use a white card with a `#E8D2FF` border; the dark `how-it-works` diagrams use a `#1A1224` card.

| Role                 | Colour                          | Source                                                |
| -------------------- | ------------------------------- | ----------------------------------------------------- |
| Logo purple          | `#6706CE`                       | `roboflow-logo.svg`                                   |
| Primary purple       | `#8315F9`                       | docs theme primary, `docs/stylesheets/rf.css`         |
| Mid purple           | `#A351FB`                       | `docs/stylesheets/cookbooks_card.css`                 |
| Light purple         | `#E8D2FF`                       | `docs/stylesheets/rf.css`                             |
| Deep purple (titles) | `#4A176F`                       | `docs/stylesheets/rf.css`                             |
| Mint                 | `#A5F9EA`                       | `docs/stylesheets/cookbooks_card.css`                 |
| Ink, grey, tint      | `#1D1D1F`, `#6F6C79`, `#F3EEFF` | shared by the docs diagrams, not defined in the theme |
| Teal, rule grey      | `#0B7A6B`, `#4FD6BF`, `#B9B4C6` | derived for the diagrams, not brand colours           |

Animated diagrams (`training-loop.svg`, `early-stopping.svg`, `how-map-works.svg`) use plain CSS keyframes, with no JavaScript or SMIL, so they play inside an `<img>`. The static markup is the final frame: with `prefers-reduced-motion` the animation is switched off and the finished picture shows. Keyframes cannot use variables, so these files are generated; the generators are not in the repo, so edit the SVG by hand only for small changes.
