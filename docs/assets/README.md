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
| Grid and frame | grid `#cccccc` 1.5 px; frame and text `#262626`                                                               |
| x-axis title   | `Latency [ms] (T4, TensorRT FP16, batch=1)`                                                                   |

## Encoding

- **One panel per dataset.** AP50:95 (the primary COCO metric) is a solid line and AP50 a dashed one (`stroke-dasharray="9 8"`), both 3.5 px with round joins and caps.

- **Colour identifies the model family** and stays the same in every chart.

    | Family                           | Colour    |
    | -------------------------------- | --------- |
    | RF-DETR, RF-DETR-Seg, Keypoint   | `#9467bd` |
    | YOLO11, YOLOv11-Seg, YOLO11-pose | `#f28e2b` |
    | YOLO26, YOLO26-Seg, YOLO26-pose  | `#000000` |
    | LW-DETR                          | `#4c78a8` |
    | D-FINE                           | `#9c7660` |
    | YOLOv8-Seg                       | `#edc948` |

- **Points are size labels, not markers.** The table's `Size` value (A, F, P, N, S, M, L, XL, 2XL, T, X) is drawn in the family colour on a 4 px white halo. A family with a single model, such as RF-DETR Keypoint, is a filled star instead of a line.

- **RF-DETR is drawn last** so its lines sit on top; dashed lines are drawn before solid ones.

## Legend

- Centred under the panels, outside the plot area.
- First row: the model families, as a solid line in the family colour.
- Second row, only when a chart shows both metrics: `mAP@50:95` as a solid line and `mAP@50` as a dashed line, both grey `#8c8c8c`, so they do not read as a family.

## Embedding

- Each SVG carries a `<title>` and a `<desc>` that name the plotted metrics.
- `docs/index.md` sets `width="840"` and a `height` matching the canvas ratio (460 for a 1096-high canvas, 442 for 1052). Update both when the canvas changes.
- Keep the alt text in `README.md`, `docs/index.md` and `docs/learn/benchmarks.md` true to the chart, for example the values it quotes.
