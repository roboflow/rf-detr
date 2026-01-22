---
hide:
- navigation
---

# RF-DETR: Real-Time SOTA Detection and Segmentation Model

RF-DETR is a real-time transformer architecture for object detection and instance segmentation developed by Roboflow. Built on a DINOv2 vision transformer backbone, RF-DETR delivers state-of-the-art accuracy and latency trade-offs on Microsoft COCO and RF100-VL.

RF-DETR uses a DINOv2 vision transformer backbone and supports both detection and instance segmentation in a single, consistent API. All core models and code are released under the Apache 2.0 license.

## Install

You can install and use `rfdetr` in a [**Python>=3.10**](https://www.python.org/) environment. For detailed installation instructions, including installing from source, and setting up a local development environment, check out our [install](learn/install/) page.


!!! example "Installation"
    [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
    [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
    [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE.md)
    [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

    === "pip"
        ```bash
        pip install rfdetr
        ```

    === "uv"
        ```bash
        uv pip install rfdetr
        ```

        For uv projects:

        ```bash
        uv add rfdetr
        ```

## Quickstart

<div class="grid cards" markdown>

- **Run Detection Models**

    ---

    Load and run pre-trained RF-DETR detection models.

    [:octicons-arrow-right-24: Tutorial](learn/run/detection/)

- **Run Segmentation Models**

    ---

    Load and run pre-trained RF-DETR-Seg segmentation models.

    [:octicons-arrow-right-24: Tutorial](learn/run/segmentation/)

- **Train Models**

    ---

    Learn how to fine-tune RF-DETR models for detection and segmentation.

    [:octicons-arrow-right-24: Tutorial](/learn/train/)

</div>


## Benchmarks

RF-DETR achieves the best accuracy–latency trade-off among real-time object detection and instance segmentation models — both on COCO and on the more demanding RF100-VL benchmark (domain adaptability). For detailed benchmark tables and methodology, check out our [benchmarks](learn/benchmarks/) page.

### Detection

<img alt="Pareto front – detection" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_object_detection.png" style="max-width: 840px; height: auto;" />

| Architecture   | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | RF100VL AP<sub>50</sub> | RF100VL AP<sub>50:95</sub> | Latency (ms)  | Params (M)  | Resolution  |
|----------------|----------------------|-------------------------|-------------------------|----------------------------|---------------|-------------|-------------|
| RF-DETR-N      | 67.6                 | 48.4                    | 85.0                    | 57.7                       | 2.3           | 30.5        | 384×384     |
| RF-DETR-S      | 72.1                 | 53.0                    | 86.7                    | 60.2                       | 3.5           | 32.1        | 512×512     |
| RF-DETR-M      | 73.6                 | 54.7                    | 87.4                    | 61.2                       | 4.4           | 33.7        | 576×576     |
| RF-DETR-L      | 75.1                 | 56.5                    | 88.2                    | 62.2                       | 6.8           | 33.9        | 704×704     |
| RF-DETR-XL     | 77.4                 | 58.6                    | 88.5                    | 62.9                       | 11.5          | 126.4       | 700×700     |
| RF-DETR-2XL    | 78.5                 | 60.1                    | 89.0                    | 63.2                       | 17.2          | 126.9       | 880×880     |


### Segmentation

<img alt="Pareto front – segmentation" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_instance_segmentation.png" style="max-width: 840px; height: auto;" />

|   Architecture   | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
|------------------|----------------------|-------------------------|--------------|------------|------------|
| RF-DETR-Seg-N    | 63.0                 | 40.3                    | 3.4          | 33.6       | 312×312    |
| RF-DETR-Seg-S    | 66.2                 | 43.1                    | 4.4          | 33.7       | 384×384    |
| RF-DETR-Seg-M    | 68.4                 | 45.3                    | 5.9          | 35.7       | 432×432    |
| RF-DETR-Seg-L    | 70.5                 | 47.1                    | 8.8          | 36.2       | 504×504    |
| RF-DETR-Seg-XL   | 72.2                 | 48.8                    | 13.5         | 38.1       | 624×624    |
| RF-DETR-Seg-2XL  | 73.1                 | 49.9                    | 21.8         | 38.6       | 768×768    |
