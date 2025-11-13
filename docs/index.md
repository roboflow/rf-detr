---
hide:
- navigation
---

# RF-DETR: SOTA Real-Time Detection and Segmentation Model

## Introduction

RF-DETR is a real-time, transformer-based object detection and instance segmentation model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO object detection benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is fastest and most accurate for its size when compared current real-time objection models.

On image segmentation, RF-DETR Seg (Preview) is 3x faster and more accurate than the largest YOLO when evaluated on the Microsoft COCO Segmentation benchmark, defining a new real-time state-of-the-art for the industry-standard benchmark in segmentation model evaluation.

RF-DETR is small enough to run on the edge using [Inference](https://github.com/roboflow/inference), making it an ideal model for deployments that need both strong accuracy and real-time performance.

RF-DETR is documented in the [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554) research paper, available on Arxiv.

## Benchmark Results

### Object Detection

We validated the performance of RF-DETR on both Microsoft COCO and the RF100-VL benchmarks.

[See our full benchmarks.](learn/benchmarks/)

<img src="https://media.roboflow.com/rfdetr/pareto1.png" style="max-height: 50rem" />

### Instance Segmentation

We benchmarked RF-DETR on the Microsoft COCO dataset for segmentation. Our results are below.

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto_seg.png)

## 💻 Install

You can install and use `rfdetr` in a
[**Python>=3.9**](https://www.python.org/) environment.

!!! example "Installation"

    === "pip (recommended)"
        [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
        [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
        [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

        ```bash
        pip install rfdetr
        ```

    === "poetry"
        [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
        [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
        [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

        ```bash
        poetry add rfdetr
        ```

    === "uv"
        [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
        [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
        [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

        ```bash
        uv pip install rfdetr
        ```

        For uv projects:

        ```bash
        uv add rfdetr
        ```

    === "rye"
        [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
        [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
        [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE.md)
        [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

        ```bash
        rye add rfdetr
        ```

!!! example "git clone (for development)"
    === "virtualenv"
        ```bash
        # clone repository and navigate to root directory
        git clone --depth 1 -b develop https://github.com/roboflow/rf-detr.git
        cd rf-detr

        # setup python environment and activate it
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip

        # installation
        pip install -e "."
        ```

    === "uv"
        ```bash
        # clone repository and navigate to root directory
        git clone --depth 1 -b develop https://github.com/roboflow/rf-detr.git
        cd rf-detr

        # setup python environment and activate it
        uv venv
        source .venv/bin/activate

        # installation
        uv pip install -r pyproject.toml -e . --all-extras

        ```

## 🚀 Quickstart

<div class="grid cards" markdown>

- **Run a Pre-Trained Model**

    ---

    Load and run a pre-trained RF-DETR model.

    [:octicons-arrow-right-24: Tutorial](/learn/pretrained)

- **Train an RF-DETR Model**

    ---

    Learn how to train an RF-DETR model with the `rfdetr` Python package.

    [:octicons-arrow-right-24: Tutorial](/learn/train/)

- **Deploy an RF-DETR Model**

    ---

    Learn how to deploy an RF-DETR model in the cloud and on your device.

    [:octicons-arrow-right-24: Tutorial](/learn/deploy/)

</div>