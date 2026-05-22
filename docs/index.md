---
description: RF-DETR is a real-time transformer for object detection and instance segmentation by Roboflow. DINOv2 backbone, SOTA on COCO (60.1 AP50:95). Apache 2.0.
hide:
  - navigation
---

# RF-DETR: Real-Time SOTA Detection and Segmentation Model

RF-DETR is a real-time transformer architecture for object detection and instance segmentation developed by Roboflow. Built on a DINOv2 vision transformer backbone, RF-DETR achieves state-of-the-art accuracy–latency trade-offs: RF-DETR-L reaches 56.5 AP50:95 on COCO at 6.8 ms (NVIDIA T4, TensorRT FP16), and RF-DETR-2XL achieves 60.1 AP50:95 — the first real-time model to exceed 60 AP on COCO. Accepted at [ICLR 2026](https://arxiv.org/abs/2511.09554).

RF-DETR uses a DINOv2 vision transformer backbone and supports both detection and instance segmentation in a single, consistent API. Core models (Nano through Large) and all code are released under the Apache 2.0 license; XL and 2XLarge detection models require `rfdetr[plus]` and are provided under PML 1.0.

Developed by Isaac Robinson, Peter Robicheaux, Fedor Popov, Deva Ramanan (CMU), and Neehar Peri (CMU) at [Roboflow](https://roboflow.com). If you use RF-DETR in your research, please cite:

```bibtex
@inproceedings{robinson2026rfdetr,
  title     = {RF-DETR: Real-Time Detection Transformer},
  author    = {Robinson, Isaac and Robicheaux, Peter and Popov, Fedor and Ramanan, Deva and Peri, Neehar},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.09554}
}
```

## Install

You can install and use `rfdetr` in a [**Python>=3.10**](https://www.python.org/) environment. For detailed installation instructions, including installing from source, and setting up a local development environment, check out our [install](getting-started/install.md) page.

!!! example "Installation"

    [![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
    [![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
    [![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)
    [![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)

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

    [:octicons-arrow-right-24: Tutorial](learn/run/detection.md)

- **Run Segmentation Models**

    ---

    Load and run pre-trained RF-DETR-Seg segmentation models.

    [:octicons-arrow-right-24: Tutorial](learn/run/segmentation.md)

- **Train Models**

    ---

    Learn how to fine-tune RF-DETR models for detection and segmentation.

    [:octicons-arrow-right-24: Tutorial](learn/train/index.md)

</div>

## Tutorials

<div class="grid cards" markdown>

- **Train RF-DETR on a Custom Dataset. Video**

    ---

    ![Train RF-DETR on a Custom Dataset](https://i.ytimg.com/vi/-OvpdLAElFA/maxresdefault.jpg){ width="1280" height="720" loading="lazy" }

    End to end walkthrough of training RF-DETR on a custom dataset.

    [:octicons-arrow-right-24: Watch the video](https://www.youtube.com/watch?v=-OvpdLAElFA)

- **Deploy RF-DETR to NVIDIA Jetson. Article**

    ---

    ![Deploy RF-DETR to NVIDIA Jetson](https://blog.roboflow.com/content/images/size/w1000/format/webp/2025/06/inst-3-.png){ width="1000" height="563" loading="lazy" }

    Instructions for deploying RF-DETR on NVIDIA Jetson with Roboflow Inference.

    [:octicons-arrow-right-24: Read the tutorial](https://blog.roboflow.com/how-to-deploy-rf-detr-to-an-nvidia-jetson/)

- **Train and Deploy RF-DETR with Roboflow**

    ---

    ![Train and Deploy RF-DETR with Roboflow](https://blog.roboflow.com/content/images/size/w1000/format/webp/2025/03/img-blog-nycerebro-2.png){ width="1000" height="563" loading="lazy" }

    Cloud training and hardware deployment workflow using Roboflow.

    [:octicons-arrow-right-24: Read the tutorial](https://blog.roboflow.com/train-and-deploy-rf-detr-models-with-roboflow/)

</div>

## Benchmarks

RF-DETR achieves the best accuracy–latency trade-off among real-time object detection and instance segmentation models — both on COCO and on the more demanding RF100-VL benchmark (domain adaptability). For detailed benchmark tables and methodology, check out our [benchmarks](learn/benchmarks.md) page.

### Detection

<img alt="Pareto front — detection accuracy vs latency: RF-DETR-2XL achieves 78.5 COCO AP50 (60.1 AP50:95) at 17.2 ms; RF-DETR-L achieves 75.1 AP50 at 6.8 ms, outperforming YOLO11x at comparable latency" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_object_detection.png" width="840" height="630" style="max-width: 840px; height: auto;" />

| Architecture | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | RF100VL AP<sub>50</sub> | RF100VL AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
| ------------ | -------------------- | ----------------------- | ----------------------- | -------------------------- | ------------ | ---------- | ---------- |
| RF-DETR-N    | 67.6                 | 48.4                    | 85.0                    | 57.7                       | 2.3          | 30.5       | 384×384    |
| RF-DETR-S    | 72.1                 | 53.0                    | 86.7                    | 60.2                       | 3.5          | 32.1       | 512×512    |
| RF-DETR-M    | 73.6                 | 54.7                    | 87.4                    | 61.2                       | 4.4          | 33.7       | 576×576    |
| RF-DETR-L    | 75.1                 | 56.5                    | 88.2                    | 62.2                       | 6.8          | 33.9       | 704×704    |
| RF-DETR-XL   | 77.4                 | 58.6                    | 88.5                    | 62.9                       | 11.5         | 126.4      | 700×700    |
| RF-DETR-2XL  | 78.5                 | 60.1                    | 89.0                    | 63.2                       | 17.2         | 126.9      | 880×880    |

### Segmentation

<img alt="Pareto front — segmentation accuracy vs latency: RF-DETR-Seg-2XL achieves 73.1 COCO AP50 (49.9 AP50:95) at 21.8 ms; RF-DETR-Seg-L achieves 70.5 AP50 at 8.8 ms" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_instance_segmentation.png" width="840" height="630" style="max-width: 840px; height: auto;" />

| Architecture    | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
| --------------- | -------------------- | ----------------------- | ------------ | ---------- | ---------- |
| RF-DETR-Seg-N   | 63.0                 | 40.3                    | 3.4          | 33.6       | 312×312    |
| RF-DETR-Seg-S   | 66.2                 | 43.1                    | 4.4          | 33.7       | 384×384    |
| RF-DETR-Seg-M   | 68.4                 | 45.3                    | 5.9          | 35.7       | 432×432    |
| RF-DETR-Seg-L   | 70.5                 | 47.1                    | 8.8          | 36.2       | 504×504    |
| RF-DETR-Seg-XL  | 72.2                 | 48.8                    | 13.5         | 38.1       | 624×624    |
| RF-DETR-Seg-2XL | 73.1                 | 49.9                    | 21.8         | 38.6       | 768×768    |

## Frequently Asked Questions

**What is RF-DETR?**
RF-DETR (Roboflow Detection Transformer) is a real-time object detection and instance segmentation model from Roboflow, accepted at ICLR 2026. It uses a DINOv2 vision transformer backbone and achieves state-of-the-art accuracy–latency trade-offs on COCO (60.1 AP50:95 for RF-DETR-2XL) and RF100-VL.

**How does RF-DETR compare to YOLOv11?**
RF-DETR-L achieves 56.5 AP50:95 on COCO at 6.8 ms latency on an NVIDIA T4, outperforming YOLOv11x (54.7 AP) at lower latency. The DINOv2 backbone gives RF-DETR stronger performance on domain-shift benchmarks such as RF100-VL.

**What GPU is required to train RF-DETR?**
A CUDA-capable GPU with at least 8 GB VRAM (e.g., NVIDIA RTX 3060, T4, A10) is recommended for fine-tuning. Smaller models (RF-DETR-N and RF-DETR-S) can fit in 6 GB VRAM with reduced batch size. CPU inference is supported for evaluation.

**Which dataset formats does RF-DETR support?**
RF-DETR supports COCO JSON and YOLO-format datasets (with `dataset_file: "yolo"`). Roboflow datasets export directly to both formats. Detection and segmentation datasets use the same format — the model variant determines the task.

**Can RF-DETR run in real time?**
Yes. RF-DETR-N runs at 2.3 ms per frame on a T4 GPU (TensorRT FP16, batch 1), and RF-DETR-L at 6.8 ms — both well within real-time thresholds. ONNX and TFLite exports are available for edge deployment.

**What is the difference between RF-DETR detection and segmentation models?**
Detection models (e.g., `RFDETRLarge`) output bounding boxes. Segmentation models (e.g., `RFDETRSegLarge`) additionally output instance masks. Both share the same backbone and training API; segmentation adds a mask head and requires COCO-format segmentation annotations.

**Is RF-DETR open source?**
Yes. Core models (Nano through Large) and all training/inference code are released under the Apache 2.0 license. XLarge and 2XLarge models require the `rfdetr[plus]` package (PML 1.0 license).

**How do I fine-tune RF-DETR on a custom dataset?**
Instantiate a model and call `model.train(...)` with your dataset directory in COCO JSON or YOLO format. Example: `model = RFDETRLarge(); model.train(dataset_dir='./dataset', epochs=50, batch_size=4)`. The model downloads pretrained weights automatically and resumes from the best checkpoint.

**How do I export RF-DETR to ONNX or TensorRT?**
Call `model.export(format="onnx")` after training or loading a checkpoint. ONNX export works on CPU and produces a single `.onnx` file compatible with ONNX Runtime and OpenCV DNN. For TensorRT deployment, first export to ONNX and then convert the `.onnx` model with TensorRT tooling or helpers such as `trtexec`; this requires TensorRT and a CUDA GPU.

**Which RF-DETR model size should I use?**
RF-DETR-Nano (2.3 ms, 67.6 AP50 on COCO) is best for edge and real-time applications. RF-DETR-Large (6.8 ms, 56.5 AP50:95) offers the best accuracy–latency trade-off for server deployment. RF-DETR-2XLarge (17.2 ms, 60.1 AP50:95) maximizes accuracy when latency allows.
