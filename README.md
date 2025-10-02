# RF-DETR: SOTA Real-Time Detection and Segmentation Model

[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)

[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time, transformer-based object detection and instance segmentation model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO object detection benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is fastest and most accurate for its size when compared current real-time objection models.

On image segmentation, RF-DETR Seg (Preview) is 3x faster and more accurate than the largest YOLO when evaluated on the Microsoft COCO Segmentation benchmark, defining a new real-time state-of-the-art for the industry-standard benchmark in segmentation model evaluation.

[![rf-detr-tutorial-banner](https://github.com/user-attachments/assets/555a45c3-96e8-4d8a-ad29-f23403c8edfd)](https://youtu.be/-OvpdLAElFA)

## News

- `2025/10/02`: We release RF-DETR-Seg (Preview), a preview of our instance segmentation head for RF-DETR.
- `2025/07/23`: We release three new checkpoints for RF-DETR: Nano, Small, and Medium.
    - RF-DETR Base is now deprecated. We recommend using RF-DETR Medium which offers subtantially better accuracy at comparable latency.
- `2025/03/20`: We release RF-DETR real-time object detection model. **Code and checkpoint for RF-DETR-large and RF-DETR-base are available.**
- `2025/04/03`: We release early stopping, gradient checkpointing, metrics saving, training resume, TensorBoard and W&B logging support.
- `2025/05/16`: We release an 'optimize_for_inference' method which speeds up native PyTorch by up to 2x, depending on platform.

## Results

RF-DETR achieves state-of-the-art performance on both the Microsoft COCO and the RF100-VL benchmarks.

The below tables shows how RF-DETR performs when validated on the Microsoft COCO benchmark for object detection and image segmentation.

### Object Detection Benchmarks

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto1.png)

| Architecture | COCO AP<sub>50</sub> |  COCO AP<sub>50:95</sub>   |  RF100VL AP<sub>50</sub>   | RF100VL AP<sub>50:95</sub>  |  Latency (ms)   |   Params (M) |   Resolution  |
|:------------:|:--------------------:|:--------------------------:|:--------------------------:|:---------------------------:|:---------------:|:------------:|:-------------:|
|  RF-DETR-N   |         67.6         |            48.4            |            84.1            |            57.1             |      2.32       |         30.5 |       384x384 |
|  RF-DETR-S   |         72.1         |            53.0            |            85.9            |            59.6             |      3.52       |         32.1 |       512x512 |
|  RF-DETR-M   |         73.6         |            54.7            |            86.6            |            60.6             |      4.52       |         33.7 |       576x576 |
|   YOLO11-N   |         52.0         |            37.4            |            81.4            |            55.3             |      2.49       |          2.6 |       640x640 |
|   YOLO11-S   |         59.7         |            44.4            |            82.3            |            56.2             |      3.16       |          9.4 |       640x640 |
|   YOLO11-M   |         64.1         |            48.6            |            82.5            |            56.5             |      5.13       |         20.1 |       640x640 |
|   YOLO11-L   |         65.3         |            50.2            |             x              |              x              |      6.65       |         25.3 |       640x640 |
|   YOLO11-X   |         66.5         |            51.2            |             x              |              x              |      11.92      |         56.9 |       640x640 |
|  LW-DETR-T   |         60.7         |            42.9            |             x              |              x              |      1.91       |         12.1 |       640x640 |
|  LW-DETR-S   |         66.8         |            48.0            |            84.5            |            58.0             |      2.62       |         14.6 |       640x640 |
|  LW-DETR-M   |         72.0         |            52.6            |            85.2            |            59.4             |      4.49       |         28.2 |       640x640 |
|   D-FINE-N   |         60.2         |            42.7            |            83.6            |            57.7             |      2.12       |          3.8 |       640x640 |
|   D-FINE-S   |         67.6         |            50.7            |            84.5            |            59.9             |      3.55       |         10.2 |       640x640 |
|   D-FINE-M   |         72.6         |            55.1            |            84.6            |            60.2             |      5.68       |         19.2 |       640x640 |

[See our benchmark notes in the RF-DETR documentation.](https://rfdetr.roboflow.com/learn/benchmarks/)

_We are actively working on RF-DETR Large and X-Large models using the same techniques we used to achieve the strong accuracy that RF-DETR Medium attains. This is why RF-DETR Large and X-Large is not yet reported on our pareto charts and why we haven't benchmarked other models at similar sizes. Check back in the next few weeks for the launch of new RF-DETR Large and X-Large models._

### Instance Segmentation Benchmarks

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto_segmentation.png)

| Model Name              | Reported Latency | Reported mAP | Measured Latency | Measured mAP |
|-------------------------|------------------|--------------|------------------|--------------|
| RF-DETR Seg-Preview@312 |                  |              | 3.3              | 39.4         |
| YOLO11n-Seg             | 1.8              | 32.0         | 3.6              | 30.0         |
| YOLOv8n-Seg             |                  | 30.5         | 3.5              | 28.3         |
| RF-DETR Seg-Preview@384 |                  |              | 4.5              | 42.7         |
| YOLO11s-Seg             | 2.9              | 37.8         | 4.6              | 35.0         |
| YOLOv8s-Seg             |                  | 36.8         | 4.2              | 34.0         |
| RF-DETR Seg-Preview@432 |                  |              | 5.6              | 44.3         |
| YOLO11m-Seg             | 6.3              | 41.5         | 6.9              | 38.5         |
| YOLOv8m-Seg             |                  | 40.8         | 7.0              | 37.3         |
| YOLO11l-Seg             | 7.8              | 42.9         | 8.3              | 39.5         |
| YOLOv8l-Seg             |                  | 42.6         | 9.7              | 39.0         |
| YOLO11x-Seg             | 15.8             | 43.8         | 13.7             | 40.1         |
| YOLOv8x-Seg             |                  | 43.4         | 14.0             | 39.5         |

For more information on measuring end-to-end latency for models, see our open source [Single Artifact Benchmarking tool](https://github.com/roboflow/single_artifact_benchmarking).

## Installation

To install RF-DETR, install the `rfdetr` package in a [**Python>=3.9**](https://www.python.org/) environment with `pip`:

```bash
pip install rfdetr
```

<details>
<summary>Install from source</summary>

<br>

By installing RF-DETR from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/rf-detr.git
```

</details>

## Inference

The easiest path to deployment is using Roboflow's [Inference](https://github.com/roboflow/inference) package. 

The code below lets you run `rfdetr-base` on an image:

```python
import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = get_model("rfdetr-base")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
```

To use segmentation, use the `rfdetr-seg-preview` model ID. This model will return segmentation masks from a RF-DETR-Seg (Preview) model trained on the Microsoft COCO dataset.

## Predict

You can also use the .predict method to perform inference during local development. The `.predict()` method accepts various input formats, including file paths, PIL images, NumPy arrays, and torch tensors. Please ensure inputs use RGB channel order. For `torch.Tensor` inputs specifically, they must have a shape of `(3, H, W)` with values normalized to the `[0..1)` range. If you don't plan to modify the image or batch size dynamically at runtime, you can also use `.optimize_for_inference()` to get up to 2x end-to-end speedup, depending on platform.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

### Train a Model

You can fine-tune an RF-DETR Nano, Small, Medium, and Base model with a custom dataset using the `rfdetr` Python package.

[Learn how to train an RF-DETR model.](https://rfdetr.roboflow.com/learn/train/)

## Documentation

Visit our [documentation website](https://rfdetr.roboflow.com) to learn more about how to use RF-DETR.

## License

Both the code and the weights pretrained on the COCO dataset are released under the [Apache 2.0 license](https://github.com/roboflow/r-flow/blob/main/LICENSE).

## Acknowledgements

Our work is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159). Thanks to their authors for their excellent work!

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```

## Contribute

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.

<div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://discuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
