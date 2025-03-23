# RF-DETR: SOTA Real-Time Object Detection Model

[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)
[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/pypi/l/rfdetr)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![replicate](https://img.shields.io/badge/Replicate-Deploy%20with%20Replicate-blue?logo=replicate)](https://replicate.com/hardikdava/rf-detr)

RF-DETR is a real-time, transformer-based object detection model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is comparable speed to current real-time objection models.

**RF-DETR is small enough to run on the edge, making it an ideal model for deployments that need both strong accuracy and real-time performance.**

## Results

We validated the performance of RF-DETR on both Microsoft COCO and the RF100-VL benchmarks.

![rf-detr-coco-rf100-vl-8](https://media.roboflow.com/rf-detr/charts.png)

| Model            | mAP<sup>COCO val<br>@0.50:0.95 | mAP<sup>RF100-VL<br>Average @0.50 | Total Latency<br><sup>T4 bs=1<br>(ms) |
|:----------------:|:------------------------------:|:---------------------------------:|:-------------------------------------:|
| D-FINE-M         | **55.1**                       | N/A                               | 6.3                                   |
| LW-DETR-M        | 52.5                           | 84.0                              | <ins>6.0</ins>                        |
| YOLO11m          | 51.5                           | 84.9                              | **5.7**                               |
| YOLOv8m          | 50.6                           | <ins>85.0</ins>                   | 6.3                                   |
| RF-DETR-B        | <ins>53.3</ins>                | **86.7**                          | <ins>6.0</ins>                        |


<details>
<summary>Benchmark notes</summary>

The "Total Latency" reported here is measured on a T4 GPU using TensorRT10 FP16 (ms/img) and was introduced by LW-DETR. Unlike transformer-based models, YOLO models perform Non-Maximum Suppression (NMS) after generating predictions to refine bounding box candidates. While NMS boosts accuracy, it also slightly reduces speed due to the additional computation required, which varies with the number of objects in an image. Notably, many YOLO benchmarks include NMS in accuracy measurements but exclude it from speed metrics. By contrast, our benchmarking—following LW-DETR’s approach—factors in NMS latency to provide a uniform measure of the total time needed to obtain a final result across all models on the same hardware.

D-FINE’s fine-tuning capability is currently unavailable, making its domain adaptability performance inaccessible. The authors [caution](https://github.com/Peterande/D-FINE) that “if your categories are very simple, it might lead to overfitting and suboptimal performance.” Furthermore, several open issues ([#108](https://github.com/Peterande/D-FINE/issues/108), [#146](https://github.com/Peterande/D-FINE/issues/146), [#169](https://github.com/Peterande/D-FINE/issues/169), [#214](https://github.com/Peterande/D-FINE/issues/214)) currently prevent successful fine-tuning. We have opened an additional issue in hopes of ultimately benchmarking D-FINE with RF100-VL.
</details>

## News

- `2025/03/20`: We release RF-DETR real-time object detection model. **Code and checkpoint for RF-DETR-large and RF-DETR-base are available.**

## Installation

```bash
pip install rfdetr
```

## Prediction

RF-DETR comes out of the box with a checkpoint trained from the Microsoft COCO dataset.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

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

![rf-detr-coco-results-2](https://media.roboflow.com/rf-detr/example_grid.png)

## Training

### Dataset structure

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: `train`, `valid`, and `test`. Each subdirectory should contain its own `_annotations.coco.json` file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch or convert existing datasets from formats like YOLO, and then export them in COCO JSON format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

### Fine-tuning

You can fine-tune RF-DETR from pre-trained COCO checkpoints. By default, the RF-DETR-B checkpoint will be used. To get started quickly, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

```python
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4)
```

### Result checkpoints

During training, two model checkpoints (the regular weights and an EMA-based set of weights) will be saved in the specified output directory. The EMA (Exponential Moving Average) file is a smoothed version of the model’s weights over time, often yielding better stability and generalization.

### Load and run fine-tuned model

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

detections = model.predict(<IMAGE_PATH>)
```

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


## Contribution

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.
