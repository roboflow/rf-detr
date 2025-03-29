# RF-DETR: SOTA Real-Time Object Detection Model

[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)

[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time, transformer-based object detection model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is comparable speed to current real-time objection models.

**RF-DETR is small enough to run on the edge, making it an ideal model for deployments that need both strong accuracy and real-time performance.**

## Results

We validated the performance of RF-DETR on both Microsoft COCO and the RF100-VL benchmarks.

![rf-detr-coco-rf100-vl-9](https://github.com/user-attachments/assets/fdb6c31d-f11f-4518-8377-5671566265a4)

<details>
<summary>RF100-VL benchmark results</summary>
<img src="https://github.com/user-attachments/assets/e61a7ba4-5294-40a9-8cd7-4fc924639924" alt="rf100-vl-map50">
</details>

| Model            | params<br><sup>(M) | mAP<sup>COCO val<br>@0.50:0.95 | mAP<sup>RF100-VL<br>Average @0.50 | mAP<sup>RF100-VL<br>Average @0.50:95 | Total Latency<br><sup>T4 bs=1<br>(ms) |
|------------------|--------------------|--------------------------------|-----------------------------------|---------------------------------------|---------------------------------------|
| D-FINE-M         | 19.3               | <ins>55.1</ins>                | N/A                               | N/A                                   | 6.3                                   |
| LW-DETR-M        | 28.2               | 52.5                           | 84.0                              | 57.5                                  | 6.0                                   |
| YOLO11m          | 20.0               | 51.5                           | 84.9                              | 59.7                                  | <ins>5.7</ins>                        |
| YOLOv8m          | 28.9               | 50.6                           | 85.0                              | 59.8                                  | 6.3                                   |
| RF-DETR-B        | 29.0               | 53.3                           | <ins>86.7</ins>                   | <ins>60.3</ins>                       | 6.0                                   |


<details>
<summary>RF100-VL benchmark notes</summary>

- The "Total Latency" reported here is measured on a T4 GPU using TensorRT10 FP16 (ms/img) and was introduced by LW-DETR. Unlike transformer-based models, YOLO models perform Non-Maximum Suppression (NMS) after generating predictions to refine bounding box candidates. While NMS boosts accuracy, it also slightly reduces speed due to the additional computation required, which varies with the number of objects in an image. Notably, many YOLO benchmarks include NMS in accuracy measurements but exclude it from speed metrics. By contrast, our benchmarking—following LW-DETR’s approach—factors in NMS latency to provide a uniform measure of the total time needed to obtain a final result across all models on the same hardware.

- D-FINE’s fine-tuning capability is currently unavailable, making its domain adaptability performance inaccessible. The authors [caution](https://github.com/Peterande/D-FINE) that “if your categories are very simple, it might lead to overfitting and suboptimal performance.” Furthermore, several open issues ([#108](https://github.com/Peterande/D-FINE/issues/108), [#146](https://github.com/Peterande/D-FINE/issues/146), [#169](https://github.com/Peterande/D-FINE/issues/169), [#214](https://github.com/Peterande/D-FINE/issues/214)) currently prevent successful fine-tuning. We have opened an additional issue in hopes of ultimately benchmarking D-FINE with RF100-VL.
</details>

## News

- `2025/03/20`: We release RF-DETR real-time object detection model. **Code and checkpoint for RF-DETR-large and RF-DETR-base are available.**

## Installation

Pip install the `rfdetr` package in a [**Python>=3.9**](https://www.python.org/) environment.

```bash
pip install rfdetr
```

<details>
<summary>From source</summary>

By installing RF-DETR from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/rf-detr.git
```

</details>


## Inference

RF-DETR comes out of the box with checkpoints pre-trained on the Microsoft COCO dataset.

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

<details>
<summary>Video inference</summary>

```python
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

def callback(frame, index):
    detections = model.predict(frame, threshold=0.5)
        
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

process_video(
    source_path=<SOURCE_VIDEO_PATH>,
    target_path=<TARGET_VIDEO_PATH>,
    callback=callback
)
```

</details>

<details>
<summary>Webcam inference</summary>

```python
import cv2
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    detections = model.predict(frame, threshold=0.5)
    
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</details>

![rf-detr-coco-results-2](https://media.roboflow.com/rf-detr/example_grid.png)


### Model Variants

RF-DETR is available in two variants: RF-DETR-B 29M [`RFDETRBase`](https://github.com/roboflow/rf-detr/blob/ed1af5144343ea52d3d26ce466719d064bb92b9c/rfdetr/detr.py#L133) and RF-DETR-L 128M [`RFDETRLarge`](https://github.com/roboflow/rf-detr/blob/ed1af5144343ea52d3d26ce466719d064bb92b9c/rfdetr/detr.py#L140). The corresponding COCO pretrained checkpoints are automatically loaded when you initialize either class.

### Input Resolution

Both model variants support configurable input resolutions. A higher resolution usually improves prediction quality by capturing more detail, though it can slow down inference. You can adjust the resolution by passing the `resolution` argument when initializing the model. `resolution` value must be divisible by `56`.

```python
model = RFDETRBase(resolution=560)
```

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

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=<OUTPUT_PATH>)
```

### Batch size

Different GPUs have different amounts of VRAM (video memory), which limits how much data they can handle at once during training. To make training work well on any machine, you can adjust two settings: `batch_size` and `grad_accum_steps`. These control how many samples are processed at a time. The key is to keep their product equal to 16 — that’s our recommended total batch size. For example, on powerful GPUs like the A100, set `batch_size=16` and `grad_accum_steps=1`. On smaller GPUs like the T4, use `batch_size=4` and `grad_accum_steps=4`. We use a method called gradient accumulation, which lets the model simulate training with a larger batch size by gradually collecting updates before adjusting the weights.

### Multi-GPU training

You can fine-tune RF-DETR on multiple GPUs using PyTorch’s Distributed Data Parallel (DDP). Create a `main.py` script that initializes your model and calls `.train()` as usual than run it in terminal.

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py
```

Replace `8` in the `--nproc_per_node argument` with the number of GPUs you want to use. This approach creates one training process per GPU and splits the workload automatically. Note that your effective batch size is multiplied by the number of GPUs, so you may need to adjust your `batch_size` and `grad_accum_steps` to maintain the same overall batch size.

### Result checkpoints

During training, two model checkpoints (the regular weights and an EMA-based set of weights) will be saved in the specified output directory. The EMA (Exponential Moving Average) file is a smoothed version of the model’s weights over time, often yielding better stability and generalization.

### Logging with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit that helps you visualize and track training metrics. With TensorBoard set up, you can train your model and keep an eye on the logs to monitor performance, compare experiments, and optimize model training.

<details>
<summary>Launch TensorBoard</summary>

- To use TensorBoard locally, navigate to your project directory and run:

    ```bash
    tensorboard --logdir <OUTPUT_DIR>
    ```

    Then open `http://localhost:6006/` in your browser to view your logs.

- To use TensorBoard in Google Colab run:

    ```bash
    %load_ext tensorboard
    %tensorboard --logdir <OUTPUT_DIR>
    ```

    This will start a TensorBoard session in the Google Colab environment.
  
</details>

### Logging with Weights and Biases (W&B)

[W&B](https://www.wandb.ai) similar to TensorBoard is a very powerful platform to monitor training metrics. It is cloud-based and offers a more complex feature set compared to TensorBoard.

<details>
<summary>Get Started</summary>

- To use W&B, make sure you are logged in:

    ```bash
    wandb login
    ```

    You will need to specify an API Key for your account that can be retrieved at https://wandb.ai/authorize.

- For those familiar with W&B, you can specify the `project_name` and `run_name` in the `model.train` method:

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    detections = model.train(dataset_dir=<DATASET_PATH>, project_name=<PROJECT_NAME>, run_name=<RUN_NAME>)
    ```

    For those unfamiliar with W&B: projects are collections of related machine learning experiments, while runs are individual units of computation within a project that record specific experiments, such as training a model or conducting hyperparameter tuning. Not specifying a name, will result in random names provided by W&B.
  
</details>

### Load and run fine-tuned model

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

detections = model.predict(<IMAGE_PATH>)
```

## ONNX export

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency. To export your model, simply initialize it and call the `.export()` method.

```python
from rfdetr import RFDETRBase

model = RFDETRBase()

model.export()
```

This command saves the ONNX model to the `output` directory.

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
