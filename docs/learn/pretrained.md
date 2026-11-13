---
description: Choose and run COCO-pretrained RF-DETR models for detection (Atto to 2XLarge), instance segmentation (Nano to 2XLarge), and keypoints (preview) with the rfdetr package.
---

# Pretrained Models

RF-DETR ships COCO-pretrained checkpoints for object detection, instance segmentation, and keypoint detection (preview). The weights download automatically the first time you create a model, so a pretrained model runs without any training.

--8<-- "docs/snippets/license-rfdetr.md"

## Choose a model

| Task                  | Model classes                                   | Predicts            | Details                                                                 |
| --------------------- | ----------------------------------------------- | ------------------- | ----------------------------------------------------------------------- |
| Object detection      | `RFDETRAtto` to `RFDETR2XLarge` (9 sizes)       | 80 COCO classes     | [Detection checkpoints](run/detection.md#pre-trained-checkpoints)       |
| Instance segmentation | `RFDETRSegNano` to `RFDETRSeg2XLarge` (6 sizes) | 80 COCO classes     | [Segmentation checkpoints](run/segmentation.md#pre-trained-checkpoints) |
| Keypoint detection    | `RFDETRKeypointPreview`                         | 17 person keypoints | [Keypoint checkpoint](run/keypoints.md#pre-trained-checkpoints)         |

`RFDETRSmall` is a good default for detection. Larger sizes are more accurate and slower; compare them on the [Benchmarks](benchmarks.md) page. Use the [COCO accuracy cookbook](../../cookbooks/coco-accuracy/) to reproduce the COCO accuracy of every checkpoint on your own hardware. Atto, Femto, Pico, XLarge and 2XLarge detection need `pip install "rfdetr[plus]"`.

## Run on an image

```python
import supervision as sv
from rfdetr import RFDETRSmall

model = RFDETRSmall()
detections = model.predict("https://media.roboflow.com/dog.jpeg", threshold=0.5)

labels = list(detections.data["class_name"])
annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"].copy(), detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

`predict()` accepts RGB PIL images and NumPy arrays. It accepts normalized CHW tensors and normalized BCHW tensor batches. It also accepts image paths, HTTP image URLs, `pathlib.Path` objects, and `os.PathLike` objects. An extensionless HTTP URL is treated as an image URL.

One image returns one `sv.Detections` or `sv.KeyPoints` object. An iterable of images, such as a list, tuple, or generator, returns a list of results from one batched forward pass. A BCHW tensor returns one result per image.

## Run on videos and streams

The model's `predict()` handles images only. Use `RFDETRInference` for videos, folders, globs, manifests, cameras, network streams, and lazy results. `RFDETRInference(model)` borrows a loaded model without copying its weights. It also loads checkpoints and exported models, and accepts every image input above.

```python
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())

results = predictor.predict("<SOURCE_VIDEO_PATH>", threshold=0.5, stream=True)
try:
    for frame_index, detections in enumerate(results):
        print(frame_index, len(detections))
finally:
    results.close()
```

Video, camera, and network stream sources need `pip install "rfdetr[stream]"`. See [Run on Videos and Streams](run/videos-and-streams.md) for webcams, RTSP streams, YouTube, folders, and annotated video output.

## Include source images

`include_source_image=True` by default. For `sv.Detections`, the source image is stored in metadata. For `sv.KeyPoints`, it is stored in each object's data.

RF-DETR copies source images for result storage. It converts them to `uint8` RGB arrays. Set `include_source_image=False` to omit source images.

```python
detections = model.predict("image.jpg", include_source_image=False)
```

## Batch Inference

You can provide `.predict()` with either a single image or a list of images. When multiple images are supplied, they are processed together in a single forward pass, resulting in a corresponding list of detections.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

model = RFDETRSmall()

urls = [
    "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog-3.jpeg",
]

images = [Image.open(io.BytesIO(requests.get(url).content)) for url in urls]

detections_list = model.predict(images, threshold=0.5)

for image, detections in zip(images, detections_list):
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections.data["class_name"], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    sv.plot_image(annotated_image)
```
