---
description: Run RF-DETR models on images, video files, webcams, and RTSP streams.
---

# Run pretrained models

You can run RF-DETR with [Inference](https://github.com/roboflow/inference), an open source computer vision inference server. The Nano, Small, Medium, and Large models are trained on the [Microsoft COCO dataset](https://universe.roboflow.com/microsoft/coco). XLarge and 2XLarge models require `pip install rfdetr[plus]` and use the PML 1.0 license.

## Run on an image

Use the Inference server to run a pretrained model and annotate an image:

```python
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = get_model("rfdetr-small")
predictions = model.infer(image, confidence=0.5)[0]
detections = sv.Detections.from_inference(predictions)
labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
sv.plot_image(annotated_image)
```

Replace the image URL with an image of your choice.

<figure markdown="span">
![](https://media.roboflow.com/rfdetr-docs/annotated_image_base.jpg){ width=300 }
<figcaption>RF-DETR predictions</figcaption>
</figure>

## Predict with the Python package

Use `RFDETRSmall` to run prediction with the RF-DETR Python package:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
detections = model.predict("https://media.roboflow.com/dog.jpeg", threshold=0.5)
```

`predict()` accepts RGB PIL images, NumPy arrays, normalized CHW tensors, image paths, HTTP URLs, `pathlib.Path` objects, and `os.PathLike` objects.

One image returns one `sv.Detections` or `sv.KeyPoints` object. A list or tuple of images uses one batched forward pass and returns a list of results.

## Predict on files and folders

Pass a directory or glob pattern to predict on multiple supported image and video files. RF-DETR sorts matches, and directory searches are not recursive.

```python
from pathlib import Path
from rfdetr import RFDETRSmall

model = RFDETRSmall()
results = model.predict(Path("images"), threshold=0.5)
results = model.predict("images/*.jpg", threshold=0.5)
```

Directories, globs, and video files return a flat list when `stream=False` (the default). Use `stream=True` to process one image or video frame at a time.

## Run on a video file and save the results

Use `supervision.process_video` to annotate a video and save the output:

```python
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSmall()


def callback(frame, index):
    detections = model.predict(frame[:, :, ::-1], threshold=0.5)
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame


sv.process_video(
    source_path="<SOURCE_VIDEO_PATH>",
    target_path="<TARGET_VIDEO_PATH>",
    callback=callback,
)
```

Set `SOURCE_VIDEO_PATH` to the input video path and `TARGET_VIDEO_PATH` to the output video path.

## Stream a video, webcam, or RTSP source

Set `stream=True` to get a lazy generator. Each item is a `sv.Detections` or `sv.KeyPoints` result for one image or video frame.

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
stream = model.predict("video.mp4", threshold=0.5, stream=True)

try:
    for frame_detections in stream:
        print(len(frame_detections))
finally:
    stream.close()
```

Pass an integer webcam index or RTSP URL as the source. Webcam and RTSP sources require `stream=True` because they can produce results without a bound. The generator closes its capture at the end, on error, or when you call `close()`.

OpenCV converts captured video frames from BGR to RGB before prediction.

## Include source images

`include_source_image=True` by default. For `sv.Detections`, the source image is stored in metadata. For `sv.KeyPoints`, it is stored in each object's data.

Writable `uint8` NumPy arrays stay attached by reference. Changes to those arrays also change the stored image. RF-DETR copies read-only `uint8` arrays and converts arrays with other data types. Set `include_source_image=False` to omit source images.

```python
detections = model.predict("image.jpg", include_source_image=False)
```

## Batch inference

Pass a list or tuple of images for one batched forward pass. The results match the input order.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSmall()
urls = [
    "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog-3.jpeg",
]
images = [Image.open(io.BytesIO(requests.get(url).content)) for url in urls]
detections_list = model.predict(images, threshold=0.5)

for image, detections in zip(images, detections_list):
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    sv.plot_image(annotated_image)
```
