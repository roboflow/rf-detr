---
description: Run RF-DETR models on images, video files, webcams, and RTSP streams.
---

# Run pretrained models

You can run RF-DETR with [Inference](https://github.com/roboflow/inference), an open source computer vision inference server. The Nano, Small, Medium, and Large models are trained on the [Microsoft COCO dataset](https://universe.roboflow.com/microsoft/coco). XLarge and 2XLarge models require `pip install rfdetr[plus]`. These models use the PML 1.0 license.

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

`predict()` accepts RGB PIL images and NumPy arrays. It accepts normalized CHW tensors and normalized BCHW tensor batches. It also accepts image paths, HTTP image URLs, `pathlib.Path` objects, and `os.PathLike` objects. An extensionless HTTP URL is treated as an image URL.

Use `PredictionInput` in type annotations. Import it from `rfdetr` or `rfdetr.prediction`.

One image returns one `sv.Detections` or `sv.KeyPoints` object. A list or tuple of images uses one batched forward pass and returns a list of results.

## Predict on files and folders

Pass a directory or glob pattern to predict on supported image and video files. RF-DETR sorts matches. Directory searches are not recursive. A glob can use `**` for recursive matching.

```python
from pathlib import Path
from rfdetr import RFDETRSmall

model = RFDETRSmall()
results = model.predict(Path("images"), threshold=0.5)
results = model.predict("images/*.jpg", threshold=0.5)
```

Directories, globs, manifests, and video files return a flat list by default (`stream=False`). Use `stream=True` to process one image or video frame at a time. A scalar still image returns one result. A list or tuple of images uses one batched forward pass.

Use `batch` to set the number of finite images or frames per forward pass. The default is `1`. RF-DETR predicts the final partial batch when the source length is not divisible by `batch`. An inference-compiled model may require the batch size used during compilation.

Text and CSV manifests list sources, one per line or cell. They can contain images, videos, and live sources. Relative paths are resolved from the manifest directory. A `.streams` file lists cameras, network streams, and local video files for concurrent reading. Each result batch follows the source order in that file. The stream stops when a source with a known frame count ends. Screen capture is not supported inside a `.streams` file.

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

## Stream a video, camera, network, or screen source

Set `stream=True` to get a lazy generator. Each item is a `sv.Detections` or `sv.KeyPoints` result for one image or video frame. The default `batch=1` predicts one frame at a time for a single source.

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

Pass an integer webcam index, a numeric camera index as text, or an explicit live stream URL as the source. RF-DETR treats RTSP, RTSPS, RTMP, and TCP URLs as live streams. For HTTP URLs, use `.m3u8`, `.mjpg`, or `.mjpeg` to identify a live stream. An extensionless HTTP URL is treated as an image by default. Put an extensionless live HTTP URL in a `.streams` file.

Live sources also run with `stream=False`. This call stores every result in memory and warns after source resolution identifies a live input. This includes live sources inside text and CSV manifests. Recorded YouTube videos are finite and do not trigger this warning. Use `stream=True` to keep memory use bounded. Capture keeps the first frame. By default, it replaces pending frames with the latest frame. This policy can skip frames when prediction is slower than capture. Set `stream_buffer=True` to queue up to 30 pending frames per source. If a stream read fails, RF-DETR tries to reconnect three times, then raises an error.

Set `vid_stride` to capture every Nth frame for prediction. It defaults to `1`. The latest-frame policy can still skip frames when prediction is slower than capture.

Video, camera, network stream, and screen capture require `uv pip install "rfdetr[stream]"`. Use `screen` to capture the desktop. Add an optional monitor index, or a monitor index and crop rectangle. For example, use `screen 1 100 100 640 480`. YouTube page URLs also require this optional extra for `yt-dlp` URL resolution. Recorded YouTube videos end at the final frame. Live YouTube URLs continue as live sources.

The generator closes its capture at the end, on error, or when you call `close()`. If an OpenCV backend blocks while it reads, capture cleanup waits for that read to return. Network read timeouts depend on backend support. When a finite network stream stops, RF-DETR logs a warning because OpenCV cannot distinguish EOF from a read failure.

OpenCV converts captured video frames from BGR to RGB before prediction.

RF-DETR supports many sources and controls that Ultralytics supports. Its `predict()` method is not a drop-in Ultralytics API. It returns Supervision objects. Video decoding depends on the installed OpenCV backends and codecs.

## Include source images

`include_source_image=True` by default. For `sv.Detections`, the source image is stored in metadata. For `sv.KeyPoints`, it is stored in each object's data.

RF-DETR copies source images for result storage. It converts them to `uint8` RGB arrays. Set `include_source_image=False` to omit source images.

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
