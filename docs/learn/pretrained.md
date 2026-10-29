---
description: Run pre-trained RF-DETR models (Nano to 2XLarge) on images, video, webcam, and RTSP streams. COCO-trained with real-time DINOv2 backbone.
---

You can run any of the four supported RF-DETR base models -- Nano, Small, Medium, Large -- with [Inference](https://github.com/roboflow/inference), an open source computer vision inference server. The base models are trained on the [Microsoft COCO dataset](https://universe.roboflow.com/microsoft/coco). These base models are Apache 2.0, so they're commercially usable out of the box. XLarge and 2XLarge detection models are also available via `pip install rfdetr[plus]` under the Roboflow Platform Model License (PML 1.0): commercial use is included when you deploy on Roboflow, and commercial self-hosting requires the Enterprise add-on.

=== "Run on an Image"

    To run RF-DETR on an image, use the following code:

    ```python
    import os
    import supervision as sv
    from inference import get_model
    from PIL import Image
    from io import BytesIO
    import requests

    url = "https://media.roboflow.com/dog.jpeg"
    image = Image.open(BytesIO(requests.get(url).content))

    model = get_model("rfdetr-large")

    predictions = model.infer(image, confidence=0.5)[0]

    detections = sv.Detections.from_inference(predictions)

    labels = [prediction.class_name for prediction in predictions.predictions]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    sv.plot_image(annotated_image)
    ```

    Above, replace the image URL with any image you want to use with the model.

    Here are the results from the code above:

    <figure markdown="span">
    ![](https://media.roboflow.com/rfdetr-docs/annotated_image_base.jpg){ width=300 }
    <figcaption>RF-DETR Base predictions</figcaption>
    </figure>

=== "Run on a Video File"

    To run RF-DETR on a video file, use the following code:

    ```python
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()


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

    Above, set your `SOURCE_VIDEO_PATH` and `TARGET_VIDEO_PATH` to the directories of the video you want to process and where you want to save the results from inference, respectively.

=== "Run on a Webcam Stream"

    To run RF-DETR on a webcam input, use the following code:

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = model.predict(frame[:, :, ::-1], threshold=0.5)

        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

=== "Run on an RTSP Stream"

    To run RF-DETR on an RTSP (Real Time Streaming Protocol) stream, use the following code:

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    cap = cv2.VideoCapture("<RTSP_STREAM_URL>")
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = model.predict(frame[:, :, ::-1], threshold=0.5)

        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RTSP Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

You can change the RF-DETR model that the code snippet above uses. To do so, update `rfdetr-small` to any of the following values:

- `rfdetr-nano`
- `rfdetr-small`
- `rfdetr-medium`
- `rfdetr-large`

## Predict with the Python package

Use `RFDETRSmall` to run prediction with the RF-DETR Python package:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()
detections = model.predict("https://media.roboflow.com/dog.jpeg", threshold=0.5)
```

`predict()` accepts RGB PIL images and NumPy arrays. It accepts normalized CHW tensors and normalized BCHW tensor batches. It also accepts image paths, HTTP image URLs, `pathlib.Path` objects, and `os.PathLike` objects. An extensionless HTTP URL is treated as an image URL.

One image returns one `sv.Detections` or `sv.KeyPoints` object. An iterable of images, such as a list, tuple, or generator, returns a list of results from one batched forward pass. A BCHW tensor returns one result per image.

The model's `predict()` handles images only. Use `RFDETRInference` for videos, folders, globs, manifests, cameras, network streams, and lazy results. `RFDETRInference(model)` borrows a loaded model without copying its weights. It also loads checkpoints and exported models, and accepts every image input above.

## Predict on files and folders

Pass a directory or glob pattern to predict on supported image and video files. RF-DETR sorts matches. Directory searches are not recursive. A glob can use `**` for recursive matching.

```python
from pathlib import Path
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())
results = predictor.predict(Path("images"), threshold=0.5)
results = predictor.predict("images/*.jpg", threshold=0.5)
```

Directories, globs, manifests, and video files return a flat list by default (`stream=False`). Use `stream=True` to process one image or video frame at a time.

The flat list keeps every result in memory. With the default `include_source_image=True`, each result also holds its full RGB source image, so one minute of 1080p video at 30 fps keeps about 11 GB. RF-DETR warns before a video, directory, or glob whose estimated source images exceed 1 GiB. It estimates a video from its reported frame count and frame size, and a directory or glob from its image count and the size of its first image. Use `stream=True` or `include_source_image=False` to keep memory use bounded. A scalar still image returns one result. A list or tuple of images uses one batched forward pass unless you pass `batch`.

Use `batch` to set the number of finite images or frames per forward pass. The default, `batch=None`, runs a BCHW tensor or a list of images as one forward pass and predicts other expanded sources one image or frame at a time. Pass an integer to split a BCHW tensor or a list of images into smaller batches and bound the memory of each pass. RF-DETR predicts the final partial batch when the source length is not divisible by `batch`. An inference-compiled model may require the batch size used during compilation.

Text manifests list one source per line. CSV manifests list one source per cell; when the first row has a `source` or `path` cell, that row is a header and only its column is read. Manifests can contain images, videos, and live sources. Relative paths are resolved from the manifest directory. A `.streams` file lists cameras, network streams, and local video files for concurrent reading. Each result batch follows the source order in that file. The stream stops when a source with a known frame count ends. Screen capture is not supported inside a `.streams` file.

An empty collection of sources is quiet; an empty location is loud. A text or CSV manifest with no entries, and a tensor batch with no frames, return `[]` with `stream=False` and yield nothing with `stream=True`. An empty list or tuple yields nothing with `stream=True` and raises `ValueError` with `stream=False`. A directory or glob without supported files raises `FileNotFoundError`, and a `.streams` file without sources raises `ValueError`, in both modes.

## Stream a video, camera, network, or screen source

Set `stream=True` to get a lazy generator. Each item is a `sv.Detections` or `sv.KeyPoints` result for one image or video frame. By default, a single source predicts one frame at a time.

```python
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())
stream = predictor.predict("video.mp4", threshold=0.5, stream=True)

try:
    for frame_detections in stream:
        print(len(frame_detections))
finally:
    stream.close()
```

Pass an integer webcam index, a numeric camera index as text, or an explicit live stream URL as the source. RF-DETR treats RTSP, RTSPS, RTMP, TCP, UDP, and SRT URLs as live streams. When the installed OpenCV cannot open the URL, for example because it lacks support for the protocol, RF-DETR raises `ValueError: Could not open the video source.` For HTTP URLs, use `.m3u8`, `.mjpg`, or `.mjpeg` to identify a live stream. An extensionless HTTP URL is treated as an image by default. Put an extensionless live HTTP URL in a `.streams` file.

Live sources also run with `stream=False`. This call stores every result in memory and warns after source resolution identifies a live input. This includes live sources inside text and CSV manifests. Recorded YouTube videos are finite and do not trigger this warning. Use `stream=True` to keep memory use bounded. Capture keeps the first frame. By default, it replaces pending frames with the latest frame. This policy can skip frames when prediction is slower than capture. Set `stream_buffer=True` to queue up to 30 pending frames per source. If a stream read fails, RF-DETR tries to reconnect three times, then raises an error.

Finite videos in `.streams` always use the bounded buffer, even with `stream_buffer=False`, so prediction speed does not discard frames.

Set `vid_stride` to capture every Nth frame for prediction. It defaults to `1`. For live sources, the latest-frame policy can still skip frames when prediction is slower than capture.

Video, camera, network stream, and screen capture require `uv pip install "rfdetr[stream]"`. Use `screen` to capture the desktop. Add an optional monitor index, or a monitor index and crop rectangle. For example, use `screen 1 100 100 640 480`. YouTube page URLs also require this optional extra for `yt-dlp` URL resolution. Recorded YouTube videos end at the final frame. Live YouTube URLs continue as live sources.

The generator closes its capture at the end, on error, or when you call `close()`. If an OpenCV backend blocks while it reads, capture cleanup waits for that read to return. Network read timeouts depend on backend support. When a finite network stream stops, RF-DETR logs a warning because OpenCV cannot distinguish EOF from a read failure.

OpenCV converts captured video frames from BGR to RGB before prediction.

RF-DETR supports many sources and controls that Ultralytics supports. `RFDETRInference.predict()` is not a drop-in Ultralytics API. It returns Supervision objects. Video decoding depends on the installed OpenCV backends and codecs.

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
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()

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
