---
description: Run RF-DETR on folders, video files, video URLs, webcams, RTSP streams, and YouTube with RFDETRInference, as an eager list or a lazy stream of Supervision results.
---

# Run on Videos and Streams

`RFDETRInference.predict()` accepts more than images. Pass it a folder, a glob, a video file or URL, a webcam index, an RTSP or other live stream URL, a YouTube page, or a manifest that lists any of these. It returns the same `sv.Detections` or `sv.KeyPoints` results as image prediction, one per image or video frame.

The model's own `predict()` handles images only. Wrap a loaded model with `RFDETRInference(model)` to use the sources on this page. The wrapper borrows the model without copying its weights. It also loads checkpoints and exported models, so the same calls work for an ONNX, TensorRT, or other exported artifact.

Video, camera, network stream, YouTube, and screen capture need the optional stream dependencies:

```bash
uv pip install "rfdetr[stream]"
```

The clips below come from RF-DETR Small at a 0.5 confidence threshold. Each one is annotated from the results of the call shown above it. `threshold` also takes one value per entry of `class_names` on every source on this page, as described in [Set a confidence threshold for each class](detection.md#set-a-confidence-threshold-for-each-class).

## Predict on a video file

Pass a video path to get one result per frame. Set `stream=True` to get a lazy generator that decodes and predicts frames as you iterate, instead of collecting every result first.

```python
import supervision as sv
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

results = predictor.predict("people-walking.mp4", threshold=0.5, stream=True, batch=4)
frames = sv.get_video_frames_generator("people-walking.mp4")
video_info = sv.VideoInfo.from_video_path("people-walking.mp4")

with sv.VideoSink("people-walking-annotated.mp4", video_info) as sink:
    try:
        for frame, detections in zip(frames, results):
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, list(detections.data["class_name"]))
            sink.write_frame(annotated)
    finally:
        results.close()
```

<video controls muted loop playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/589e6c18-17e4-4f7b-b63b-9d7c410134a3"></video>

_A 13.6-second 1080p clip at 25 fps (341 frames). `batch=4` runs four frames per forward pass and predicts the final single frame on its own._

## Predict on a video URL

An HTTP or HTTPS URL that ends in a video suffix, such as `.mp4`, is read as a video. RF-DETR streams it through OpenCV without downloading the whole file first.

```python
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())
source = "https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4"

results = predictor.predict(source, threshold=0.5, stream=True, include_source_image=False)
try:
    for frame_index, detections in enumerate(results):
        print(frame_index, len(detections))
finally:
    results.close()
```

<video controls muted loop playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/a2fdd960-a6f0-4a0a-b50d-dcfb79e1a613"></video>

_The 42.5-second `vehicles-2.mp4` clip, 1080p at 29.97 fps (1,275 frames)._

## Predict on a webcam

Pass an integer camera index, or the index as a numeric string such as `"0"`. The example below reads from an OBS Virtual Camera on device `1` and runs on the CPU.

```python
import torch
from rfdetr import RFDETRInference, RFDETRSmall

torch.set_num_threads(4)
predictor = RFDETRInference(RFDETRSmall(device="cpu"))

results = predictor.predict(1, threshold=0.5, stream=True)
try:
    for detections in results:
        print(len(detections))
finally:
    results.close()
```

<video controls muted loop playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/75d511b7-167c-43f4-95a9-ce0be02b55fc"></video>

_OBS Virtual Camera playing a highway video into RF-DETR Small on the CPU._

## Predict on an RTSP stream

RTSP, RTSPS, RTMP, TCP, UDP, and SRT URLs are read as live streams. If a read fails, RF-DETR reopens the stream up to three times before it raises, so a publisher restart does not end the generator.

```python
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())

results = predictor.predict("rtsp://127.0.0.1:8554/vehicles", threshold=0.5, stream=True)
try:
    for detections in results:
        print(len(detections))
finally:
    results.close()
```

<video controls muted loop playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/1212b7cb-73b8-4dbf-a957-cfd5a51e4631"></video>

_A local MediaMTX server looping `vehicles-2.mp4` over RTSP. The same generator kept predicting after the publisher was restarted._

## Predict on a YouTube video

Pass a YouTube page URL. RF-DETR resolves it to a video stream with `yt-dlp`, which `rfdetr[stream]` installs, and never downloads the full file. Recorded videos end at their last frame. Live YouTube URLs continue as live sources.

```python
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())

results = predictor.predict(
    "https://www.youtube.com/watch?v=3FXUw98rrUY", threshold=0.5, stream=True, include_source_image=False
)
try:
    for detections in results:
        print(len(detections))
finally:
    results.close()
```

<video controls muted loop playsinline preload="metadata" width="100%" src="https://github.com/user-attachments/assets/9c1909c6-c732-45c5-ab17-528851a529e5"></video>

_A 45.6-second 1080p YouTube video at 23.976 fps (1,093 frames)._

## Predict on files and folders

Pass a directory or glob pattern to predict on supported image and video files. RF-DETR sorts matches. Directory searches are not recursive. A glob can use `**` for recursive matching.

```python
from pathlib import Path
from rfdetr import RFDETRInference, RFDETRSmall

predictor = RFDETRInference(RFDETRSmall())
results = predictor.predict(Path("images"), threshold=0.5)
results = predictor.predict("images/*.jpg", threshold=0.5)
```

Text manifests list one source per line. CSV manifests list one source per cell; when the first row has a `source` or `path` cell, that row is a header and only its column is read. Manifests can contain images, videos, and live sources. Relative paths are resolved from the manifest directory. A `.streams` file lists cameras, network streams, and local video files for concurrent reading. Each result batch follows the source order in that file. The stream stops when a source with a known frame count ends. Screen capture is not supported inside a `.streams` file.

## Eager lists and lazy streams

Directories, globs, manifests, and video files return a flat list by default (`stream=False`). Use `stream=True` to process one image or video frame at a time. Each item of the generator is a `sv.Detections` or `sv.KeyPoints` result for one image or video frame.

The flat list keeps every result in memory. With the default `include_source_image=True`, each result also holds its full RGB source image, so one minute of 1080p video at 30 fps keeps about 11 GB. RF-DETR warns before a video, directory, or glob whose estimated source images exceed 1 GiB. It estimates a video from its reported frame count and frame size, and a directory or glob from its image count and the size of its first image. Use `stream=True` or `include_source_image=False` to keep memory use bounded. A scalar still image returns one result. A list or tuple of images uses one batched forward pass unless you pass `batch`.

Live sources also run with `stream=False`. This call stores every result in memory and warns after source resolution identifies a live input. This includes live sources inside text and CSV manifests. Recorded YouTube videos are finite and do not trigger this warning. Use `stream=True` to keep memory use bounded.

An empty collection of sources is quiet; an empty location is loud. A text or CSV manifest with no entries, and a tensor batch with no frames, return `[]` with `stream=False` and yield nothing with `stream=True`. An empty list or tuple yields nothing with `stream=True` and raises `ValueError` with `stream=False`. A directory or glob without supported files raises `FileNotFoundError`, and a `.streams` file without sources raises `ValueError`, in both modes.

## Batching and frame stride

Use `batch` to set the number of finite images or frames per forward pass. The default, `batch=None`, runs a BCHW tensor or a list of images as one forward pass and predicts other expanded sources one image or frame at a time. Pass an integer to split a BCHW tensor or a list of images into smaller batches and bound the memory of each pass. RF-DETR predicts the final partial batch when the source length is not divisible by `batch`. An inference-compiled model may require the batch size used during compilation.

Set `vid_stride` to capture every Nth frame for prediction. It defaults to `1`. For live sources, the latest-frame policy can still skip frames when prediction is slower than capture.

## Live capture behavior

Pass an integer webcam index, a numeric camera index as text, or an explicit live stream URL as the source. RF-DETR treats RTSP, RTSPS, RTMP, TCP, UDP, and SRT URLs as live streams. When the installed OpenCV cannot open the URL, for example because it lacks support for the protocol, RF-DETR raises `ValueError: Could not open the video source.` For HTTP URLs, use `.m3u8`, `.mjpg`, or `.mjpeg` to identify a live stream. An extensionless HTTP URL is treated as an image by default. Put an extensionless live HTTP URL in a `.streams` file.

Capture keeps the first frame. By default, it replaces pending frames with the latest frame. This policy can skip frames when prediction is slower than capture. Set `stream_buffer=True` to queue up to 30 pending frames per source. If a stream read fails, RF-DETR tries to reconnect three times, then raises an error. Finite videos in `.streams` always use the bounded buffer, even with `stream_buffer=False`, so prediction speed does not discard frames.

Use `screen` to capture the desktop. Add an optional monitor index, or a monitor index and crop rectangle. For example, use `screen 1 100 100 640 480`.

The generator closes its capture at the end, on error, or when you call `close()`. If an OpenCV backend blocks while it reads, capture cleanup waits for that read to return. Network read timeouts depend on backend support. When a finite network stream stops, RF-DETR logs a warning because OpenCV cannot distinguish EOF from a read failure.

OpenCV converts captured video frames from BGR to RGB before prediction.

RF-DETR supports many sources and controls that Ultralytics supports. `RFDETRInference.predict()` is not a drop-in Ultralytics API. It returns Supervision objects. Video decoding depends on the installed OpenCV backends and codecs.
