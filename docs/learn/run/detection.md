---
description: Run RF-DETR object detection on images, video, and streams. Nano to 2XLarge models with 2.3-17.2 ms latency and up to 60.1 AP on COCO.
---

# Run an RF-DETR Object Detection Model

RF-DETR is a real-time transformer architecture for object detection, built on a DINOv2 vision transformer backbone (a PE-Core-T backbone for Atto, Femto and Pico). The base models are trained on the Microsoft COCO dataset and achieve state-of-the-art accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR offers model sizes from Atto to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. Nano to Large are licensed under Apache 2.0, so they're commercially usable out of the box: you can ship them in closed-source, commercial products, including self-hosted deployments, without paying a license fee or open-sourcing your code. Atto, Femto, Pico, XLarge and 2XLarge (marked with △) are provided by the [`rfdetr_plus`](https://github.com/roboflow/rf-detr_plus) extension (`pip install rfdetr[plus]`) under the [Roboflow Platform Model License (PML 1.0)](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE): commercial use is included when you deploy on Roboflow, and commercial self-hosting requires the Enterprise add-on.

<!-- COPIES: the three AP columns come from the results table of docs/cookbooks/coco-accuracy.ipynb (one run, provenance in the note below); params and resolution repeat the Detection table of docs/learn/benchmarks.md. -->

|      Size       | PyTorch FP32 AP<sub>50:95</sub> | TensorRT FP16 AP<sub>50:95</sub> | TensorRT INT8 AP<sub>50:95</sub> | Params (M) | Resolution |  License   |
| :-------------: | :-----------------------------: | :------------------------------: | :------------------------------: | :--------: | :--------: | :--------: |
|   Atto (A) △    |              30.7               |               30.6               |               23.7               |    7.4     |  380x380   |  PML 1.0   |
|   Femto (F) △   |              37.6               |               37.6               |               28.9               |    7.9     |  384x384   |  PML 1.0   |
|   Pico (P) △    |              41.7               |               41.6               |               33.2               |    8.4     |  560x560   |  PML 1.0   |
|    Nano (N)     |              48.0               |               48.0               |               47.6               |    30.5    |  384x384   | Apache 2.0 |
|    Small (S)    |              52.8               |               52.8               |               52.3               |    32.1    |  512x512   | Apache 2.0 |
|   Medium (M)    |              54.7               |               54.6               |               54.0               |    33.7    |  576x576   | Apache 2.0 |
|    Large (L)    |              56.6               |               56.5               |               56.2               |    33.9    |  704x704   | Apache 2.0 |
|  XLarge (XL) △  |              58.7               |               58.7               |               58.3               |   126.4    |  700x700   |  PML 1.0   |
| 2XLarge (2XL) △ |              60.1               |               60.0               |               59.6               |   126.9    |  880x880   |  PML 1.0   |

> △ Requires the `rfdetr_plus` extension: `pip install rfdetr[plus]`. Atto, Femto and Pico lose 7.0 to 8.7 AP in TensorRT INT8 against PyTorch FP32, so prefer FP16 engines for them.

--8<-- "docs/snippets/reproduced-coco-ap.md"

## Fastest Configuration on Apple Silicon

The configuration with the lowest end-to-end latency for each size on an Apple M4 Max, batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own device to find yours.

<!-- COPIES: every number comes from the results table of docs/cookbooks/measure-latency.ipynb (HARDWARE = "apple", one run, provenance in the note below). -->

|      Size       | PyTorch inductor FP32 | PyTorch inductor FP16 | PyTorch TorchScript FP16 |   CoreML FP32   |   CoreML FP16   | CoreML GPU FP32 |  CoreML GPU FP16  |
| :-------------: | :-------------------: | :-------------------: | :----------------------: | :-------------: | :-------------: | :-------------: | :---------------: |
|   Atto (A) △    |     12.46 (14.93)     |     12.00 (15.02)     |       9.81 (11.79)       |   5.91 (8.77)   | **3.76 (4.23)** |   7.26 (9.54)   |    7.49 (9.84)    |
|   Femto (F) △   |     14.03 (17.83)     |     13.35 (17.17)     |       8.07 (10.49)       | **6.01 (8.85)** |  25.02 (26.39)  |   6.18 (9.37)   |    6.43 (9.26)    |
|   Pico (P) △    |     15.14 (18.75)     |     15.73 (18.45)     |       9.80 (10.67)       |   6.69 (9.23)   | **5.74 (8.28)** |   7.13 (9.26)   |    6.74 (9.43)    |
|    Nano (N)     |      7.03 (7.32)      |    **6.73 (7.02)**    |       9.97 (10.67)       |   7.75 (8.65)   |  11.23 (11.74)  |   8.32 (9.40)   |    8.95 (9.57)    |
|    Small (S)    |     10.44 (10.71)     |    **9.71 (9.95)**    |      11.46 (11.77)       |  10.71 (11.07)  |  22.97 (23.60)  |  10.90 (11.18)  |   9.89 (10.22)    |
|   Medium (M)    |     13.28 (13.74)     |   **12.07 (12.39)**   |      13.56 (14.20)       |  13.25 (13.50)  |  31.75 (32.40)  |  13.50 (14.00)  |   12.14 (12.73)   |
|    Large (L)    |     17.59 (18.04)     |   **15.76 (16.09)**   |      17.09 (17.56)       |  17.77 (18.24)  |  55.29 (55.89)  |  20.39 (21.22)  |   17.02 (17.66)   |
|  XLarge (XL) △  |     46.18 (50.49)     |     42.80 (44.66)     |      36.03 (36.53)       |  40.51 (45.02)  |  91.58 (93.94)  |  42.23 (46.82)  | **33.61 (35.14)** |
| 2XLarge (2XL) △ |     60.42 (64.14)     |     51.34 (53.36)     |      45.51 (49.41)       |  54.76 (59.14)  | 127.39 (129.93) |  56.62 (62.00)  | **44.69 (46.85)** |

Each cell is median ms (p95): the median latency in milliseconds, with the 95th percentile of the same 500 calls, also in milliseconds, in brackets. Lower is better, and the lowest median of each row is in bold.

Femto's CoreML FP16 time (25.02 ms) is not noise: the Neural Engine compiler fails on its FP16 package under the default compute units, as the [CoreML page](../../exports/coreml.md#measured-latency-on-apple-silicon) explains, and Core ML then runs it slowly. CoreML GPU FP16 avoids that plan and takes 6.43 ms.

--8<-- "docs/snippets/reproduced-latency.md"

## Run on an Image

Perform inference on an image with the `rfdetr` package. To use a different model size, select the corresponding class from the table above.

```python
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSmall()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

!!! note "Using COCO classes vs. fine-tuned model classes"

    `COCO_CLASSES` works for COCO-pretrained models (80 COCO classes, indexed 0-79). For fine-tuned models, use `detections.data["class_name"]` instead — it resolves class names from the checkpoint and works for both COCO and custom datasets.

`predict()` resizes without antialiasing by default (`antialias=False`), which matches checkpoints trained with the default CPU augmentation backend when `rfdetr[augment]` is installed (Albumentations). Pass `antialias=True` for checkpoints trained with torchvision resizing: the Kornia/GPU augmentation backend, the CPU backend without `rfdetr[augment]`, or older RF-DETR releases. Antialiasing costs a little extra preprocessing time per image. `RFDETRInference` uses the artifact metadata's `resize_antialias` by default for exported models; pass an explicit `antialias=True` or `antialias=False` to override it. The lower-level `rfdetr.export` runtime preprocessing helpers always resize without antialiasing. On MPS with PyTorch older than 2.7, `antialias=True` resizes on the CPU because MPS has no antialiased resize kernel there.

For repeated inference with the `rfdetr` package on CUDA, a fixed batch size, and a fixed resolution, the direct CUDA Graph backend records the TorchScript forward once and replays it. Capture allocates a graph-private memory pool holding the captured forward's intermediates plus the static input/output buffers; it persists for the graph's lifetime and scales with batch size, resolution and model, so this backend uses more device memory than the TorchScript backend. The backend was measured on detection RF-DETR Nano at batch size 1; segmentation and keypoint models use the same path but are unmeasured, and replay/clone memory cost grows with masks and batch size. Outputs are cloned before they leave the graph, so predictions returned by an earlier call are not overwritten by the next one. It is the default on CUDA; pass `compile_backend="torchscript"` to trace without capturing:

```python
predictor = model.inference(compile_backend="cudagraph", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

An operator that CUDA Graphs cannot capture makes `inference()` raise a `RuntimeError`. A failed capture can leave CUDA random-number state unusable, so restart the process before choosing another backend.

PyTorch Inductor is another opt-in backend for long-running inference. Cold compilation can have a higher one-time setup cost, but Inductor can apply broader graph optimizations and later processes may reuse its disk cache. Both examples require a compatible CUDA device, operators, and installed PyTorch version; `dtype="float16"` also requires FP16 support:

```python
predictor = model.inference(compile_backend="inductor", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

The native factory returns a predictor that owns one independent copy of the current model weights. The source model remains unchanged; deleting the predictor releases its runtime when no other references remain. Use the same pattern for CPU inference without compilation:

```python
predictor = model.inference(compile=False, dtype="float16")
detections = predictor.predict("image.jpg")
```

## Set a confidence threshold for each class

One `threshold` holds every class to the same bar, but a trained model rarely finds all classes equally well: for some classes the scores of correct detections and of false positives overlap more than for others. Pass a `ClassThresholds` instead, and `predict()` keeps a detection when its score exceeds the threshold of its own class. Name the classes that need their own bar; every other class takes `default`:

```python
from rfdetr import ClassThresholds, RFDETRSmall

model = RFDETRSmall()

thresholds = ClassThresholds({"person": 0.35, "traffic light": 0.6}, default=0.5)

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=thresholds)
```

- A name that is not in `model.class_names` raises a `ValueError` that lists the valid names, before the model runs. A NaN raises a `ValueError` when the `ClassThresholds` is built; use `float("inf")` to drop a class.
- `ClassThresholds` also takes one threshold per entry of `model.class_names`, in the same order, as a list, tuple, [NumPy](https://pypi.org/project/numpy/) array or tensor: `ClassThresholds([0.4, 0.55, 0.45])`. It must hold exactly `len(model.class_names)` entries.
- `threshold` takes a single number or a `ClassThresholds`. A plain list or `dict` raises a `TypeError` that points to `ClassThresholds`, since a `dict` alone says nothing about the classes it leaves out.
- The comparison is strict, like the single-number form, so a `ClassThresholds` that gives every class `t` keeps exactly the detections of the listed classes that `t` keeps.
- A `ClassThresholds` drops detections that belong to no entry of `model.class_names`: the no-object slot of a fine-tuned model and the unused category IDs of a pretrained COCO model. A single number keeps them.
- Segmentation and keypoint models take the same argument. For keypoint models the thresholds apply to `key_points.detection_confidence`.

The [per-class confidence thresholds cookbook](../../../cookbooks/per-class-confidence-thresholds/) shows how to choose the values from held-out images, and measures when a list of thresholds beats a single one and when it does not.

## Run on video, webcam, or RTSP stream

!!! tip "Built-in video and stream sources"

    `RFDETRInference(model).predict()` reads video files, video URLs, webcams, RTSP streams, and YouTube directly and can yield results lazily with `stream=True`. See [Run on Videos and Streams](videos-and-streams.md). The examples below keep full control of decoding and display with OpenCV.

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video source: <SOURCE_VIDEO_PATH>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    WEBCAM_INDEX = 0
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open webcam: {WEBCAM_INDEX}")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "stream"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream: <RTSP_STREAM_URL>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
