---
description: Run RF-DETR instance segmentation on images, video, and streams. Mask predictions with 3.4-21.8 ms latency using DINOv2 backbone.
---

# Run an RF-DETR Instance Segmentation Model

RF-DETR is a real-time transformer architecture for instance segmentation, built on a DINOv2 vision transformer backbone. The base models are trained on the Microsoft COCO dataset and achieve strong accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR-Seg offers model sizes from Nano to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. Every RF-DETR-Seg size, including XLarge and 2XLarge, is part of the core `rfdetr` package under Apache 2.0, so it's commercially usable out of the box: you can ship it in closed-source, commercial products, including self-hosted deployments, without paying a license fee or open-sourcing your code. No `rfdetr[plus]` install is needed.

<!-- COPIES: the three AP columns come from the results table of docs/cookbooks/coco-accuracy.ipynb (one run, provenance in the note below); params and resolution repeat the Segmentation table of docs/learn/benchmarks.md. -->

|     Size      | PyTorch FP32 mask AP<sub>50:95</sub> | TensorRT FP16 mask AP<sub>50:95</sub> | TensorRT INT8 mask AP<sub>50:95</sub> | Params (M) | Resolution |  License   |
| :-----------: | :----------------------------------: | :-----------------------------------: | :-----------------------------------: | :--------: | :--------: | :--------: |
|   Nano (N)    |                 40.1                 |                 40.1                  |                 39.6                  |    33.6    |  312x312   | Apache 2.0 |
|   Small (S)   |                 42.9                 |                 43.0                  |                 42.1                  |    33.7    |  384x384   | Apache 2.0 |
|  Medium (M)   |                 44.9                 |                 44.9                  |                 44.0                  |    35.7    |  432x432   | Apache 2.0 |
|   Large (L)   |                 46.4                 |                 46.4                  |                 46.0                  |    36.2    |  504x504   | Apache 2.0 |
|  XLarge (XL)  |                 48.5                 |                 48.4                  |                 48.0                  |    38.1    |  624x624   | Apache 2.0 |
| 2XLarge (2XL) |                 49.7                 |                 49.7                  |                 49.3                  |    38.6    |  768x768   | Apache 2.0 |

--8<-- "docs/snippets/reproduced-coco-ap.md"

## Fastest Configuration on Apple Silicon

The configuration with the lowest end-to-end latency for each size on an Apple M4 Max, batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own device to find yours.

<!-- COPIES: every number comes from the results table of docs/cookbooks/measure-latency.ipynb (HARDWARE = "apple", one run, provenance in the note below). -->

|     Size      | PyTorch inductor FP32 | PyTorch inductor FP16 | PyTorch TorchScript FP16 |  CoreML FP32  |   CoreML FP16   | CoreML GPU FP32 | CoreML GPU FP16 |
| :-----------: | :-------------------: | :-------------------: | :----------------------: | :-----------: | :-------------: | :-------------: | :-------------: |
|   Nano (N)    |     10.44 (10.86)     |   **9.82 (10.24)**    |      12.08 (12.67)       | 11.12 (11.97) |  24.64 (25.38)  |  11.03 (11.74)  |  10.30 (11.16)  |
|   Small (S)   |     12.79 (13.32)     |   **11.90 (12.32)**   |      14.20 (14.62)       | 13.65 (14.50) |  24.26 (27.45)  |  13.25 (13.90)  |  12.06 (12.70)  |
|  Medium (M)   |     15.92 (16.23)     |   **14.54 (14.91)**   |      16.68 (17.04)       | 17.39 (18.09) |  31.51 (35.07)  |  17.43 (18.16)  |  15.77 (16.41)  |
|   Large (L)   |     19.81 (20.21)     |   **18.25 (18.59)**   |      19.96 (20.41)       | 21.28 (21.99) |  45.28 (48.40)  |  21.36 (22.07)  |  19.05 (19.80)  |
|  XLarge (XL)  |     31.03 (31.56)     |   **27.38 (27.75)**   |      28.71 (29.13)       | 36.23 (37.23) | 125.48 (130.74) |  36.08 (37.05)  |  30.75 (31.53)  |
| 2XLarge (2XL) |     53.35 (55.11)     |     44.99 (45.59)     |    **43.55 (44.03)**     | 62.08 (63.98) | 224.85 (226.62) |  61.81 (68.96)  |  51.73 (53.44)  |

Each cell is median ms (p95): the median latency in milliseconds, with the 95th percentile of the same 500 calls, also in milliseconds, in brackets. Lower is better, and the lowest median of each row is in bold.

PyTorch inductor FP16 is fastest for every size except 2XLarge. Several rows are ties between their two fastest configurations, where a gap under about 5% counts as level: Nano, where CoreML GPU FP16 (10.30 ms) is 4.9% behind inductor FP16 (9.82 ms), Small, where CoreML GPU FP16 (12.06 ms) is within 1.3% of inductor FP16 (11.90 ms), Large, where it is within 4.3% (19.05 and 18.25 ms), and XLarge, where PyTorch TorchScript FP16 (28.71 ms) is within 4.9% (27.38 ms). For 2XLarge, TorchScript FP16 (43.55 ms) is within 3.3% of inductor FP16 (44.99 ms).

--8<-- "docs/snippets/reproduced-latency.md"

## Run on an Image

Perform inference on an image with the `rfdetr` package. To use a different model size, select the corresponding class from the table above.

```python
import supervision as sv
from rfdetr import RFDETRSegSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegSmall()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

!!! note "Using COCO classes vs. fine-tuned model classes"

    `COCO_CLASSES` maps sparse COCO category IDs (1–90, with gaps) to names. COCO-pretrained models return those sparse IDs as `class_id`; for fine-tuned models, use `detections.data["class_name"]` to get names from the checkpoint.

### Read the masks

Segmentation models return one binary mask per instance in addition to the box, score and class. The masks are stored in `detections.mask`, a boolean array of shape `(N, H, W)` for `N` instances, which `supervision` annotators and utilities consume directly:

```python
import supervision as sv
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium()
detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

print(detections.mask.shape)  # (N, H, W)
print(sv.mask_to_polygons(detections.mask[0]))  # polygon outline of the first instance
print(detections.area)  # per-instance pixel area, computed from the masks
```

Instances scoring below `threshold` are dropped before their masks are upsampled to the image size, so a higher threshold also means less mask memory and post-processing time. With per-class thresholds (a `ClassThresholds`, see [Set a confidence threshold for each class](detection.md#set-a-confidence-threshold-for-each-class)), the lowest of them sets only the mask pre-filter cut-off before upsampling; each instance is then filtered by its own class's threshold. Use `sv.MaskAnnotator` to draw filled masks, `sv.PolygonAnnotator` to draw outlines only, or `sv.BoxAnnotator` to overlay the boxes as well.

### Preprocessing and inference backends

`predict()` resizes without antialiasing by default (`antialias=False`), which matches checkpoints trained with the default CPU augmentation backend when `rfdetr[augment]` is installed (Albumentations). Pass `antialias=True` for checkpoints trained with torchvision resizing: the Kornia/GPU augmentation backend, the CPU backend without `rfdetr[augment]`, or older RF-DETR releases. Antialiasing costs a little extra preprocessing time per image. `RFDETRInference` uses the artifact metadata's `resize_antialias` by default for exported models; pass an explicit `antialias=True` or `antialias=False` to override it. The lower-level `rfdetr.export` runtime preprocessing helpers always resize without antialiasing.

For repeated inference with the `rfdetr` package on CUDA, a fixed batch size, and a fixed resolution, the direct CUDA Graph backend records the TorchScript forward once and replays it. Capture allocates a graph-private memory pool that persists for the graph's lifetime and scales with batch size, resolution and model. Segmentation models also carry the mask outputs in that pool, so expect higher memory cost than for detection. The backend was measured on detection RF-DETR Nano at batch size 1; segmentation models use the same path but are unmeasured. It is the default on CUDA; pass `compile_backend="torchscript"` to trace without capturing:

```python
predictor = model.inference(compile_backend="cudagraph", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

An operator that CUDA Graphs cannot capture makes `inference()` raise a `RuntimeError`. A failed capture can leave CUDA random-number state unusable, so restart the process before choosing another backend.

PyTorch Inductor is another opt-in backend for long-running inference. Cold compilation can have a higher one-time setup cost, but later processes may reuse its disk cache. Inductor compiles with `torch.compile` on a device supported by the installed PyTorch version and its operators; it does not require CUDA. `dtype="float16"` also requires FP16 support on that device:

```python
predictor = model.inference(compile_backend="inductor", batch_size=1, dtype="float16")
detections = predictor.predict("image.jpg")
```

The native factory returns a predictor that owns one independent copy of the current model weights. The source model remains unchanged; deleting the predictor releases its runtime when no other references remain. Use the same pattern for CPU inference without compilation:

```python
predictor = model.inference(compile=False, dtype="float16")
detections = predictor.predict("image.jpg")
```

## Run on video, webcam, or RTSP stream

!!! tip "Built-in video and stream sources"

    `RFDETRInference(model).predict()` reads video files, video URLs, webcams, RTSP streams, and YouTube directly and can yield results lazily with `stream=True`. See [Run on Videos and Streams](videos-and-streams.md). The examples below keep full control of decoding and display with OpenCV.

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRSegMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

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

        annotated_frame = sv.MaskAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR-Seg Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRSegMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

    WEBCAM_INDEX = 0  # Change this to the desired webcam index (e.g., 1, 2, ...)
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

        annotated_frame = sv.MaskAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR-Seg Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "stream"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRSegMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

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

        annotated_frame = sv.MaskAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR-Seg RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
