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

## Fastest Configuration on NVIDIA GPUs

The configuration with the lowest end-to-end latency for each size on four NVIDIA GPUs, batch 1. Run the [latency cookbook](../../../cookbooks/measure-latency/) on your own GPU to find yours.

<!-- COPIES: every number comes from the results of docs/cookbooks/measure-latency.ipynb (HARDWARE = "cuda", one run per GPU, provenance in the note below). -->

=== "Tesla T4"

    |     Size      | PyTorch CUDA graph FP16 | TensorRT FP16 | TensorRT FP16 graph |   TensorRT INT8   | TensorRT INT8 graph |
    | :-----------: | :---------------------: | :-----------: | :-----------------: | :---------------: | :-----------------: |
    |   Nano (N)    |       9.04 (9.63)       |  7.22 (7.70)  |     7.59 (8.39)     |  **6.87 (7.39)**  |     6.98 (8.01)     |
    |   Small (S)   |      11.32 (11.76)      |  9.22 (9.98)  |     9.19 (9.73)     |  **8.35 (8.90)**  |     8.62 (9.40)     |
    |  Medium (M)   |      16.13 (16.73)      | 12.00 (12.83) |    12.19 (12.84)    | **11.30 (11.85)** |    11.44 (12.16)    |
    |   Large (L)   |      20.94 (21.83)      | 16.80 (17.55) |    17.13 (18.03)    | **15.16 (15.81)** |    15.64 (16.21)    |
    |  XLarge (XL)  |      35.44 (37.11)      | 26.47 (27.93) |    26.47 (27.81)    | **24.41 (25.54)** |    24.73 (26.01)    |
    | 2XLarge (2XL) |      61.49 (65.45)      | 43.51 (47.45) |    40.39 (41.38)    |   40.43 (41.68)   |  **39.21 (41.05)**  |

=== "NVIDIA L4"

    |     Size      | PyTorch CUDA graph FP16 | TensorRT FP16 | TensorRT FP16 graph | TensorRT INT8 | TensorRT INT8 graph |
    | :-----------: | :---------------------: | :-----------: | :-----------------: | :-----------: | :-----------------: |
    |   Nano (N)    |     **4.48 (4.84)**     |  4.97 (5.60)  |     4.65 (5.23)     |  4.71 (5.17)  |     4.55 (5.02)     |
    |   Small (S)   |       5.14 (5.50)       |  5.28 (5.71)  |     5.09 (5.92)     |  5.15 (5.85)  |   **4.83 (5.19)**   |
    |  Medium (M)   |       6.57 (6.93)       |  6.16 (6.59)  |     5.94 (6.35)     |  5.83 (6.40)  |   **5.65 (6.17)**   |
    |   Large (L)   |       7.58 (7.92)       |  7.15 (7.60)  |     7.06 (7.55)     |  6.86 (7.24)  |   **6.72 (7.31)**   |
    |  XLarge (XL)  |      11.82 (12.26)      | 10.78 (11.35) |    10.37 (10.95)    | 9.93 (10.38)  |  **9.90 (10.40)**   |
    | 2XLarge (2XL) |      18.18 (19.13)      | 16.36 (17.13) |    16.01 (16.69)    | 14.93 (15.39) |  **14.81 (15.37)**  |

=== "A100"

    |     Size      | PyTorch CUDA graph FP16 | TensorRT FP16 | TensorRT FP16 graph | TensorRT INT8 | TensorRT INT8 graph |
    | :-----------: | :---------------------: | :-----------: | :-----------------: | :-----------: | :-----------------: |
    |   Nano (N)    |       4.42 (4.89)       |  4.72 (5.48)  |     4.32 (4.69)     |  4.64 (5.17)  |   **4.25 (4.59)**   |
    |   Small (S)   |       4.69 (5.10)       |  4.80 (5.49)  |     4.51 (4.87)     |  4.79 (5.37)  |   **4.49 (4.85)**   |
    |  Medium (M)   |       5.28 (5.63)       |  5.39 (5.86)  |     5.02 (5.34)     |  5.36 (6.00)  |   **5.01 (5.46)**   |
    |   Large (L)   |       5.68 (6.04)       |  5.73 (6.47)  |   **5.38 (5.77)**   |  5.92 (6.59)  |     5.38 (5.70)     |
    |  XLarge (XL)  |       7.22 (7.51)       |  7.23 (7.54)  |     6.57 (6.95)     |  7.01 (7.44)  |   **6.50 (6.82)**   |
    | 2XLarge (2XL) |       9.16 (9.60)       |  8.44 (8.90)  |   **7.66 (8.20)**   |  8.11 (8.46)  |     7.69 (8.09)     |

=== "RTX PRO 6000"

    |     Size      | PyTorch CUDA graph FP16 | TensorRT FP16 | TensorRT FP16 graph | TensorRT INT8 | TensorRT INT8 graph |
    | :-----------: | :---------------------: | :-----------: | :-----------------: | :-----------: | :-----------------: |
    |   Nano (N)    |       2.25 (2.39)       |  2.16 (2.24)  |   **1.89 (1.97)**   |  2.18 (2.26)  |     1.89 (1.98)     |
    |   Small (S)   |       2.41 (2.49)       |  2.24 (2.33)  |     2.01 (2.09)     |  2.17 (2.25)  |   **1.90 (1.98)**   |
    |  Medium (M)   |       2.75 (2.84)       |  2.51 (2.59)  |     2.30 (2.40)     |  2.34 (2.43)  |   **2.11 (2.21)**   |
    |   Large (L)   |       2.98 (3.10)       |  2.72 (2.81)  |     2.49 (2.61)     |  2.59 (2.67)  |   **2.39 (2.49)**   |
    |  XLarge (XL)  |       3.92 (4.00)       |  3.32 (3.40)  |     3.08 (3.16)     |  3.24 (3.36)  |   **3.02 (3.12)**   |
    | 2XLarge (2XL) |       5.08 (5.20)       |  4.31 (4.40)  |     4.03 (4.15)     |  4.07 (4.16)  |   **3.80 (3.90)**   |

Each cell is median ms (p95): the median latency in milliseconds, with the 95th percentile of the same 500 calls, also in milliseconds, in brackets. Lower is better, and the lowest median of each row is in bold. The configurations are explained in the note below.

- On the RTX PRO 6000 INT8 graph is fastest for Small to 2XLarge. At Nano, FP16 graph and INT8 graph both take 1.89 ms, and at Large and XLarge FP16 graph is within 4.2% and 2.0% of INT8 graph. PyTorch CUDA graph FP16 takes 1.19 to 1.34 times as long as the fastest.
- On the L4, PyTorch CUDA graph FP16 is fastest for Nano (4.48 ms), level with INT8 graph (4.55 ms) and FP16 graph (4.65 ms). INT8 graph is fastest from Small up, 5% to 7% below FP16 graph.
- On the A100 the fastest configuration leads the second by less than 2% at every size, so no configuration is clearly faster.
- On the T4, TensorRT INT8 is fastest from Nano to XLarge and INT8 graph at 2XLarge (39.21 ms, 3.0% below FP16 graph). PyTorch CUDA graph FP16 takes 1.32 to 1.57 times as long as the fastest.

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
