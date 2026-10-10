---
description: Run RF-DETR instance segmentation on images, video, and streams. Mask predictions with 3.4-21.8 ms latency using DINOv2 backbone.
---

# Run an RF-DETR Instance Segmentation Model

RF-DETR is a real-time transformer architecture for instance segmentation, built on a DINOv2 vision transformer backbone. The base models are trained on the Microsoft COCO dataset and achieve strong accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR-Seg offers model sizes from Nano to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1.

| Size | RF-DETR package class | Inference package alias | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
| :--: | :-------------------: | :---------------------- | :------------------: | :---------------------: | :----------: | :--------: | :--------: |
|  N   |    `RFDETRSegNano`    | `rfdetr-seg-nano`       |         63.0         |          40.3           |     3.4      |    33.6    |  312x312   |
|  S   |   `RFDETRSegSmall`    | `rfdetr-seg-small`      |         66.2         |          43.1           |     4.4      |    33.7    |  384x384   |
|  M   |   `RFDETRSegMedium`   | `rfdetr-seg-medium`     |         68.4         |          45.3           |     5.9      |    35.7    |  432x432   |
|  L   |   `RFDETRSegLarge`    | `rfdetr-seg-large`      |         70.5         |          47.1           |     8.8      |    36.2    |  504x504   |
|  XL  |   `RFDETRSegXLarge`   | `rfdetr-seg-xlarge`     |         72.2         |          48.8           |     13.5     |    38.1    |  624x624   |
| 2XL  |  `RFDETRSeg2XLarge`   | `rfdetr-seg-2xlarge`    |         73.1         |          49.9           |     21.8     |    38.6    |  768x768   |

## Run on an Image

Perform inference on an image using either the `rfdetr` package or the `inference` package. To use a different model size, select the corresponding class or alias from the table above.

=== "rfdetr"

    ```python
    import supervision as sv
    from rfdetr import RFDETRSegMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

    detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    ```

=== "inference"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from inference import get_model

    model = get_model("rfdetr-seg-medium")

    image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
    predictions = model.infer(image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)

    annotated_image = sv.MaskAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
    ```

!!! note "Using COCO classes vs. fine-tuned model classes"

    `COCO_CLASSES` works for COCO-pretrained models (80 COCO classes, indexed 0-79). For fine-tuned models, use `detections.data["class_name"]` instead — it resolves class names from the checkpoint and works for both COCO and custom datasets.

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

Instances scoring below `threshold` are dropped before their masks are upsampled to the image size, so a higher threshold also means less mask memory and post-processing time. Use `sv.MaskAnnotator` to draw filled masks, `sv.PolygonAnnotator` to draw outlines only, or `sv.BoxAnnotator` to overlay the boxes as well.

### Preprocessing and inference backends

`predict()` resizes without antialiasing by default (`antialias=False`), which matches checkpoints trained with the default CPU augmentation backend when `rfdetr[augment]` is installed (Albumentations). Pass `antialias=True` for checkpoints trained with torchvision resizing: the Kornia/GPU augmentation backend, the CPU backend without `rfdetr[augment]`, or older RF-DETR releases. Antialiasing costs a little extra preprocessing time per image. Exported models and the `rfdetr.export` runtime helpers always resize without antialiasing, so `antialias=True` results will not match them unless you pre-resize the image with antialiasing yourself.

For repeated inference with the `rfdetr` package on CUDA, a fixed batch size, and a fixed resolution, the direct CUDA Graph backend records the TorchScript forward once and replays it. Capture allocates a graph-private memory pool that persists for the graph's lifetime and scales with batch size, resolution and model. Segmentation models also carry the mask outputs in that pool, so expect higher memory cost than for detection. The backend was measured on detection RF-DETR Nano at batch size 1; segmentation models use the same path but are unmeasured. The default stays `"torchscript"`:

```python
model.inference(compile_backend="cudagraph", batch_size=1, dtype="float16")
```

An operator that CUDA Graphs cannot capture makes `inference()` raise a `RuntimeError`. A failed capture can leave CUDA random-number state unusable, so restart the process before choosing another backend.

PyTorch Inductor is another opt-in backend for long-running inference. Cold compilation can have a higher one-time setup cost, but later processes may reuse its disk cache. It requires a compatible CUDA device, operators, and installed PyTorch version; `dtype="float16"` also requires FP16 support. The external `inference` package API shown above does not expose `RFDETR.inference()`:

```python
model.inference(compile_backend="inductor", batch_size=1, dtype="float16")
```

For memory-constrained inference-only deployments with the `rfdetr` package, optimize the loaded model in place before calling `predict()`. Pass `dtype="float16"` to halve weight memory in addition to clearing the base model reference. This operation is irreversible — to restore the original model, create a new `RFDETR` instance:

```python
model.inference(compile=False, inplace=True, dtype="float16")
```

## Run on video, webcam, or RTSP stream

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
