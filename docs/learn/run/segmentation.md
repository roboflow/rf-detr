# Run an RF-DETR Instance Segmentation Model

RF-DETR is a real-time transformer architecture for instance segmentation, built on a DINOv2 vision transformer backbone. The base models are trained on the Microsoft COCO dataset and achieve strong accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR-Seg offers model sizes from Nano to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1.

| Size | RF-DETR package class | Inference package alias     | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
|:----:|:---------------------:|:----------------------------|:--------------------:|:------------------------:|:------------:|:----------:|:----------:|
| N    | `RFDETRSegNano`       | `rfdetr-seg-nano`           | 63.0                 | 40.3                     | 3.4          | 33.6       | 312x312    |
| S    | `RFDETRSegSmall`      | `rfdetr-seg-small`          | 66.2                 | 43.1                     | 4.4          | 33.7       | 384x384    |
| M    | `RFDETRSegMedium`     | `rfdetr-seg-medium`         | 68.4                 | 45.3                     | 5.9          | 35.7       | 432x432    |
| L    | `RFDETRSegLarge`      | `rfdetr-seg-large`          | 70.5                 | 47.1                     | 8.8          | 36.2       | 504x504    |
| XL   | `RFDETRSegXLarge`     | `rfdetr-seg-xlarge`         | 72.2                 | 48.8                     | 13.5         | 38.1       | 624x624    |
| 2XL  | `RFDETRSeg2XLarge`    | `rfdetr-seg-2xlarge`        | 73.1                 | 49.9                     | 21.8         | 38.6       | 768x768    |

## Run on an Image

Perform inference on an image using either the `rfdetr` package or the `inference` package. To use a different model size, select the corresponding class or alias from the table above.

=== "rfdetr"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from rfdetr import RFDETRSegMedium
    from rfdetr.util.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

    image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
    detections = model.predict(image, threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]}"
        for class_id
        in detections.class_id
    ]

    annotated_image = sv.MaskAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    ```

=== "inference"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from inference import get_model

    model = get_model("rfdetr-seg-medium")

    image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
    predictions = model.infer(image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)

    annotated_image = sv.MaskAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
    ```

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRSegMedium
    from rfdetr.util.coco_classes import COCO_CLASSES

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

        labels = [
            COCO_CLASSES[class_id]
            for class_id in detections.class_id
        ]

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
    from rfdetr.util.coco_classes import COCO_CLASSES

    model = RFDETRSegMedium()

    video_capture = cv2.VideoCapture(<WEBCAM_INDEX>)
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open webcam: <WEBCAM_INDEX>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [
            COCO_CLASSES[class_id]
            for class_id in detections.class_id
        ]

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
    from rfdetr.util.coco_classes import COCO_CLASSES

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

        labels = [
            COCO_CLASSES[class_id]
            for class_id in detections.class_id
        ]

        annotated_frame = sv.MaskAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR-Seg RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
