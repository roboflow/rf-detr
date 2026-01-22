# Run an RF-DETR Object Detection Model

RF-DETR is a real-time transformer architecture for object detection, built on a DINOv2 vision transformer backbone. The base models are trained on the Microsoft COCO dataset and achieve state-of-the-art accuracy and latency trade-offs.

## Pre-trained Checkpoints

RF-DETR offers model sizes from Nano to 2XLarge, allowing trade-offs between accuracy, latency, and parameter count. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1. Core models (Nano to Large) are licensed under Apache 2.0, while XLarge and 2XLarge use the Platform Model License 1.0 and require a Roboflow account.

| Size | RF-DETR package class | Inference package alias | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub>   | Latency (ms) | Params (M) | Resolution |  License   |
|:----:|:---------------------:|:------------------------|:--------------------:|:-------------------------:|:------------:|:----------:|:----------:|:----------:|
| N    | `RFDETRNano`          | `rfdetr-nano`           | 67.6                 |           48.4            | 2.3          | 30.5       | 384x384    | Apache 2.0 |
| S    | `RFDETRSmall`         | `rfdetr-small`          | 72.1                 |           53.0            | 3.5          | 32.1       | 512x512    | Apache 2.0 |
| M    | `RFDETRMedium`        | `rfdetr-medium`         | 73.6                 |           54.7            | 4.4          | 33.7       | 576x576    | Apache 2.0 |
| L    | `RFDETRLarge`         | `rfdetr-large`          | 75.1                 |           56.5            | 6.8          | 33.9       | 704x704    | Apache 2.0 |
| XL   | `RFDETRXLarge`        | `rfdetr-xlarge`         | 77.4                 |           58.6            | 11.5         | 126.4      | 700x700    |  PML 1.0   |
| 2XL  | `RFDETR2XLarge`       | `rfdetr-2xlarge`        | 78.5                 |           60.1            | 17.2         | 126.9      | 880x880    |  PML 1.0   |

## Run on an Image

Perform inference on an image using either the `rfdetr` package or the `inference` package. To use a different model size, select the corresponding class or alias from the table above.

=== "rfdetr"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from rfdetr import RFDETRMedium
    from rfdetr.util.coco_classes import COCO_CLASSES
    
    model = RFDETRMedium()
    
    image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
    detections = model.predict(image, threshold=0.5)
    
    labels = [
        f"{COCO_CLASSES[class_id]}"
        for class_id
        in detections.class_id
    ]
    
    annotated_image = sv.BoxAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    ```

=== "inference"

    ```python
    import requests
    import supervision as sv
    from PIL import Image
    from inference import get_model
    
    model = get_model("rfdetr-medium")
    
    image = Image.open(requests.get('https://media.roboflow.com/dog.jpg', stream=True).raw)
    predictions = model.infer(image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)
    
    annotated_image = sv.BoxAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
    ```

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from rfdetr.util.coco_classes import COCO_CLASSES

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

        labels = [
            COCO_CLASSES[class_id]
            for class_id in detections.class_id
        ]

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
    from rfdetr.util.coco_classes import COCO_CLASSES

    model = RFDETRMedium()

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
    from rfdetr.util.coco_classes import COCO_CLASSES

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

        labels = [
            COCO_CLASSES[class_id]
            for class_id in detections.class_id
        ]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```