---
description: Run RF-DETR keypoint detection on images, video, and streams. COCO-pretrained preview model predicts 17 person keypoints using a DINOv2 backbone.
---

# Run an RF-DETR Keypoint Model

RF-DETR Keypoint is a transformer architecture for human pose estimation, built on a DINOv2 vision transformer backbone. The preview model is pretrained on the Microsoft COCO dataset and predicts 17 body keypoints for the "person" class.

!!! note "Preview model"

    `RFDETRKeypointPreview` is an early-access release. Fine-tuning on custom keypoint datasets is the primary intended use case. API surface and checkpoint weights may change before the stable release.

## Pre-trained Checkpoints

|     Size     |  RF-DETR package class  | COCO OKS AP | Params (M) | Resolution |
| :----------: | :---------------------: | :---------: | :--------: | :--------: |
| XL (Preview) | `RFDETRKeypointPreview` |      —      |   126.4    |  700x700   |

> The keypoint model is available only in the `rfdetr` package. It is not yet available via the `inference` package.

## Run on an Image

Perform inference on an image using the `rfdetr` package. `model.predict()` returns an `sv.KeyPoints` object. The source image is stored per detection in `key_points.data["source_image"]`; all entries reference the same frame so index `[0]` retrieves it.

=== "rfdetr"

    ```python
    import supervision as sv
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()

    key_points = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

    source_image = key_points.data["source_image"][0]
    annotated_image = sv.VertexAnnotator().annotate(source_image, key_points)
    ```

    !!! tip "Best results with person images"

        The model is trained on COCO person keypoints. Images containing people will produce the most meaningful predictions. Non-person images (such as the dog placeholder above) may return zero or low-confidence keypoints.

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually `0` for the default camera.

=== "video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()

    video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video source: <SOURCE_VIDEO_PATH>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(frame_rgb, threshold=0.5)

        annotated_frame = sv.VertexAnnotator().annotate(frame_bgr, key_points)

        cv2.imshow("RF-DETR Keypoint Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()

    WEBCAM_INDEX = 0  # Change this to the desired webcam index (e.g., 1, 2, ...)
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open webcam: {WEBCAM_INDEX}")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(frame_rgb, threshold=0.5)

        annotated_frame = sv.VertexAnnotator().annotate(frame_bgr, key_points)

        cv2.imshow("RF-DETR Keypoint Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "stream"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream: <RTSP_STREAM_URL>")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(frame_rgb, threshold=0.5)

        annotated_frame = sv.VertexAnnotator().annotate(frame_bgr, key_points)

        cv2.imshow("RF-DETR Keypoint RTSP", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
