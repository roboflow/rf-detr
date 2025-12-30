# Run an RF-DETR Pose Estimation Model

You can run pose estimation models with RF-DETR to detect keypoints and body poses on people and objects. The pose model outputs bounding boxes along with keypoints in (x, y, visibility) format for each detection, following a similar approach to YOLOv11 pose estimation.

!!! note "Training Required"
    RF-DETR Pose requires training on a keypoint dataset before inference. By default, it loads RF-DETR Medium detection weights (`rf-detr-medium.pth`) as a starting point - the backbone and detection heads are initialized from this checkpoint, while the keypoint head is randomly initialized and learned during fine-tuning.

## Run a Model

=== "Run on an Image"

    To run RF-DETR Pose on an image, use the following code:

    ```python
    import io
    import requests
    import supervision as sv
    import numpy as np
    from PIL import Image
    from rfdetr import RFDETRPose

    model = RFDETRPose(pretrain_weights="path/to/pose_weights.pth")

    url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
    image = Image.open(io.BytesIO(requests.get(url).content))

    detections = model.predict(image, threshold=0.5)

    # Access keypoints from detections
    keypoints = detections.data.get("keypoints")  # [N, K, 3] where K=17 for COCO

    # Annotate image with boxes
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)

    # Draw keypoints manually
    if keypoints is not None:
        annotated_image = draw_keypoints(annotated_image, keypoints)

    sv.plot_image(annotated_image)
    ```

=== "Run on a Video File"

    To run RF-DETR Pose on a video file, use the following code:

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRPose

    model = RFDETRPose(pretrain_weights="path/to/pose_weights.pth")

    def callback(frame, index):
        detections = model.predict(frame[:, :, ::-1], threshold=0.5)
        keypoints = detections.data.get("keypoints")

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)

        # Draw keypoints on frame
        if keypoints is not None:
            annotated_frame = draw_keypoints(annotated_frame, keypoints)

        return annotated_frame

    sv.process_video(
        source_path=<SOURCE_VIDEO_PATH>,
        target_path=<TARGET_VIDEO_PATH>,
        callback=callback
    )
    ```

=== "Run on a Webcam Stream"

    To run RF-DETR Pose on a webcam input, use the following code:

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRPose

    model = RFDETRPose(pretrain_weights="path/to/pose_weights.pth")

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = model.predict(frame[:, :, ::-1], threshold=0.5)
        keypoints = detections.data.get("keypoints")

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)

        if keypoints is not None:
            annotated_frame = draw_keypoints(annotated_frame, keypoints)

        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

## Keypoint Output Format

RF-DETR Pose outputs keypoints in the `detections.data["keypoints"]` field as a NumPy array with shape `[N, K, 3]`:

- `N` = number of detections
- `K` = number of keypoints (default: 17 for COCO pose)
- `3` = (x, y, visibility) for each keypoint

The visibility value ranges from 0 to 1:
- Values close to 0 indicate the keypoint is not visible or not confident
- Values close to 1 indicate the keypoint is visible and confident

## COCO Keypoint Format

By default, RF-DETR Pose uses the COCO 17-keypoint format:

| Index | Keypoint Name    |
|-------|------------------|
| 0     | nose             |
| 1     | left_eye         |
| 2     | right_eye        |
| 3     | left_ear         |
| 4     | right_ear        |
| 5     | left_shoulder    |
| 6     | right_shoulder   |
| 7     | left_elbow       |
| 8     | right_elbow      |
| 9     | left_wrist       |
| 10    | right_wrist      |
| 11    | left_hip         |
| 12    | right_hip        |
| 13    | left_knee        |
| 14    | right_knee       |
| 15    | left_ankle       |
| 16    | right_ankle      |

## Drawing Keypoints

Here's a helper function to draw keypoints and skeleton connections:

```python
import cv2
import numpy as np
from rfdetr.models.keypoint_head import COCO_SKELETON

def draw_keypoints(image, keypoints, threshold=0.3):
    """
    Draw keypoints and skeleton on image.

    Args:
        image: PIL Image or numpy array
        keypoints: [N, K, 3] array of keypoints (x, y, visibility)
        threshold: Minimum visibility to draw keypoint

    Returns:
        Annotated image as numpy array
    """
    if hasattr(image, 'copy'):
        image = np.array(image)

    image = image.copy()
    h, w = image.shape[:2]

    colors = [
        (255, 0, 0),    # Red
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 255, 255),  # Cyan
        (0, 0, 255),    # Blue
        (127, 0, 255),  # Purple
    ]

    for person_kpts in keypoints:
        # Draw skeleton connections
        for i, (start_idx, end_idx) in enumerate(COCO_SKELETON):
            start_kpt = person_kpts[start_idx]
            end_kpt = person_kpts[end_idx]

            if start_kpt[2] > threshold and end_kpt[2] > threshold:
                start_pos = (int(start_kpt[0]), int(start_kpt[1]))
                end_pos = (int(end_kpt[0]), int(end_kpt[1]))
                color = colors[i % len(colors)]
                cv2.line(image, start_pos, end_pos, color, 2)

        # Draw keypoints
        for kpt in person_kpts:
            if kpt[2] > threshold:
                pos = (int(kpt[0]), int(kpt[1]))
                cv2.circle(image, pos, 5, (0, 255, 0), -1)
                cv2.circle(image, pos, 5, (0, 0, 0), 1)

    return image
```

## Custom Keypoint Configurations

RF-DETR Pose supports custom keypoint configurations. You can specify:

- `num_keypoints`: Number of keypoints to detect
- `keypoint_names`: List of keypoint names
- `skeleton`: List of keypoint index pairs for skeleton connections

```python
from rfdetr import RFDETRPose

# Custom configuration for hand keypoints (21 keypoints)
model = RFDETRPose(
    num_keypoints=21,
    keypoint_names=[
        "wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ],
    skeleton=[
        [0, 1], [1, 2], [2, 3], [3, 4],  # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  # index
        # ... additional connections
    ],
    pretrain_weights=None  # Train from scratch for custom keypoints
)
```

## Batch Inference

You can provide `.predict()` with a list of images for batch inference:

```python
import io
import requests
from PIL import Image
from rfdetr import RFDETRPose

model = RFDETRPose(pretrain_weights="path/to/pose_weights.pth")

urls = [
    "https://media.roboflow.com/notebooks/examples/dog-2.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog-3.jpeg"
]

images = [Image.open(io.BytesIO(requests.get(url).content)) for url in urls]

detections_list = model.predict(images, threshold=0.5)

for image, detections in zip(images, detections_list):
    keypoints = detections.data.get("keypoints")
    print(f"Detected {len(detections)} people with keypoints shape: {keypoints.shape if keypoints is not None else None}")
```
