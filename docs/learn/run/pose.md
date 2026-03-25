# Run an RF-DETR Pose Estimation Model

You can run pose estimation models with RF-DETR to detect keypoints and body poses on people and objects. The pose model outputs bounding boxes along with keypoints in (x, y, visibility) format for each detection, following a similar approach to YOLOv11 pose estimation.

!!! note "Training Required"
    RF-DETR Pose requires training on a keypoint dataset before inference. By default, it loads detection weights as a starting point - the backbone and detection heads are initialized from this checkpoint, while the keypoint head is randomly initialized and learned during fine-tuning.

## Model Sizes

RF-DETR Pose is available in multiple sizes to balance speed and accuracy:

| Model | Class | Resolution | Decoder Layers | Use Case |
|-------|-------|------------|----------------|----------|
| Nano | `RFDETRPoseNano` | 384 | 2 | Real-time, edge devices |
| Small | `RFDETRPoseSmall` | 512 | 3 | Good speed/accuracy balance |
| Medium | `RFDETRPoseMedium` | 576 | 4 | Default, good accuracy |
| Large | `RFDETRPoseLarge` | 768 | 6 | Highest accuracy |

```python
from rfdetr import RFDETRPoseNano, RFDETRPoseSmall, RFDETRPoseMedium, RFDETRPoseLarge

# Choose the size that fits your needs
model = RFDETRPoseNano()   # Fastest, lowest accuracy
model = RFDETRPoseSmall()  # Balanced
model = RFDETRPoseMedium() # Default
model = RFDETRPoseLarge()  # Slowest, highest accuracy
```

!!! tip "Choosing a Model Size"
    - Use **Nano** for real-time applications or when running on edge devices
    - Use **Small** for a good balance of speed and accuracy
    - Use **Medium** (default) for most use cases
    - Use **Large** when accuracy is critical and speed is less important

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
    keypoints = detections.data.get("keypoints")  # [N, K, 3] where K=num_keypoints

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
- `K` = number of keypoints (configured via `num_keypoints`, default: 17 for COCO pose)
- `3` = (x, y, visibility) for each keypoint

The visibility value follows the COCO format:
- `0` = keypoint not visible / not confident
- `2` = keypoint visible and confident

For the raw confidence scores (0.0 to 1.0), use `detections.data["keypoints_confidence"]`:

```python
keypoints = detections.data["keypoints"]  # [N, K, 3] - (x, y, visibility)
confidence = detections.data["keypoints_confidence"]  # [N, K] - raw scores 0.0-1.0
```

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

## Technical Architecture

This section explains the keypoint implementation for those interested in the underlying design.

### Query-Based Keypoint Regression

RF-DETR Pose predicts keypoints directly from transformer decoder query features using lightweight MLP heads:

```
Decoder Query Features [B, N, hidden_dim]
           │
           ├──► Coordinate MLP ──► [B, N, K×2] ──► (x, y) per keypoint
           │
           └──► Visibility MLP ──► [B, N, K]   ──► visibility logits
```

**Why this works well for DETR:**

1. **Query-Object Correspondence**: Each decoder query already attends to a specific object instance via cross-attention. The query feature encodes both *where* the object is and *what* it looks like—ideal context for predicting instance-specific keypoints.

2. **End-to-End Learning**: Unlike anchor-based methods that require keypoint-to-anchor assignment heuristics, DETR's Hungarian matching naturally extends to keypoints. The matched query predicts all attributes (class, box, keypoints) for its assigned ground truth.

3. **Global Context via Attention**: Transformer self-attention allows queries to reason about other objects in the scene, helping with occluded keypoints and crowded scenarios where local CNN features struggle.

### Coordinate Prediction: Box-Relative Regression

Keypoint coordinates are predicted as **offsets relative to the detection bounding box**, then converted to absolute coordinates:

```python
# Predicted: normalized offsets in [0, 1] relative to box
kpt_offsets = coord_head(query_features).sigmoid()  # [B, N, K, 2]

# Convert to absolute: kpt = box_topleft + offset * box_size
x = box_x + kpt_offsets[..., 0] * box_w
y = box_y + kpt_offsets[..., 1] * box_h
```

**Advantages:**

- **Translation invariance**: Model learns relative positions, generalizes across image locations
- **Scale invariance**: Offsets normalized by box size, handles objects of varying scales
- **Bounded output**: Sigmoid ensures predictions stay within reasonable range

### Separate Visibility Head

Visibility is predicted by an independent MLP rather than sharing weights with coordinates:

```python
self.coord_head = MLP(hidden_dim, hidden_dim, num_keypoints * 2, num_layers=3)
self.visibility_head = MLP(hidden_dim, hidden_dim, num_keypoints, num_layers=3)
```

**Rationale:**

- Visibility is a **classification task** (visible vs. not visible)
- Coordinates are a **regression task**
- Separate heads allow independent gradient flow and task-specific optimization
- Prevents visibility predictions from interfering with coordinate precision

### Loss Functions

Three complementary losses for keypoint supervision:

| Loss | Purpose | Details |
|------|---------|---------|
| **L1 Loss** | Coordinate regression | Applied only on visible keypoints |
| **BCE Loss** | Visibility classification | Applied on all keypoints |
| **OKS Loss** | COCO-compatible similarity | Incorporates scale and per-keypoint σ |

**Why OKS Loss:**

- Object Keypoint Similarity (OKS) is the COCO evaluation metric
- Training with OKS loss directly optimizes for the evaluation target
- Incorporates per-keypoint σ values (some keypoints are harder to localize)
- Scale-aware: larger objects tolerate more absolute error

### Comparison to Alternative Methods

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **RF-DETR Pose** | Query → MLP | End-to-end, global context, no anchors | Needs more epochs |
| **YOLO Pose** | Grid cell → Conv | Fast, well-optimized | Anchor assignment heuristics, local features only |
| **HRNet** | High-res heatmaps | Very accurate | Expensive, separate from detection |
| **ViTPose** | ViT + heatmaps | Strong features | Two-stage, not end-to-end |

### Design Benefits for RF-DETR

1. **Minimal Architecture Changes**: Adds only two MLP heads (~100K params), reuses existing decoder features
2. **Unified Detection + Pose**: Single forward pass produces boxes and keypoints
3. **Leverages DETR Strengths**: Hungarian matching, global attention, no NMS post-processing
4. **Configurable**: `num_keypoints`, `keypoint_names`, `skeleton` all user-defined
5. **COCO-Compatible**: Same output format and evaluation as standard pose benchmarks
