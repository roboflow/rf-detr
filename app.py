import streamlit as st
from rfdetr import RFDETRNano
from PIL import Image
import supervision as sv
import numpy as np
import tempfile
import os
import cv2
import torch

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RF-DETR Object Detection",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔍 RF-DETR Object Detection")

st.write(
    "Real-time object detection using RF-DETR Nano "
    "with NVIDIA GPU acceleration."
)

# --------------------------------------------------
# GPU Information
# --------------------------------------------------

if torch.cuda.is_available():

    st.success(
        f"GPU Acceleration: {torch.cuda.get_device_name(0)}"
    )

else:

    st.warning(
        "CUDA GPU is not available. "
        "RF-DETR will run on CPU."
    )

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = RFDETRNano()

    # Enable optimized FP16 inference on NVIDIA GPU
    if torch.cuda.is_available():

        model.inference(
            dtype=torch.float16
        )

    return model


with st.spinner(
    "Loading RF-DETR Nano and optimizing inference..."
):

    model = load_model()

st.success(
    "RF-DETR Nano is ready."
)

# --------------------------------------------------
# Helper: Get Correct RF-DETR Class Names
# --------------------------------------------------

def get_class_names(detections):
    """
    RF-DETR's pretrained COCO model can use sparse
    COCO category IDs.

    Therefore, do NOT use:

        model.class_names[class_id]

    Instead, use the class_name mapping created
    internally by RF-DETR's predict() method.
    """

    if (
        hasattr(detections, "data")
        and "class_name" in detections.data
    ):

        return list(
            detections.data["class_name"]
        )

    # Fallback
    return [
        str(class_id)
        for class_id in detections.class_id
    ]


# --------------------------------------------------
# Helper: Remove Very Small Detections
# --------------------------------------------------

def filter_small_detections(
    detections,
    min_width=12,
    min_height=12
):

    if len(detections) == 0:
        return detections

    boxes = detections.xyxy

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    keep = (
        (widths >= min_width)
        &
        (heights >= min_height)
    )

    return detections[keep]


# --------------------------------------------------
# Image Detection Function
# --------------------------------------------------

def detect_image(image, threshold):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name

    try:

        detections = model.predict(
            temp_path,
            threshold=threshold
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Remove extremely small detections
    detections = filter_small_detections(
        detections,
        min_width=12,
        min_height=12
    )

    image_np = np.array(image)

    box_annotator = sv.BoxAnnotator()

    label_annotator = sv.LabelAnnotator()

    class_names = get_class_names(
        detections
    )

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            class_names,
            detections.confidence
        )
    ]

    annotated_image = box_annotator.annotate(
        scene=image_np.copy(),
        detections=detections
    )

    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )

    return annotated_image, detections


# --------------------------------------------------
# Live Webcam Processor
# --------------------------------------------------

class RFDETRVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.threshold = 0.5

    def recv(self, frame):

        try:

            # ------------------------------------------
            # Convert WebRTC frame to BGR
            # ------------------------------------------

            img = frame.to_ndarray(
                format="bgr24"
            )

            # ------------------------------------------
            # Resize large webcam frames
            # ------------------------------------------

            height, width = img.shape[:2]

            max_width = 960

            if width > max_width:

                scale = max_width / width

                new_width = int(
                    width * scale
                )

                new_height = int(
                    height * scale
                )

                img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )

            # ------------------------------------------
            # BGR → RGB
            # ------------------------------------------

            rgb_image = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------
            # Convert to PIL
            # ------------------------------------------

            pil_image = Image.fromarray(
                rgb_image
            )

            temp_path = None

            try:

                # --------------------------------------
                # Create temporary image
                # --------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp_file:

                    pil_image.save(
                        temp_file.name,
                        quality=90
                    )

                    temp_path = temp_file.name

                # --------------------------------------
                # RF-DETR Detection
                # --------------------------------------

                detections = model.predict(
                    temp_path,
                    threshold=self.threshold
                )

            finally:

                # --------------------------------------
                # Always remove temporary file
                # --------------------------------------

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)

            # ------------------------------------------
            # Remove tiny detections
            # ------------------------------------------

            detections = filter_small_detections(
                detections,
                min_width=12,
                min_height=12
            )

            # ------------------------------------------
            # Correct Class Names
            # ------------------------------------------

            class_names = get_class_names(
                detections
            )

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(
                    class_names,
                    detections.confidence
                )
            ]

            # ------------------------------------------
            # Create Annotators
            # ------------------------------------------

            box_annotator = sv.BoxAnnotator()

            label_annotator = sv.LabelAnnotator()

            # ------------------------------------------
            # Draw Bounding Boxes
            # ------------------------------------------

            annotated = box_annotator.annotate(
                scene=img.copy(),
                detections=detections
            )

            # ------------------------------------------
            # Draw Labels
            # ------------------------------------------

            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels
            )

            # ------------------------------------------
            # Return Webcam Frame
            # ------------------------------------------

            return frame.from_ndarray(
                annotated,
                format="bgr24"
            )

        except Exception as e:

            print(
                "Detection error:",
                e
            )

            return frame


# --------------------------------------------------
# Detection Mode
# --------------------------------------------------

mode = st.radio(
    "Select Detection Mode",
    [
        "Image Detection",
        "Live Webcam"
    ],
    horizontal=True
)

# --------------------------------------------------
# Confidence Threshold
# --------------------------------------------------

threshold = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# ==================================================
# IMAGE DETECTION
# ==================================================

if mode == "Image Detection":

    st.subheader(
        "📷 Image Detection"
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

        if st.button(
            "🚀 Detect Objects",
            key="image_detect"
        ):

            with st.spinner(
                "Running RF-DETR detection..."
            ):

                annotated_image, detections = (
                    detect_image(
                        image,
                        threshold
                    )
                )

            st.subheader(
                "Detection Results"
            )

            st.image(
                annotated_image,
                use_container_width=True
            )

            st.subheader(
                "Detected Objects"
            )

            if len(detections) == 0:

                st.warning(
                    "No objects were detected."
                )

            else:

                st.success(
                    f"{len(detections)} "
                    "object(s) detected"
                )

                class_names = get_class_names(
                    detections
                )

                for class_name, confidence in zip(
                    class_names,
                    detections.confidence
                ):

                    st.write(
                        f"**{class_name}** — "
                        f"Confidence: "
                        f"**{confidence:.2f}**"
                    )


# ==================================================
# LIVE WEBCAM
# ==================================================

else:

    st.subheader(
        "🎥 Live Webcam Detection"
    )

    st.info(
        "Start the camera below and "
        "allow browser camera access."
    )

    # --------------------------------------------------
    # WebRTC Configuration
    # --------------------------------------------------

    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    # --------------------------------------------------
    # Start Webcam
    # --------------------------------------------------

    ctx = webrtc_streamer(
        key="rfdetr-webcam",
        video_processor_factory=(
            RFDETRVideoProcessor
        ),
        rtc_configuration=(
            RTC_CONFIGURATION
        ),
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    # --------------------------------------------------
    # Update Detection Threshold
    # --------------------------------------------------

    if ctx.video_processor:

        ctx.video_processor.threshold = (
            threshold
        )