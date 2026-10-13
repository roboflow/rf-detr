# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Streamlit demo for RF-DETR Nano.

Install the demo dependencies with ``pip install 'rfdetr[demo]'`` and start the app with ``streamlit run app.py``.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import supervision as sv
from PIL import Image


class Detector(Protocol):
    """Prediction interface used by the image demo."""

    def predict(self, image: Image.Image, threshold: float) -> sv.Detections:
        """Return detections for an in-memory image."""


def get_class_names(detections: sv.Detections) -> list[str]:
    """Return the display name for each detection, preserving sparse class IDs.

    Examples:
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[0, 0, 20, 20]]),
        ...     class_id=np.array([17]),
        ...     data={"class_name": np.array(["cat"])},
        ... )
        >>> get_class_names(detections)
        ['cat']
    """
    class_names = detections.data.get("class_name")
    if class_names is not None:
        return [str(name) for name in class_names]
    if detections.class_id is None:
        return ["unknown"] * len(detections)
    return [str(class_id) for class_id in detections.class_id]


def filter_small_detections(
    detections: sv.Detections,
    min_width: int = 12,
    min_height: int = 12,
) -> sv.Detections:
    """Remove detections whose boxes are smaller than the requested dimensions.

    Examples:
        >>> detections = sv.Detections(
        ...     xyxy=np.array([[0, 0, 20, 20], [0, 0, 5, 5]]),
        ...     class_id=np.array([1, 2]),
        ... )
        >>> len(filter_small_detections(detections))
        1
    """
    if not len(detections):
        return detections

    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    return detections[(widths >= min_width) & (heights >= min_height)]


def detect_image(image: Image.Image, threshold: float, model: Detector) -> tuple[np.ndarray, sv.Detections]:
    """Predict and annotate an image without writing it to a temporary file.

    Examples:
        >>> class Model:
        ...     def predict(self, image, threshold):
        ...         return sv.Detections(
        ...             xyxy=np.empty((0, 4)),
        ...             class_id=np.empty((0,), dtype=int),
        ...             confidence=np.empty((0,)),
        ...         )
        >>> result, detections = detect_image(Image.new("RGB", (32, 32)), 0.5, Model())
        >>> result.shape
        (32, 32, 3)
    """
    detections = filter_small_detections(model.predict(image, threshold=threshold))
    image_array = np.asarray(image)
    class_names = get_class_names(detections)
    labels = [f"{name} {confidence:.2f}" for name, confidence in zip(class_names, detections.confidence)]

    annotated = sv.BoxAnnotator().annotate(scene=image_array.copy(), detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)
    return annotated, detections


def main() -> None:
    """Run the Streamlit app.

    Examples:
        The live UI launches a server and loads pretrained model weights, so it is not run as a doctest.

        >>> main()  # doctest: +SKIP
    """
    import cv2
    import streamlit as st
    import torch
    from av import VideoFrame
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    from rfdetr import RFDETRNano

    st.set_page_config(page_title="RF-DETR Object Detection", page_icon="🔍", layout="wide")
    st.title("🔍 RF-DETR Object Detection")
    st.write("Real-time object detection using RF-DETR Nano.")

    if torch.cuda.is_available():
        st.success(f"GPU acceleration: {torch.cuda.get_device_name(0)}")
    else:
        st.info("CUDA is unavailable; RF-DETR will run on CPU.")

    @st.cache_resource
    def load_model() -> RFDETRNano:
        """Load and cache the pretrained detector."""
        model = RFDETRNano()
        if torch.cuda.is_available():
            model.inference(dtype=torch.float16)
        return model

    with st.spinner("Loading RF-DETR Nano..."):
        model = load_model()
    st.success("RF-DETR Nano is ready.")

    mode = st.radio("Detection mode", ["Image", "Webcam"], horizontal=True)
    threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if mode == "Image":
        st.subheader("Image detection")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is None:
            return

        with Image.open(uploaded_file) as uploaded_image:
            image = uploaded_image.convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)

        if st.button("Detect objects", key="image_detect"):
            with st.spinner("Running RF-DETR detection..."):
                annotated_image, detections = detect_image(image, threshold, model)
            st.subheader("Detection results")
            st.image(annotated_image, use_container_width=True)
            if not len(detections):
                st.warning("No objects were detected.")
                return

            st.success(f"{len(detections)} object(s) detected")
            for name, confidence in zip(get_class_names(detections), detections.confidence):
                st.write(f"**{name}** — Confidence: **{confidence:.2f}**")
        return

    st.subheader("Live webcam detection")
    st.info("Start the camera below and allow browser camera access.")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class RFDETRVideoProcessor(VideoProcessorBase):
        """Run RF-DETR on incoming webcam frames."""

        def __init__(self) -> None:
            """Initialize the frame processor."""
            self.threshold = 0.5

        def recv(self, frame: VideoFrame) -> VideoFrame:
            """Detect objects in one BGR frame and return its annotated image."""
            try:
                image_bgr = frame.to_ndarray(format="bgr24")
                height, width = image_bgr.shape[:2]
                if width > 960:
                    scale = 960 / width
                    image_bgr = cv2.resize(
                        image_bgr,
                        (960, int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
                detections = filter_small_detections(model.predict(image, threshold=self.threshold))
                labels = [
                    f"{name} {confidence:.2f}"
                    for name, confidence in zip(get_class_names(detections), detections.confidence)
                ]
                annotated = sv.BoxAnnotator().annotate(scene=image_bgr, detections=detections)
                annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)
                return VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception:
                logging.getLogger(__name__).exception("Webcam inference failed")
                return frame

    context = webrtc_streamer(
        key="rfdetr-webcam",
        video_processor_factory=RFDETRVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if context.video_processor is not None:
        context.video_processor.threshold = threshold


if __name__ == "__main__":
    main()
