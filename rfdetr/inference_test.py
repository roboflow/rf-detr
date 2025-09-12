
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

""" Demo Inference script for RF-DETR with easy switch between DINOv2 and DINOv3(local repo)."""
import os
import io
import requests
import supervision as sv
from PIL import Image

ENCODER_ALIASES = {
    "dinov2": "dinov2_windowed_small",
    "v2": "dinov2_windowed_small",
    "dinov2_small": "dinov2_windowed_small",
    "dinov2_base": "dinov2_windowed_base",
    "dinov3": "dinov3_base",
    "v3": "dinov3_base",
}

VALID_ENCODERS = {
    "dinov2_windowed_small",
    "dinov2_windowed_base",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
}

def resolve_encoder(enc_str: str) -> str:
    """Resolve the encoder string to a valid encoder name.
    Args:
        enc_str (str): The encoder string to resolve.

    Returns:
            str: The resolved encoder name.
    Examples:
        resolve_encoder("v2")  # returns "dinov2_windowed_small"
    """
    enc_str = enc_str.strip().lower()
    enc = ENCODER_ALIASES.get(enc_str, enc_str)
    if enc not in VALID_ENCODERS:
        raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
    return enc

## If using DINOv3, ensure you have the local repo and weights set up.
def main(encoder:str = "v3", repo_dir: str = "./dinov3_repo", dino_v_weights_path: str = "./dinov3_weights.pth"):
    """Main function to run the inference demo.

    Args:
        encoder (str): The encoder to use, e.g., "v2", "v3", or exact name.
        repo_dir (str): Path to the local DINOv3 repository.
        dino_v_weights_path (str): Path to the DINOv3 weights file.

    Returns:
        None
    
    Examples:
        main(encoder="v2")
        main(encoder="v3", repo_dir="D:/repos/dinov3", dino_v_weights_path="D:/repos/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    """
    encoder = resolve_encoder(encoder)

    if encoder.startswith("dinov3"):
        print("Using DINOv3 encoder:", encoder)
        # Set the environment variables for DINOv3 repo and weights
        os.environ["DINOV3_REPO"] = repo_dir
        os.environ["DINOV3_WEIGHTS"] = dino_v_weights_path
    elif encoder.startswith("dinov2"):
        print("Using DINOv2 encoder:", encoder)

    # Set env *before* importing your package (your Pydantic defaults read env)
    os.environ["RFD_ENCODER"] = encoder


    # Optional: ensure we don't try Hugging Face first (use Hub fallback).
    # If you DO have HF access+token and want to prefer HF, just remove this line.
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)

    from rfdetr import RFDETRMedium, RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES

    model = RFDETRMedium()              # uses encoder="dinov3_base" per your config defaults
    model.optimize_for_inference()

    while True:
        url = input("Enter image URL (or 'exit' to quit): ")
        if url.lower() == 'exit':
            break
        #url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
        image = Image.open(io.BytesIO(requests.get(url).content))

        detections = model.predict(image, threshold=0.5)

        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

        sv.plot_image(annotated_image)

if __name__ == "__main__":
    main(encoder="v2", repo_dir="..\dinov3", dino_v_weights_path="..\dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")