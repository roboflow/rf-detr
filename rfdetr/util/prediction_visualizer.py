import argparse
import os
from typing import List
import cv2
import supervision as sv
from PIL import Image
import numpy as np
from rfdetr import RFDETRBase

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on sample images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to pre-trained RF-DETR model weights.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save annotated images.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for predictions.")
    return parser.parse_args()

def load_images(input_dir: str) -> List[Image.Image]:
    supported_extensions = (".jpg", ".jpeg", ".png")
    return [Image.open(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]

def save_annotated_image(image: Image.Image, detections: sv.Detections, output_path: str):
    annotated_image = np.array(image)
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    labels = [f"{pred.class_name} {pred.confidence:.2f}" for pred in detections]
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist.")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = RFDETRBase(pretrain_weights=args.weights)
    
    # Load images
    images = load_images(args.input_dir)
    if not images:
        raise ValueError(f"No supported images found in {args.input_dir}.")
    
    # Process each image
    for idx, image in enumerate(images):
        try:
            detections = model.predict(image, threshold=args.confidence)
            output_path = os.path.join(args.output_dir, f"annotated_{idx}.png")
            save_annotated_image(image, detections, output_path)
            print(f"Saved annotated image to {output_path}.")
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}.")

if __name__ == "__main__":
    main()
