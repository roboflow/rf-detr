# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
TensorRT export helpers: trtexec invocation, output parsing, and engine inference.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    import supervision as sv

logger = get_logger()


def run_command_shell(command, dry_run: bool = False) -> subprocess.CompletedProcess:
    if dry_run:
        logger.info(f"\nCUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')} {command}\n")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        raise


def trtexec(onnx_dir: str, args) -> str:
    """Convert an ONNX model to a TensorRT engine using trtexec.

    Args:
        onnx_dir: Path to the input ONNX file.
        args: Namespace with ``verbose`` (bool), ``profile`` (bool), and
            ``dry_run`` (bool) attributes.

    Returns:
        Path to the generated ``.engine`` file.
    """
    engine_dir = onnx_dir.replace(".onnx", ".engine")

    # Base trtexec command
    trt_command = " ".join(
        [
            "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            "--memPoolSize=workspace:4096 --fp16",
            "--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10",
            f"{'--verbose' if args.verbose else ''}",
        ]
    )

    if args.profile:
        profile_dir = onnx_dir.replace(".onnx", ".nsys-rep")
        # Wrap with nsys profile command
        command = " ".join(
            ["nsys profile", f"--output={profile_dir}", "--trace=cuda,nvtx", "--force-overwrite true", trt_command]
        )
        logger.info(f"Profile data will be saved to: {profile_dir}")
    else:
        command = trt_command

    output = run_command_shell(command, args.dry_run)
    parse_trtexec_output(output.stdout)
    return engine_dir


def parse_trtexec_output(output_text):
    logger.info(output_text)
    # Common patterns in trtexec output
    gpu_compute_pattern = (
        r"GPU Compute Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms, median = (\d+\.\d+) ms"
    )
    h2d_pattern = r"Host to Device Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    d2h_pattern = r"Device to Host Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    latency_pattern = r"Latency: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    throughput_pattern = r"Throughput: (\d+\.\d+) qps"

    stats = {}

    # Extract compute times
    if match := re.search(gpu_compute_pattern, output_text):
        stats.update(
            {
                "compute_min_ms": float(match.group(1)),
                "compute_max_ms": float(match.group(2)),
                "compute_mean_ms": float(match.group(3)),
                "compute_median_ms": float(match.group(4)),
            }
        )

    # Extract H2D times
    if match := re.search(h2d_pattern, output_text):
        stats.update(
            {
                "h2d_min_ms": float(match.group(1)),
                "h2d_max_ms": float(match.group(2)),
                "h2d_mean_ms": float(match.group(3)),
            }
        )

    # Extract D2H times
    if match := re.search(d2h_pattern, output_text):
        stats.update(
            {
                "d2h_min_ms": float(match.group(1)),
                "d2h_max_ms": float(match.group(2)),
                "d2h_mean_ms": float(match.group(3)),
            }
        )

    if match := re.search(latency_pattern, output_text):
        stats.update(
            {
                "latency_min_ms": float(match.group(1)),
                "latency_max_ms": float(match.group(2)),
                "latency_mean_ms": float(match.group(3)),
            }
        )

    # Extract throughput
    if match := re.search(throughput_pattern, output_text):
        stats["throughput_qps"] = float(match.group(1))

    return stats


class RFDETRTensorRT:
    """RF-DETR inference using a pre-built TensorRT engine file (``.engine``).

    Provides the same :meth:`predict` interface as :class:`~rfdetr.RFDETR` but
    uses a TensorRT engine for accelerated GPU inference.

    .. note::
        Requires the ``trt`` extras: ``pip install "rfdetr[trt]"``

    Example::

        from rfdetr.export.tensorrt import RFDETRTensorRT

        detector = RFDETRTensorRT("output/inference_model.engine")
        detections = detector.predict(image)

    Args:
        engine_path: Path to the ``.engine`` file produced by ``trtexec`` or
            :meth:`~rfdetr.RFDETR.export` with ``tensorrt=True``.
        resolution: Input image resolution (square side length in pixels).
            Must match the resolution used when the ONNX model was exported.
        num_classes: Number of output object classes.  Defaults to 80 (COCO).
        num_select: Number of top-scoring queries to keep from the raw model
            output before threshold filtering.
        class_names: Optional list of class name strings (0-indexed).
        device: CUDA device string (e.g. ``"cuda:0"``).
    """

    means: List[float] = [0.485, 0.456, 0.406]
    stds: List[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        engine_path: str,
        resolution: int = 560,
        num_classes: int = 80,
        num_select: int = 300,
        class_names: Optional[List[str]] = None,
        device: str = "cuda:0",
    ) -> None:
        try:
            from rfdetr.export.benchmark import TRTInference
        except ImportError as exc:
            raise ImportError(
                "TensorRT inference requires additional packages. Install them with: pip install 'rfdetr[trt]'"
            ) from exc

        from rfdetr.models import PostProcess

        self.engine_path = engine_path
        self.resolution = resolution
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = torch.device(device)

        self._trt = TRTInference(engine_path=engine_path, device=device)
        self._postprocess = PostProcess(num_select=num_select).to(self.device)

    def predict(
        self,
        images: Union[str, torch.Tensor, List[Union[str, torch.Tensor]]],
        threshold: float = 0.5,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Run object detection on one or more images using the TensorRT engine.

        Accepts the same image formats as :meth:`~rfdetr.RFDETR.predict`:
        file paths, URLs, PIL Images, NumPy arrays (HWC, uint8 or float), or
        ``torch.Tensor`` (CHW, values in ``[0, 1]``).

        Args:
            images: A single image or a list of images to process.
            threshold: Minimum confidence score for a detection to be kept.

        Returns:
            A single :class:`supervision.Detections` when *images* is a single
            image, or a list of :class:`supervision.Detections` for a batch.
        """
        import requests
        import supervision as sv
        import torchvision.transforms.functional as F
        from PIL import Image

        if not isinstance(images, list):
            images = [images]

        orig_sizes: List[tuple] = []
        processed: List[torch.Tensor] = []

        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    img = requests.get(img, stream=True).raw
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(f"Invalid image shape. Expected 3 channels (RGB), but got {img.shape[0]} channels.")

            h, w = img.shape[1:]
            orig_sizes.append((h, w))

            img = img.to(self.device)
            img = F.resize(img, (self.resolution, self.resolution))
            img = F.normalize(img, self.means, self.stds)
            processed.append(img)

        batch_tensor = torch.stack(processed).float()

        with torch.no_grad():
            trt_outputs: Dict[str, torch.Tensor] = self._trt({"input": batch_tensor})

            # Remap TRT output names to PostProcess expected keys.
            # The ONNX export uses "dets" / "labels" / (optionally) "masks".
            outputs = {
                "pred_logits": trt_outputs["labels"],
                "pred_boxes": trt_outputs["dets"],
            }
            if "masks" in trt_outputs:
                outputs["pred_masks"] = trt_outputs["masks"]

            target_sizes = torch.tensor(orig_sizes, device=self.device)
            results = self._postprocess(outputs, target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]
            keep = scores > threshold

            if "masks" in result:
                masks = result["masks"]
                detections = sv.Detections(
                    xyxy=boxes[keep].float().cpu().numpy(),
                    confidence=scores[keep].float().cpu().numpy(),
                    class_id=labels[keep].cpu().numpy(),
                    mask=masks[keep].squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes[keep].float().cpu().numpy(),
                    confidence=scores[keep].float().cpu().numpy(),
                    class_id=labels[keep].cpu().numpy(),
                )
            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]
