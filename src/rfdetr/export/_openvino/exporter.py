"""Direct PyTorch → OpenVINO IR export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def export_openvino(
    output_dir: str,
    model: nn.Module,
    input_tensors: torch.Tensor,
    backbone_only: bool = False,
    verbose: bool = True,
    variant_name: str | None = None,
    output_names: list[str] | None = None,
) -> str:
    """Export a PyTorch model directly to OpenVINO IR format.

    Args:
        output_dir: Directory where the exported model will be saved.
        model: PyTorch model to export.
        input_tensors: Example input tensor(s) for tracing.
        backbone_only: Whether exporting only the backbone.
        verbose: Whether to print verbose export information.
        variant_name: Optional model variant name (e.g., "nano", "small", "medium").
        output_names: List of output names (e.g., ["dets", "labels", "masks"]).

    Returns:
        Path to the exported OpenVINO IR model (.xml file).

    Raises:
        ImportError: If OpenVINO is not installed.
        RuntimeError: If export fails.
    """
    try:
        from openvino import convert_model, save_model
    except ImportError:
        logger.error(
            "OpenVINO is not installed. Please run `pip install openvino` and try again.",
        )
        raise

    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    if variant_name is not None:
        export_name = f"{variant_name}-backbone" if backbone_only else variant_name
    else:
        export_name = "backbone_model" if backbone_only else "inference_model"
    
    output_xml = os.path.join(output_dir, f"{export_name}.xml")
    output_bin = os.path.join(output_dir, f"{export_name}.bin")

    if verbose:
        logger.info(f"Converting PyTorch model to OpenVINO IR...")
        logger.info(f"Input shape: {input_tensors.shape}")

    # Ensure model is in eval mode and on CPU
    model.eval()
    model = model.cpu()
    input_tensors = input_tensors.cpu()

    # Determine output names if not provided
    if output_names is None:
        output_names = ["dets", "labels"]

    try:
        # Create a wrapper to handle dictionary outputs
        class ModelWrapper(nn.Module):
            """Wrapper to convert dictionary outputs to tuple of tensors."""
            
            def __init__(self, model: nn.Module, output_names: list[str]):
                super().__init__()
                self.model = model
                self.output_names = output_names
            
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                output = self.model(x)
                
                # Handle backbone-only case (returns tensor directly)
                if not isinstance(output, dict):
                    return (output,)
                
                # Extract outputs in the specified order
                result = []
                for name in self.output_names:
                    # Map output names to model's keys
                    if name == "dets":
                        result.append(output.get("pred_boxes", output.get("dets")))
                    elif name == "labels":
                        result.append(output.get("pred_logits", output.get("labels")))
                    elif name == "masks":
                        masks = output.get("pred_masks", output.get("masks"))
                        if isinstance(masks, torch.Tensor):
                            result.append(masks)
                    elif name == "keypoints":
                        keypoints = output.get("pred_keypoints", output.get("keypoints"))
                        if isinstance(keypoints, torch.Tensor):
                            result.append(keypoints)
                    elif name == "features":
                        result.append(output)
                
                # Filter out None values
                result = [r for r in result if r is not None]
                return tuple(result)
        
        # Wrap the model
        wrapped_model = ModelWrapper(model, output_names)
        wrapped_model.eval()
        
        # Convert directly to OpenVINO
        with torch.no_grad():
            ov_model = convert_model(wrapped_model, example_input=input_tensors)
        
        # Save the model
        save_model(ov_model, output_xml)
        
        if verbose:
            logger.info(f"✓ OpenVINO IR model saved to {output_xml}")
            logger.info(f"✓ Model binary saved to {output_bin}")
        
        return output_xml
        
    except Exception as e:
        logger.error(f"OpenVINO export failed: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        raise RuntimeError(f"Failed to export model to OpenVINO IR: {e}") from e
