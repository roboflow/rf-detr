# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
CLI orchestrator for ONNX and TensorRT model export.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from rfdetr.datasets.transforms import Normalize
from rfdetr.export._onnx.exporter import export_onnx, onnx_simplify
from rfdetr.export.tensorrt import trtexec
from rfdetr.models import build_model
from rfdetr.utilities.distributed import get_rank
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_sha, get_version

logger = get_logger()


def make_infer_image(infer_dir, shape, batch_size, device="cuda"):
    if infer_dir is None:
        dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
        image = Image.fromarray(dummy, mode="RGB")
    else:
        image = Image.open(infer_dir).convert("RGB")

    transforms = Compose(
        [
            Resize((shape[0], shape[0])),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    # inps = utils.nested_tensor_from_tensor_list([inps for _ in range(args.batch_size)])
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps


def no_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("BatchNorm2d found in the model. Please remove it.")


def main(args):
    git_info = get_sha()
    if git_info != "unknown":
        logger.info(f"Running from git repository: {git_info}")
    else:
        version = get_version()
        logger.info(f"Running RF-DETR version: {version or 'unknown'}")
    logger.info(f"Export config: {vars(args)}")
    # convert device to device_id
    if args.device == "cuda":
        device_id = "0"
    elif args.device == "cpu":
        device_id = ""
    else:
        device_id = str(int(args.device))
        args.device = f"cuda:{device_id}"

    # device for export onnx
    # TODO: export onnx with cuda failed with onnx error
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    result = build_model(args)
    model = result[0] if isinstance(result, tuple) else result
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_parameters}")
    n_backbone_parameters = sum(p.numel() for p in model.backbone.parameters())
    logger.info(f"number of backbone parameters: {n_backbone_parameters}")
    n_projector_parameters = sum(p.numel() for p in model.backbone[0].projector.parameters())
    logger.info(f"number of projector parameters: {n_projector_parameters}")
    n_backbone_encoder_parameters = sum(p.numel() for p in model.backbone[0].encoder.parameters())
    logger.info(f"number of backbone encoder parameters: {n_backbone_encoder_parameters}")
    n_transformer_parameters = sum(p.numel() for p in model.transformer.parameters())
    logger.info(f"number of transformer parameters: {n_transformer_parameters}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        logger.info(f"load checkpoints {args.resume}")

    if args.layer_norm:
        no_batch_norm(model)

    model.to(device)

    input_tensors = make_infer_image(args.infer_dir, args.shape, args.batch_size, device)
    input_names = ["input"]
    if args.backbone_only:
        output_names = ["features"]
    elif args.segmentation_head:
        output_names = ["dets", "labels", "masks"]
    else:
        output_names = ["dets", "labels"]
    dynamic_axes = None
    # Run model inference in pytorch mode
    model.eval().to("cuda")
    input_tensors = input_tensors.to("cuda")
    with torch.no_grad():
        if args.backbone_only:
            features = model(input_tensors)
            logger.debug(f"PyTorch inference output shape: {features.shape}")
        elif args.segmentation_head:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            masks = outputs["pred_masks"]
            if isinstance(masks, torch.Tensor):
                logger.debug(
                    f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                    f"Masks: {masks.shape}"
                )
            else:
                # masks is a dict with spatial_features, query_features, bias
                logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
                logger.debug(
                    "Mask spatial_features: "
                    f"{masks['spatial_features'].shape}, "
                    f"query_features: {masks['query_features'].shape}, "
                    f"bias: {masks['bias'].shape}"
                )
        else:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
    model.cpu()
    input_tensors = input_tensors.cpu()

    output_file = export_onnx(
        args.output_dir,
        model,
        input_names,
        input_tensors,
        output_names,
        dynamic_axes,
        backbone_only=args.backbone_only,
        verbose=args.verbose,
        opset_version=args.opset_version,
    )

    if args.simplify:
        output_file = onnx_simplify(output_file, input_names, input_tensors, args)

    if args.tensorrt:
        output_file = trtexec(output_file, args)
