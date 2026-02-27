# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
export ONNX model and TensorRT engine for deployment
"""

import inspect
import os
import random
import re
import subprocess

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from rfdetr.datasets.transforms import Normalize
from rfdetr.models import build_model
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import get_rank, get_sha
from rfdetr.util.package import get_version

logger = get_logger()


def run_command_shell(command, dry_run: bool = False) -> subprocess.CompletedProcess:
    if dry_run:
        logger.info(f"\nCUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}\n")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        raise


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


def export_onnx(
    output_dir,
    model,
    input_names,
    input_tensors,
    output_names,
    dynamic_axes,
    backbone_only=False,
    verbose=True,
    opset_version=17,
):
    export_name = "backbone_model" if backbone_only else "inference_model"
    output_file = os.path.join(output_dir, f"{export_name}.onnx")

    # Prepare model for export
    if hasattr(model, "export"):
        model.export()

    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # Torch 2.10+ may default to the dynamo exporter which requires extra deps
        # (e.g. onnxscript). Use the legacy path for compatibility.
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        input_tensors,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        verbose=verbose,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )

    logger.info(f"\nSuccessfully exported ONNX model: {output_file}")
    return output_file


def onnx_simplify(onnx_dir: str, input_names, input_tensors, force=False):
    import onnx
    import onnxsim

    from rfdetr.deploy._onnx import OnnxOptimizer

    sim_onnx_dir = onnx_dir.replace(".onnx", ".sim.onnx")
    if os.path.isfile(sim_onnx_dir) and not force:
        return sim_onnx_dir

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]

    logger.info(f"start simplify ONNX model: {onnx_dir}")
    opt = OnnxOptimizer(onnx_dir)
    opt.info("Model: original")
    opt.common_opt()
    opt.info("Model: optimized")
    opt.save_onnx(sim_onnx_dir)
    input_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    model_opt, check_ok = onnxsim.simplify(onnx_dir, check_n=3, input_data=input_dict, dynamic_input_shape=False)
    if check_ok:
        onnx.save(model_opt, sim_onnx_dir)
    else:
        raise RuntimeError("Failed to simplify ONNX model.")
    logger.info(f"Successfully simplified ONNX model: {sim_onnx_dir}")
    return sim_onnx_dir


def trtexec(onnx_dir: str, args) -> None:
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
