# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""This tool provides performance benchmarks by using ONNX Runtime and TensorRT to run inference on a given model with
the COCO validation set.

It offers reliable measurements of inference latency using ONNX Runtime or TensorRT on the device.
"""

import importlib
import json
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import torch
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None

from rfdetr.export._tensorrt.inference import TimeProfiler, TRTInference
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class _JsonArgparseCLI(Protocol):
    """Minimal jsonargparse CLI callable interface used by this script."""

    def __call__(self, component: Callable[..., Any]) -> Any:
        """Run jsonargparse CLI for a callable component."""
        ...


def get_image_list(ann_file: str) -> list[dict[str, Any]]:
    with open(ann_file) as fin:
        data = json.load(fin)
    return list(data["images"])


def load_image(file_path: str) -> Image.Image:
    return Image.open(file_path).convert("RGB")


_DEFAULT_INPUT_SIZE = (640, 640)
_DEFAULT_NUM_QUERIES = 300


def _static_dim(value: Any, fallback: int) -> int:
    """Coerce a tensor-shape entry to a positive int, falling back for dynamic/unknown axes.

    ONNX Runtime and TensorRT report dynamic axes as strings (e.g. ``"height"``), ``None``, or ``-1``. Such values
    cannot drive a fixed preprocessing size, so the caller-supplied *fallback* is used instead.

    Args:
        value: A single entry from a model input/output shape.
        fallback: Value to return when *value* is not a concrete positive integer.

    Returns:
        The integer dimension, or *fallback* for dynamic axes.
    """
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return fallback
    return dim if dim > 0 else fallback


def _ensure_contiguous(image: Tensor, target: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any] | None]:
    """Materialize *image* as a contiguous tensor, leaving *target* untouched.

    ``ToImage`` permutes a decoded HWC buffer to CHW without copying, so the pipeline otherwise
    yields a channels_last view. Runtimes that read the input buffer directly rather than honoring
    strides — the ExecuTorch runtime among them — then misread the image and silently return wrong
    predictions, so the copy is forced here where every export inference path picks it up.

    Args:
        image: CHW image tensor, possibly a non-contiguous view.
        target: Optional annotation dict, passed through unchanged.

    Returns:
        Tuple of ``(contiguous_image, target)``.
    """
    return image.contiguous(), target


def infer_transforms(size: tuple[int, int] = _DEFAULT_INPUT_SIZE) -> Any:
    """Build the benchmark preprocessing pipeline for a given model input size.

    Args:
        size: Target ``(height, width)`` the image is resized to before inference. Defaults to
            :data:`_DEFAULT_INPUT_SIZE` for dynamic-axis models where a static size cannot be read.

    Returns:
        A ``torchvision.transforms.v2.Compose`` that tensorizes, resizes, normalizes, and returns
        the image as a contiguous tensor.

    Note:
        Tensorize-then-resize with ``antialias=False`` mirrors ``RFDETR.predict()``'s preprocessing
        (``detr.py``): resizing the PIL image first would apply PIL's adaptive antialias filter and
        benchmark the model on inputs predict() never produces.
    """
    from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

    from rfdetr.datasets.transforms import Normalize

    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=False),
            Normalize(),
            _ensure_contiguous,
        ]
    )


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def post_process(
    outputs: Mapping[str, Tensor], target_sizes: Tensor, num_queries: int = _DEFAULT_NUM_QUERIES
) -> list[dict[str, Tensor]]:
    out_logits, out_bbox = outputs["labels"], outputs["dets"]

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    flat_scores = prob.view(out_logits.shape[0], -1)
    # Clamp k to the flattened dimension: when num_queries is a fallback for a dynamic-axis model
    # it may exceed num_queries*num_classes and trigger a topk runtime error.
    k = min(num_queries, flat_scores.shape[1])
    topk_values, topk_indexes = torch.topk(flat_scores, k, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).expand(-1, -1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)]

    return results


def infer_onnx(
    sess: Any,
    coco_evaluator: Any,
    time_profile: "TimeProfiler",
    prefix: str,
    img_list: Sequence[dict[str, Any]],
    device: str | torch.device,
    repeats: int = 1,
) -> None:
    input_shape = sess.get_inputs()[0].shape
    # fallback for dynamic-axis models (dynamic H/W report as strings or None)
    input_h = _static_dim(input_shape[2] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[0])
    input_w = _static_dim(input_shape[3] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[1])
    output_shape = sess.get_outputs()[0].shape
    num_queries = _static_dim(output_shape[1] if len(output_shape) > 1 else None, _DEFAULT_NUM_QUERIES)
    transforms = infer_transforms((input_h, input_w))

    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = transforms(image, None)  # target is None

        samples = image_tensor[None].numpy()

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                res = sess.run(None, {"input": samples})
        time_list.append(time_profile.total / repeats)
        outputs = {}
        outputs["labels"] = torch.Tensor(res[1]).to(device)
        outputs["dets"] = torch.Tensor(res[0]).to(device)

        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        results = post_process(outputs, orig_target_sizes, num_queries=num_queries)
        res = {img_dict["id"]: results[0]}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    logger.info(f"Model latency with ONNX Runtime: {1000 * sum(time_list) / len(img_list)}ms")

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


def infer_engine(
    model: "TRTInference",
    coco_evaluator: Any,
    time_profile: "TimeProfiler",
    prefix: str,
    img_list: Sequence[dict[str, Any]],
    device: str | torch.device,
    repeats: int = 1,
) -> None:
    input_shape = list(model.bindings[model.input_names[0]].shape)
    # fallback for dynamic-axis models
    input_h = _static_dim(input_shape[2] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[0])
    input_w = _static_dim(input_shape[3] if len(input_shape) > 3 else None, _DEFAULT_INPUT_SIZE[1])
    output_shape = list(model.bindings[model.output_names[0]].shape)
    num_queries = _static_dim(output_shape[1] if len(output_shape) > 1 else None, _DEFAULT_NUM_QUERIES)
    transforms = infer_transforms((input_h, input_w))

    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = transforms(image, None)  # target is None

        samples = image_tensor[None].to(device)
        _, _, h, w = samples.shape
        # torch.Tensor(np.array([h, w]).reshape((1, 2)).astype(np.float32)).to(device)
        # torch.Tensor(np.array([h / height, w / width]).reshape((1, 2)).astype(np.float32)).to(device)

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                outputs = model({"input": samples})

        time_list.append(time_profile.total / repeats)
        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        if coco_evaluator is not None:
            results = post_process(outputs, orig_target_sizes, num_queries=num_queries)
            res = {img_dict["id"]: results[0]}
            coco_evaluator.update(res)

    logger.info(f"Model latency with TensorRT: {1000 * sum(time_list) / len(img_list)}ms")

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


def main(
    path: str,
    coco_path: str = "data/coco",
    device: int = 0,
    run_benchmark: bool = False,
    disable_eval: bool = False,
) -> None:
    """Performance benchmark tool for ONNX/TRT models.

    Args:
        path: Engine file path (.onnx or .trt/.engine).
        coco_path: COCO dataset path.
        device: CUDA device index.
        run_benchmark: Repeat inference 10x to measure latency.
        disable_eval: Skip COCO evaluation.
    """
    logger.info(
        {
            "path": path,
            "coco_path": coco_path,
            "device": device,
            "run_benchmark": run_benchmark,
            "disable_eval": disable_eval,
        }
    )
    coco_gt = osp.join(coco_path, "annotations/instances_val2017.json")
    img_list = get_image_list(coco_gt)
    prefix = osp.join(coco_path, "val2017")
    if run_benchmark:
        repeats = 10
        logger.info("Inference for each image will be repeated 10 times ...")
    else:
        repeats = 1

    if not disable_eval:
        from rfdetr.evaluation.coco_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(coco_gt, ["bbox"])
    else:
        coco_evaluator = None
    time_profile = TimeProfiler()

    if path.endswith(".onnx"):
        import onnxruntime as nxrun

        sess = nxrun.InferenceSession(
            path,
            providers=[("CUDAExecutionProvider", {"device_id": device})],
        )
        infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    elif path.endswith((".trt", ".engine")):
        model = TRTInference(path, sync_mode=True, device=f"cuda:{device}")
        infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    else:
        raise NotImplementedError('Only model file names ending with ".onnx", ".trt", or ".engine" are supported.')


if __name__ == "__main__":
    jsonargparse = importlib.import_module("jsonargparse")

    cast(_JsonArgparseCLI, getattr(jsonargparse, "CLI"))(main)
