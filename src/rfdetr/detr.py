# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

import glob
import json
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import requests
import torch

if TYPE_CHECKING:
    import supervision as sv
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,  # DEPRECATED
    RFDETRLargeConfig,
    RFDETRLargeDeprecatedConfig,  # DEPRECATED
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,  # DEPRECATED
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.datasets.coco import is_valid_coco_dataset
from rfdetr.datasets.yolo import is_valid_yolo_dataset
from rfdetr.models import PostProcess, build_model
from rfdetr.utilities.logger import get_logger

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = get_logger()


class _ModelContext:
    """Lightweight model wrapper returned by RFDETR.get_model().

    Provides the same attribute interface as the legacy ``main.py:Model`` but
    without importing or depending on ``populate_args()`` or the legacy stack.

    Args:
        model: The underlying ``nn.Module`` (LWDETR instance).
        postprocess: PostProcess instance for converting raw outputs to boxes.
        device: Device the model lives on.
        resolution: Input resolution (square side length in pixels).
        args: Namespace produced by :func:`build_namespace`.
        class_names: Optional list of class name strings loaded from checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: List[str] = None,
    ) -> None:
        self.model = model
        self.postprocess = postprocess
        self.device = device
        self.resolution = resolution
        self.args = args
        self.class_names = class_names
        self.inference_model = None

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a different number of classes.

        Args:
            num_classes: New number of output classes (including background).
        """
        self.model.reinitialize_detection_head(num_classes)
        self.args.num_classes = num_classes


def _load_pretrain_weights_into(nn_model: torch.nn.Module, args: Any) -> List[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Mirrors ``Model.__init__`` and ``RFDETRModule._load_pretrain_weights``
    checkpoint loading logic: validates hash, re-downloads on corruption, and
    trims query embeddings to match the configured query count.

    Args:
        nn_model: The model to load weights into.
        args: Namespace with ``pretrain_weights``, ``num_classes``,
            ``num_queries``, and ``group_detr`` attributes.

    Returns:
        List of class names extracted from the checkpoint, or empty list.
    """
    class_names: List[str] = []

    download_pretrain_weights(args.pretrain_weights)
    if not os.path.isfile(args.pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying without MD5 validation.")
        download_pretrain_weights(args.pretrain_weights, redownload=True, validate_md5=False)
    validate_pretrain_weights(args.pretrain_weights, strict=False)

    try:
        checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)
    except Exception:
        logger.info("Failed to load pretrain weights, re-downloading")
        download_pretrain_weights(args.pretrain_weights, redownload=True, validate_md5=False)
        checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)

    if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
        class_names = checkpoint["args"].class_names or []

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    if checkpoint_num_classes != args.num_classes + 1:
        logger.warning(
            "Reinitializing detection head: checkpoint has %d classes, configured for %d.",
            checkpoint_num_classes - 1,
            args.num_classes,
        )
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    num_desired_queries = args.num_queries * args.group_detr
    query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in query_param_names):
            checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

    nn_model.load_state_dict(checkpoint["model"], strict=False)

    if checkpoint_num_classes != args.num_classes + 1:
        nn_model.reinitialize_detection_head(args.num_classes + 1)

    return class_names


def _apply_lora_to(nn_model: torch.nn.Module) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Args:
        nn_model: LWDETR model whose backbone encoder will receive LoRA.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        use_dora=True,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        ],
    )
    nn_model.backbone[0].encoder = get_peft_model(nn_model.backbone[0].encoder, lora_config)


def _build_model_context(model_config: ModelConfig) -> "_ModelContext":
    """Build a _ModelContext from ModelConfig without using legacy main.py:Model.

    Replicates ``Model.__init__`` logic: builds the nn.Module, optionally loads
    pretrain weights and applies LoRA, then moves the model to the target device.

    Args:
        model_config: Architecture configuration.

    Returns:
        Fully initialised _ModelContext ready for inference or training.
    """
    from rfdetr._namespace import build_namespace

    # A dummy TrainConfig is needed only for build_namespace's required fields;
    # dataset_dir/output_dir are unused during model construction.
    args = build_namespace(model_config, TrainConfig(dataset_dir=".", output_dir="."))
    nn_model = build_model(args)

    class_names: List[str] = []
    if args.pretrain_weights is not None:
        class_names = _load_pretrain_weights_into(nn_model, args)

    if args.backbone_lora:
        _apply_lora_to(nn_model)

    device = torch.device(args.device)
    nn_model = nn_model.to(device)
    postprocess = PostProcess(num_select=args.num_select)

    return _ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
    )


class RFDETR:
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        pretrain_weights = self.model_config.pretrain_weights
        if pretrain_weights is None:
            return
        download_pretrain_weights(pretrain_weights)

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """Train an RF-DETR model via the PyTorch Lightning stack.

        All keyword arguments are forwarded to :meth:`get_train_config` to build
        a :class:`~rfdetr.config.TrainConfig`.  Several legacy kwargs are absorbed
        so existing call-sites do not break:

        * ``device`` — mapped to ``TrainConfig.accelerator``; ``"cpu"`` becomes
          ``accelerator="cpu"``, all others default to ``"auto"``.
        * ``callbacks`` — if the dict contains any non-empty lists a
          :class:`DeprecationWarning` is emitted; the dict is then discarded.
          Use PTL :class:`~pytorch_lightning.Callback` objects passed via
          :func:`~rfdetr.training.build_trainer` instead.
        * ``start_epoch`` — emits :class:`DeprecationWarning` and is dropped.
        * ``do_benchmark`` — emits :class:`DeprecationWarning` and is dropped.

        After training completes the underlying ``nn.Module`` is synced back
        onto ``self.model.model`` so that :meth:`predict` and :meth:`export`
        continue to work without reloading the checkpoint.
        """
        from rfdetr.training import RFDETRDataModule, RFDETRModule, build_trainer

        # Absorb legacy `callbacks` dict — warn if non-empty, then discard.
        callbacks_dict = kwargs.pop("callbacks", None)
        if callbacks_dict and any(callbacks_dict.values()):
            warnings.warn(
                "Custom callbacks dict is not forwarded to PTL. Use PTL Callback objects instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Absorb legacy `device` kwarg.  When the caller explicitly requests CPU
        # (e.g. in tests or CPU-only environments), honour it by forwarding it as
        # the PTL accelerator.  All other device strings (cuda, mps) are ignored
        # so PTL can auto-select the best available device.
        _device = kwargs.pop("device", None)
        _accelerator = "cpu" if _device == "cpu" else None

        # Absorb legacy `start_epoch` — PTL resumes automatically via ckpt_path.
        if "start_epoch" in kwargs:
            warnings.warn(
                "`start_epoch` is deprecated and ignored; PTL resumes automatically via `resume`.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("start_epoch")

        # Pop `do_benchmark`; benchmarking via `.train()` is deprecated.
        run_benchmark = bool(kwargs.pop("do_benchmark", False))
        if run_benchmark:
            warnings.warn(
                "`do_benchmark` in `.train()` is deprecated; use `rfdetr benchmark`.",
                DeprecationWarning,
                stacklevel=2,
            )

        config = self.get_train_config(**kwargs)
        module = RFDETRModule(self.model_config, config)
        datamodule = RFDETRDataModule(self.model_config, config)
        trainer = build_trainer(config, self.model_config, accelerator=_accelerator)
        trainer.fit(module, datamodule, ckpt_path=config.resume or None)

        # Sync the trained weights back so predict() / export() see the updated model.
        self.model.model = module.model

    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution, device=self.model.device, dtype=dtype
                ),
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self):
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False

    def export(
        self,
        output_dir: str = "output",
        infer_dir: str = None,
        simplify: bool = False,
        backbone_only: bool = False,
        opset_version: int = 17,
        verbose: bool = True,
        force: bool = False,
        shape: tuple = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """Export the trained model to ONNX format.

        See the `ONNX export documentation <https://rfdetr.roboflow.com/learn/export/>`_
        for more information.

        Args:
            output_dir: Directory to write the ONNX file to.
            infer_dir: Optional directory of sample images for dynamic-axes inference.
            simplify: Whether to run onnx-simplifier on the exported graph.
            backbone_only: Export only the backbone (feature extractor).
            opset_version: ONNX opset version to target.
            verbose: Print export progress information.
            force: Force re-export even if output already exists.
            shape: ``(height, width)`` tuple; defaults to square at model resolution.
            batch_size: Static batch size to bake into the ONNX graph.
            **kwargs: Additional keyword arguments forwarded to export_onnx.
        """
        logger.info("Exporting model to ONNX format")
        try:
            from rfdetr.export.main import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            logger.error(
                "It seems some dependencies for ONNX export are missing."
                " Please run `pip install rfdetr[onnx]` and try again."
            )
            raise

        device = self.model.device
        model = deepcopy(self.model.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir_path = Path(output_dir)
        if shape is None:
            shape = (self.model.resolution, self.model.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ["input"]
        if backbone_only:
            output_names = ["features"]
        elif self.model_config.segmentation_head:
            output_names = ["dets", "labels", "masks"]
        else:
            output_names = ["dets", "labels"]

        dynamic_axes = None
        model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                logger.debug(f"PyTorch inference output shape: {features.shape}")
            elif self.model_config.segmentation_head:
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
                    logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs["pred_boxes"]
                labels = outputs["pred_logits"]
                logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")

        model.cpu()
        input_tensors = input_tensors.cpu()

        output_file = export_onnx(
            output_dir=str(output_dir_path),
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )

        logger.info(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file, input_names=input_names, input_tensors=input_tensors, force=force
            )
            logger.info(f"Successfully simplified ONNX model to: {sim_output_file}")

        logger.info("ONNX export completed successfully")
        self.model.model = self.model.model.to(device)

    @staticmethod
    def _load_classes(dataset_dir: str) -> List[str]:
        """Load class names from a COCO or YOLO dataset directory."""
        if is_valid_coco_dataset(dataset_dir):
            coco_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
            with open(coco_path, "r") as f:
                anns = json.load(f)
            categories = sorted(anns["categories"], key=lambda category: category.get("id", float("inf")))

            # Catch possible placeholders for no supercategory
            placeholders = {"", "none", "null", None}

            # If no meaningful supercategory exists anywhere, treat as flat dataset
            has_any_sc = any(c.get("supercategory", "none") not in placeholders for c in categories)
            if not has_any_sc:
                return [c["name"] for c in categories]

            # Mixed/Hierarchical: keep only categories that are not parents of other categories.
            # Both leaves (with a real supercategory) and standalone top-level nodes (supercategory is a
            # placeholder) satisfy this condition — neither appears as another category's supercategory.
            parents = {c.get("supercategory") for c in categories if c.get("supercategory", "none") not in placeholders}
            has_children = {c["name"] for c in categories if c["name"] in parents}

            class_names = [c["name"] for c in categories if c["name"] not in has_children]
            # Safety fallback for pathological inputs
            return class_names or [c["name"] for c in categories]

        # list all YAML files in the folder
        if is_valid_yolo_dataset(dataset_dir):
            yaml_paths = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
            # any YAML file starting with data e.g. data.yaml, dataset.yaml
            yaml_data_files = [yp for yp in yaml_paths if os.path.basename(yp).startswith("data")]
            yaml_path = yaml_data_files[0]
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if "names" in data:
                if isinstance(data["names"], dict):
                    return [data["names"][i] for i in sorted(data["names"].keys())]
                return data["names"]
            else:
                raise ValueError(f"Found {yaml_path} but it does not contain 'names' field.")
        raise FileNotFoundError(
            f"Could not find class names in {dataset_dir}."
            " Checked for COCO (train/_annotations.coco.json) and YOLO (data.yaml, data.yml) styles."
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig) -> "_ModelContext":
        """Retrieve a model context from the provided architecture configuration.

        Args:
            config: Architecture configuration.

        Returns:
            _ModelContext with model, postprocess, device, resolution, args,
            and class_names attributes.
        """
        return _build_model_context(config)

    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, "class_names") and self.model.class_names:
            return {i + 1: name for i, name in enumerate(self.model.class_names)}

        return COCO_CLASSES

    def predict(
        self,
        images: Union[
            str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]
        ],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, image url, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images:
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold:
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            A single or multiple Detections objects, each containing bounding box
            coordinates, confidence scores, and class IDs.
        """
        import supervision as sv

        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. Latency may be higher than expected."
                " You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

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
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                # this could happen if someone manually changes self.model.resolution after optimizing the model
                raise ValueError(
                    f"Resolution mismatch. "
                    f"Model was optimized for resolution {self._optimized_resolution}, "
                    f"but got {batch_tensor.shape[2]}."
                    " You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch. "
                        f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                        f"but got {batch_tensor.shape[0]}."
                        " You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                        " Alternatively, you can recompile the optimized model for a different batch size"
                        " by calling model.optimize_for_inference(batch_size=<new_batch_size>)."
                    )

        with torch.no_grad():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                return_predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    return_predictions["pred_masks"] = predictions[2]
                predictions = return_predictions
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

    def deploy_to_roboflow(
        self,
        workspace: str,
        project_id: str,
        version: str,
        api_key: Optional[str] = None,
        size: Optional[str] = None,
    ) -> None:
        """
        Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in
        Roboflow Workflows and on-device deployment.

        Args:
            workspace: The name of the Roboflow workspace to deploy to.
            project_id: The project ID to which the model will be deployed.
            version: The project version to which the model will be deployed.
            api_key: Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size: The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).

        Raises:
            ValueError: If the `api_key` is not provided and not found in the
                environment variable `ROBOFLOW_API_KEY`, or if the `size` is
                not set for custom architectures.
        """
        import shutil

        from roboflow import Roboflow

        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")

        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save({"model": self.model.model.state_dict(), "args": self.model.args}, outpath)
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(model_type=size, model_path=tmp_out_dir, filename="weights.pt")
        shutil.rmtree(tmp_out_dir)


class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """

    size = "rfdetr-base"

    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """

    size = "rfdetr-nano"

    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """

    size = "rfdetr-small"

    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """

    size = "rfdetr-medium"

    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRLargeNew(RFDETR):
    size = "rfdetr-large"

    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRLargeDeprecated(RFDETR):
    """
    Train an RF-DETR Large model.
    """

    size = "rfdetr-large"

    def __init__(self, **kwargs):
        warnings.warn(
            "RFDETRLargeDeprecated is deprecated and will be removed in a future version."
            " Please use RFDETRLarge instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)

    def get_model_config(self, **kwargs):
        return RFDETRLargeDeprecatedConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRLarge(RFDETR):
    size = "rfdetr-large"

    def __init__(self, **kwargs):
        self.init_error = None
        self.is_deprecated = False
        try:
            super().__init__(**kwargs)
        except Exception as e:
            self.init_error = e
            self.is_deprecated = True
            try:
                super().__init__(**kwargs)
                logger.warning(
                    "\n"
                    "=" * 100 + "\n"
                    "WARNING: Automatically switched to deprecated model configuration,"
                    " due to using deprecated weights."
                    " This will be removed in a future version.\n"
                    " Please retrain your model with the new weights and configuration.\n"
                    "=" * 100 + "\n"
                )
            except Exception:
                raise self.init_error

    def get_model_config(self, **kwargs):
        if not self.is_deprecated:
            return RFDETRLargeConfig(**kwargs)
        else:
            return RFDETRLargeDeprecatedConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRSegPreview(RFDETR):
    size = "rfdetr-seg-preview"

    def get_model_config(self, **kwargs):
        return RFDETRSegPreviewConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegNano(RFDETR):
    size = "rfdetr-seg-nano"

    def get_model_config(self, **kwargs):
        return RFDETRSegNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegSmall(RFDETR):
    size = "rfdetr-seg-small"

    def get_model_config(self, **kwargs):
        return RFDETRSegSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegMedium(RFDETR):
    size = "rfdetr-seg-medium"

    def get_model_config(self, **kwargs):
        return RFDETRSegMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegLarge(RFDETR):
    size = "rfdetr-seg-large"

    def get_model_config(self, **kwargs):
        return RFDETRSegLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegXLarge(RFDETR):
    size = "rfdetr-seg-xlarge"

    def get_model_config(self, **kwargs):
        return RFDETRSegXLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSeg2XLarge(RFDETR):
    size = "rfdetr-seg-2xlarge"

    def get_model_config(self, **kwargs):
        return RFDETRSeg2XLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)
