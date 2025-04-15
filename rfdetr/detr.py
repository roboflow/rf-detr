# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import json
import os
from collections import defaultdict
from rfdetr.util.logger import get_logger
from typing import Union, List

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

from rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig, TrainConfig, ModelConfig
from rfdetr.main import Model, download_pretrain_weights
from rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink
from rfdetr.util.coco_classes import COCO_CLASSES

logger = get_logger()

class RFDETR:
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

    def maybe_download_pretrain_weights(self):
        download_pretrain_weights(self.model_config.pretrain_weights)

    def get_model_config(self, **kwargs):
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)
    
    def export(self, **kwargs):
        self.model.export(**kwargs)

    def train_from_config(self, config: TrainConfig, **kwargs):
        with open(
            os.path.join(config.dataset_dir, "train", "_annotations.coco.json"), "r"
        ) as f:
            anns = json.load(f)
            num_classes = len(anns["categories"])
            class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
            self.model.class_names = class_names

        if self.model_config.num_classes != num_classes:
            logger.info(
                f"Model configured with {self.model_config.num_classes} classes, dataset contains {num_classes} classes. "
                f"Reinitializing detection head to match dataset ({num_classes} classes)."
            )
            self.model.reinitialize_detection_head(num_classes)
        
        
        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")
        
        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)
        
        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump()
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback
            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        return Model(**config.dict())
    
    # Get class_names from the model
    @property
    def class_names(self):
        if hasattr(self.model, 'class_names') and self.model.class_names:
            return {i+1: name for i, name in enumerate(self.model.class_names)}
            
        return COCO_CLASSES

    def predict(
            self,
            images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
            threshold: float = 0.5,
            **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                img_tensor = F.to_tensor(img)
            else:
                if (img > 1).any():
                    raise ValueError(
                        "Image has pixel values above 1. Please ensure the image is "
                        "normalized (scaled to [0, 1])."
                    )
                if img.shape[0] != 3:
                    raise ValueError(
                        f"Invalid image shape. Expected 3 channels (RGB), but got "
                        f"{img.shape[0]} channels."
                    )
                img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        with torch.inference_mode():
            predictions = self.model.model(batch_tensor)
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocessors["bbox"](predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            detections = sv.Detections(
                xyxy=boxes.cpu().numpy(),
                confidence=scores.cpu().numpy(),
                class_id=labels.cpu().numpy(),
            )
            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]


class RFDETRBase(RFDETR):
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRLarge(RFDETR):
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)
