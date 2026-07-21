# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T  # noqa: N812
from matplotlib.axes import Axes
from supervision import BoxAnnotator, Color, Detections, LabelAnnotator, MaskAnnotator
from torch import Tensor
from torch.utils.data import DataLoader

from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class DatasetGridSaver:
    """Utility for saving 3x3 image grids to visualize augmentation effects.

    Args:
        data_loader: Dataloader of the dataset to sample images from.
        output_dir: Directory where grid images will be saved.
        max_batches: Number of batches to draw samples from.
        dataset_type: Dataset split label, e.g. ``'train'`` or ``'val'``.
    """

    def __init__(
        self, data_loader: DataLoader, output_dir: Path, max_batches: int = 3, dataset_type: str = "train"
    ) -> None:
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.max_batches = max_batches
        self.dataset_type = dataset_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_grid(self) -> None:
        """Create and save image grids to ``output_dir``.

        Each grid is a 3x3 JPEG containing up to 9 images from a single batch, with bounding boxes and class labels
        drawn on top.
        """
        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        mask_annotator = MaskAnnotator(opacity=0.45)
        box_annotator = BoxAnnotator(thickness=2)
        label_annotator = LabelAnnotator(
            text_color=Color.BLACK,
            text_scale=0.5,
            text_padding=3,
        )

        for batch_idx, (sample, target) in enumerate(self.data_loader):
            if batch_idx >= self.max_batches:
                break

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f"{self.dataset_type} dataset, batch {batch_idx}")
            axes = axes.flatten()

            num_plotted = 0
            for single_image, single_target in zip(sample.tensors, target):
                if num_plotted >= 9:
                    break
                self._annotate_and_plot(
                    single_image,
                    single_target,
                    axes[num_plotted],
                    inv_normalize,
                    mask_annotator,
                    box_annotator,
                    label_annotator,
                )
                num_plotted += 1

            for i in range(num_plotted, 9):
                axes[i].axis("off")

            fig.tight_layout()
            plt.savefig(self.output_dir / f"{self.dataset_type}_batch{batch_idx}_grid.jpg", dpi=200)
            plt.close()

        logger.info(f"Saved {self.dataset_type} grids with augmented images to: {self.output_dir.resolve()}")

    @staticmethod
    def _annotate_and_plot(
        single_image: Tensor,
        single_target: dict[str, Any],
        ax: Axes,
        inv_normalize: T.Normalize,
        mask_annotator: MaskAnnotator,
        box_annotator: BoxAnnotator,
        label_annotator: LabelAnnotator,
    ) -> None:
        """De-normalize a single image tensor, annotate it with masks, boxes, and labels, and plot it on ``ax``.

        Args:
            single_image: Normalized image tensor of shape ``(C, H, W)``.
            single_target: Target dict with keys ``'size'``, ``'boxes'`` (cx,cy,wh normalized), ``'labels'``, and
                optionally ``'masks'``.
            ax: Matplotlib axis to plot the annotated image on.
            inv_normalize: Inverse normalization transform to convert the tensor back to pixel values.
            mask_annotator: ``MaskAnnotator`` instance for drawing instance masks.
            box_annotator: ``BoxAnnotator`` instance for drawing bounding boxes.
            label_annotator: ``LabelAnnotator`` instance for drawing class labels.
        """
        from PIL import Image as PILImage

        resized_size = single_target["size"]
        if isinstance(resized_size, Tensor):
            resized_size = resized_size.detach().cpu()
        h, w = int(resized_size[0]), int(resized_size[1])

        de_normalized_img = inv_normalize(single_image[:, :h, :w])
        if isinstance(de_normalized_img, Tensor):
            de_normalized_img = de_normalized_img.detach().cpu().numpy()
        scene = PILImage.fromarray(
            np.ascontiguousarray((np.clip(de_normalized_img.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))
        )

        if len(single_target["boxes"]) > 0:
            labels_tensor = single_target["labels"]
            if isinstance(labels_tensor, Tensor):
                class_ids = labels_tensor.detach().cpu().numpy().astype(int)
            else:
                class_ids = np.asarray(labels_tensor, dtype=int)

            boxes = single_target["boxes"]
            if isinstance(boxes, Tensor):
                boxes_iter = boxes.detach().cpu()
            else:
                boxes_iter = boxes

            xyxy = np.asarray(
                [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for box in boxes_iter for b in [box_cxcywh_to_xyxy(box)]],
                dtype=np.float32,
            )
            masks = DatasetGridSaver._extract_masks(single_target, image_shape=(h, w), num_instances=len(class_ids))
            detections = Detections(xyxy=xyxy, mask=masks, class_id=class_ids)
            labels = [str(c) for c in class_ids]
            if masks is not None:
                scene = mask_annotator.annotate(scene=scene, detections=detections)
            scene = box_annotator.annotate(scene=scene, detections=detections)
            scene = label_annotator.annotate(scene=scene, detections=detections, labels=labels)

        ax.imshow(scene)
        ax.axis("off")

    @staticmethod
    def _extract_masks(
        single_target: dict[str, Any],
        image_shape: tuple[int, int],
        num_instances: int,
    ) -> np.ndarray | None:
        """Return target masks as ``(N, H, W)`` booleans aligned to the rendered image shape."""
        masks = single_target.get("masks")
        if masks is None or len(masks) == 0:
            return None

        if isinstance(masks, Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        if masks_np.ndim != 3:
            logger.warning("Skipping dataset-grid masks with invalid shape %s; expected (N, H, W).", masks_np.shape)
            return None

        if masks_np.shape[0] < num_instances:
            logger.warning(
                "Skipping dataset-grid masks with %d masks for %d instances.",
                masks_np.shape[0],
                num_instances,
            )
            return None

        masks_np = masks_np[:num_instances].astype(bool, copy=False)
        height, width = image_shape
        if masks_np.shape[1:] == image_shape:
            return masks_np

        aligned_masks = np.zeros((masks_np.shape[0], height, width), dtype=bool)
        copy_height = min(height, masks_np.shape[1])
        copy_width = min(width, masks_np.shape[2])
        aligned_masks[:, :copy_height, :copy_width] = masks_np[:, :copy_height, :copy_width]
        return aligned_masks
