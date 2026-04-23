# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""RF-DETR training package (PyTorch Lightning).

Provides the Lightning module, data module, callbacks, and CLI for
training and evaluation.

Exports:
    RFDETRModelModule: LightningModule wrapping the RF-DETR model and training loop.
    RFDETRDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from RF-DETR configs.
"""

from pytorch_lightning import seed_everything

from rfdetr.training.callbacks import (
    BestModelCallback,
    COCOEvalCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.training.cli import RFDETRCli
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer
from rfdetr.utilities.logger import get_logger

_logger = get_logger()

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "RFDETRCli",
    "RFDETRDataModule",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
    "RFDETRModelModule",
    "build_trainer",
    "convert_legacy_checkpoint",
    "seed_everything",
]
