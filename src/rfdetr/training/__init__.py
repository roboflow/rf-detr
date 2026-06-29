# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR training package (PyTorch Lightning).

Provides the Lightning module, data module, callbacks, and CLI for training and evaluation.

Exports:
    RFDETRModelModule: LightningModule wrapping the RF-DETR model and training loop.
    RFDETRDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from RF-DETR configs.
"""

from typing import TYPE_CHECKING, Any

from pytorch_lightning import seed_everything

from rfdetr.training.callbacks import (
    BestModelCallback,
    COCOEvalCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from rfdetr.training.checkpoint import convert_legacy_checkpoint
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.trainer import build_trainer
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from rfdetr.cli.train import RFDETRCli

_logger = get_logger()


def __getattr__(name: str) -> Any:
    # ``RFDETRCli`` is defined in ``rfdetr.cli.train`` and re-exported here.  It
    # is imported lazily to avoid a circular import: ``rfdetr.cli.train`` imports
    # ``rfdetr.training`` submodules at module load time.
    if name == "RFDETRCli":
        from rfdetr.cli.train import RFDETRCli

        return RFDETRCli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
