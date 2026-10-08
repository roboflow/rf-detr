# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Install probes for the optional training, dataset, evaluation and plotting packages.

Each flag answers "is this package installed" without importing it, so a caller can gate on it at collection time (for
example ``pytest.mark.skipif``) and import the package locally where it is used. Export-format packages live in
:mod:`rfdetr.export.imports`.

The flags are private and exist for the test suite's collection-time gating; no runtime code reads them. Runtime code
that needs an optional package keeps its own import boundary, so the ``_IS_*_AVAILABLE`` names defined next to that
boundary are a separate concept and are not replaced by these.
"""

from __future__ import annotations

from rfdetr.utilities.package import is_installed

#: Whether the optional torch-xla package is installed.
_IS_TORCH_XLA_INSTALLED = is_installed("torch_xla")
#: Whether the optional Transformer Engine package is installed. The probe is deliberately the top-level package: a
#: ``transformer_engine.pytorch`` probe would import the parent at collection, and a Transformer Engine install whose
#: PyTorch binding fails to import must surface as a test error rather than a silent skip.
_IS_TRANSFORMER_ENGINE_INSTALLED = is_installed("transformer_engine")
#: Whether the optional PyTorch Lightning package is installed.
_IS_PYTORCH_LIGHTNING_INSTALLED = is_installed("pytorch_lightning")
#: Whether the optional pytorch-optimizer package is installed.
_IS_PYTORCH_OPTIMIZER_INSTALLED = is_installed("pytorch_optimizer")
#: Whether the optional PEFT package is installed.
_IS_PEFT_INSTALLED = is_installed("peft")

#: Whether the optional Kornia package is installed.
_IS_KORNIA_INSTALLED = is_installed("kornia")
#: Whether the optional Albumentations package is installed.
_IS_ALBUMENTATIONS_INSTALLED = is_installed("albumentations")
#: Whether the optional WebDataset package is installed.
_IS_WEBDATASET_INSTALLED = is_installed("webdataset")

#: Whether the optional pycocotools package is installed.
_IS_PYCOCOTOOLS_INSTALLED = is_installed("pycocotools")
#: Whether the optional hotcoco package is installed.
_IS_HOTCOCO_INSTALLED = is_installed("hotcoco")
#: Whether the optional faster-coco-eval package is installed.
_IS_FASTER_COCO_EVAL_INSTALLED = is_installed("faster_coco_eval")
#: Whether the optional ultrafast-pycocotools package is installed.
_IS_UFCOCO_INSTALLED = is_installed("ultrafast_pycocotools")
#: Whether the optional Vernier package is installed.
_IS_VERNIER_INSTALLED = is_installed("vernier")

#: Whether the optional Matplotlib package is installed.
_IS_MATPLOTLIB_INSTALLED = is_installed("matplotlib")
#: Whether the optional pandas package is installed.
_IS_PANDAS_INSTALLED = is_installed("pandas")
#: Whether the optional seaborn package is installed.
_IS_SEABORN_INSTALLED = is_installed("seaborn")
