# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Static INT8 post-training quantization for an OpenVINO IR graph, via NNCF.

``quantization="int8"`` on ``format="openvino"`` compresses the converted IR before it is written, so the ``.xml`` /
``.bin`` pair on disk is already 8-bit and no extra runtime step is needed.

NNCF is asked for its transformer recipe (``ModelType.TRANSFORMER``) rather than its generic defaults. The difference
matters on this architecture: measured on a 500-image COCO val2017 subset, plain ``nncf.quantize(model, dataset)`` cost
5.34 mAP on Nano and 6.18 on Small, worse than a comparable ONNX Runtime configuration, because it quantizes
normalization and the elementwise math around attention along with the matrix multiplies. The transformer recipe keeps
those paths in float, which is the same conclusion the ONNX path reached by restricting its op list -- see
:mod:`rfdetr.export._onnx.quantize`.

NNCF is an extra install (``pip install nncf``) rather than part of the ``openvino`` extra: it is only needed for this
one mode, and it carries its own heavyweight dependency set that an FP32 or FP16 export has no use for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from rfdetr.export._runtime.calibration import calibration_batches, warn_if_too_few_samples
from rfdetr.utilities.logger import get_logger

logger = get_logger()

#: Quantization modes ``format="openvino"`` accepts. ``None`` and ``"fp32"`` leave the converted IR uncompressed.
VALID_QUANTIZATIONS: frozenset[str | None] = frozenset({None, "fp32", "int8"})


def _require_nncf() -> Any:
    """Import NNCF, or refuse with the install command for the one mode that needs it.

    Returns:
        The imported ``nncf`` module.

    Raises:
        ImportError: If ``nncf`` is not installed.
    """
    try:
        import nncf  # optional, and only for this mode
    except ImportError as exc:
        raise ImportError(
            "quantization='int8' for format='openvino' needs NNCF to collect activation ranges. "
            "Install it: `pip install nncf`. It is not part of the `rfdetr[openvino]` extra because only this "
            "mode uses it."
        ) from exc
    return nncf


def quantize_int8(
    ov_model: Any,
    calibration_data: str | Path | NDArray[Any],
    *,
    height: int,
    width: int,
    channels: int = 3,
    max_images: int = 100,
) -> Any:
    """Return an INT8 copy of *ov_model*, calibrated on *calibration_data*.

    Args:
        ov_model: The converted OpenVINO model, before it is saved.
        calibration_data: Directory of representative images, a ``.npy`` path, or a preprocessed array.
        height: Spatial height the graph was traced at.
        width: Spatial width the graph was traced at.
        channels: Channel count the graph expects.
        max_images: Maximum images read from a *calibration_data* directory.

    Returns:
        The quantized model, ready for ``openvino.save_model``.

    Raises:
        ImportError: If ``nncf`` is not installed.
        ValueError: If *calibration_data* yields no usable sample.

    Examples:
        Needs a converted IR graph and representative images, so this is documentation only (not a doctest):

        ```python
        quantize_int8(ov_model, "calibration_images/", height=512, width=512)
        # -> <Model: 'Model0'>
        ```
    """
    nncf = _require_nncf()

    batches = list(
        calibration_batches(
            calibration_data,
            height=height,
            width=width,
            channels=channels,
            max_images=max_images,
        )
    )
    if not batches:
        raise ValueError("Calibration data produced no samples.")
    warn_if_too_few_samples(len(batches))

    logger.info(f"Quantizing OpenVINO IR to INT8 from {len(batches)} calibration samples")
    # subset_size defaults to 300; left unset NNCF warns that the dataset is smaller than it wanted and the number
    # in the log no longer describes what was actually used.
    return nncf.quantize(
        ov_model,
        nncf.Dataset(batches),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(batches),
    )
