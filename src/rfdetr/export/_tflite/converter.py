# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ONNX → TFLite conversion using the ``onnx2tf`` library.

``onnx2tf`` (PINTO0309) converts an ONNX graph to TFLite.  Version 2.0+
uses a fast ``flatbuffer_direct`` backend; earlier 1.x releases go through
the TensorFlow ``TFLiteConverter``.

The converter uses the ``onnx2tf`` Python API directly (rather than
shelling out to the CLI) so that we can:

* Apply a compatibility shim for older ``onnx2tf`` releases that call
  :func:`numpy.load` on pickled data without ``allow_pickle=True``.
* Redirect ``onnx2tf``'s built-in ``download_test_image_data()`` to use
  locally-prepared calibration data instead of downloading from GitHub
  (which can fail in many environments).

``onnx2tf`` uses ``download_test_image_data()`` in two contexts:

1. **Output validation** — always runs to compare ONNX-vs-TF outputs.
2. **INT8 calibration** — when ``output_integer_quantized_tflite=True``,
   uses the same function as a representative dataset source.

Both calls are redirected to local data via the
``_patch_validation_download()`` context manager.  This avoids the
network dependency and lets the caller supply proper calibration images
for INT8 quantization.

INT8 quantization
-----------------
When ``quantization="int8"`` the caller **should** supply representative
calibration images via *calibration_data*.  Accepted formats:

* A **directory path** containing JPEG/PNG images — the converter
  automatically loads, resizes, and converts them to the correct format.
  This is the easiest approach: just point to your dataset folder.
* A ``.npy`` file path — shape ``(N, H, W, 3)``, dtype ``float32``,
  values in ``[0, 1]``.
* A :class:`numpy.ndarray` with the same constraints.

Pixel values must be in ``[0, 1]`` (divided by 255 but **not**
ImageNet-normalized — the converter applies ImageNet normalization
automatically via ``onnx2tf``'s default ``quant_norm_mean`` /
``quant_norm_std`` parameters).

If no calibration data is provided, random noise is used instead and a
warning is emitted.  This is sufficient for ``fp32`` / ``fp16`` conversion
but will produce **poor accuracy** for ``int8``.

Note:
    The resulting ``.tflite`` model expects the same input normalization as
    the ONNX model: ImageNet mean/std (``mean=[0.485, 0.456, 0.406]``,
    ``std=[0.229, 0.224, 0.225]``).  The caller is responsible for applying
    this normalization at inference time.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Generator, cast

import numpy as np
from numpy.typing import NDArray

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# Supported quantization modes.
_VALID_QUANTIZATIONS: set[str | None] = {None, "fp32", "fp16", "int8"}

# Number of random calibration samples generated when none are provided.
_DEFAULT_CALIB_SAMPLES: int = 20

# Supported image file extensions for directory-based calibration.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Default number of images to sample from a directory for calibration.
_DEFAULT_DIR_CALIB_SAMPLES: int = 100


def _check_onnx2tf_available() -> None:
    """Verify that the ``onnx2tf`` package is importable.

    Raises:
        ImportError: If ``onnx2tf`` cannot be imported.
    """
    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx2tf is not installed. TFLite export requires both ONNX and "
            "TFLite export dependencies. Install them with: "
            "pip install rfdetr[onnx,tflite]"
        ) from exc


@contextlib.contextmanager
def _numpy_allow_pickle() -> Generator[None, None, None]:
    """Temporarily patch :func:`numpy.load` to set ``allow_pickle=True``.

    ``onnx2tf`` 1.x calls ``np.load()`` on its bundled calibration data
    without passing ``allow_pickle=True``.  NumPy ≥ 1.16.3 defaults that
    flag to ``False`` and raises :class:`ValueError` for pickled files.

    This context manager monkey-patches ``np.load`` for the duration of the
    ``onnx2tf`` conversion and restores the original afterwards.
    """
    _original_load = np.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("allow_pickle", True)
        return _original_load(*args, **kwargs)

    np.load = _patched_load  # type: ignore[assignment,unused-ignore]
    try:
        yield
    finally:
        np.load = _original_load  # type: ignore[assignment,unused-ignore]


@contextlib.contextmanager
def _patch_validation_download(npy_path: str) -> Generator[None, None, None]:
    """Redirect ``download_test_image_data()`` to use local calibration data.

    ``onnx2tf`` calls ``download_test_image_data()`` during conversion to
    fetch test images from GitHub.  The function is called in two places:

    1. **Validation** — compares ONNX-vs-TF outputs (all conversions).
    2. **INT8 calibration** — builds a representative dataset when
       ``output_integer_quantized_tflite=True``.

    This download can fail in many environments (firewalls, CI, air-gapped
    systems, or when the upstream file is unavailable).  This context
    manager monkey-patches the function in all known module locations to
    return the data from the calibration ``.npy`` file we already prepared.

    We intentionally do **not** use ``custom_input_op_name_np_data_path``
    because that code path triggers a ``tf.tile`` rank mismatch in onnx2tf
    1.x when processing models with DINOv2-style embeddings and N > 1
    calibration samples.  Patching the download function achieves the same
    goal without that issue.

    Args:
        npy_path: Path to the ``.npy`` file containing calibration data in
            NHWC format.
    """

    def _replacement() -> NDArray[Any]:
        # Calibration data prepared by _prepare_calibration_data() is always
        # a plain float32 ndarray — never pickled.  allow_pickle=False is
        # intentional here; allow_pickle=True is handled by _numpy_allow_pickle()
        # for onnx2tf's own internal np.load calls.
        return cast(NDArray[Any], np.load(npy_path, allow_pickle=False))

    originals: dict[str, Any] = {}
    modules = [
        "onnx2tf.utils.common_functions",
        "onnx2tf.onnx2tf",
    ]
    for mod_name in modules:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "download_test_image_data"):
            originals[mod_name] = getattr(mod, "download_test_image_data")
            setattr(mod, "download_test_image_data", _replacement)

    try:
        yield
    finally:
        for mod_name, original in originals.items():
            mod = sys.modules.get(mod_name)
            if mod:
                setattr(mod, "download_test_image_data", original)


def _load_calibration_images(
    image_dir: Path,
    height: int,
    width: int,
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> NDArray[np.float32]:
    """Load images from a directory and prepare them for calibration.

    Images are loaded, resized to ``(height, width)``, converted to
    ``float32`` in ``[0, 1]``, and stacked into an NHWC array.

    Args:
        image_dir: Directory containing image files (JPEG, PNG, etc.).
        height: Target image height matching the model input.
        width: Target image width matching the model input.
        max_images: Maximum number of images to load.  Files are sorted
            alphabetically and the first *max_images* are used.

    Returns:
        A ``float32`` NumPy array of shape ``(N, height, width, 3)`` with
        pixel values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If *image_dir* does not exist or contains no
            supported image files.
    """
    from PIL import Image

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)

    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {image_dir}. Supported extensions: {sorted(_IMAGE_EXTENSIONS)}"
        )

    image_paths = image_paths[:max_images]
    logger.info(f"Loading {len(image_paths)} calibration images from {image_dir} (resizing to {height}x{width})")

    arrays: list[NDArray[np.float32]] = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB").resize((width, height))
            image_array = np.asarray(img, dtype=np.float32)
            image_array /= np.float32(255.0)
            arrays.append(image_array)
        except Exception:
            logger.debug(f"Skipping unreadable image: {img_path}")
            continue

    if not arrays:
        raise FileNotFoundError(f"No readable images found in {image_dir}")

    logger.info(f"Loaded {len(arrays)} calibration images")
    return np.stack(arrays).astype(np.float32, copy=False)


def _get_onnx_input_info(onnx_path: Path) -> tuple[str, list[int]]:
    """Read the first input tensor's name and shape from an ONNX model.

    Args:
        onnx_path: Path to the ``.onnx`` file.

    Returns:
        A ``(name, dims)`` tuple where *dims* is the NCHW shape list,
        e.g. ``("input", [1, 3, 560, 560])``.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is not installed. TFLite export requires both ONNX and "
            "TFLite export dependencies. Install them with: "
            "pip install rfdetr[onnx,tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    return name, dims


def _prepare_calibration_data(
    onnx_path: Path,
    calibration_data: str | os.PathLike[str] | np.ndarray | None,
    output_dir: Path,
    quantization: str | None,
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> Path:
    """Prepare calibration data as a ``.npy`` file for ``onnx2tf``.

    The returned path points to a ``.npy`` file containing an NHWC float32
    array with pixel values in ``[0, 1]``.  This file is loaded by the
    ``_patch_validation_download()`` context manager, which replaces
    ``onnx2tf``'s built-in ``download_test_image_data()`` call.  ``onnx2tf``
    uses this data for both ONNX-vs-TF output validation and (when INT8 is
    requested) as a representative calibration dataset.

    Args:
        onnx_path: Path to the source ``.onnx`` file (used to read the
            input tensor NCHW shape for random data generation and for
            determining the target resolution when loading images from
            a directory).
        calibration_data: One of:

            * ``None`` — generate random calibration data.  Sufficient for
              fp32/fp16 but emits a warning for int8.
            * A **directory path** containing JPEG/PNG images — images are
              loaded, resized to the model input resolution, and converted
              to the correct format automatically.
            * A path to a ``.npy`` file containing an array of shape
              ``(N, H, W, 3)``, dtype float32, values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same constraints.
        output_dir: Directory where a temporary ``.npy`` file may be
            written when *calibration_data* is ``None``, a directory, or
            an ndarray.
        quantization: The requested quantization mode (used only to decide
            whether to emit a warning).
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Ignored for other
            calibration data formats.

    Returns:
        Path to the ``.npy`` calibration data file.

    Raises:
        FileNotFoundError: If *calibration_data* is a path that does not
            exist, or a directory with no supported images.
    """
    if calibration_data is None:
        if quantization == "int8":
            logger.warning(
                "No calibration_data provided for INT8 quantization. Using "
                "random data — this will produce poor quantization accuracy. "
                "For best results, pass calibration_data with representative "
                "images from your dataset."
            )
        _, input_dims = _get_onnx_input_info(onnx_path)
        # input_dims is NCHW, e.g. [1, 3, 384, 384].
        _, c, h, w = input_dims
        # NHWC, float32, [0, 1] range — onnx2tf applies ImageNet norm.
        calib = np.random.rand(_DEFAULT_CALIB_SAMPLES, h, w, c).astype(np.float32)
        npy_path = output_dir / "_rfdetr_calib_data.npy"
        np.save(str(npy_path), calib)
        logger.debug(f"Generated random calibration data: shape={calib.shape}, saved to {npy_path}")
    elif isinstance(calibration_data, np.ndarray):
        npy_path = output_dir / "_rfdetr_calib_data.npy"
        np.save(str(npy_path), calibration_data)
        logger.info(f"Using provided calibration array: shape={calibration_data.shape}")
    else:
        data_path = Path(calibration_data)
        if data_path.is_dir():
            # Directory of images — load, resize, and convert.
            _, input_dims = _get_onnx_input_info(onnx_path)
            _, _c, h, w = input_dims
            calib = _load_calibration_images(data_path, height=h, width=w, max_images=max_images)
            npy_path = output_dir / "_rfdetr_calib_data.npy"
            np.save(str(npy_path), calib)
            logger.info(f"Prepared calibration data from image directory: shape={calib.shape}, saved to {npy_path}")
        elif data_path.is_file():
            npy_path = data_path
            logger.info(f"Using calibration data from: {npy_path}")
        else:
            raise FileNotFoundError(f"Calibration data path not found: {data_path}")

    return npy_path


def export_tflite(
    onnx_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    quantization: str | None = None,
    calibration_data: str | os.PathLike[str] | np.ndarray | None = None,
    verbosity: str = "error",
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> Path:
    """Convert an ONNX model to TFLite via ``onnx2tf``.

    Uses the ``onnx2tf`` Python API with a NumPy compatibility shim so
    that both 1.x and 2.x releases of ``onnx2tf`` work correctly.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_dir: Directory where TFLite artifacts will be written.
            ``onnx2tf`` creates files named ``{stem}_float32.tflite`` and
            ``{stem}_float16.tflite`` (plus ``{stem}_integer_quant.tflite``
            when ``quantization="int8"``).
        quantization: Quantization mode.

            * ``None`` / ``"fp32"`` — default FP32 + FP16 output.
            * ``"fp16"`` — same as above (onnx2tf always emits both).
            * ``"int8"`` — additionally produce an INT8-quantized model.
        calibration_data: Representative data used by ``onnx2tf`` for
            output validation (fp32/fp16) and INT8 calibration.  Accepts:

            * ``None`` — auto-generate random data (warns for int8).
            * A **directory path** containing JPEG/PNG images — images
              are loaded, resized, and converted automatically.
            * A path to a ``.npy`` file — shape ``(N, H, W, 3)``,
              dtype float32, pixel values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same format.

            For INT8 quantization, provide real images from your dataset
            for best accuracy.
        verbosity: Log verbosity passed to ``onnx2tf``.  One of
            ``"debug"``, ``"info"``, ``"warn"``, ``"error"`` (default).
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Defaults to 100.
            Ignored for other calibration data formats.

    Returns:
        The path to the primary ``*_float32.tflite`` file.

    Raises:
        FileNotFoundError: If *onnx_path* does not exist or
            *calibration_data* points to a missing file.
        ImportError: If ``onnx2tf`` is not installed.
        ValueError: If *quantization* is not a recognized mode.
        RuntimeError: If the conversion fails.

    Note:
        This function is **not thread-safe**.  It globally monkey-patches
        :func:`numpy.load` (via :func:`_numpy_allow_pickle`) and
        ``onnx2tf.download_test_image_data`` (via
        :func:`_patch_validation_download`) for the duration of the
        conversion.  Concurrent calls from multiple threads will interfere
        with each other.  Run conversion in a subprocess if isolation is
        required.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if quantization not in _VALID_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported quantization mode {quantization!r}. "
            f"Choose from: {sorted(q for q in _VALID_QUANTIZATIONS if q is not None)}"
        )

    _check_onnx2tf_available()

    # Force-import onnx2tf submodules so that _patch_validation_download()
    # can patch them.  onnx2tf's __init__.py may not import all submodules
    # eagerly in all versions, so we ensure they are in sys.modules before
    # entering the patching context manager.
    import onnx2tf.onnx2tf as _onnx2tf_mod
    import onnx2tf.utils.common_functions as _onnx2tf_common

    del _onnx2tf_mod, _onnx2tf_common  # imported for side-effect only

    output_dir.mkdir(parents=True, exist_ok=True)

    calib_npy_path = _prepare_calibration_data(
        onnx_path, calibration_data, output_dir, quantization, max_images=max_images
    )

    logger.info(f"Converting ONNX → TFLite (quantization={quantization!r}, verbosity={verbosity!r}): {onnx_path}")

    try:
        # _patch_validation_download redirects onnx2tf's
        # download_test_image_data() to return our calibration data.
        # onnx2tf uses this data for both ONNX/TF output validation and
        # (when int8 is requested) as a representative calibration dataset.
        #
        # We intentionally do NOT pass custom_input_op_name_np_data_path
        # because that code path in onnx2tf 1.x triggers a tf.tile rank
        # mismatch when processing the DINOv2 backbone with N > 1 samples.
        # The patched download function achieves the same goal without that
        # issue.
        #
        # output_signaturedefs=True is required because segmentation
        # models produce ONNX node names (e.g.
        # "/segmentation_head/blocks.2/dwconv/Conv/kernel") that contain
        # leading "/" characters which violate the saved_model naming
        # pattern. Enabling signature defs bypasses this restriction.
        with (
            _numpy_allow_pickle(),
            _patch_validation_download(str(calib_npy_path)),
        ):
            from onnx2tf import convert

            convert_kwargs: dict[str, Any] = {
                "input_onnx_file_path": str(onnx_path),
                "output_folder_path": str(output_dir),
                "output_signaturedefs": True,
                "non_verbose": True,
                "verbosity": verbosity,
            }

            if quantization == "int8":
                convert_kwargs["output_integer_quantized_tflite"] = True

            convert(**convert_kwargs)

    except Exception as exc:
        logger.error(f"onnx2tf conversion failed: {exc}")
        raise RuntimeError(f"onnx2tf conversion failed: {exc}") from exc

    # onnx2tf names output files based on the ONNX model stem.
    model_stem = onnx_path.stem
    primary = output_dir / f"{model_stem}_float32.tflite"

    if not primary.is_file():
        # Fallback: look for any .tflite file produced from this specific ONNX stem.
        # Scoped to {stem}_*.tflite to avoid returning a stale artifact from a
        # previous export in a reused output directory (review C2).
        tflite_files = sorted(output_dir.glob(f"{model_stem}_*.tflite"))
        if tflite_files:
            primary = tflite_files[0]
            logger.info(f"Expected {model_stem}_float32.tflite not found; using {primary.name} instead.")
        else:
            raise RuntimeError(
                f"onnx2tf completed but no .tflite file matching '{model_stem}_*.tflite' was found in {output_dir}"
            )

    logger.info(f"TFLite model exported to: {primary}")
    return primary
