# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ONNX → TFLite conversion using the ``onnx2tf`` library.

``onnx2tf`` (PINTO0309) converts an ONNX graph to TFLite.  **Version 2.4.0 or later is required** — earlier 1.x releases
cannot lower three op patterns in the RF-DETR graph (constant Expand, 1-D TopK, rank-3 Tile). Although the onnx2tf 2.x
default backend is ``flatbuffer_direct``, RF-DETR unconditionally forces ``tflite_backend="tf_converter"`` to avoid a
runtime error in the TFLite TopK_V2 kernel (``flatbuffer_direct`` trips a "k > internal dimension" check at
``AllocateTensors()`` time).  ``Erf`` and ``GeLU`` activations are replaced with TFLite-native pseudo-operators
(``replace_to_pseudo_operators=["Erf", "GeLU"]``) so the produced model does not require the TensorFlow Flex delegate at
inference time.

GridSample rewrite
------------------
RF-DETR's deformable cross-attention uses :func:`torch.nn.functional.grid_sample` once per decoder layer (6 calls
total).  ``onnx2tf``'s built-in GridSample handler lowers the op to ``tf.gather_nd(batch_dims=1)``, which TFLite's
``GatherNd`` kernel does not support — the kernel silently accepts the call during ``AllocateTensors()`` but produces
numerically wrong output at inference time, causing all detection scores to collapse from ~0.6 to ~0.1. The
``replace_to_pseudo_operators=["GridSample"]`` pseudo-op path also produces numerically wrong logit magnitudes in both
FP32 and FP16 (the pseudo-op itself is broken, not a quantization issue).  An earlier ONNX-level rewrite using
``GatherElements(axis=2)`` was lowered to ``tf.gather_nd(batch_dims=2)``, which TFLite does not support and crashes with
index out-of-bounds at inference time.

Before invoking ``onnx2tf``, :func:`_replace_gridsample_for_tflite` rewrites every ``GridSample`` node in the ONNX graph
into an equivalent bilinear sampling subgraph built from ``Gather(axis=0)`` on a transposed and flattened
``(N*(H+2)*(W+2), C)`` image tensor.  ``onnx2tf`` lowers ``Gather(axis=0)`` to TFLite's ``GATHER`` op with no
``batch_dims`` — the only TFLite gather path that is unconditionally supported, neither crashing on
``AllocateTensors()`` nor producing wrong values.

Import order
------------
TensorFlow must be imported before ONNX's C extension or the conversion deadlocks, so :func:`export_tflite` calls
:func:`~rfdetr.export._backend.preload_tensorflow_before_onnx` first — see there for why.

The converter uses the ``onnx2tf`` Python API directly (rather than shelling out to the CLI) to:

* Apply a compatibility shim for older ``onnx2tf`` releases that call
  :func:`numpy.load` on pickled data without ``allow_pickle=True``.
* Redirect the legacy ``download_test_image_data()`` validation hook to locally prepared data when an ``onnx2tf``
  version exposes it. Releases 2.4.0–2.4.3 do not expose this hook, so the patch is currently inert.
* Put the running interpreter's script directory on ``PATH`` (:func:`_interpreter_scripts_on_path`) so the bare
  ``onnxsim`` console script that ``onnx2tf`` shells out to resolves under a non-activated virtual environment.

INT8 quantization
-----------------
``quantization="int8"`` produces a **dynamic-range** INT8 model (INT8 weights, float activations, roughly 4x smaller
than FP32, no calibration data needed), built from the ``onnx2tf`` SavedModel.

Static (full-integer) INT8 is not supported and raises ``ValueError``: RF-DETR's transformer activations do not survive
8-bit post-training quantization.  Because dynamic-range quantization derives its weight scales from the weights
themselves, *calibration_data* cannot change the INT8 artifact — TFLite only consumes representative data through
``representative_dataset``, which this path never sets.

Note:
    The resulting ``.tflite`` model expects the same input normalization as the ONNX model: ImageNet mean/std
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).  The caller is responsible for applying this
    normalization at inference time.

Note:
    Segmentation models additionally emit a ``masks`` output.  FP32, FP16, and dynamic-range INT8 all match the PyTorch
    baseline closely (INT8 mask fidelity is marginally lower).  Verified on the non-plus segmentation
    variants: Nano, Small, Medium, Large, and Preview.
"""

from __future__ import annotations

import contextlib
import os
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any, Generator, cast

import numpy as np
from numpy.typing import NDArray

from rfdetr.export._resize import _bilinear_resize_half_pixel
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


def _replace_single_gridsample(node: Any, graph: Any, *, index: int) -> None:
    """Rewrite one GridSample ONNX node into a TFLite-safe bilinear subgraph.

    Replaces ``GridSample(im, grid)`` with an equivalent bilinear sampling subgraph that performs four
    ``Gather(axis=0)`` lookups on a transposed and flattened ``(N*(H+2)*(W+2), C)`` image tensor.  ``onnx2tf`` lowers
    ``Gather(axis=0)`` to TFLite's ``GATHER`` op with no ``batch_dims`` — the only TFLite gather path that is
    unconditionally supported, neither crashing on ``AllocateTensors()`` nor producing wrong values. Per-sample batch
    offsets are added to the flat index so that a single rank-1 ``Gather`` covers the entire batch.

    The replacement is mathematically identical to PyTorch's ``F.grid_sample`` for ``mode="bilinear"``,
    ``padding_mode="zeros"``, ``align_corners=0``.  Out-of-bounds sample positions are clamped to the zero-padded
    border, which has the same effect as PyTorch's zero padding.

    Shape-dependent values are computed at runtime via ONNX Shape/Gather/ Concat/Cast ops so the subgraph works for any
    static or dynamic input shape.

    Args:
        node: The ``gs.Node`` to replace (must be a ``GridSample`` node).
        graph: The ``gs.Graph`` that owns *node*.
        index: Unique integer suffix used to namespace tensor/constant names.

    Raises:
        NotImplementedError: If the node has unsupported attributes.
        ValueError: If ``im`` or ``grid`` are not rank-4 tensors.
    """
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs

    im = node.inputs[0]  # [N, C, H, W]
    grid = node.inputs[1]  # [N, H_out, W_out, 2]
    out = node.outputs[0]  # [N, C, H_out, W_out]

    mode = node.attrs.get("mode", "bilinear")
    padding_mode = node.attrs.get("padding_mode", "zeros")
    align_corners = node.attrs.get("align_corners", 0)

    if mode != "bilinear" or padding_mode != "zeros" or align_corners != 0:
        raise NotImplementedError(
            f"GridSample TFLite patch only supports mode='bilinear', "
            f"padding_mode='zeros', align_corners=0; "
            f"got mode={mode!r}, padding_mode={padding_mode!r}, align_corners={align_corners}"
        )

    if im.shape is None or len(im.shape) != 4:
        raise ValueError(f"GridSample TFLite patch requires rank-4 im; got im.shape={im.shape}")
    if grid.shape is None or len(grid.shape) != 4:
        raise ValueError(f"GridSample TFLite patch requires rank-4 grid; got grid.shape={grid.shape}")

    pfx = f"_gsrepl{index}"
    uid: list[int] = [0]

    def v(name: str, dtype: Any = np.float32, shape: list[int] | None = None) -> Any:
        uid[0] += 1
        return gs.Variable(f"{pfx}_{uid[0]}_{name}", dtype=dtype, shape=shape)

    def c(val: Any, dtype: Any = np.float32, name: str = "") -> Any:
        uid[0] += 1
        return gs.Constant(f"{pfx}_{uid[0]}_c_{name}", np.array(val, dtype=dtype))

    nodes: list[Any] = []

    def op(kind: str, ins: list[Any], outs: list[Any], attrs: dict[str, Any] | None = None) -> None:
        nodes.append(gs.Node(op=kind, inputs=ins, outputs=outs, attrs=attrs or {}))

    i64 = int(onnx.TensorProto.INT64)
    f32 = int(onnx.TensorProto.FLOAT)

    # ── Scalar / vector constants reused throughout the subgraph ────────────
    zero_i = c(np.int64(0), np.int64, "zero_i")
    one_i = c(np.int64(1), np.int64, "one_i")
    two_i = c(np.int64(2), np.int64, "two_i")

    zero_f = c(np.float32(0.0), np.float32, "zero_f")
    one_f = c(np.float32(1.0), np.float32, "one_f")
    two_f = c(np.float32(2.0), np.float32, "two_f")
    half_f = c(np.float32(0.5), np.float32, "half_f")

    ax0_1d = c([0], np.int64, "ax0_1d")  # Unsqueeze axes=[0] (vector form, opset-13)
    ax3_1d = c([3], np.int64, "ax3_1d")  # Slice/Squeeze axes=[3]
    neg1_1d = c([-1], np.int64, "neg1_1d")  # Reshape to rank-1 flat vector

    # ── Step 0: Runtime shape extraction via Shape + Gather ─────────────────
    # Dimensions may be symbolic strings when the ONNX was exported with
    # dynamic spatial axes.  Extract them as scalar int64 tensors at runtime.
    im_shape_t = v("im_shape", dtype=np.int64, shape=[4])
    op("Shape", [im], [im_shape_t])
    grid_shape_t = v("grid_shape", dtype=np.int64, shape=[4])
    op("Shape", [grid], [grid_shape_t])

    def gather_dim(shape_t: Any, axis_idx: int, name: str) -> Any:
        result = v(name, dtype=np.int64)
        op("Gather", [shape_t, c(np.int64(axis_idx), np.int64, f"i{axis_idx}_{name}")], [result], {"axis": 0})
        return result

    # Variables below mirror the tensor names in the algorithm spec (capitalised
    # dimension symbols N, C, H, W and the padded variants pH, pW) — ruff N806
    # is suppressed locally to preserve algorithmic readability.
    N_t = gather_dim(im_shape_t, 0, "N")  # noqa: N806
    C_t = gather_dim(im_shape_t, 1, "C")  # noqa: N806
    H_t = gather_dim(im_shape_t, 2, "H")  # noqa: N806
    W_t = gather_dim(im_shape_t, 3, "W")  # noqa: N806
    H_out_t = gather_dim(grid_shape_t, 1, "H_out")  # noqa: N806
    W_out_t = gather_dim(grid_shape_t, 2, "W_out")  # noqa: N806

    # Padded dims (scalars, int64): pH = H + 2, pW = W + 2, pH_pW = pH * pW
    pH_t = v("pH", dtype=np.int64)  # noqa: N806
    op("Add", [H_t, two_i], [pH_t])
    pW_t = v("pW", dtype=np.int64)  # noqa: N806
    op("Add", [W_t, two_i], [pW_t])
    pH_pW_t = v("pH_pW", dtype=np.int64)  # noqa: N806
    op("Mul", [pH_t, pW_t], [pH_pW_t])

    # Per-batch stride (N * pH * pW) used to build the flat-data reshape target.
    N_pH_pW_t = v("N_pH_pW", dtype=np.int64)  # noqa: N806
    op("Mul", [N_t, pH_pW_t], [N_pH_pW_t])

    # ── Float casts for coordinate arithmetic ───────────────────────────────
    W_f = v("W_f", dtype=np.float32)  # noqa: N806
    op("Cast", [W_t], [W_f], {"to": f32})
    H_f = v("H_f", dtype=np.float32)  # noqa: N806
    op("Cast", [H_t], [H_f], {"to": f32})
    pW_f = v("pW_f", dtype=np.float32)  # noqa: N806
    op("Cast", [pW_t], [pW_f], {"to": f32})
    pH_f = v("pH_f", dtype=np.float32)  # noqa: N806
    op("Cast", [pH_t], [pH_f], {"to": f32})

    W_half = v("W_half", dtype=np.float32)  # noqa: N806
    op("Div", [W_f, two_f], [W_half])  # W / 2
    H_half = v("H_half", dtype=np.float32)  # noqa: N806
    op("Div", [H_f, two_f], [H_half])  # H / 2

    pW_max_f = v("pW_max_f", dtype=np.float32)  # noqa: N806
    op("Sub", [pW_f, one_f], [pW_max_f])  # pW - 1  (max clamped x index)
    pH_max_f = v("pH_max_f", dtype=np.float32)  # noqa: N806
    op("Sub", [pH_f, one_f], [pH_max_f])  # pH - 1  (max clamped y index)

    # ── Dynamic reshape targets built via Unsqueeze + Concat ────────────────
    def unsq0(scalar_t: Any, name: str) -> Any:
        """Promote a 0-D int64 scalar to a 1-D [1] int64 vector for Concat."""
        result = v(name, dtype=np.int64, shape=[1])
        op("Unsqueeze", [scalar_t, ax0_1d], [result])
        return result

    N_1d = unsq0(N_t, "N_1d")  # noqa: N806
    C_1d = unsq0(C_t, "C_1d")  # noqa: N806
    H_out_1d = unsq0(H_out_t, "H_out_1d")  # noqa: N806
    W_out_1d = unsq0(W_out_t, "W_out_1d")  # noqa: N806
    N_pH_pW_1d = unsq0(N_pH_pW_t, "N_pH_pW_1d")  # noqa: N806

    # Flat-data reshape target: (N*(H+2)*(W+2), C)
    flat_shape_t = v("flat_shape", dtype=np.int64, shape=[2])
    op("Concat", [N_pH_pW_1d, C_1d], [flat_shape_t], {"axis": 0})

    # Output NHWC reshape target: (N, H_out, W_out, C) — built fresh for the
    # final Reshape that precedes the back-to-NCHW Transpose.
    nhwc_shape_t = v("nhwc_shape", dtype=np.int64, shape=[4])
    op("Concat", [N_1d, H_out_1d, W_out_1d, C_1d], [nhwc_shape_t], {"axis": 0})

    # ── Step 1: Build flat data tensor (N*(H+2)*(W+2), C) ───────────────────
    # Pad the NCHW image by 1 on each spatial side (zero padding), then move
    # channels to the last axis so a single Reshape produces the row-per-pixel
    # layout that Gather(axis=0) needs.
    # pads layout: [dim0_beg, dim1_beg, dim2_beg, dim3_beg, dim0_end, ...]
    pads = c([0, 0, 1, 1, 0, 0, 1, 1], np.int64, "pads")
    im_pad = v("im_pad")  # (N, C, H+2, W+2)
    op("Pad", [im, pads], [im_pad])
    im_nhwc = v("im_nhwc")  # (N, H+2, W+2, C)
    op("Transpose", [im_pad], [im_nhwc], {"perm": [0, 2, 3, 1]})
    im_flat = v("im_flat")  # (N*(H+2)*(W+2), C)
    op("Reshape", [im_nhwc, flat_shape_t], [im_flat])

    # ── Step 2: Extract gx, gy from grid last dim ───────────────────────────
    gx_raw = v("gx_raw")
    op("Slice", [grid, c([0], np.int64, "s0_gx"), c([1], np.int64, "e1_gx"), ax3_1d], [gx_raw])
    gx = v("gx")
    op("Squeeze", [gx_raw, ax3_1d], [gx])

    gy_raw = v("gy_raw")
    op("Slice", [grid, c([1], np.int64, "s1_gy"), c([2], np.int64, "e2_gy"), ax3_1d], [gy_raw])
    gy = v("gy")
    op("Squeeze", [gy_raw, ax3_1d], [gy])

    # Unnormalize (align_corners=False): px = (gx + 1) * W/2 - 0.5
    gxp1 = v("gxp1")
    op("Add", [gx, one_f], [gxp1])
    pxr = v("pxr")
    op("Mul", [gxp1, W_half], [pxr])
    px = v("px")
    op("Sub", [pxr, half_f], [px])

    gyp1 = v("gyp1")
    op("Add", [gy, one_f], [gyp1])
    pyr = v("pyr")
    op("Mul", [gyp1, H_half], [pyr])
    py = v("py")
    op("Sub", [pyr, half_f], [py])

    # ── Step 3: Floor + bilinear weights ────────────────────────────────────
    x0_f = v("x0_f")
    op("Floor", [px], [x0_f])
    y0_f = v("y0_f")
    op("Floor", [py], [y0_f])
    x1_f = v("x1_f")
    op("Add", [x0_f, one_f], [x1_f])
    y1_f = v("y1_f")
    op("Add", [y0_f, one_f], [y1_f])

    wx1 = v("wx1")
    op("Sub", [px, x0_f], [wx1])
    wy1 = v("wy1")
    op("Sub", [py, y0_f], [wy1])
    wx0 = v("wx0")
    op("Sub", [one_f, wx1], [wx0])
    wy0 = v("wy0")
    op("Sub", [one_f, wy1], [wy0])

    # ── Step 4: Shifted + clamped integer coords in the padded image ────────
    def int_shifted_clamped(coord_f: Any, clamp_max: Any, name: str) -> Any:
        shifted = v(f"sh_{name}")
        op("Add", [coord_f, one_f], [shifted])  # shift by 1 for the padding band
        clipped = v(f"cl_{name}")
        op("Clip", [shifted, zero_f, clamp_max], [clipped])
        cast_int = v(f"ca_{name}", dtype=np.int64)
        op("Cast", [clipped], [cast_int], {"to": i64})
        return cast_int

    x0c = int_shifted_clamped(x0_f, pW_max_f, "x0c")
    x1c = int_shifted_clamped(x1_f, pW_max_f, "x1c")
    y0c = int_shifted_clamped(y0_f, pH_max_f, "y0c")
    y1c = int_shifted_clamped(y1_f, pH_max_f, "y1c")

    # ── Step 5: Batch offset via Range ──────────────────────────────────────
    # batch_range = [0, 1, ..., N-1]; batch_offset = batch_range * pH_pW
    # reshaped to (N, 1, 1) so it broadcasts against (N, H_out, W_out).
    batch_range = v("batch_range", dtype=np.int64)
    op("Range", [zero_i, N_t, one_i], [batch_range])
    batch_offset_flat = v("batch_offset_flat", dtype=np.int64)
    op("Mul", [batch_range, pH_pW_t], [batch_offset_flat])
    batch_offset_shape = c([-1, 1, 1], np.int64, "batch_offset_shape")
    batch_offset = v("batch_offset", dtype=np.int64)
    op("Reshape", [batch_offset_flat, batch_offset_shape], [batch_offset])

    # ── Step 6: Global 1-D flat indices per corner ──────────────────────────
    # local  = yc * pW + xc                            → (N, H_out, W_out)
    # global = local + batch_offset                    → (N, H_out, W_out)
    # gidx   = Reshape(global, [-1])                   → (N*H_out*W_out,)
    def flat_global_index(xc: Any, yc: Any, name: str) -> Any:
        ypw = v(f"ypw_{name}", dtype=np.int64)
        op("Mul", [yc, pW_t], [ypw])  # pW broadcasts as scalar
        local = v(f"local_{name}", dtype=np.int64)
        op("Add", [ypw, xc], [local])
        global_idx = v(f"glob_{name}", dtype=np.int64)
        op("Add", [local, batch_offset], [global_idx])  # batch_offset broadcasts on (N,1,1)
        flat1d = v(f"gidx_{name}", dtype=np.int64)
        op("Reshape", [global_idx, neg1_1d], [flat1d])
        return flat1d

    gidx_aa = flat_global_index(x0c, y0c, "aa")
    gidx_ba = flat_global_index(x1c, y0c, "ba")
    gidx_ab = flat_global_index(x0c, y1c, "ab")
    gidx_bb = flat_global_index(x1c, y1c, "bb")

    # ── Step 7: Gather(axis=0) on the flat data tensor ──────────────────────
    # onnx2tf lowers Gather(axis=0) to TFLite GATHER with no batch_dims.
    def gather_flat(gidx: Any, name: str) -> Any:
        sampled = v(f"samp_{name}")  # (N*H_out*W_out, C)
        op("Gather", [im_flat, gidx], [sampled], {"axis": 0})
        return sampled

    saa = gather_flat(gidx_aa, "aa")
    sba = gather_flat(gidx_ba, "ba")
    sab = gather_flat(gidx_ab, "ab")
    sbb = gather_flat(gidx_bb, "bb")

    # ── Step 8: Bilinear weighted sum on (N*K, C) ───────────────────────────
    # Each weight is reshaped to (N*K, 1) so it broadcasts over the channel dim.
    wflat_shape = c([-1, 1], np.int64, "wflat_shape")

    def contrib(gathered: Any, wx: Any, wy: Any, name: str) -> Any:
        w_2d = v(f"w2d_{name}")  # (N, H_out, W_out)
        op("Mul", [wy, wx], [w_2d])
        w_flat = v(f"wflat_{name}")  # (N*K, 1)
        op("Reshape", [w_2d, wflat_shape], [w_flat])
        out_contrib = v(f"c_{name}")  # (N*K, C)
        op("Mul", [gathered, w_flat], [out_contrib])
        return out_contrib

    caa = contrib(saa, wx0, wy0, "aa")
    cba = contrib(sba, wx1, wy0, "ba")
    cab = contrib(sab, wx0, wy1, "ab")
    cbb = contrib(sbb, wx1, wy1, "bb")

    s1 = v("s1")
    op("Add", [caa, cba], [s1])
    s2 = v("s2")
    op("Add", [s1, cab], [s2])
    total = v("total")
    op("Add", [s2, cbb], [total])  # (N*K, C)

    # ── Step 9: Reshape + transpose back to (N, C, H_out, W_out) ────────────
    total_nhwc = v("total_nhwc")  # (N, H_out, W_out, C)
    op("Reshape", [total, nhwc_shape_t], [total_nhwc])
    op("Transpose", [total_nhwc], [out], {"perm": [0, 3, 1, 2]})

    # Disconnect the original node so graph.cleanup() removes it.
    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.extend(nodes)


def _replace_gridsample_for_tflite(onnx_path: Path, output_dir: Path) -> Path:
    """Rewrite every GridSample node in *onnx_path* to use TFLite-safe ops.

    ``onnx2tf``'s built-in GridSample handler lowers to ``tf.gather_nd(batch_dims=1)``, which TFLite's ``GatherNd``
    kernel does not support — the kernel silently accepts the model during ``AllocateTensors()`` but produces
    numerically wrong output at inference time.  The ``replace_to_pseudo_operators=["GridSample"]`` pseudo-op path also
    produces wrong logit magnitudes in both FP32 and FP16 (the pseudo-op itself is broken, independent of quantization).
    This function rewrites the ONNX graph *before* calling ``onnx2tf.convert()``, replacing each ``GridSample`` node
    with an equivalent bilinear subgraph that performs four ``Gather(axis=0)`` lookups on a transposed and flattened
    ``(N*(H+2)*(W+2), C)`` image tensor.  ``onnx2tf`` lowers ``Gather(axis=0)`` to TFLite's ``GATHER`` op with no
    ``batch_dims`` — the only TFLite gather path that is unconditionally supported.

    Only ``mode="bilinear"``, ``padding_mode="zeros"``, ``align_corners=0`` nodes are patched — the only variant emitted
    by RF-DETR's exporter.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_dir: Directory where the patched ``.onnx`` is written.  Must
            already exist.

    Returns:
        Path to the patched ``.onnx`` file if any ``GridSample`` nodes were found; *onnx_path* unchanged if the graph
        contains no such nodes.

    Raises:
        ImportError: If ``onnx`` or ``onnx_graphsurgeon`` are not available.
        NotImplementedError: If a ``GridSample`` node has unsupported attrs.
        RuntimeError: If the patched graph fails ONNX shape-inference or
            model validation.
    """
    try:
        import onnx
        import onnx.shape_inference
        import onnx_graphsurgeon as gs
    except ImportError as exc:
        raise ImportError(
            "onnx and onnx_graphsurgeon are required for the GridSample TFLite "
            "patch.  Install with: pip install rfdetr[tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    gs_nodes = [n for n in graph.nodes if n.op == "GridSample"]
    if not gs_nodes:
        logger.debug("No GridSample nodes found; skipping TFLite-safe patch.")
        return onnx_path

    logger.info(
        "Patching %d GridSample node(s) → TFLite-safe Gather(axis=0) subgraph.",
        len(gs_nodes),
    )
    for i, node in enumerate(gs_nodes):
        _replace_single_gridsample(node, graph, index=i)

    graph.cleanup().toposort()

    try:
        patched = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
        onnx.checker.check_model(patched)
    except Exception as exc:
        raise RuntimeError(f"GridSample ONNX patch produced an invalid graph: {exc}") from exc

    out_path = output_dir / (onnx_path.stem + "_gs_patched.onnx")
    onnx.save(patched, str(out_path))
    logger.debug("GridSample-patched ONNX saved to: %s", out_path)
    return out_path


def _check_onnx2tf_available() -> None:
    """Verify that a compatible ``onnx2tf`` package is importable.

    onnx2tf 2.4.0 or later is required — earlier 1.x releases cannot lower the constant ``Expand``, 1-D ``TopK``, and
    rank-3 ``Tile`` ops present in RF-DETR's ONNX graph.

    Raises:
        ImportError: If ``onnx2tf`` cannot be imported or is below 2.4.0.
    """
    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx2tf is not installed. TFLite export requires the tflite extra. "
            "Install it with: pip install rfdetr[tflite]"
        ) from exc

    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version

    import onnx2tf as _onnx2tf_mod
    from packaging.version import Version as _Version

    try:
        installed: str | None = _pkg_version("onnx2tf")
    except _PkgNotFound:
        # Dist-info absent (e.g. editable install without metadata, or a
        # test that injects a fake onnx2tf into sys.modules).
        # Fall back to the __version__ attribute; if missing, skip version check.
        installed = getattr(_onnx2tf_mod, "__version__", None)
        if installed is None:
            return

    assert installed is not None  # guaranteed by the early return above
    if _Version(installed) < _Version("2.4.0"):
        raise ImportError(
            f"onnx2tf {installed} is installed but RF-DETR requires >= 2.4.0. "
            "Earlier 1.x releases cannot lower the Expand, TopK, and Tile ops "
            "in RF-DETR's ONNX graph. Upgrade with: pip install 'onnx2tf>=2.4.0'"
        )


# Serializes the global ``np.load`` monkey-patch below so concurrent conversions in the
# same process cannot clobber each other's patch/restore cycle.
_NUMPY_LOAD_PATCH_LOCK = threading.Lock()


@contextlib.contextmanager
def _numpy_allow_pickle() -> Generator[None, None, None]:
    """Temporarily patch :func:`numpy.load` to set ``allow_pickle=True``.

    ``onnx2tf`` 1.x calls ``np.load()`` on its bundled calibration data without passing ``allow_pickle=True``.  NumPy ≥
    1.16.3 defaults that flag to ``False`` and raises :class:`ValueError` for pickled files.

    This context manager monkey-patches ``np.load`` for the duration of the ``onnx2tf`` conversion and restores the
    original afterwards.  Because ``np.load`` is process-global, the patch/restore cycle is guarded by a module-level
    lock so it is safe when multiple conversions run on separate threads.
    """
    with _NUMPY_LOAD_PATCH_LOCK:
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
def _interpreter_scripts_on_path() -> Generator[None, None, None]:
    """Temporarily put the running interpreter's script directory first on ``PATH``.

    ``onnx2tf`` simplifies the ONNX graph by shelling out to the bare ``onnxsim`` console script.  The lookup relies on
    ``PATH``, so it fails whenever RF-DETR runs under a virtual environment that was never *activated*:
    calling ``<venv>/bin/python`` by absolute path (the norm for subprocess-driven tooling, IDE run
    configurations, and desktop app integrations) leaves the venv's ``bin`` / ``Scripts`` directory
    off ``PATH`` even though ``onnxsim`` is installed there.

    Prepending the interpreter's own script directory makes the subprocess lookup resolve to the ``onnxsim``
    installed by the ``rfdetr[tflite]`` extra, matching the interpreter that is driving the export. Candidate
    directories already present in ``PATH`` are moved to the front and duplicate preferred-directory entries are
    removed.

    Yields:
        ``None``.  ``PATH`` is restored to its previous value on exit, including on exception.

    Note:
        ``os.environ`` is process-global, so this is not safe against concurrent mutation of ``PATH`` from other
        threads.  Restore is unconditional rather than merged, so a concurrent ``PATH`` edit made inside the context
        is lost.
    """
    scripts_dirs: list[str] = []
    if sys.executable:
        executable_path = Path(sys.executable)
        if executable_path.is_absolute():
            scripts_dirs.append(str(executable_path.parent))

    sysconfig_scripts = sysconfig.get_path("scripts")
    if sysconfig_scripts and Path(sysconfig_scripts).is_absolute():
        scripts_dirs.append(sysconfig_scripts)

    scripts_dirs = list(dict.fromkeys(scripts_dirs))
    original_path = os.environ.get("PATH", "")
    path_was_set = "PATH" in os.environ
    entries = original_path.split(os.pathsep) if original_path else []
    remaining = [entry for entry in entries if entry not in scripts_dirs]
    updated_path = os.pathsep.join(scripts_dirs + remaining)

    if not scripts_dirs or updated_path == original_path:
        yield
        return

    logger.debug(f"Prepending {scripts_dirs} to PATH so onnx2tf's onnxsim subprocess resolves.")
    os.environ["PATH"] = updated_path
    try:
        yield
    finally:
        if path_was_set:
            os.environ["PATH"] = original_path
        else:
            os.environ.pop("PATH", None)


@contextlib.contextmanager
def _patch_validation_download(npy_path: str) -> Generator[None, None, None]:
    """Redirect onnx2tf's legacy validation-data download when the hook exists.

    onnx2tf 2.4.0–2.4.3 do not define ``download_test_image_data``. The conditional patch remains for
    compatible 2.x releases that expose it. This is an output-validation hook, not an INT8 calibration path: RF-DETR
    never passes ``output_integer_quantized_tflite``.

    Args:
        npy_path: Path to locally prepared NHWC validation data.
    """

    def _replacement() -> NDArray[np.float32]:
        # Calibration data prepared by _prepare_calibration_data() is always
        # a plain float32 ndarray — never pickled.  allow_pickle=False is
        # intentional here; allow_pickle=True is handled by _numpy_allow_pickle()
        # for onnx2tf's own internal np.load calls.
        return cast(NDArray[np.float32], np.load(npy_path, allow_pickle=False))

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

    Images are loaded, resized to ``(height, width)``, converted to ``float32`` in ``[0, 1]``, and stacked into an NHWC
    array.

    Args:
        image_dir: Directory containing image files (JPEG, PNG, etc.).
        height: Target image height matching the model input.
        width: Target image width matching the model input.
        max_images: Maximum number of images to load.  Files are sorted
            alphabetically and the first *max_images* are used.

    Returns:
        A ``float32`` NumPy array of shape ``(N, height, width, 3)`` with pixel values in ``[0, 1]``.

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
            with Image.open(img_path) as _img:
                img = _img.convert("RGB")
            # Resize with predict()'s convention (bilinear, half-pixel, no antialias) so the INT8
            # calibration ranges come from the same pixel distribution the model sees at inference.
            # PIL's default resize (BICUBIC + adaptive antialias) diverges from that distribution.
            chw = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
            image_array = _bilinear_resize_half_pixel(chw, height, width).transpose(1, 2, 0)
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
        A ``(name, dims)`` tuple where *dims* is the NCHW shape list, e.g. ``("input", [1, 3, 560, 560])``.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is not installed. TFLite export requires the tflite extra. "
            "Install it with: pip install rfdetr[tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    return name, dims


def _prepare_calibration_data(
    onnx_path: Path,
    calibration_data: str | os.PathLike[str] | NDArray[np.float32] | None,
    output_dir: Path,
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> Path:
    """Prepare calibration data as a ``.npy`` file for ``onnx2tf``.

    The returned path points to an NHWC float32 array with pixel values in ``[0, 1]``. Generated or reformatted data is
    saved as ``_rfdetr_calib_data.npy`` in *output_dir*; an existing ``.npy`` path is returned unchanged. The data is
    offered to onnx2tf's conditional legacy validation hook and does not influence the generated ``.tflite`` models.

    Args:
        onnx_path: Path to the source ``.onnx`` file, read for the input tensor's
            NCHW shape (random data generation, and the target resolution when loading images from a directory).
        calibration_data: One of:

            * ``None`` — generate random data.  Adequate for every quantization
              mode, INT8 included: INT8 is dynamic-range, so no calibration is involved.
            * A **directory path** containing JPEG/PNG images — images are
              loaded, resized to the model input resolution, and converted to the correct format automatically.
            * A path to a ``.npy`` file containing an array of shape
              ``(N, H, W, 3)``, dtype float32, values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same constraints.
        output_dir: Directory where a temporary ``.npy`` file may be
            written when *calibration_data* is ``None``, a directory, or an ndarray.
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Ignored for other calibration data formats.

    Returns:
        Path to the ``.npy`` calibration data file.

    Raises:
        FileNotFoundError: If *calibration_data* is a path that does not
            exist, or a directory with no supported images.
    """
    if calibration_data is None:
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


def _rename_precision_outputs(output_dir: Path, model_stem: str) -> None:
    """Rename onnx2tf's hardcoded ``_float32``/``_float16`` outputs to the ``_fp32``/``_fp16`` tokens.

    ``onnx2tf`` always writes ``{model_stem}_float32.tflite`` and ``{model_stem}_float16.tflite`` — that
    naming is internal to the library and not configurable. Rename both (when present) so the RF-DETR
    export surface uses the same ``fp32``/``fp16`` vocabulary as the ONNX/CoreML/ExecuTorch/TensorRT
    backends. Missing files (e.g. onnx2tf skipped one precision on a given run) are silently ignored.

    Args:
        output_dir: Directory onnx2tf wrote its ``.tflite`` outputs to.
        model_stem: Stem of the (possibly GridSample-patched) ONNX file passed to onnx2tf.
    """
    for old_token, new_token in (("float32", "fp32"), ("float16", "fp16")):
        old_path = output_dir / f"{model_stem}_{old_token}.tflite"
        if old_path.is_file():
            old_path.replace(output_dir / f"{model_stem}_{new_token}.tflite")


def _quantize_dynamic_range(saved_model_dir: Path, model_stem: str) -> Path:
    """Build a dynamic-range INT8 TFLite model from the onnx2tf SavedModel.

    Dynamic-range quantization stores weights as INT8 and keeps activations in float, so it needs no calibration data.

    Args:
        saved_model_dir: Directory holding the SavedModel ``onnx2tf`` wrote.
        model_stem: Stem of the source ONNX file, used to name the output.

    Returns:
        Path to the written ``{model_stem}_dynamic_range_quant.tflite`` file.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    out_path = saved_model_dir / f"{model_stem}_dynamic_range_quant.tflite"
    out_path.write_bytes(converter.convert())
    return out_path


def export_tflite(
    onnx_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    quantization: str | None = None,
    calibration_data: str | os.PathLike[str] | NDArray[np.float32] | None = None,
    verbosity: str = "error",
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
    *,
    verbose: bool = False,
) -> Path:
    """Convert an ONNX model to TFLite via ``onnx2tf``.

    Requires ``onnx2tf >= 2.4.0``.  Uses the Python API with a NumPy compatibility shim.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_dir: Directory where TFLite artifacts will be written.
            ``onnx2tf`` creates ``{stem}_float32.tflite`` and ``{stem}_float16.tflite`` (its own hardcoded
            naming); both are renamed to ``{stem}_fp32.tflite`` / ``{stem}_fp16.tflite`` to match the naming
            used by the other export backends.  When ``quantization="int8"`` a ``{stem}_dynamic_range_quant.tflite``
            is additionally written.
        quantization: Quantization mode.

            * ``None`` / ``"fp32"`` / ``"fp16"`` — FP32 + FP16 output
              (``onnx2tf`` always emits both).
            * ``"int8"`` — additionally produce a dynamic-range INT8 model
              (INT8 weights, float activations, ~4x smaller than FP32). Static / full-integer INT8 is not supported.
        calibration_data: Optional data not consumed when building the generated ``.tflite`` models. Accepts:

            * ``None`` — auto-generate random data.
            * A **directory path** containing JPEG/PNG images — images
              are loaded, resized, and converted automatically.
            * A path to a ``.npy`` file — shape ``(N, H, W, 3)``,
              dtype float32, pixel values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same format.

            This argument does not affect the generated ``.tflite`` models. INT8 is dynamic-range, so no representative
            data is involved. ``None``, directories, and arrays produce ``_rfdetr_calib_data.npy`` in *output_dir*;
            existing ``.npy`` paths are reused without writing a copy.
        verbosity: Log verbosity passed to ``onnx2tf``.  One of
            ``"debug"``, ``"info"``, ``"warn"``, ``"error"`` (default).
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Defaults to 100. Ignored for other calibration data formats.
        verbose: When ``True``, stream ``onnx2tf`` per-node progress —
            useful for monitoring long conversions (5–15 min on transformer-based models).  Defaults to ``False``
            (silent).

    Returns:
        Path to the primary artifact.  ``onnx2tf`` always writes both an FP32 and FP16 model (renamed to
        ``{stem}_fp32.tflite`` / ``{stem}_fp16.tflite``) to *output_dir*; ``quantization="int8"`` adds
        ``{stem}_dynamic_range_quant.tflite``. The returned path is the dynamic-range file for ``int8``,
        otherwise the fp32 file.

    Raises:
        FileNotFoundError: If *onnx_path* does not exist or
            *calibration_data* points to a missing file.
        ImportError: If ``onnx2tf`` is not installed.
        ValueError: If *quantization* is not a recognized mode.
        RuntimeError: If the conversion fails.

    Note:
        This function is **not thread-safe**. It globally monkey-patches :func:`numpy.load` (via
        :func:`_numpy_allow_pickle`), ``onnx2tf.download_test_image_data`` when available (via
        :func:`_patch_validation_download`), and ``os.environ["PATH"]`` (via
        :func:`_interpreter_scripts_on_path`) for the duration of the conversion. Concurrent calls from multiple
        threads will interfere with each other. Run conversion in a subprocess if isolation is required.

        ``tf_converter`` backend is forced unconditionally (overriding onnx2tf's 2.x ``flatbuffer_direct`` default) to
        avoid a runtime error in the TFLite TopK_V2 kernel.  ``Erf`` and ``GeLU`` ops are substituted with TFLite-native
        pseudo-operators to avoid a missing TensorFlow Flex delegate at inference time.

        Segmentation models additionally emit a ``masks`` output, decoded by
        :func:`rfdetr.export._tflite.inference._run_inference`.  Verified on the non-plus segmentation variants (Nano,
        Small, Medium, Large, Preview).
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if quantization not in _VALID_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported quantization mode {quantization!r}. "
            f"Choose from: {sorted(q for q in _VALID_QUANTIZATIONS if q is not None)}. "
            "Static / full-integer INT8 is not supported; 'int8' is dynamic-range."
        )

    # Load TensorFlow before the GridSample rewrite below imports onnx: a wrong load order makes
    # TensorFlow's SavedModel restore deadlock (see preload_tensorflow_before_onnx).  RFDETR.export()
    # already calls this before its ONNX export; repeating it here covers direct export_tflite() calls.
    from rfdetr.export._backend import preload_tensorflow_before_onnx

    preload_tensorflow_before_onnx()

    _check_onnx2tf_available()

    # Force-import onnx2tf submodules so that _patch_validation_download()
    # can patch them.  onnx2tf's __init__.py may not import all submodules
    # eagerly in all versions, so we ensure they are in sys.modules before
    # entering the patching context manager.
    import onnx2tf.onnx2tf as _onnx2tf_mod
    import onnx2tf.utils.common_functions as _onnx2tf_common

    del _onnx2tf_mod, _onnx2tf_common  # imported for side-effect only

    output_dir.mkdir(parents=True, exist_ok=True)

    # Rewrite every GridSample node into a TFLite-safe Gather(axis=0) subgraph
    # before invoking onnx2tf.  onnx2tf's default GridSample lowering produces
    # wrong values in TFLite, and its pseudo-op replacement is independently
    # broken.  The patched path becomes the input for everything downstream.
    # Best-effort: skip the rewrite when onnx/onnx_graphsurgeon are not installed
    # (e.g. test environments that only mock onnx2tf).
    try:
        onnx_path = _replace_gridsample_for_tflite(onnx_path, output_dir)
    except ImportError as exc:
        logger.warning(
            "GridSample TFLite patch skipped — onnx/onnx_graphsurgeon not available (%s). "
            "TFLite inference may produce incorrect scores if the model contains GridSample nodes. "
            "Install with: pip install rfdetr[tflite]",
            exc,
        )

    calib_npy_path = _prepare_calibration_data(onnx_path, calibration_data, output_dir, max_images=max_images)

    logger.info(f"Converting ONNX → TFLite (quantization={quantization!r}, verbosity={verbosity!r}): {onnx_path}")

    try:
        # We intentionally do NOT pass custom_input_op_name_np_data_path
        # because that code path in onnx2tf 1.x triggers a tf.tile rank
        # mismatch when processing the DINOv2 backbone with N > 1 samples.
        #
        # output_signaturedefs=True is required because segmentation
        # models produce ONNX node names (e.g.
        # "/segmentation_head/blocks.2/dwconv/Conv/kernel") that contain
        # leading "/" characters which violate the saved_model naming
        # pattern. Enabling signature defs bypasses this restriction.
        with (
            _numpy_allow_pickle(),
            _patch_validation_download(str(calib_npy_path)),
            _interpreter_scripts_on_path(),
        ):
            from onnx2tf import convert

            convert_kwargs: dict[str, Any] = {
                "input_onnx_file_path": str(onnx_path),
                "output_folder_path": str(output_dir),
                "output_signaturedefs": True,
                "non_verbose": not verbose,
                "verbosity": verbosity,
                # Replace Erf / GeLU with TFLite-native pseudo-operators so the
                # produced .tflite does not require the TensorFlow Flex delegate
                # at inference time.  Without this, AllocateTensors() fails with
                # "FlexErf failed to prepare".  GridSample is handled by the
                # ONNX-level rewrite in _replace_gridsample_for_tflite() and
                # therefore intentionally omitted here.
                "replace_to_pseudo_operators": ["Erf", "GeLU"],
            }

            # Prefer tf_converter backend (SavedModel → TFLiteConverter path).
            # onnx2tf 2.x defaults to flatbuffer_direct, which handles
            # GatherElements incorrectly for RF-DETR's deformable attention,
            # producing wrong inference results.  tf_converter routes around this.
            # inspect.signature() cannot probe tflite_backend= at import time on
            # onnx2tf 2.x (the function is wrapped with *args/**kwargs), so we
            # probe at call time via try/except instead.
            # tflite_backend is intentionally NOT in convert_kwargs so that the
            # except-TypeError fallback path calls convert() without it.
            try:
                convert(**convert_kwargs, tflite_backend="tf_converter")
            except TypeError:
                logger.warning(
                    "onnx2tf does not support tflite_backend= — proceeding with "
                    "default backend. If TFLite inference produces wrong results, "
                    "upgrade to onnx2tf>=2.4.0."
                )
                convert(**convert_kwargs)

    except Exception as exc:
        logger.error(f"onnx2tf conversion failed: {exc}")
        raise RuntimeError(f"onnx2tf conversion failed: {exc}") from exc

    # onnx2tf names output files based on the input ONNX stem.
    # If GridSample patching wrote a _gs_patched.onnx, onnx_path.stem
    # reflects that new name and must match the TFLite files onnx2tf created.
    model_stem = onnx_path.stem

    # onnx2tf always writes both "{stem}_float32.tflite" and "{stem}_float16.tflite"
    # (its own hardcoded naming, regardless of the requested quantization mode) — rename
    # to the RF-DETR-wide "fp32"/"fp16" token before either return path below, so both
    # files carry the same vocabulary as the other export backends.
    _rename_precision_outputs(output_dir, model_stem)

    if quantization == "int8":
        # Dynamic-range INT8; static full-integer INT8 is rejected as unsupported.
        primary = _quantize_dynamic_range(output_dir, model_stem)
        logger.info(f"TFLite model exported to: {primary}")
        return primary

    primary = output_dir / f"{model_stem}_fp32.tflite"

    if not primary.is_file():
        # Fallback: look for any .tflite file produced from this specific ONNX stem.
        # Scoped to {stem}_*.tflite to avoid returning a stale artifact from a
        # previous export in a reused output directory (review C2).
        tflite_files = sorted(output_dir.glob(f"{model_stem}_*.tflite"))
        if tflite_files:
            primary = tflite_files[0]
            logger.warning(
                f"Expected TFLite output {output_dir / f'{model_stem}_fp32.tflite'} not found; "
                f"searched for '{model_stem}_*.tflite' in {output_dir} and using {primary.name} instead. "
                "The returned model may have a different dtype (e.g. int8) than the caller expects."
            )
        else:
            raise RuntimeError(
                f"onnx2tf completed but no .tflite file matching '{model_stem}_*.tflite' was found in {output_dir}"
            )

    logger.info(f"TFLite model exported to: {primary}")
    return primary
