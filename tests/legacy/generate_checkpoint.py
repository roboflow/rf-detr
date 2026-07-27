# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Generate a minimal RF-DETR checkpoint for backward-compatibility CI testing.

Intended to be run with a *specific released version* of the ``rfdetr`` package
already installed (the version under test).  Produces a ``.pth`` checkpoint file
in the format that version would save during training, which the
``test_checkpoint_compat.py`` test suite then loads with the *current* code.

Usage::

    pip install rfdetr==1.5.0
    python tests/legacy/generate_checkpoint.py --output checkpoint_v1.5.0.pth

Arguments
---------
--output : str
    Destination path for the generated ``.pth`` checkpoint file.
--num-classes : int, optional
    Number of foreground classes to embed in the checkpoint (default: 2).
--model : str, optional
    Class name to try first (default: ``RFDETRSmall``).  Falls back
    automatically through ``RFDETRBase`` → ``RFDETR`` if the preferred class
    is unavailable in the installed version.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_OutIndices = list[int] | tuple[int, ...]


def _get_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """Extract the bare model state-dict from an rfdetr facade instance.

    Tries the current layout (``model.model.model.state_dict()``) then the
    legacy single-wrapper layout (``model.model.state_dict()``).

    Args:
        model: An rfdetr facade instance (e.g. ``RFDETRSmall``).

    Returns:
        State dictionary of the underlying ``nn.Module``.

    Raises:
        RuntimeError: If neither layout yields a non-empty state dict.
    """
    # Current layout: RFDETR → .model (Model) → .model (nn.Module)
    sd: dict[str, torch.Tensor] | None = None
    with contextlib.suppress(AttributeError):
        sd = dict(model.model.model.state_dict())
    if sd:
        return sd

    # Legacy single-wrapper layout: RFDETR → .model (nn.Module directly)
    with contextlib.suppress(AttributeError):
        sd = dict(model.model.state_dict())
    if sd:
        return sd

    raise RuntimeError(
        "Cannot extract state_dict from the rfdetr model instance. "
        "Neither model.model.model nor model.model resolved to a non-empty Module."
    )


def _get_patch_size(model: Any) -> int:
    """Extract patch_size from an rfdetr facade instance.

    Args:
        model: An rfdetr facade instance.

    Returns:
        Patch size (default 16 when not resolvable).
    """
    for attr_path in ("model_config.patch_size", "config.patch_size"):
        obj: Any = model
        with contextlib.suppress(AttributeError):
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return int(obj)
    return 16


def _find_pruneable_heads_and_indices(
    heads: set[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """Return pruneable heads and the flattened index mask.

    Mirrors the helper that older RF-DETR releases imported from
    ``transformers.pytorch_utils`` before transformers v5 removed that public
    export.

    Args:
        heads: Attention head indices requested for pruning.
        n_heads: Number of heads currently present in the layer.
        head_size: Width of each attention head.
        already_pruned_heads: Heads removed by previous pruning calls.

    Returns:
        Adjusted pruneable head indices and a tensor selecting retained rows.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head -= sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _align_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: _OutIndices | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    """Fill missing backbone output names or indices from stage names.

    Args:
        out_features: Names of backbone stages to emit, or ``None``.
        out_indices: Indices of backbone stages to emit, or ``None``.
        stage_names: Ordered backbone stage names.

    Returns:
        Normalized output feature names and indices.
    """
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    assert out_features is not None
    assert out_indices is not None
    return out_features, list(out_indices)


def _get_aligned_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: _OutIndices | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]:
    """Align backbone output feature names and indices for old RF-DETR.

    Mirrors the transformers helper that RF-DETR 1.4 imports from
    ``transformers.utils.backbone_utils``.  Newer transformers releases removed
    that public export.

    Args:
        out_features: Names of backbone stages to emit, or ``None``.
        out_indices: Indices of backbone stages to emit, or ``None``.
        stage_names: Ordered backbone stage names.

    Returns:
        Normalized output feature names and indices.

    Raises:
        ValueError: If both arguments are set and do not describe the same
            stages.
    """
    out_features, out_indices = _align_output_features_output_indices(out_features, out_indices, stage_names)
    if len(out_features) != len(out_indices):
        raise ValueError("out_features and out_indices must have the same length")
    if out_features != [stage_names[idx] for idx in out_indices]:
        raise ValueError("out_features and out_indices must describe the same stages")
    return out_features, out_indices


def _init_backbone_shim(self: Any, config: Any) -> None:
    """Compatibility shim for ``BackboneMixin._init_backbone`` removed in transformers v5.

    RF-DETR 1.4's ``WindowedDinov2WithRegistersBackbone.__init__`` calls
    ``super()._init_backbone(config)`` explicitly.  Transformers v5 removed that
    method from ``BackboneMixin`` (replacing it with cooperative-MRO
    ``__init__`` → ``_init_transformers_backbone``).  This shim reinstates the
    v4 behaviour: populate ``stage_names``, ``_out_features``, ``_out_indices``,
    and ``num_features`` from *config*.

    Args:
        config: Backbone configuration object.  Expected to carry
            ``stage_names``, ``out_features``, and ``out_indices`` attributes.
    """
    self.stage_names = list(getattr(config, "stage_names", None) or [])
    out_features = getattr(config, "out_features", None)
    out_indices = getattr(config, "out_indices", None)
    aligned_features, aligned_indices = _get_aligned_output_features_output_indices(
        out_features=list(out_features) if out_features is not None else None,
        out_indices=list(out_indices) if out_indices is not None else None,
        stage_names=self.stage_names,
    )
    self._out_features = aligned_features
    self._out_indices = aligned_indices
    # num_features is immediately overridden by the subclass; set sentinel here.
    if not hasattr(self, "num_features"):
        self.num_features = None


def _install_transformers_compat() -> None:
    """Install transformers compatibility shims required by old RF-DETR.

    RF-DETR 1.4 imports helpers from transformer modules where newer transformers releases no longer expose them, and
    calls ``BackboneMixin._init_backbone`` which was removed in transformers v5. These shims are intentionally scoped to
    this legacy generator process.
    """
    pytorch_utils = importlib.import_module("transformers.pytorch_utils")
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing `Backbone.*` from `utils/backbone_utils.py` is deprecated.*",
            category=FutureWarning,
        )
        backbone_utils = importlib.import_module("transformers.utils.backbone_utils")
    if not hasattr(backbone_utils, "get_aligned_output_features_output_indices"):
        backbone_utils.get_aligned_output_features_output_indices = _get_aligned_output_features_output_indices

    # RF-DETR 1.4's WindowedDinov2WithRegistersBackbone calls super()._init_backbone(config)
    # explicitly.  Transformers v5 removed this method from BackboneMixin.
    backbone_module = importlib.import_module("transformers.backbone_utils")
    backbone_mixin_cls = getattr(backbone_module, "BackboneMixin", None)
    if backbone_mixin_cls is not None and not hasattr(backbone_mixin_cls, "_init_backbone"):
        backbone_mixin_cls._init_backbone = _init_backbone_shim


def _build_model(preferred_class: str, num_classes: int, device: str) -> Any:
    """Instantiate an rfdetr model, falling back through available classes.

    Tries *preferred_class* first, then ``RFDETRBase``, then ``RFDETR`` in
    sequence to handle API differences across released versions.

    Args:
        preferred_class: Preferred rfdetr class name (e.g. ``"RFDETRSmall"``).
        num_classes: Number of foreground classes.
        device: PyTorch device string (e.g. ``"cpu"``).

    Returns:
        Instantiated rfdetr facade.

    Raises:
        RuntimeError: If none of the candidate classes are importable.
    """
    candidates = [preferred_class, "RFDETRBase", "RFDETR"]
    # Deduplicate while preserving order
    seen: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)

    rfdetr_module = sys.modules.get("rfdetr") or __import__("rfdetr")

    errors: list[str] = []
    for class_name in seen:
        cls = getattr(rfdetr_module, class_name, None)
        if cls is None:
            errors.append(f"{class_name}: not found in rfdetr module")
            continue
        try:
            model = cls(pretrain_weights=None, num_classes=num_classes, device=device)
            return model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
            continue

    raise RuntimeError("Could not instantiate any rfdetr model class.\n" + "\n".join(f"  {e}" for e in errors))


def generate_checkpoint(
    output_path: str,
    num_classes: int = 2,
    preferred_class: str = "RFDETRSmall",
) -> None:
    """Create a minimal rfdetr checkpoint at *output_path*.

    The checkpoint uses the legacy ``{model, args}`` format which all released
    rfdetr versions can produce and which :func:`rfdetr.models.weights.load_pretrain_weights`
    can consume both directly and after PTL-key normalisation.

    Args:
        output_path: File path where the checkpoint is written.
        num_classes: Number of foreground classes to store in the checkpoint.
        preferred_class: rfdetr facade class to attempt first.
    """
    _install_transformers_compat()

    import rfdetr

    installed_version: str = getattr(rfdetr, "__version__", "unknown")
    print(f"[generate_checkpoint] rfdetr {installed_version} installed")

    model = _build_model(preferred_class, num_classes, device="cpu")
    state_dict = _get_state_dict(model)
    patch_size = _get_patch_size(model)

    # Simulate the args SimpleNamespace that rfdetr training attaches to checkpoints.
    # Keys reflect the union of fields checked by validate_checkpoint_compatibility
    # and class-name extraction in load_pretrain_weights across all versions.
    args = SimpleNamespace(
        class_names=[f"class_{i}" for i in range(num_classes)],
        patch_size=patch_size,
        num_classes=num_classes,
        segmentation_head=False,
    )

    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "args": args,
        "epoch": 0,
        # Record provenance so test failure messages can identify the source version.
        "rfdetr_version": installed_version,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(f"[generate_checkpoint] saved checkpoint → {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    """Entry-point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Generate a minimal RF-DETR checkpoint for backward-compat CI testing.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the generated .pth checkpoint file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        dest="num_classes",
        help="Number of foreground classes (default: 2).",
    )
    parser.add_argument(
        "--model",
        default="RFDETRSmall",
        dest="model",
        help="rfdetr class name to try first (default: RFDETRSmall).",
    )
    args = parser.parse_args()
    generate_checkpoint(
        output_path=args.output,
        num_classes=args.num_classes,
        preferred_class=args.model,
    )


if __name__ == "__main__":
    main()
