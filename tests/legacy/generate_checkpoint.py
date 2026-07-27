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
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


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
