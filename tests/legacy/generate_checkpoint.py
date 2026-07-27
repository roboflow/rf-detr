# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Generate an RF-DETR checkpoint for backward-compatibility CI testing.

Intended to be run with a *specific released version* of the ``rfdetr`` package
already installed (the version under test).  Produces a ``.pth`` checkpoint file
in the format that version would save during training, which the
``test_checkpoint_compat.py`` test suite then loads with the *current* code.

Usage::

    pip install rfdetr==1.5.0
    python tests/legacy/generate_checkpoint.py --output checkpoint_v1.5.0.pth --use-pretrained

Arguments
---------
--output : str
    Destination path for the generated ``.pth`` checkpoint file.
--num-classes : int, optional
    Number of foreground classes to embed in the checkpoint (default: 2).
    Ignored when ``--use-pretrained`` is set.
--model : str, optional
    Class name to try first (default: ``RFDETRSmall``).  Falls back
    automatically through ``RFDETRBase`` → ``RFDETR`` if the preferred class
    is unavailable in the installed version.
--use-pretrained : flag, optional
    Build with real (COCO-)pretrained weights instead of a fast random init,
    and store a reference prediction (on a fixed real image) in the
    checkpoint for the compat suite to check against after reload.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# supervision's own "people-walking.jpg" example image (same file served by
# supervision.assets.ImageAssets.PEOPLE_WALKING). Fetched by direct URL rather
# than via supervision.assets: that module's API is not stable across the
# legacy version matrix — e.g. supervision 0.27.0 (pulled transitively by
# rfdetr 1.6.0) only ships VideoAssets, no ImageAssets/download_assets(enum)
# support at all — while a plain URL + MD5 check works identically everywhere.
_REFERENCE_IMAGE_URL = "https://media.roboflow.com/supervision/image-examples/people-walking.jpg"
_REFERENCE_IMAGE_MD5 = "e6bda00b47f2908eeae7df86ef995dcd"


def _get_reference_image_path() -> Path:
    """Download (once) and return the path to the fixed reference test image.

    Caches the file in a stable temp directory and verifies its MD5 hash
    before reuse, so repeated calls (across matrix versions, or local reruns)
    skip re-downloading once the cached copy is confirmed intact.

    Returns:
        Absolute path to the downloaded reference JPEG.

    Raises:
        ValueError: If the downloaded content's MD5 does not match the
            expected hash.
    """
    import requests

    cache_dir = Path(tempfile.gettempdir()) / "rfdetr-legacy-test-assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "people-walking.jpg"

    if cached.is_file() and hashlib.md5(cached.read_bytes()).hexdigest() == _REFERENCE_IMAGE_MD5:
        return cached.resolve()

    response = requests.get(_REFERENCE_IMAGE_URL, timeout=30)
    response.raise_for_status()
    digest = hashlib.md5(response.content).hexdigest()
    if digest != _REFERENCE_IMAGE_MD5:
        raise ValueError(f"Downloaded reference image MD5 mismatch: expected {_REFERENCE_IMAGE_MD5}, got {digest}")
    cached.write_bytes(response.content)
    return cached.resolve()


def _top_detection(detections: Any) -> dict[str, Any]:
    """Extract the highest-confidence detection as a plain, picklable reference.

    ``class_id`` (not ``class_name``) is the identity used for comparison by
    callers: class-name mapping logic has changed across rfdetr releases
    (e.g. issues #988/#1051), while the raw category id predicted by the
    model is what a load-path regression would actually corrupt. Confidence
    and box are included as continuous checks; ``class_name`` is kept for
    human-readable diagnostics only.

    Args:
        detections: A ``supervision.Detections`` instance returned by ``predict()``.

    Returns:
        Dict with ``class_id`` (int), ``class_name`` (str, informational),
        ``confidence`` (float), and ``xyxy`` (list of 4 floats).

    Raises:
        ValueError: If *detections* is empty (no detection above threshold).
    """
    if len(detections) == 0:
        raise ValueError("No detections above threshold on the reference image.")
    # Near-tie flakiness: on a near-equal-confidence pair, argmax can pick a different index
    # across gen vs. reload, tripping the class_id/IoU comparison in the caller. Fails *closed*
    # (a spurious mismatch assertion, not a silent pass), so it is a rare-flake risk, not a
    # correctness bug — not worth a tie-break heuristic given the reference image is fixed.
    top_idx = int(detections.confidence.argmax())
    class_names = detections.data.get("class_name")
    return {
        "class_id": int(detections.class_id[top_idx]),
        "class_name": str(class_names[top_idx]) if class_names is not None else "",
        "confidence": float(detections.confidence[top_idx]),
        "xyxy": [float(v) for v in detections.xyxy[top_idx]],
    }


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


def _build_model(preferred_class: str, device: str, *, num_classes: int | None = None) -> Any:
    """Instantiate an rfdetr model, falling back through available classes.

    Tries *preferred_class* first, then ``RFDETRBase``, then ``RFDETR`` in
    sequence to handle API differences across released versions.

    Args:
        preferred_class: Preferred rfdetr class name (e.g. ``"RFDETRSmall"``).
        device: PyTorch device string (e.g. ``"cpu"``).
        num_classes: When given, builds a random-init model with this many
            foreground classes (``pretrain_weights=None`` — no network access).
            When ``None`` (default), builds with the class's own real
            (COCO-)pretrained weights and its natural class count instead.

    Returns:
        Instantiated rfdetr facade.

    Raises:
        RuntimeError: If none of the candidate classes are importable, or if
            instantiation fails for a reason other than a network/infra error.
        TransientFetchError: If every candidate fails to instantiate and every
            failure looks like a network/infra error (e.g. the pretrained-
            weights download timing out) rather than a real code regression.
    """
    candidates = [preferred_class, "RFDETRBase", "RFDETR"]
    # Deduplicate while preserving order
    seen: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)

    rfdetr_module = sys.modules.get("rfdetr") or __import__("rfdetr")

    build_kwargs: dict[str, Any] = {"device": device}
    if num_classes is not None:
        build_kwargs["pretrain_weights"] = None
        build_kwargs["num_classes"] = num_classes

    errors: list[str] = []
    saw_non_transient_error = False
    for class_name in seen:
        cls = getattr(rfdetr_module, class_name, None)
        if cls is None:
            errors.append(f"{class_name}: not found in rfdetr module")
            saw_non_transient_error = True
            continue
        try:
            model = cls(**build_kwargs)
            return model
        except Exception as exc:
            logger.debug("Failed to instantiate %s during fallback probing: %s", class_name, exc)
            errors.append(f"{class_name}: {exc}")
            if not _is_transient_network_error(exc):
                saw_non_transient_error = True
            continue

    message = "Could not instantiate any rfdetr model class.\n" + "\n".join(f"  {e}" for e in errors)
    if saw_non_transient_error:
        raise RuntimeError(message)
    # Every candidate failed with what looks like a network/infra error (e.g. the
    # pretrained-weights download used by use_pretrained=True) rather than a real
    # code regression — raise a distinguishable type so main() can exit with a
    # "temporary failure" code instead of a generic one.
    raise TransientFetchError(message)


class TransientFetchError(RuntimeError):
    """Raised when model instantiation fails only due to what looks like a network/infra error.

    Distinguishes a real code regression (any other exception) from a likely-transient fetch failure (e.g. the
    pretrained-weights download), so callers like ``main()`` can report and exit differently — mirroring the reference-
    image fetch's existing MD5-guard skip-vs-fail posture.
    """


def _is_transient_network_error(exc: BaseException) -> bool:
    """Best-effort check for whether *exc* looks like a network/infra failure, not a code bug.

    Walks the exception's ``__cause__``/``__context__`` chain (a wrapped download error
    surfaces its original ``requests`` exception this way) looking for
    ``requests.exceptions.RequestException`` or a bare ``TimeoutError``/``ConnectionError``.

    Args:
        exc: The caught exception to classify.

    Returns:
        True if *exc* (or something in its cause/context chain) is a recognized
        network-related exception type.
    """
    import requests

    seen_ids: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        if isinstance(current, requests.exceptions.RequestException | TimeoutError | ConnectionError):
            return True
        current = current.__cause__ or current.__context__
    return False


def generate_checkpoint(
    output_path: str,
    num_classes: int = 2,
    preferred_class: str = "RFDETRSmall",
    use_pretrained: bool = False,
) -> None:
    """Create an rfdetr checkpoint at *output_path*.

    The checkpoint uses the legacy ``{model, args}`` format which all released
    rfdetr versions can produce and which :func:`rfdetr.models.weights.load_pretrain_weights`
    can consume both directly and after PTL-key normalisation.

    By default (``use_pretrained=False``) builds a random-init model with
    ``num_classes`` foreground classes — fast and offline, no network access.
    With ``use_pretrained=True`` (used by the legacy-checkpoint-compat CI
    workflow) builds the model with its real (COCO-)pretrained weights
    instead — ``num_classes`` is then ignored, since overriding it would
    discard the real pretrained detection head — and additionally runs one
    fixed reference-image prediction, storing the top detection in the
    checkpoint under ``reference_prediction`` so the compat suite can assert
    the same detection reproduces after reload with current code.

    Args:
        output_path: File path where the checkpoint is written.
        num_classes: Number of foreground classes for the random-init path.
            Ignored when ``use_pretrained=True``.
        preferred_class: rfdetr facade class to attempt first.
        use_pretrained: Use real pretrained weights and capture a reference
            prediction instead of a fast random init.
    """
    import rfdetr

    installed_version: str = getattr(rfdetr, "__version__", "unknown")
    print(f"[generate_checkpoint] rfdetr {installed_version} installed")

    if use_pretrained:
        model = _build_model(preferred_class, device="cpu")
    else:
        model = _build_model(preferred_class, device="cpu", num_classes=num_classes)
    state_dict = _get_state_dict(model)
    patch_size = _get_patch_size(model)

    # Read the real class count back from the saved head rather than trusting
    # num_classes, which is meaningless on the use_pretrained path.
    class_embed_bias = state_dict.get("class_embed.bias")
    resolved_num_classes = int(class_embed_bias.shape[0]) - 1 if class_embed_bias is not None else num_classes

    # Simulate the args SimpleNamespace that rfdetr training attaches to checkpoints.
    # Keys reflect the union of fields checked by validate_checkpoint_compatibility
    # and class-name extraction in load_pretrain_weights across all versions.
    args = SimpleNamespace(
        class_names=[f"class_{i}" for i in range(resolved_num_classes)],
        patch_size=patch_size,
        num_classes=resolved_num_classes,
        segmentation_head=False,
    )

    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "args": args,
        "epoch": 0,
        # Record provenance so test failure messages can identify the source version.
        "rfdetr_version": installed_version,
    }

    if use_pretrained:
        reference_image = _get_reference_image_path()
        detections = model.predict(str(reference_image), threshold=0.5)
        reference_prediction = _top_detection(detections)
        print(
            f"[generate_checkpoint] reference prediction: {reference_prediction['class_name']!r} "
            f"(class_id={reference_prediction['class_id']}, confidence={reference_prediction['confidence']:.4f})"
        )
        checkpoint["reference_prediction"] = reference_prediction

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
        help="Number of foreground classes (default: 2). Ignored when --use-pretrained is set.",
    )
    parser.add_argument(
        "--model",
        default="RFDETRSmall",
        dest="model",
        help="rfdetr class name to try first (default: RFDETRSmall).",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        dest="use_pretrained",
        help=(
            "Build with real (COCO-)pretrained weights instead of a random init, and capture a "
            "reference prediction for regression checking against the reloaded model."
        ),
    )
    args = parser.parse_args()
    try:
        generate_checkpoint(
            output_path=args.output,
            num_classes=args.num_classes,
            preferred_class=args.model,
            use_pretrained=args.use_pretrained,
        )
    except TransientFetchError as exc:
        # Looks like a network/infra failure (e.g. the pretrained-weights download), not a
        # real code regression — exit with the POSIX "temporary failure" code so CI logs and
        # any future retry logic can tell this apart from a genuine test failure.
        print(f"[generate_checkpoint] TRANSIENT INFRA ERROR (not a code regression): {exc}", file=sys.stderr)
        sys.exit(getattr(os, "EX_TEMPFAIL", 75))


if __name__ == "__main__":
    main()
