# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint conversion and serializability helpers for the PTL training stack.

Provides :func:`convert_legacy_checkpoint` to convert RF-DETR ``*.pth`` checkpoints (produced by the pre-PTL
``engine.py`` training loop) into the ``*.ckpt`` format expected by ``pytorch_lightning.Trainer``.

Auto-detection of legacy format at load time is handled by
:meth:`rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint`.

:func:`_loads_weights_only` and :func:`_weights_only_loadable` keep what a checkpoint records about its model
readable by a weights-only :func:`torch.load`. They live here rather than next to either writer because both
:meth:`rfdetr.training.callbacks.best_model.BestModelCallback._model_description` (the ``.pth`` files) and
:meth:`rfdetr.training.module_model.RFDETRModelModule.on_save_checkpoint` (the ``.ckpt`` files) go through them.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["convert_legacy_checkpoint"]


def _loads_weights_only(value: object) -> bool:
    """Return whether ``value`` survives ``torch.save`` followed by ``torch.load(weights_only=True)``.

    Examples:
        >>> from pathlib import PurePosixPath
        >>> _loads_weights_only({"epochs": 3}), _loads_weights_only(PurePosixPath("exp"))
        (True, False)
    """
    buffer = io.BytesIO()
    try:
        torch.save(value, buffer)
        buffer.seek(0)
        torch.load(buffer, weights_only=True)
    # Any exception at all means "not loadable", which is the only thing the caller asks about. Pickling and
    # weights-only unpickling raise unrelated types (PicklingError, AttributeError, TypeError, UnpicklingError)
    # depending on the object, and an object whose own __reduce__/__getstate__ is broken raises whatever that code
    # raises — the verdict is the same for all of them, and none is worth ending a training run over.
    except Exception:
        return False
    return True


def _weights_only_loadable(fields: dict[str, Any], name: str) -> dict[str, Any]:
    """Return ``fields`` with each value a weights-only ``torch.load`` cannot read replaced by its ``repr``.

    ``Trainer.fit(ckpt_path=...)`` loads with ``weights_only=None``, which means ``True`` on torch 2.6 and newer, and
    :func:`rfdetr.utilities.io._safe_torch_load` tries the same mode first for a ``.pth``. One such value in a
    checkpoint, for example a ``Path`` in ``TrainConfig.notes`` or an optimizer callable that ``TrainConfig`` could
    not turn into a dotted path, would make the file impossible to resume from or reload without full pickling.

    Args:
        fields: Serialized config, keyed by field name.
        name: Checkpoint key the config is stored under, used in the warning.

    Returns:
        ``fields`` itself when it all loads weights-only, otherwise a copy with each offending value as its ``repr``,
        or as ``<unrepresentable <type name>>`` when ``repr`` itself raises.

    Examples:
        >>> import warnings
        >>> from pathlib import PurePosixPath
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     _weights_only_loadable({"epochs": 3, "notes": PurePosixPath("exp")}, "args")
        {'epochs': 3, 'notes': "PurePosixPath('exp')"}
    """
    if _loads_weights_only(fields):
        return fields
    loadable: dict[str, Any] = {}
    for key, value in fields.items():
        if not _loads_weights_only(value):
            warnings.warn(
                f"checkpoint[{name!r}][{key!r}] holds a {type(value).__name__}, which a weights-only torch.load "
                "cannot read, so RF-DETR checkpoints store its repr() to stay loadable.",
                UserWarning,
                stacklevel=2,
            )
            try:
                value = repr(value)
            # The repr is the last resort for a value no checkpoint can hold as it is, so it must not be what ends
            # the run: an object whose own __repr__ raises would otherwise abort the write at the end of an epoch.
            except Exception:
                value = f"<unrepresentable {type(value).__name__}>"
        loadable[key] = value
    return loadable


def convert_legacy_checkpoint(old_path: str, new_path: str) -> None:
    """Convert a legacy RF-DETR ``.pth`` checkpoint to PTL ``.ckpt`` format.

    Loads a checkpoint saved by the pre-PTL ``engine.py`` training loop and rewrites it in the structure expected by
    ``pytorch_lightning.Trainer``:

    * ``state_dict`` keys are prefixed with ``"model."`` to match the
      attribute path inside :class:`~rfdetr.training.module_model.RFDETRModelModule`.
    * ``args`` (``argparse.Namespace`` or ``dict``) is normalised to a plain
      ``dict`` and stored as ``hyper_parameters``.
    * ``legacy_checkpoint_format: True`` is written so
      :meth:`~rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint` can distinguish converted files from
      native PTL checkpoints.
    * If an ``ema_model`` key is present it is preserved verbatim under
      ``legacy_ema_state_dict`` for optional EMA weight restoration.

    Args:
        old_path: Path to the source legacy ``.pth`` checkpoint.
        new_path: Destination path for the converted ``.ckpt`` file.
    """
    # trust=True: this function converts internally-produced legacy .pth files;
    # allow pickle fallback if safe deserialization fails due to non-tensor/custom objects.
    from rfdetr.utilities.io import _safe_torch_load

    old: dict[str, Any] = _safe_torch_load(old_path, trust=True)

    if "model" not in old:
        raise ValueError(
            f"The checkpoint at {old_path!r} does not contain a 'model' key."
            " Only RF-DETR legacy .pth files produced by engine.py are supported."
        )

    args_obj = old.get("args")
    if isinstance(args_obj, dict):
        hyper_parameters: dict[str, Any] = args_obj
    elif args_obj is None:
        hyper_parameters = {}
    else:
        try:
            hyper_parameters = vars(args_obj)
        except TypeError:
            logger.warning(
                "Cannot extract hyper_parameters from args of type %s; storing empty dict.",
                type(args_obj).__name__,
            )
            hyper_parameters = {}

    new: dict[str, Any] = {
        "state_dict": {"model." + k: v for k, v in old["model"].items()},
        "epoch": old.get("epoch", 0),
        "global_step": 0,
        "hyper_parameters": hyper_parameters,
        "legacy_checkpoint_format": True,
    }

    if "ema_model" in old:
        # Preserve EMA weights under a dedicated key.  Callback-specific state
        # keys are framework-internal and must not be written here.
        new["legacy_ema_state_dict"] = old["ema_model"]

    torch.save(new, new_path)
