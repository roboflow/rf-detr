# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Confidence thresholds for prediction: one number for every class, or one per class with :class:`ClassThresholds`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from rfdetr._prediction.core import PredictionContext

#: The scalars a threshold may be. A single threshold and every per-class threshold must be one of them, or a 0-d NumPy
#: array holding one (see ``_is_threshold_number``), so a bool, for instance, counts as a number in both forms and a
#: string counts in neither.
_THRESHOLD_NUMBER_TYPES = (int, float, np.integer, np.floating, np.bool_, torch.Tensor)

#: How the single-threshold refusal tells a caller where per-class values go.
_PER_CLASS_HINT = (
    "pass per-class thresholds as ClassThresholds({'person': 0.35}, default=0.5), "
    "or as ClassThresholds([...]) with one threshold per entry of class_names"
)


def _is_threshold_number(value: object) -> TypeGuard[float | np.generic | np.ndarray[Any, Any] | torch.Tensor]:
    """Report whether a value is a number a threshold may be, a 0-d NumPy array of one included.

    Args:
        value: A single threshold, or one per-class threshold.

    Returns:
        ``True`` for an instance of ``_THRESHOLD_NUMBER_TYPES``, or a 0-d array whose element is one.

    Examples:
        >>> _is_threshold_number(np.array(0.5)), _is_threshold_number(np.array("0.5")), _is_threshold_number("0.5")
        (True, False, False)
    """
    if isinstance(value, np.ndarray):
        # A 0-d array is NumPy's boxed scalar. Its element decides, so that an array of a string is refused as the
        # string is, instead of being parsed by `float()`.
        return value.ndim == 0 and isinstance(value[()], _THRESHOLD_NUMBER_TYPES)
    return isinstance(value, _THRESHOLD_NUMBER_TYPES)


def _as_float(value: float | np.generic | np.ndarray[Any, Any] | torch.Tensor) -> float:
    """Read one threshold number as a Python float.

    Args:
        value: A number accepted as a threshold, see ``_THRESHOLD_NUMBER_TYPES``.

    Returns:
        The value as a float. A tensor is detached first, so that one that requires grad does not warn.

    Examples:
        >>> _as_float(np.float32(0.5)), _as_float(True), _as_float(torch.tensor(0.25, requires_grad=True))
        (0.5, 1.0, 0.25)
    """
    return float(value.detach() if isinstance(value, torch.Tensor) else value)


def _detach_tensors(threshold: Any) -> Any:
    """Copy every tensor of a threshold, at any depth of nested lists and tuples, to the CPU without its autograd graph.

    NumPy reads a tensor through ``.numpy()``, which refuses one that requires grad or sits on CUDA, so the shape check
    in :meth:`ClassThresholds._read_values` would fail with PyTorch's error instead of the threshold's own. Copying each
    tensor once here also spares ``_as_float`` a second copy.

    Args:
        threshold: Per-class thresholds, or a part of them.

    Returns:
        The same structure, with lists and tuples rebuilt as lists and each tensor detached on the CPU.

    Examples:
        >>> _detach_tensors([[torch.tensor(0.5, requires_grad=True)], 0.25])
        [[tensor(0.5000)], 0.25]
    """
    if isinstance(threshold, torch.Tensor):
        return threshold.detach().cpu()
    if isinstance(threshold, (list, tuple)):
        return [_detach_tensors(entry) for entry in threshold]
    return threshold


def _threshold_entries(threshold: Sequence[Any] | np.ndarray[Any, Any] | torch.Tensor) -> list[Any]:
    """List the entries of a one-dimensional threshold sequence, reading a tensor or array in one call.

    Iterating a tensor and calling ``float()`` on each element costs one device-to-host copy per class on CUDA and warns
    for a tensor that requires grad. ``tolist()`` copies it once, and ``detach`` spares the warning.

    Args:
        threshold: A tuple, list, one-dimensional array or one-dimensional tensor.

    Returns:
        The entries as a list. A tensor or array gives Python numbers; any other sequence keeps its own entries.

    Examples:
        >>> _threshold_entries(torch.tensor([1, 2], requires_grad=False))
        [1, 2]
        >>> _threshold_entries(np.array([0.5, 0.25]))
        [0.5, 0.25]
        >>> _threshold_entries((0.1, 0.2))
        [0.1, 0.2]
    """
    if isinstance(threshold, torch.Tensor):
        return threshold.detach().cpu().tolist()
    if isinstance(threshold, np.ndarray):
        return cast("list[Any]", threshold.tolist())
    return list(threshold)


def _threshold_number(value: object, name: str, expected: str = "a number") -> float:
    """Read one threshold as a float, refusing what ``scores > threshold`` could not apply.

    Args:
        value: The threshold to read.
        name: How the threshold is called in an error message, such as ``"threshold"`` or ``"default"``.
        expected: What the caller should have passed, for the ``TypeError`` message.

    Returns:
        The threshold as a float.

    Raises:
        TypeError: If ``value`` is not a number, or is a tensor with dimensions.
        ValueError: If ``value`` is NaN.

    Examples:
        >>> _threshold_number(np.float32(0.25), "threshold")
        0.25
        >>> _threshold_number(float("nan"), "threshold")
        Traceback (most recent call last):
        ...
        ValueError: threshold is NaN, so no detection can score above it; pass a number.
    """
    if not _is_threshold_number(value) or (isinstance(value, torch.Tensor) and value.ndim != 0):
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}.")
    number = _as_float(value)
    # No score is above NaN, so `scores > nan` would silently empty every result.
    if math.isnan(number):
        raise ValueError(f"{name} is NaN, so no detection can score above it; pass a number.")
    return number


@dataclass(frozen=True, repr=False)
class ClassThresholds:
    """Confidence thresholds for individual classes, passed to ``predict(threshold=...)``.

    Name the classes that need their own bar and let every other class take ``default``, or list one threshold per
    entry of the model's ``class_names``, in that order. A detection is kept when its score is strictly above the
    threshold of its own class, and ``float("inf")`` drops a class. Class names and the number of entries are checked
    against the model when ``predict`` runs, so one object serves every model with the same classes.

    Args:
        values: Class name to threshold, or one threshold per entry of ``class_names`` as a sequence, a one-dimensional
            array or a one-dimensional tensor.
        default: The threshold of every class that a mapping does not name. A sequence names every class, so it does
            not use ``default``.

    Raises:
        TypeError: If ``values`` is neither a mapping nor a sequence, a mapping key is not a string, or a threshold is
            not a number.
        ValueError: If a sequence is ragged or not one-dimensional, or a threshold is NaN.

    Examples:
        >>> thresholds = ClassThresholds({"person": 0.35, "traffic light": 0.6}, default=0.5)
        >>> thresholds.for_classes(["person", "car", "traffic light"])
        [0.35, 0.5, 0.6]
        >>> ClassThresholds([0.4, 0.55]).for_classes(["cat", "dog"])
        [0.4, 0.55]
    """

    values: Mapping[str, float] | Sequence[float] | np.ndarray[Any, Any] | torch.Tensor
    default: float = 0.5

    def __post_init__(self) -> None:
        """Check every threshold where it is written, not on each batch of every ``predict`` call."""
        default = _threshold_number(self.default, "ClassThresholds default")
        values: Mapping[str, float] | tuple[float, ...]
        if isinstance(self.values, Mapping):
            by_name: dict[str, float] = {}
            for name, value in self.values.items():
                if not isinstance(name, str):
                    raise TypeError(f"ClassThresholds keys must be class names, got a {type(name).__name__} key.")
                by_name[name] = _threshold_number(value, f"ClassThresholds threshold for {name!r}")
            values = MappingProxyType(by_name)
        else:
            values = self._read_values(self.values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "default", default)

    def __repr__(self) -> str:
        """Show the thresholds as they were written, in a form that rebuilds the object.

        Examples:
            >>> ClassThresholds({"person": 0.35}, default=0.4)
            ClassThresholds({'person': 0.35}, default=0.4)
            >>> ClassThresholds((0.25, 0.75))
            ClassThresholds([0.25, 0.75])
        """
        if isinstance(self.values, Mapping):
            return f"ClassThresholds({dict(self.values)!r}, default={self.default!r})"
        return f"ClassThresholds({list(cast('tuple[float, ...]', self.values))!r})"

    def for_classes(self, class_names: Sequence[str]) -> list[float]:
        """Lay the thresholds out with one per entry of ``class_names``, in that order.

        Args:
            class_names: The model's class names, in class-id order.

        Returns:
            One threshold per class name.

        Raises:
            ValueError: If the model has no classes, a mapping names a class that ``class_names`` does not hold, or a
                sequence has a different number of entries than ``class_names``.

        Examples:
            >>> ClassThresholds({"dgo": 0.4}).for_classes(["cat", "dog"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ClassThresholds names ['dgo'], which the model does not have; its class_names are ['cat', ...
        """
        if not class_names:
            # `min` of no thresholds, which the mask pre-filter needs, would raise after the forward pass.
            raise ValueError("The model has no classes to give a threshold to; pass a single number as threshold.")
        if isinstance(self.values, Mapping):
            known = set(class_names)
            unknown = [name for name in self.values if name not in known]
            if unknown:
                raise ValueError(
                    f"ClassThresholds names {unknown}, which the model does not have; "
                    f"its class_names are {list(class_names)}."
                )
            return [self.values.get(name, self.default) for name in class_names]
        values = cast("tuple[float, ...]", self.values)
        if len(values) != len(class_names):
            raise ValueError(
                f"ClassThresholds has {len(values)} entries but the model has {len(class_names)} classes; "
                "pass one threshold per entry of class_names, or name the classes in a mapping."
            )
        return list(values)

    def _table(self, context: PredictionContext) -> tuple[float, torch.Tensor]:
        """Fit the thresholds to a model's classes and lay them out by raw class id.

        An export's metadata is used as written, so its class-id map is checked here and not assumed.

        Args:
            context: The prediction rules whose ``class_names`` and class-id map the thresholds must fit.

        Returns:
            The lowest threshold, which is the loosest bound and so the score floor of the pre-filter in
            ``postprocess``, and the table from :meth:`_by_class_id` that a detection's label indexes.

        Raises:
            ValueError: As :meth:`for_classes` and :meth:`_check_layout` do.

        Examples:
            >>> from types import SimpleNamespace
            >>> context = SimpleNamespace(class_names=["a", "b"], class_id_to_name={1: "a", 2: "b"}, num_classes=2)
            >>> floor, table = ClassThresholds({"b": 0.75}, default=0.25)._table(context)
            >>> floor, table.tolist()
            (0.25, [inf, 0.25, 0.75, inf])
        """
        class_thresholds = self.for_classes(context.class_names)
        self._check_layout(context.class_names, context.class_id_to_name, context.num_classes)
        return min(class_thresholds), self._by_class_id(class_thresholds, context.class_id_to_name)

    @staticmethod
    def _read_values(values: Any) -> tuple[float, ...]:
        """Read per-class thresholds given in ``class_names`` order as floats.

        Args:
            values: A sequence, one-dimensional array or one-dimensional tensor of numbers.

        Returns:
            The thresholds as floats, in the given order.

        Raises:
            TypeError: If ``values`` has no dimensions, such as a single number or a string, or an entry is not a
                number.
            ValueError: If ``values`` is ragged or has more than one dimension, or an entry is NaN.

        Examples:
            >>> ClassThresholds._read_values(torch.tensor([0.25, 0.5], requires_grad=True))
            (0.25, 0.5)
            >>> ClassThresholds._read_values([[0.5, 0.5]])
            Traceback (most recent call last):
            ...
            ValueError: ClassThresholds values must be one-dimensional, got 2-dimensional input.
        """
        values = _detach_tensors(values)
        try:
            dimensions = np.ndim(values)
        except ValueError:
            # NumPy cannot give a ragged sequence such as [[0.5, 0.5], [0.5]] one shape and says so in its own words.
            raise ValueError("ClassThresholds values must be one-dimensional, got a ragged sequence.") from None
        if dimensions == 0:
            # A string, a set or a single number also has no dimensions.
            raise TypeError(
                "ClassThresholds takes a mapping of class name to threshold, or a sequence with one threshold per "
                f"entry of class_names, got {type(values).__name__}; pass a single number to predict(threshold=...)."
            )
        if dimensions != 1:
            raise ValueError(f"ClassThresholds values must be one-dimensional, got {dimensions}-dimensional input.")
        entries = _threshold_entries(values)
        for index, entry in enumerate(entries):
            # `float()` would take "0.5" and refuse None with its own words; the single threshold takes neither.
            if not _is_threshold_number(entry):
                raise TypeError(
                    f"ClassThresholds values must be numbers, got a {type(entry).__name__} at index {index}."
                )
        numbers = tuple(_as_float(entry) for entry in entries)
        if any(math.isnan(number) for number in numbers):
            # No score is above NaN, so the class would silently lose every detection, and `min` would carry the NaN
            # into the segmentation pre-filter and empty the whole result. `inf` is the way to say "drop this class".
            raise ValueError("ClassThresholds values hold a NaN; use float('inf') to drop a class.")
        return numbers

    @staticmethod
    def _by_class_id(class_thresholds: list[float], class_id_to_name: dict[int, str]) -> torch.Tensor:
        """Lay per-class thresholds out by raw class id, so that a detection's label indexes its own threshold.

        ``class_names`` lists the classes in ascending class-id order for every layout (plain, sparse COCO, legacy
        background-first keypoints, and the ``class_id_to_name`` of an export's metadata), so the *i*-th smallest id
        in ``class_id_to_name`` belongs to ``class_thresholds[i]``. Positions are taken from the sorted ids rather than
        from the names, which two classes may share. Call :meth:`_check_layout` first: the layout above is checked
        there, not here, and an id that is negative or far beyond the model's logit slots would index or size the
        table wrongly.

        Args:
            class_thresholds: One threshold per entry of ``class_names``.
            class_id_to_name: Every raw class id the model emits, mapped to its name.

        Returns:
            A float64 CPU tensor holding ``inf`` at every id missing from ``class_id_to_name`` (the no-object slot, gaps
            in the COCO category ids) and in its last entry. No score exceeds ``inf``, so such a detection is dropped.
            Cast it to the score dtype before moving it to the device (MPS has no float64) and index it with
            ``labels.clamp(min=-1, max=table.numel() - 1)``: negative and out-of-table ids then read the last entry.

        Examples:
            >>> ClassThresholds._by_class_id([0.25, 0.75], {3: "b", 1: "a"}).tolist()
            [inf, 0.25, inf, 0.75, inf]
        """
        table = torch.full((max(class_id_to_name, default=-1) + 2,), float("inf"), dtype=torch.float64, device="cpu")
        for index, class_id in enumerate(sorted(class_id_to_name)):
            table[class_id] = class_thresholds[index]
        return table

    @staticmethod
    def _check_layout(class_names: list[str], class_id_to_name: dict[int, str], num_logit_slots: int) -> None:
        """Refuse a class-id map that per-class thresholds cannot be laid out against.

        Per-class thresholds are matched to raw class ids by position (see :meth:`_by_class_id`), which is only right
        when the *i*-th smallest id of ``class_id_to_name`` is named ``class_names[i]``. A native model always
        satisfies that. An export's metadata does not have to: ``metadata=`` mappings and hand-edited sidecars are used
        as written. A map that has fewer ids than ``class_names`` is accepted, because a model may have fewer active
        slots than names (legacy background-first keypoints). Pretrained COCO is accepted too: its sparse ids 1..90
        name 80 classes.

        Args:
            class_names: Class names in ascending class-id order, one per threshold.
            class_id_to_name: Every raw class id the model emits, mapped to its name.
            num_logit_slots: Number of class logits of the model; the largest raw class id a model can emit.

        Raises:
            ValueError: If the map has more ids than ``class_names`` has entries, if the names in ascending id order are
                not the leading entries of ``class_names``, or if an id is negative or exceeds ``num_logit_slots``.

        Examples:
            >>> ClassThresholds._check_layout(["a", "b", "c"], {5: "b", 1: "a"}, 5)
            >>> ClassThresholds._check_layout(["a", "b"], {0: "b", 1: "a"}, 2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: class_id_to_name names ['b', 'a'] in ascending class-id order, but class_names starts ['a', ...
            >>> ClassThresholds._check_layout(["a", "b"], {0: "a", 1: "b", 2: "c"}, 2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: class_id_to_name has 3 class ids but class_names has 2 entries. ...
            >>> ClassThresholds._check_layout(["a", "b"], {0: "a", 10**9: "b"}, 2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: class_id_to_name holds class id 1000000000, outside 0..2 ...
        """
        hint = (
            "Per-class thresholds are matched to class ids by position; pass a single float, or correct the metadata."
        )
        class_ids = sorted(class_id_to_name)
        if len(class_ids) > len(class_names):
            raise ValueError(
                f"class_id_to_name has {len(class_ids)} class ids but class_names has {len(class_names)} entries. "
                f"{hint}"
            )
        outside = [class_id for class_id in class_ids if not 0 <= class_id <= num_logit_slots]
        if outside:
            raise ValueError(
                f"class_id_to_name holds class id {outside[0]}, outside 0..{num_logit_slots} (the model's class "
                f"logits). {hint}"
            )
        mapped_names = [class_id_to_name[class_id] for class_id in class_ids]
        if mapped_names != class_names[: len(class_ids)]:
            raise ValueError(
                f"class_id_to_name names {mapped_names} in ascending class-id order, "
                f"but class_names starts {class_names[: len(class_ids)]}. {hint}"
            )


def _class_thresholds(
    threshold: float | ClassThresholds, context: PredictionContext
) -> tuple[float, torch.Tensor | None]:
    """Check the ``threshold`` argument of ``predict`` against the model, in its single-number or per-class form.

    Args:
        threshold: A single threshold for every class, or a :class:`ClassThresholds`.
        context: The prediction rules whose ``class_names`` and class-id map per-class thresholds must fit.

    Returns:
        The score floor for the pre-filter in ``postprocess``, and the table that a detection's label indexes, or
        ``None`` for a single threshold, which ``predict`` compares with every score.

    Raises:
        TypeError: If ``threshold`` is neither a number nor a ``ClassThresholds``, such as a list or a dict.
        ValueError: If a number is NaN, or as :meth:`ClassThresholds._table` does.

    Examples:
        >>> from types import SimpleNamespace
        >>> context = SimpleNamespace(class_names=["a", "b"], class_id_to_name={0: "a", 1: "b"}, num_classes=2)
        >>> _class_thresholds(0.5, context)
        (0.5, None)
        >>> _class_thresholds([0.5, 0.5], context)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: threshold must be a number or a ClassThresholds; pass per-class thresholds as ClassThresholds(...
    """
    if isinstance(threshold, ClassThresholds):
        return threshold._table(context)
    return _threshold_number(threshold, "threshold", f"a number or a ClassThresholds; {_PER_CLASS_HINT}"), None
