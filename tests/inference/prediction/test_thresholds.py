# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``ClassThresholds``, the per-class form of ``predict(threshold=...)``."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr import ClassThresholds


class TestClassThresholds:
    """``ClassThresholds`` checks its values when built and fits them to ``class_names`` when ``predict`` runs."""

    def test_mapping_gives_unnamed_classes_the_default(self) -> None:
        """Named classes keep their own threshold, and every other class takes ``default``.

        This is the form that replaces a list built with ``class_names.index(...)``: only the classes that need their
        own bar are written down.
        """
        thresholds = ClassThresholds({"person": 0.35, "traffic light": 0.6}, default=0.4)

        assert thresholds.for_classes(["person", "car", "traffic light"]) == [0.35, 0.4, 0.6]

    def test_mapping_default_is_the_single_threshold_default(self) -> None:
        """Without ``default``, an unnamed class takes 0.5, the default single threshold of ``predict``."""
        thresholds = ClassThresholds({"a": 0.1})

        assert thresholds.for_classes(["a", "b"]) == [0.1, 0.5]

    @pytest.mark.parametrize(
        "to_values",
        [
            pytest.param(list, id="list"),
            pytest.param(tuple, id="tuple"),
            pytest.param(np.array, id="ndarray"),
            pytest.param(torch.tensor, id="tensor"),
        ],
    )
    def test_sequence_is_read_in_class_names_order(self, to_values: Callable[[list[float]], Any]) -> None:
        """A list, tuple, array or tensor gives one threshold per entry of ``class_names``, in that order."""
        thresholds = ClassThresholds(to_values([0.25, 0.5, 0.75]))

        assert thresholds.for_classes(["a", "b", "c"]) == [0.25, 0.5, 0.75]

    def test_values_are_copied_when_built(self) -> None:
        """Changing the caller's dict after building leaves the thresholds the object applies unchanged.

        The object is frozen, so it must not share a mutable dict with its caller.
        """
        by_name = {"a": 0.1}
        thresholds = ClassThresholds(by_name)
        by_name["a"] = 0.9

        assert thresholds.for_classes(["a"]) == [0.1]

    def test_unknown_class_name_raises_listing_class_names(self) -> None:
        """A misspelled name is refused together with the model's class names, instead of being silently ignored.

        Ignoring it would leave the intended class at the default threshold with no sign of the typo.
        """
        thresholds = ClassThresholds({"dgo": 0.4})

        with pytest.raises(ValueError, match=r"names \['dgo'\].*class_names are \['cat', 'dog'\]"):
            thresholds.for_classes(["cat", "dog"])

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([0.5], id="too-few"),
            pytest.param([0.5] * 3, id="too-many"),
        ],
    )
    def test_sequence_of_another_length_raises(self, values: list[float]) -> None:
        """A sequence must give exactly one threshold per class; the error names both counts."""
        thresholds = ClassThresholds(values)

        with pytest.raises(ValueError, match=rf"{len(values)} entries but the model has 2 classes"):
            thresholds.for_classes(["a", "b"])

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param({}, id="mapping"),
            pytest.param([], id="sequence"),
        ],
    )
    def test_model_without_classes_raises(self, values: Any) -> None:
        """No class can take a threshold, and the mask pre-filter would need the lowest of none.

        An empty sequence has as many entries as the zero classes, so only this check stops it before ``min([])``.
        """
        thresholds = ClassThresholds(values)

        with pytest.raises(ValueError, match="no classes"):
            thresholds.for_classes([])

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(["0.5", "0.5"], id="str-entries"),
            pytest.param([None, 0.5], id="none-entry"),
            pytest.param([b"0.5", 0.5], id="bytes-entry"),
            pytest.param(np.array(["0.5", "0.5"]), id="str-ndarray"),
            pytest.param([np.array("0.5"), 0.5], id="0-d-str-ndarray-entry"),
        ],
    )
    def test_non_numeric_entry_raises(self, values: Any) -> None:
        """An entry that is not a number is refused, not coerced: ``float("0.5")`` would parse the string."""
        with pytest.raises(TypeError, match="must be numbers"):
            ClassThresholds(values)

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param({"a": "0.5"}, id="str-value"),
            pytest.param({"a": None}, id="none-value"),
        ],
    )
    def test_non_numeric_mapping_value_raises(self, values: dict[str, Any]) -> None:
        """A mapping value is held to the types a single threshold takes, and the error names its class."""
        with pytest.raises(TypeError, match="threshold for 'a' must be a number"):
            ClassThresholds(values)

    def test_non_string_key_raises(self) -> None:
        """Keys are class names; an integer key would never match an entry of ``class_names``."""
        with pytest.raises(TypeError, match="keys must be class names"):
            ClassThresholds({0: 0.5})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        ("values", "default"),
        [
            pytest.param([float("nan"), 0.1], 0.5, id="sequence-entry"),
            pytest.param({"a": float("nan")}, 0.5, id="mapping-value"),
            pytest.param({"a": 0.1}, float("nan"), id="default"),
        ],
    )
    def test_nan_raises_when_built(self, values: Any, default: float) -> None:
        """A NaN would silently drop its class, and as the lowest threshold it would empty a segmentation result."""
        with pytest.raises(ValueError, match="NaN"):
            ClassThresholds(values, default=default)

    @pytest.mark.parametrize(
        "values",
        [
            0.5,
            "abc",
            pytest.param({0.5}, id="set"),
        ],
    )
    def test_values_without_dimensions_raise(self, values: Any) -> None:
        """A single number goes straight to ``predict(threshold=...)``, and a string or set is no list of thresholds."""
        with pytest.raises(TypeError, match="ClassThresholds takes a mapping"):
            ClassThresholds(values)

    def test_ragged_sequence_raises(self) -> None:
        """A ragged list is refused with the project's message instead of NumPy's inhomogeneous-shape error."""
        with pytest.raises(ValueError, match="ragged"):
            ClassThresholds([[0.5, 0.5], [0.5]])

    @pytest.mark.parametrize(
        "to_array",
        [
            pytest.param(list, id="nested-list"),
            pytest.param(np.array, id="ndarray"),
            pytest.param(torch.tensor, id="tensor"),
            pytest.param(
                lambda rows: [[torch.tensor(value, requires_grad=True) for value in row] for row in rows],
                id="nested-list-of-tensors-requiring-grad",
            ),
        ],
    )
    def test_two_dimensional_values_raise(self, to_array: Callable[[list[list[float]]], Any]) -> None:
        """A nested shape is refused before conversion, so ``tolist()`` never hides it as a list of lists.

        Tensors that require grad would otherwise surface PyTorch's ``RuntimeError`` from NumPy's shape check.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            ClassThresholds(to_array([[0.5, 0.5]]))

    def test_tensor_that_requires_grad_is_read_without_a_warning(self) -> None:
        """A tensor of thresholds that requires grad is read without PyTorch's scalar-conversion warning.

        Reading it one element at a time with ``float()`` warned once per class; a single ``detach().cpu().tolist()``
        does not, and on CUDA it is one device-to-host copy instead of one per class.
        """
        values = torch.tensor([0.7, 0.4, 0.6], requires_grad=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            thresholds = ClassThresholds(values)

        assert thresholds.for_classes(["a", "b", "c"]) == pytest.approx([0.7, 0.4, 0.6])

    @pytest.mark.parametrize(
        ("thresholds", "expected"),
        [
            pytest.param(
                ClassThresholds({"person": 0.35}, default=0.4),
                "ClassThresholds({'person': 0.35}, default=0.4)",
                id="mapping",
            ),
            pytest.param(ClassThresholds((0.25, 0.75)), "ClassThresholds([0.25, 0.75])", id="sequence"),
        ],
    )
    def test_repr_is_the_call_that_builds_it(self, thresholds: ClassThresholds, expected: str) -> None:
        """The repr reads as the constructor call, not as the internal read-only mapping or tuple."""
        assert repr(thresholds) == expected
