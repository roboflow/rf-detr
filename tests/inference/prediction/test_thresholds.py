# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the confidence threshold of ``predict()``: one number for every class, or ``ClassThresholds``."""

import warnings
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch

from rfdetr import ClassThresholds, RFDETRInference, RFDETRSegNano
from rfdetr._prediction import core

from ..helpers import _DummyModel, _DummyRFDETR, _record_model_inputs


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


class TestPredictScoreThresholdEquivalence:
    """The mask pre-filter perf path must not change ``predict()``'s final output."""

    def test_mask_prefilter_output_equivalent_at_predict_boundary(self) -> None:
        """``predict()`` masks/boxes/scores must be bit-identical whether the mask pre-filter runs or not.

        The optimization forwards ``predict(threshold=...)`` into ``PostProcess`` so below-threshold masks skip
        upsampling. Disabling the pre-filter (``score_threshold=None``) makes ``PostProcess`` upsample every mask and
        lets ``predict()``'s own downstream filter drop them instead. The two paths must produce the same
        ``Detections``, proving the pre-filter drops only rows ``predict()`` itself discards — the equivalence that the
        unit test at ``_postprocess_masks`` level asserts, now verified end-to-end through the real ``predict()`` path.
        """
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
        model = RFDETRSegNano(pretrain_weights=None)
        threshold = 0.3

        optimized = model.predict(img, threshold=threshold)

        real_postprocess = model.model.postprocess

        def postprocess_without_prefilter(predictions, target_sizes, score_threshold=None):
            return real_postprocess(predictions, target_sizes, score_threshold=None)

        model.model.postprocess = postprocess_without_prefilter
        baseline = model.predict(img, threshold=threshold)

        np.testing.assert_array_equal(optimized.xyxy, baseline.xyxy)
        np.testing.assert_array_equal(optimized.confidence, baseline.confidence)
        np.testing.assert_array_equal(optimized.class_id, baseline.class_id)
        np.testing.assert_array_equal(optimized.mask, baseline.mask)


def _make_scored_model(
    labels: list[int],
    scores: list[float],
    class_names: list[str] | None = None,
    score_dtype: torch.dtype = torch.float32,
    **model_args: Any,
) -> _DummyRFDETR:
    """Build a weight-free model whose ``postprocess`` returns the given labels with the given scores.

    Args:
        labels: Raw class ids, one per detection.
        scores: Scores, one per detection.
        class_names: Class names the model reports. ``None`` keeps the 80 COCO names.
        score_dtype: Dtype of the scores, ``torch.float16`` for a model run in half precision.
        **model_args: Attributes for ``model.model.args``. Omitted when empty.

    Returns:
        A model whose ``predict()`` applies its own threshold filter to those detections.

    Examples:
        >>> model = _make_scored_model([0, 1], [0.9, 0.2], class_names=["a", "b"])
        >>> model.class_names
        ['a', 'b']
    """
    model = _DummyRFDETR()
    model.model = _DummyModel(class_names=class_names, labels=labels, scores=scores, score_dtype=score_dtype)
    if model_args:
        model.model.args = SimpleNamespace(**model_args)
    return model


#: One tiny RGB image; the stub model ignores its pixels, only ``predict()``'s filtering is under test.
_PER_CLASS_IMAGE = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))


class TestPerClassThresholdValidation:
    """``predict(threshold=...)`` refuses what it cannot apply to the model's classes, before the network runs."""

    #: Two detections of every class, scored 0.9 .. 0.4 in label order, as in ``TestPredictPerClassThreshold``.
    _LABELS = [0, 1, 2, 0, 1, 2]
    _SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    @pytest.fixture
    def scored_model(self) -> tuple[_DummyRFDETR, list[torch.Tensor]]:
        """A three-class model, and the list that records every batch its network receives."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])
        batches: list[torch.Tensor] = []
        model.model.model.register_forward_pre_hook(_record_model_inputs(batches))
        return model, batches

    @pytest.mark.parametrize(
        ("class_id_to_name", "message"),
        [
            pytest.param({0: "b", 1: "a", 2: "c"}, "ascending class-id order", id="names-out-of-order"),
            pytest.param(
                {0: "a", 1: "b", 2: "c", 3: "d"}, "4 class ids but class_names has 3", id="more-ids-than-names"
            ),
            pytest.param({-1: "a", 1: "b", 2: "c"}, "class id -1, outside 0..3", id="negative-id"),
            pytest.param({0: "a", 1: "b", 10**9: "c"}, "class id 1000000000, outside 0..3", id="id-beyond-the-logits"),
        ],
    )
    def test_class_id_map_that_does_not_match_class_names_raises_before_inference(
        self,
        scored_model: tuple[_DummyRFDETR, list[torch.Tensor]],
        class_id_to_name: dict[int, str],
        message: str,
    ) -> None:
        """A hand-written or hand-edited export map must not misplace a threshold, run the model, or size a huge table.

        Per-class thresholds are matched to raw class ids by position, which an export's metadata does not guarantee: a
        reordered map would hand a class the threshold of another, more ids than names would fail on a missing threshold
        after the forward pass, and an id of a billion would allocate a table of that many entries.
        """
        model, batches = scored_model
        context = replace(model._prediction_context(), class_id_to_name=class_id_to_name, num_classes=3)

        with pytest.raises(ValueError, match=message):
            core.predict(lambda: context, _PER_CLASS_IMAGE, threshold=ClassThresholds([0.5, 0.5, 0.5]))

        assert batches == []

    def test_single_float_does_not_read_the_class_id_map(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]
    ) -> None:
        """A single float never reads the class-id map, so the check cannot regress scalar-threshold artifacts.

        The same unordered, out-of-range map that refuses a ``ClassThresholds`` is harmless to the float form, which
        compares every score with one number. An export that predicts today with ``threshold=0.5`` must keep doing so.
        """
        model, batches = scored_model
        context = replace(model._prediction_context(), class_id_to_name={0: "b", 1: "a", 10**9: "c"}, num_classes=3)

        detections = core.predict(lambda: context, _PER_CLASS_IMAGE, threshold=0.55)

        assert detections.class_id.tolist() == [0, 1, 2, 0]
        assert len(batches) == 1

    @pytest.mark.parametrize(
        "threshold",
        [
            pytest.param(float("nan"), id="float"),
            pytest.param(np.float32("nan"), id="np-float32"),
            pytest.param(torch.tensor(float("nan")), id="0-d-tensor"),
        ],
    )
    def test_nan_single_threshold_raises_before_inference(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]], threshold: Any
    ) -> None:
        """A NaN single threshold is refused, as a NaN in ``ClassThresholds`` is, instead of silently emptying results.

        No score is above NaN, so ``scores > nan`` kept nothing and said nothing.
        """
        model, batches = scored_model

        with pytest.raises(ValueError, match="NaN"):
            model.predict(_PER_CLASS_IMAGE, threshold=threshold)

        assert batches == []

    def test_unknown_class_name_raises_before_inference(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]
    ) -> None:
        """A class name the model does not have is refused before the network runs, with the model's class names."""
        model, batches = scored_model

        with pytest.raises(ValueError, match=r"names \['d'\], which the model does not have"):
            model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds({"d": 0.5}))

        assert batches == []

    def test_named_classes_keep_their_own_threshold_and_others_take_the_default(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]
    ) -> None:
        """Class ``"b"`` needs > 0.4 and the unnamed classes ``"a"`` and ``"c"`` take the default 0.65.

        That keeps the two detections of class ``"b"`` and the class ``"a"`` and ``"c"`` detections scored 0.9 and 0.7.
        """
        model, _ = scored_model

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds({"b": 0.4}, default=0.65))

        assert detections.class_id.tolist() == [0, 1, 2, 1]

    @pytest.mark.parametrize(
        "entries",
        [
            pytest.param([np.float32(0.7), np.float64(0.4), 0.6], id="numpy-scalars"),
            pytest.param([torch.tensor(0.7), torch.tensor(0.4), torch.tensor(0.6)], id="0-d-tensors"),
            pytest.param([np.array(0.7), np.array(0.4), np.array(0.6)], id="0-d-ndarrays"),
            pytest.param(
                [torch.tensor(0.7, requires_grad=True), torch.tensor(0.4), torch.tensor(0.6, requires_grad=True)],
                id="0-d-tensors-requiring-grad",
            ),
        ],
    )
    def test_entries_may_be_any_number_a_single_threshold_may_be(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]], entries: list[Any]
    ) -> None:
        """``ClassThresholds`` entries take every type the single threshold takes."""
        model, _ = scored_model

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(entries))

        assert detections.class_id.tolist() == [0, 1, 2, 1]

    def test_bool_entries_count_as_numbers_like_a_bool_single_threshold(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]
    ) -> None:
        """``False`` is the threshold 0.0 and ``True`` is 1.0, per class as in the single form.

        The single threshold accepts a bool, so rejecting it only per class would make the two forms disagree. ``True``
        drops class 1 (no score reaches 1.0) and ``False`` keeps every detection of classes 0 and 2.
        """
        model, _ = scored_model

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([False, True, False]))

        assert detections.class_id.tolist() == [0, 2, 0, 2]

    def test_class_thresholds_for_a_model_without_classes_raises_before_inference(
        self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]
    ) -> None:
        """Per-class thresholds for a model without classes are refused before the network runs.

        ``ClassThresholds([])`` has as many entries as the zero classes; the lowest threshold, which the mask pre-filter
        needs, would then be the minimum of nothing.
        """
        model, batches = scored_model
        context = replace(model._prediction_context(), class_names=[], class_id_to_name={}, num_classes=0)

        with pytest.raises(ValueError, match="no classes"):
            core.predict(lambda: context, _PER_CLASS_IMAGE, threshold=ClassThresholds([]))

        assert batches == []

    def test_integer_tensor_is_read_as_thresholds(self, scored_model: tuple[_DummyRFDETR, list[torch.Tensor]]) -> None:
        """A tensor of integers converts to integers, which are numbers like any other threshold.

        ``[1, 0, 1]`` drops classes 0 and 2, since no score reaches 1, and keeps every detection of class 1.
        """
        model, _ = scored_model

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(torch.tensor([1, 0, 1])))

        assert detections.class_id.tolist() == [1, 1]


class TestPredictPerClassThreshold:
    """``predict(threshold=ClassThresholds(...))`` keeps a detection whose score exceeds its own class threshold."""

    #: Two detections of every class, scored 0.9 .. 0.4 in label order, so a threshold looked up for the wrong
    #: class changes which rows survive.
    _LABELS = [0, 1, 2, 0, 1, 2]
    _SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    def test_each_detection_is_filtered_by_its_own_class_threshold(self) -> None:
        """Class 0 needs > 0.7, class 1 > 0.4 and class 2 > 0.6, which keeps four of the six detections."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.7, 0.4, 0.6]))

        assert detections.class_id.tolist() == [0, 1, 2, 1]
        np.testing.assert_allclose(detections.confidence, [0.9, 0.8, 0.7, 0.5])

    def test_score_equal_to_class_threshold_is_dropped(self) -> None:
        """The comparison is strict, like the float form's ``score > threshold``."""
        model = _make_scored_model([0, 1], [0.5, 0.75], class_names=["a", "b"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.5, 0.5]))

        assert detections.class_id.tolist() == [1]

    def test_uniform_list_matches_the_float_form(self) -> None:
        """``[t] * n`` selects exactly the rows ``t`` selects."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        from_list = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.55] * 3))
        from_float = model.predict(_PER_CLASS_IMAGE, threshold=0.55)

        assert from_list.class_id.tolist() == from_float.class_id.tolist() == [0, 1, 2, 0]
        np.testing.assert_array_equal(from_list.confidence, from_float.confidence)

    def test_class_id_table_is_float64_on_cpu_under_another_default_device(self) -> None:
        """The table is a float64 CPU tensor even when the process default device is not the CPU.

        ``torch.set_default_device("mps")`` makes an unpinned float64 allocation fail, because MPS has no float64. The
        ``meta`` device stands in for it here: it is an available non-CPU default on every host, and an unpinned table
        would be created on it instead of on the CPU.
        """
        with torch.device("meta"):
            table = ClassThresholds._by_class_id([0.25, 0.75], {3: "b", 1: "a"})

        assert (table.device.type, table.dtype) == ("cpu", torch.float64)

    @pytest.mark.parametrize(
        ("score_dtype", "score", "expected_kept"),
        [
            pytest.param(torch.float16, 0.40014648, 0, id="float16"),
            pytest.param(torch.bfloat16, 0.400390625, 0, id="bfloat16"),
            pytest.param(torch.float32, 0.40012, 1, id="float32"),
            pytest.param(torch.float64, 0.4001, 0, id="float64"),
        ],
    )
    def test_scores_compare_like_the_float_form_in_every_dtype(
        self, score_dtype: torch.dtype, score: float, expected_kept: int
    ) -> None:
        """A score is compared with the threshold rounded to the score dtype, as the float form compares it.

        ``0.4001`` rounds to the float16 value ``0.40014648``, so a float16 score of exactly that value is not above it,
        while the unrounded ``0.4001`` would be below it. ``0.4001`` rounds to the bfloat16 value ``0.400390625``, so a
        bfloat16 score of exactly that value is not above it, which a float32 table (``0.4000999927...``) would get
        wrong. A float64 score of ``0.4001`` is not above ``0.4001`` either, which a float32 table (``0.4000999927...``)
        would get wrong, and a float32 score of ``0.40012`` is above it, which a table rounded to float16 would get
        wrong.
        """
        model = _make_scored_model([0], [score], class_names=["a"], score_dtype=score_dtype)

        from_list = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.4001]))
        from_float = model.predict(_PER_CLASS_IMAGE, threshold=0.4001)

        assert len(from_float) == expected_kept
        assert from_list.class_id.tolist() == from_float.class_id.tolist()

    @pytest.mark.parametrize(
        "to_array",
        [
            pytest.param(tuple, id="tuple"),
            pytest.param(np.array, id="ndarray"),
            pytest.param(torch.tensor, id="tensor"),
        ],
    )
    def test_any_one_dimensional_array_like_is_accepted(self, to_array: Callable[[list[float]], Any]) -> None:
        """A tuple, array or tensor of thresholds behaves like the list."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(to_array([0.7, 0.4, 0.6])))

        assert detections.class_id.tolist() == [0, 1, 2, 1]

    @pytest.mark.parametrize(
        "threshold",
        [
            pytest.param([0.5], id="too-few"),
            pytest.param([0.5] * 4, id="too-many"),
            pytest.param([], id="empty"),
        ],
    )
    def test_wrong_length_raises_before_inference(self, threshold: list[float]) -> None:
        """One threshold per class is required; the error names both counts and the model never runs."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])
        batches: list[torch.Tensor] = []

        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            with pytest.raises(ValueError, match=rf"{len(threshold)} entries.*3 classes"):
                model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(threshold))

        assert batches == []

    @pytest.mark.parametrize(
        ("threshold", "expected_class_ids"),
        [
            pytest.param([float("inf"), 0.1, 0.1], [1, 2, 1, 2], id="inf-drops-the-class"),
            pytest.param([-float("inf"), 0.95, 0.95], [0, 0], id="minus-inf-keeps-the-class"),
        ],
    )
    def test_infinite_threshold_drops_or_keeps_its_whole_class(
        self, threshold: list[float], expected_class_ids: list[int]
    ) -> None:
        """``inf`` is the documented way to drop a class, and ``-inf`` keeps every detection of one."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(threshold))

        assert detections.class_id.tolist() == expected_class_ids

    def test_each_image_of_a_batch_is_filtered_by_its_own_labels(self) -> None:
        """The thresholds are looked up with the labels of the image being filtered, not of the first one."""
        model = _make_scored_model([0], [0.0], class_names=["a", "b", "c"])
        per_image = [
            {"scores": torch.tensor([0.9, 0.3]), "labels": torch.tensor([0, 1]), "boxes": torch.zeros(2, 4)},
            {"scores": torch.tensor([0.9, 0.9]), "labels": torch.tensor([1, 2]), "boxes": torch.zeros(2, 4)},
        ]

        with patch.object(model.model, "postprocess", return_value=per_image):
            first, second = model.predict(
                [_PER_CLASS_IMAGE, _PER_CLASS_IMAGE], threshold=ClassThresholds([0.5, 0.5, 0.95])
            )

        assert (first.class_id.tolist(), second.class_id.tolist()) == ([0], [1])

    @pytest.mark.parametrize(
        "threshold",
        [
            pytest.param({"a": 0.5}, id="dict-by-class-name"),
            pytest.param([0.5, 0.5, 0.5], id="list"),
            pytest.param(np.array([0.5, 0.5, 0.5]), id="1-d-ndarray"),
            pytest.param(torch.tensor([0.5, 0.5, 0.5]), id="1-d-tensor"),
            "0.5",
            pytest.param({0.5}, id="set"),
            pytest.param(np.array("0.5"), id="0-d-str-ndarray"),
        ],
    )
    def test_threshold_that_is_neither_a_number_nor_class_thresholds_raises_before_inference(
        self, threshold: object
    ) -> None:
        """A plain dict or list is refused with a pointer to ``ClassThresholds``, before the network runs.

        A dict keyed by class name is the natural guess, but it says nothing about the classes it leaves out, so per-
        class values go through ``ClassThresholds``, which carries that ``default``.
        """
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])
        batches: list[torch.Tensor] = []

        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            with pytest.raises(TypeError, match="a number or a ClassThresholds"):
                model.predict(_PER_CLASS_IMAGE, threshold=threshold)  # type: ignore[arg-type]

        assert batches == []

    @pytest.mark.parametrize(
        ("threshold", "expected_class_ids"),
        [
            pytest.param(0, [0, 1, 2, 0, 1, 2], id="int"),
            pytest.param(np.float32(0.55), [0, 1, 2, 0], id="np-float32"),
            pytest.param(torch.tensor(0.55), [0, 1, 2, 0], id="0-d-tensor"),
            pytest.param(np.array(0.55), [0, 1, 2, 0], id="0-d-ndarray"),
        ],
    )
    def test_numeric_scalar_threshold_still_takes_the_float_form(
        self, threshold: Any, expected_class_ids: list[int]
    ) -> None:
        """The type guard lets every numeric scalar through, a 0-d NumPy array included.

        ``scores > np.array(0.55)`` raises ``TypeError``, so the single threshold is read as a float before it is used.
        """
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=threshold)

        assert detections.class_id.tolist() == expected_class_ids

    def test_segmentation_masks_follow_the_per_class_filter(self) -> None:
        """With the real ``PostProcess``, a list keeps the rows and masks of a low-threshold run that clear their class.

        The lowest entry is the mask pre-filter's cut-off, so a pre-filter fed any other entry would drop rows of the
        classes with a lower threshold before the per-class filter sees them.
        """
        model = RFDETRSegNano(pretrain_weights=None)
        low = model.predict(_PER_CLASS_IMAGE, threshold=0.0)
        index_by_name = {name: index for index, name in enumerate(model.class_names)}
        indexes = np.array([index_by_name.get(name, -1) for name in low.data["class_name"]])
        # Spread over the scores the untrained model produces, so some classes keep their rows and others lose them.
        lowest, highest = np.quantile(low.confidence, [0.1, 0.9])
        thresholds = np.random.default_rng(0).uniform(lowest, highest, len(model.class_names)).astype(np.float32)

        kept = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds(thresholds))

        expected = low[(indexes >= 0) & (low.confidence > thresholds[indexes])]
        assert 0 < len(kept) < len(low)
        np.testing.assert_array_equal(kept.confidence, expected.confidence)
        np.testing.assert_array_equal(kept.class_id, expected.class_id)
        np.testing.assert_array_equal(kept.xyxy, expected.xyxy)
        np.testing.assert_array_equal(kept.mask, expected.mask)

    @pytest.mark.parametrize("label_without_entry", [-2, 3, 7])
    def test_detection_without_a_class_entry_is_dropped(self, label_without_entry: int) -> None:
        """The no-object slot (id 3 here), an id beyond the table (7) and a negative id have no threshold.

        Such rows never pass. A negative id must not wrap around: ``-2`` would read the second-to-last table entry, the
        threshold of class ``"c"``.
        """
        model = _make_scored_model([0, label_without_entry], [0.9, 0.9], class_names=["a", "b", "c"])

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.1, 0.1, 0.1]))

        assert detections.class_id.tolist() == [0]

    def test_pretrained_coco_thresholds_are_indexed_by_class_name_position(self) -> None:
        """For COCO checkpoints the raw ids are sparse category ids, so thresholds still follow ``class_names``.

        Category 18 is ``"dog"``, ``class_names[17]``. Laying the thresholds out by raw id would give the dog
        ``class_names[18]``'s threshold and keep it.
        """
        model = _make_scored_model([1, 18, 3], [0.9, 0.9, 0.9], num_classes=90)

        detections = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds({"dog": 0.95}, default=0.1))

        assert detections.data["class_name"].tolist() == ["person", "car"]

    def test_legacy_background_first_keypoint_slots_map_to_class_names(self) -> None:
        """Slot 1 is ``class_names[0]`` and slot 0 is the background, which has no threshold."""
        model = _DummyRFDETR()
        model.model = _DummyModel(class_names=["person"], labels=[0, 1], include_keypoints=True)
        model.model.args = SimpleNamespace(num_classes=1, num_keypoints_per_class=[0, 17])

        key_points = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.5]))

        assert key_points.data["class_name"].tolist() == ["person"]

    def test_lowest_threshold_is_forwarded_to_postprocess(self) -> None:
        """The segmentation pre-filter must not drop a row the per-class filter keeps, so it gets the smallest one."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        with patch.object(model.model, "postprocess", wraps=model.model.postprocess) as spy:
            model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.7, 0.4, 0.6]))

        assert spy.call_args.kwargs["score_threshold"] == 0.4

    def test_streamed_batches_are_filtered_by_class_threshold(self) -> None:
        """A streamed call keeps the same rows as an eager one, batch by batch.

        ``stream=True`` routes every bounded batch through ``_predict_stream``, so the per-class filter must live in the
        batch step both routes share rather than in the eager entry point alone.
        """
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        results = RFDETRInference(model).predict(
            [_PER_CLASS_IMAGE, _PER_CLASS_IMAGE], threshold=ClassThresholds([0.7, 0.4, 0.6]), stream=True
        )

        assert [result.class_id.tolist() for result in results] == [[0, 1, 2, 1], [0, 1, 2, 1]]

    def test_explicit_batch_filters_every_image_by_class_threshold(self) -> None:
        """``batch=2`` over three images runs two batches and filters each image by the per-class list.

        An explicit ``batch`` routes an eager image list through ``_predict_stream``; the list threshold must reach
        every bounded batch, including the final partial one, and the call still returns a list of three results.
        """
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])
        batches: list[torch.Tensor] = []

        with model.model.model.register_forward_pre_hook(_record_model_inputs(batches)):
            results = RFDETRInference(model).predict(
                [_PER_CLASS_IMAGE] * 3, threshold=ClassThresholds([0.7, 0.4, 0.6]), batch=2
            )

        assert [len(batch) for batch in batches] == [2, 1]
        assert [result.class_id.tolist() for result in results] == [[0, 1, 2, 1]] * 3

    def test_wrong_length_list_with_stream_raises_when_predict_is_called(self) -> None:
        """A wrong-length list with ``stream=True`` raises when ``predict`` is called, not on the first ``next``.

        The generator body only runs from its first ``next``, after it has opened its source and decoded a batch, so a
        threshold the model cannot apply is refused up front, like the eager ``batch`` and ``vid_stride`` checks.
        """
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])
        predictor = RFDETRInference(model)

        with pytest.raises(ValueError, match=r"1 entries.*3 classes"):
            predictor.predict([_PER_CLASS_IMAGE], threshold=ClassThresholds([0.5]), stream=True)

    def test_bad_threshold_with_stream_never_opens_the_source(self) -> None:
        """A NaN threshold with ``stream=True`` is refused before the source reader is created or a frame decoded."""
        model = _make_scored_model(self._LABELS, self._SCORES, class_names=["a", "b", "c"])

        with patch.object(core, "iter_source_batches", autospec=True) as source_batches:
            with pytest.raises(ValueError, match="NaN"):
                RFDETRInference(model).predict([_PER_CLASS_IMAGE], threshold=float("nan"), stream=True)

        source_batches.assert_not_called()

    def test_per_class_threshold_drops_keypoint_detection_below_its_class_bar(self) -> None:
        """Two equally scored keypoint detections: the class held to a 0.95 bar is dropped, the 0.5 class is kept.

        Active-first layout, so slots 0 and 1 are ``person`` and ``bicycle``. The kept detection keeps its class name
        and the ``detection_confidence`` that the per-class filter compared with its own class threshold.
        """
        model = _DummyRFDETR()
        model.model = _DummyModel(
            class_names=["person", "bicycle"], labels=[0, 1], scores=[0.9, 0.9], include_keypoints=True
        )
        model.model.args = SimpleNamespace(num_classes=2, num_keypoints_per_class=[17, 17])

        key_points = model.predict(_PER_CLASS_IMAGE, threshold=ClassThresholds([0.5, 0.95]))

        assert isinstance(key_points, sv.KeyPoints)
        assert key_points.data["class_name"].tolist() == ["person"]
        np.testing.assert_allclose(key_points.detection_confidence, [0.9])
