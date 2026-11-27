# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the public export facades ``rfdetr.export.inference`` and ``rfdetr.export.benchmark``.

The export cookbooks import runtimes and timing helpers from these two modules, so their names are public API. Each name
must resolve to the object its private module defines, and importing ``rfdetr.export.inference`` must stay free of the
optional runtime packages those objects wrap.

Tests for explicit exported class-layout handling.

Discovery surface of the public export facades: ``__all__``, star-imports and ``dir()``.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from rfdetr.export import benchmark, inference
from rfdetr.export import benchmark as public_benchmark
from rfdetr.export import inference as public_inference
from rfdetr.export._class_layout import _exclude_background_class
from rfdetr.export.inference import _LAZY_EXPORTS

#: Child-interpreter script: records (without blocking) every real import of an optional runtime, prints them as JSON.
_RECORD_OPTIONAL_IMPORTS = """
import importlib.abc
import json
import sys

OPTIONAL_RUNTIMES = {"openvino", "tensorrt", "executorch"}
attempted = []


def is_import_statement():
    # importlib.util.find_spec() probes without importing, and rfdetr probes tensorrt that way. A real import reaches
    # the finder through importlib's frozen frames only (via _find_and_load); a probe goes through importlib.util.
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith("<frozen importlib"):
        if frame.f_code.co_name == "_find_and_load":
            return True
        frame = frame.f_back
    return False


class RecordingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in OPTIONAL_RUNTIMES and is_import_statement():
            attempted.append(fullname)
        return None


sys.meta_path.insert(0, RecordingFinder())
from rfdetr.export import inference

for name in ("DecodedDetections", "decode_detections", "preprocess_to_nchw"):
    getattr(inference, name)
print(json.dumps(attempted))
"""


class TestInferenceFacade:
    """``rfdetr.export.inference`` re-exports runtimes and pre/post-processing helpers lazily."""

    @pytest.mark.parametrize(
        ("name", "private_module"),
        [
            ("DecodedDetections", "rfdetr.export._runtime.decode"),
            ("OpenVINOInference", "rfdetr.export._openvino.inference"),
            ("TRTInference", "rfdetr.export._tensorrt.inference"),
            ("decode_detections", "rfdetr.export._runtime.decode"),
            ("load_executorch_method", "rfdetr.export._executorch.inference"),
            ("preprocess_to_nchw", "rfdetr.export._runtime.preprocess"),
        ],
    )
    def test_name_is_the_private_object(self, name: str, private_module: str) -> None:
        """Each public name is the very object its private module defines, not a copy or a wrapper.

        A copy would let the two drift apart; identity keeps the private module the single definition.
        """
        assert getattr(public_inference, name) is getattr(importlib.import_module(private_module), name)

    def test_all_lists_every_public_name(self) -> None:
        """``__all__`` names exactly the lazily resolved objects, and ``dir()`` shows them for tab completion."""
        assert set(public_inference.__all__) == set(_LAZY_EXPORTS)
        assert set(public_inference.__all__) <= set(dir(public_inference))

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """A name outside ``__all__`` raises ``AttributeError`` rather than importing something unexpected."""
        with pytest.raises(AttributeError) as exc_info:
            _ = public_inference.NotARuntime
        assert "NotARuntime" in str(exc_info.value)

    def test_import_attempts_no_optional_runtime_import(self) -> None:
        """Importing the facade and resolving its runtime-free names never even tries to import an optional runtime.

        Runs in a fresh interpreter: other tests in this suite import those packages, so an in-process check would
        pass for the wrong reason. The child records every import of ``openvino``, ``tensorrt`` or ``executorch``
        (not a mere ``importlib.util.find_spec`` availability probe) and lets it proceed or fail as it normally
        would. Recording attempts rather than checking ``sys.modules`` keeps the test meaningful where the package is
        not installed (the CPU CI matrix), and survives an eagerly imported runtime being hidden by a guarded
        ``try``/``except ImportError``.
        Only the names that wrap no optional runtime are touched; resolving ``OpenVINOInference`` itself imports
        ``rfdetr.export._openvino``, whose guarded availability probe legitimately attempts ``import openvino``.
        """
        result = subprocess.run(
            [sys.executable, "-c", _RECORD_OPTIONAL_IMPORTS],
            check=False,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, f"child interpreter failed (exit {result.returncode}):\n{result.stderr}"
        attempted = json.loads(result.stdout.strip().splitlines()[-1])
        assert attempted == [], f"importing the facade attempted optional imports {attempted}; stderr:\n{result.stderr}"


class TestBenchmarkFacade:
    """``rfdetr.export.benchmark`` exposes the timing and memory helpers the export cookbooks report with."""

    @pytest.mark.parametrize("name", ["BenchmarkResult", "MemoryResult", "measure_latency", "measure_memory"])
    def test_name_is_the_private_object(self, name: str) -> None:
        """Each public timing helper is the object defined in ``rfdetr.export._benchmark``.

        The cookbooks moved from the private module to this one; identity guarantees they time exactly as before.
        """
        assert getattr(public_benchmark, name) is getattr(importlib.import_module("rfdetr.export._benchmark"), name)


def test_no_background_class_preserves_every_exported_slot() -> None:
    """``None`` keeps the score grid and every original class ID."""
    scores_all = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    scores, class_ids = _exclude_background_class(scores_all, None)

    np.testing.assert_array_equal(scores, scores_all)
    assert class_ids.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("background_class_id", "expected_scores", "expected_class_ids"),
    [
        pytest.param(0, [[0.2, 0.3]], [1, 2], id="background-first"),
        pytest.param(-1, [[0.1, 0.2]], [0, 1], id="background-last"),
    ],
)
def test_background_exclusion_preserves_original_class_ids(
    background_class_id: int,
    expected_scores: list[list[float]],
    expected_class_ids: list[int],
) -> None:
    """Removing a slot does not renumber the remaining exported classes."""
    scores_all = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    scores, class_ids = _exclude_background_class(scores_all, background_class_id)

    np.testing.assert_array_equal(scores, np.array(expected_scores, dtype=np.float32))
    assert class_ids.tolist() == expected_class_ids


@pytest.mark.parametrize("background_class_id", [2, -3])
def test_out_of_range_background_class_is_rejected(background_class_id: int) -> None:
    """Background IDs outside the exported class axis fail with a clear error."""
    with pytest.raises(ValueError, match="background_class_id"):
        _exclude_background_class(np.ones((1, 2), dtype=np.float32), background_class_id)


def test_non_matrix_scores_are_rejected() -> None:
    """The layout helper rejects a score tensor without query and class axes."""
    with pytest.raises(ValueError) as exc_info:
        _exclude_background_class(np.ones(2, dtype=np.float32), None)

    assert "shape (Q, C)" in str(exc_info.value)


@pytest.mark.parametrize(
    "background_class_id",
    [pytest.param(0.0, id="non-integral-float"), pytest.param(True, id="bool")],
)
def test_non_integer_background_class_is_rejected(background_class_id: object) -> None:
    """A float or bool background_class_id fails fast instead of silently mis-selecting a slot.

    ``bool`` is an ``int`` subclass in Python, so it would otherwise act as index 0/1; a float would compare unequal to
    every integer class ID and silently keep every slot instead of excluding one.
    """
    with pytest.raises(TypeError, match="background_class_id"):
        _exclude_background_class(np.ones((1, 2), dtype=np.float32), background_class_id)


class TestBenchmarkFacadeAll:
    """``rfdetr.export.benchmark`` exports only the timing API, not the LW-DETR CLI internals."""

    def test_all_names_only_the_timing_api(self) -> None:
        """``__all__`` lists exactly the four re-exported benchmarking names.

        Without it a star-import also pulled in ``main``, ``post_process``, ``TRTInference`` and ``logger`` — CLI script
        internals that are not part of the public surface.
        """
        assert benchmark.__all__ == ["BenchmarkResult", "MemoryResult", "measure_latency", "measure_memory"]

    def test_star_import_binds_only_all(self) -> None:
        """A star-import binds the ``__all__`` names and nothing else.

        This is the behavior ``__all__`` exists to control; the namespace check catches a stray public name leaking.
        """
        namespace: dict[str, Any] = {}

        exec("from rfdetr.export.benchmark import *", namespace)

        assert set(namespace) - {"__builtins__"} == set(benchmark.__all__)


class TestInferenceFacadeDir:
    """``rfdetr.export.inference`` advertises its lazily resolved names to ``dir()`` and ``__all__``."""

    def test_all_is_derived_from_lazy_exports(self) -> None:
        """``__all__`` is the sorted key set of the lazy-import table, so the two cannot drift apart."""
        assert inference.__all__ == sorted(inference._LAZY_EXPORTS)

    def test_dir_lists_every_lazy_name(self) -> None:
        """``dir()`` includes every lazily resolved name, so tab-completion offers what the cookbooks import.

        The names are not module globals until first access, so without a module ``__dir__`` they were invisible.
        """
        assert set(inference.__all__) <= set(dir(inference))
