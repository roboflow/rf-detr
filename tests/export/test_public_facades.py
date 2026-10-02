# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the public export facades ``rfdetr.export.inference`` and ``rfdetr.export.benchmark``.

The export cookbooks import runtimes and timing helpers from these two modules, so their names are public API. Each name
must resolve to the object its private module defines, and importing ``rfdetr.export.inference`` must stay free of the
optional runtime packages those objects wrap.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from rfdetr.export import benchmark as public_benchmark
from rfdetr.export import inference as public_inference
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
        with pytest.raises(AttributeError, match="NotARuntime"):
            _ = public_inference.NotARuntime

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
