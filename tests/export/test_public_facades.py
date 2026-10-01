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
import subprocess
import sys

import pytest

from rfdetr.export import benchmark as public_benchmark
from rfdetr.export import inference as public_inference


class TestInferenceFacade:
    """``rfdetr.export.inference`` re-exports runtimes and pre/post-processing helpers lazily."""

    @pytest.mark.parametrize(
        ("name", "private_module"),
        [
            pytest.param("DecodedDetections", "rfdetr.export._runtime.decode", id="decoded-detections"),
            pytest.param("OpenVINOInference", "rfdetr.export._openvino.inference", id="openvino"),
            pytest.param("TRTInference", "rfdetr.export._tensorrt.inference", id="tensorrt"),
            pytest.param("decode_detections", "rfdetr.export._runtime.decode", id="decode"),
            pytest.param("load_executorch_method", "rfdetr.export._executorch.inference", id="executorch"),
            pytest.param("preprocess_to_nchw", "rfdetr.export._runtime.preprocess", id="preprocess"),
        ],
    )
    def test_name_is_the_private_object(self, name: str, private_module: str) -> None:
        """Each public name is the very object its private module defines, not a copy or a wrapper.

        A copy would let the two drift apart; identity keeps the private module the single definition.
        """
        assert getattr(public_inference, name) is getattr(importlib.import_module(private_module), name)

    def test_all_lists_every_public_name(self) -> None:
        """``__all__`` names exactly the lazily resolved objects, so star-imports and docs see the same surface."""
        assert public_inference.__all__ == [
            "DecodedDetections",
            "OpenVINOInference",
            "TRTInference",
            "decode_detections",
            "load_executorch_method",
            "preprocess_to_nchw",
        ]

    def test_import_loads_no_optional_runtime(self) -> None:
        """Importing the facade leaves ``openvino``, ``tensorrt`` and ``executorch`` out of ``sys.modules``.

        Runs in a fresh interpreter: other tests in this suite import those packages, so an in-process check would
        pass for the wrong reason.
        """
        code = (
            "import sys, rfdetr.export.inference; "
            "print(sorted(m for m in ('openvino', 'tensorrt', 'executorch') if m in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
        assert result.stdout.strip() == "[]"


class TestBenchmarkFacade:
    """``rfdetr.export.benchmark`` exposes the timing and memory helpers the export cookbooks report with."""

    @pytest.mark.parametrize("name", ["BenchmarkResult", "MemoryResult", "measure_latency", "measure_memory"])
    def test_name_is_the_private_object(self, name: str) -> None:
        """Each public timing helper is the object defined in ``rfdetr.export._benchmark``.

        The cookbooks moved from the private module to this one; identity guarantees they time exactly as before.
        """
        assert getattr(public_benchmark, name) is getattr(importlib.import_module("rfdetr.export._benchmark"), name)
