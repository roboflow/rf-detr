# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the custom ONNX symbolic optimizer registry.

``CustomOpSymbolicRegistry`` and ``register_optimizer`` back ``OnnxExporter``'s optimizer pass
(``src/rfdetr/export/_onnx/exporter.py``), which iterates ``CustomOpSymbolicRegistry._OPTIMIZER`` after export. That
list is process-global class state, so every test here restores it via an autouse fixture to avoid leaking registrations
into other export tests running in the same session.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rfdetr.export._onnx.symbolic import CustomOpSymbolicRegistry, register_optimizer


@pytest.fixture(autouse=True)
def _clean_optimizer_registry() -> Iterator[None]:
    """Snapshot and restore ``CustomOpSymbolicRegistry._OPTIMIZER`` around each test.

    A ``@pytest.fixture`` generator can't be invoked standalone outside pytest's fixture protocol.

    Examples:
        >>> callable(_clean_optimizer_registry)  # doctest: +SKIP
        True
    """
    original = list(CustomOpSymbolicRegistry._OPTIMIZER)
    CustomOpSymbolicRegistry._OPTIMIZER.clear()
    yield
    CustomOpSymbolicRegistry._OPTIMIZER[:] = original


class TestCustomOpSymbolicRegistryOptimizer:
    """Behavior of ``CustomOpSymbolicRegistry.optimizer``."""

    def test_appends_callback_to_registry(self) -> None:
        """A single registered callback is stored in ``_OPTIMIZER``."""

        def callback(graph: object) -> object:
            return graph

        CustomOpSymbolicRegistry.optimizer(callback)

        assert CustomOpSymbolicRegistry._OPTIMIZER == [callback]

    def test_preserves_registration_order_across_multiple_callbacks(self) -> None:
        """Callbacks registered in sequence keep their registration order."""

        def first(graph: object) -> object:
            return graph

        def second(graph: object) -> object:
            return graph

        CustomOpSymbolicRegistry.optimizer(first)
        CustomOpSymbolicRegistry.optimizer(second)

        assert CustomOpSymbolicRegistry._OPTIMIZER == [first, second]


class TestRegisterOptimizer:
    """Behavior of the ``register_optimizer`` decorator factory."""

    def test_decorated_function_is_added_to_registry(self) -> None:
        """Applying the decorator registers the wrapped function as an optimizer callback."""

        @register_optimizer()
        def optimize(graph: object) -> object:
            return graph

        assert CustomOpSymbolicRegistry._OPTIMIZER == [optimize]

    def test_decorator_returns_the_original_function_unchanged(self) -> None:
        """The decorator returns the same function object rather than wrapping it."""

        def optimize(graph: object) -> object:
            return graph

        decorated = register_optimizer()(optimize)

        assert decorated is optimize

    def test_each_call_returns_an_independent_decorator(self) -> None:
        """Two separate ``register_optimizer()`` calls each register their own function."""

        def first(graph: object) -> object:
            return graph

        def second(graph: object) -> object:
            return graph

        register_optimizer()(first)
        register_optimizer()(second)

        assert CustomOpSymbolicRegistry._OPTIMIZER == [first, second]
