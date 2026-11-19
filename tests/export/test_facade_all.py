# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Discovery surface of the public export facades: ``__all__``, star-imports and ``dir()``."""

from __future__ import annotations

from typing import Any

from rfdetr.export import benchmark, inference


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
