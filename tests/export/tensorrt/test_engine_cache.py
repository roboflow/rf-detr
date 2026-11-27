# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the timing cache and the portable (hardware-compatible, version-compatible) engines of ``build_engine``.

They monkeypatch the polygraphy entry points, so they run without TensorRT, a GPU, or ``polygraphy`` installed.
"""

from __future__ import annotations

import inspect
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
import torch

from rfdetr.detr import RFDETR
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter
from tests._markers import (
    _requires_mkfifo,
    _requires_posix_permissions,
    _requires_symlink_privilege,
    _requires_tensorrt,
)
from tests.export.tensorrt.helpers import (
    _STATIC_INPUT,
    _fake_tensorrt,
    _FakeNetwork,
    _FakeNetworkInput,
    _FakeProfile,
    _minimal_export_graph,
)


def _patch_polygraphy_chain_recording(monkeypatch: pytest.MonkeyPatch, network: _FakeNetwork | None = None) -> dict:
    """Stub the polygraphy chain and record every keyword the build hands to ``CreateConfig`` and the engine build.

    The other stubs bind their signature to the exact keywords ``_compile`` passes today, so a renamed keyword fails
    loudly. The keywords a feature adds are optional, so this one accepts any keyword and lets the test assert on
    exactly which were passed, and which were not.

    Args:
        monkeypatch: Fixture used to replace the polygraphy entry points on the module under test.
        network: The parsed-network stand-in the loader hands back, or ``None`` for one fixed-batch input.

    Returns:
        Dict with the ``CreateConfig`` keywords under ``"config"`` and the keywords ``engine_from_network`` received
        besides the parsed network and the configuration under ``"build"``.

    Examples:
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     captured = _patch_polygraphy_chain_recording(monkeypatch)
        ...     _ = tensorrt_export.CreateConfig(fp16=True, option=1)
        ...     _ = tensorrt_export.engine_from_network("network", config="config", build_option=2)
        >>> captured
        {'config': {'fp16': True, 'option': 1}, 'build': {'build_option': 2}}
    """
    captured: dict = {"config": {}, "build": {}}
    parsed = network if network is not None else _FakeNetwork(_STATIC_INPUT)

    def _create_config(**kwargs: object) -> str:
        captured["config"].update(kwargs)
        return "config"

    def _engine_from_network(_parsed: object, config: object, **kwargs: object) -> str:
        captured["build"].update(kwargs)
        return types.SimpleNamespace(serialize=lambda: b"engine")

    monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", parsed, "parser"))
    monkeypatch.setattr(tensorrt_export, "Profile", _FakeProfile)
    monkeypatch.setattr(tensorrt_export, "CreateConfig", _create_config)
    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)
    return captured


class TestTimingCache:
    """``timing_cache`` reaches Polygraphy's config and engine build, and nothing is passed unless it is set."""

    def test_no_timing_cache_keyword_is_passed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``timing_cache`` the build calls Polygraphy exactly as it always did."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False}
        assert captured["build"] == {}

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    def test_an_existing_cache_is_loaded_and_saved_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dynamic_batch: bool
    ) -> None:
        """A cache file that exists seeds the build, and the same file receives the merged result.

        The exporter probes the file before the build to refuse an unwritable cache, and that probe must leave the
        timings a user already collected untouched; emptying them would silently throw the warm cache away.
        """
        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)))
        captured = _patch_polygraphy_chain_recording(monkeypatch, network)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        config = TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=4, timing_cache=cache)

        TensorRTExporter(config).build_engine("model.onnx")

        assert captured["config"]["load_timing_cache"] == str(cache)
        assert captured["build"] == {"save_timing_cache": str(cache)}
        assert cache.read_bytes() == b"previous timings"

    def test_a_missing_cache_is_saved_but_not_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The first use has no file yet: loading it would only make Polygraphy warn about a missing cache."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "not-yet.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert "load_timing_cache" not in captured["config"]
        assert captured["build"] == {"save_timing_cache": str(cache)}

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    @_requires_tensorrt
    def test_the_keywords_handed_to_polygraphy_exist_in_its_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dynamic_batch: bool
    ) -> None:
        """Every keyword the exporter passes with a timing cache is one the installed Polygraphy accepts.

        The recording stub takes any keyword, so a keyword Polygraphy renamed or removed would leave the CPU suite green
        while only a GPU run broke. The keywords the exporter really passes are recorded for a fixed-batch and a
        dynamic-batch build (the latter adds a profile next to the cache), then checked against the signatures of the
        real ``CreateConfig`` and ``engine_from_network``. Skipped where TensorRT or Polygraphy is not installed.
        """
        from polygraphy.backend import trt as trt_backend

        network = _FakeNetwork(_FakeNetworkInput("input", (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)))
        captured = _patch_polygraphy_chain_recording(monkeypatch, network)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        config = TensorRTConfig(fp16=False, dynamic_batch=dynamic_batch, max_batch_size=4, timing_cache=cache)

        TensorRTExporter(config).build_engine("model.onnx")

        # ``CreateConfig`` forwards most of its options to a base class through ``**kwargs``, so collect them all.
        create_config_keywords = {
            name
            for cls in trt_backend.CreateConfig.__mro__
            if cls is not object and "__init__" in vars(cls)
            for name in inspect.signature(vars(cls)["__init__"]).parameters
        }
        build_keywords = set(inspect.signature(trt_backend.engine_from_network).parameters)
        assert "load_timing_cache" in captured["config"], "the exporter no longer loads the cache it was given"
        assert set(captured["config"]) <= create_config_keywords, "CreateConfig does not accept every keyword passed"
        assert set(captured["build"]) <= build_keywords, "engine_from_network does not accept every keyword passed"

    @pytest.mark.parametrize("value", [pytest.param("", id="empty"), b"engine.cache", 3, True])
    def test_a_value_that_is_not_a_path_is_refused(self, value: object) -> None:
        """A non-path value is refused when the exporter is built, before any work on the model, and says why."""
        with pytest.raises(ValueError, match="trt_timing_cache must be a non-empty file path"):
            TensorRTExporter(TensorRTConfig(timing_cache=value))

    def test_a_missing_directory_is_created(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Polygraphy opens the lock file beside the cache before it creates the directory, so the export does it."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "not" / "yet" / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert cache.parent.is_dir()
        assert captured["build"] == {"save_timing_cache": str(cache)}

    def test_a_directory_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A directory cannot hold the cache, and Polygraphy would only say so after the whole build."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=tmp_path))

        with pytest.raises(ValueError, match=r"trt_timing_cache.*directory"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_a_location_that_cannot_be_created_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A file where a directory is needed cannot be worked around by any user, on any operating system.

        The refusal names the setting and still raises the errno's own ``OSError`` subclass (``FileExistsError`` on
        every operating system); a caller catching that subclass must not lose it to a plain ``OSError`` re-wrap.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=blocker / "engine.cache"))

        with pytest.raises(FileExistsError, match="trt_timing_cache"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_a_path_ending_in_a_separator_is_refused_when_the_exporter_is_built(self, tmp_path: Path) -> None:
        """A path ending in a separator is refused when the exporter is built, before any work on the model.

        ``out/cache/`` names a directory, and that is plain from the value alone, so no filesystem access is needed.
        """
        with pytest.raises(ValueError, match=r"trt_timing_cache.*names a directory"):
            TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=str(tmp_path / "cache") + os.sep))

    @_requires_symlink_privilege
    def test_a_link_to_a_missing_file_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A symbolic link to a missing file is refused before the build, not after it.

        Polygraphy would create the file at the link's target after the build, and a missing target folder would lose
        the timings.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        link = tmp_path / "engine.cache"
        link.symlink_to(tmp_path / "unmounted" / "engine.cache")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=link))

        with pytest.raises(OSError, match=r"trt_timing_cache.*symbolic link"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    @_requires_posix_permissions
    def test_a_read_only_cache_file_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The build would save into the file after the engine exists, so a file it cannot write is refused now.

        The refusal must not leave behind a lock file that would then belong to the user who was refused.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        cache.chmod(0o444)
        if os.access(cache, os.W_OK):
            pytest.skip("this user can write a read-only file (root)")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache))

        with pytest.raises(OSError, match="trt_timing_cache"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}
        assert sorted(path.name for path in tmp_path.iterdir()) == ["engine.cache"]

    @_requires_posix_permissions
    def test_a_read_only_directory_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The most common bad location: a folder the user cannot write, with no cache in it yet."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        folder = tmp_path / "shared"
        folder.mkdir()
        folder.chmod(0o555)
        if os.access(folder, os.W_OK):
            pytest.skip("this user can write a read-only directory (root)")
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=folder / "engine.cache"))

        try:
            with pytest.raises(OSError, match="trt_timing_cache"):
                exporter.build_engine("model.onnx")
        finally:
            folder.chmod(0o755)

        assert captured == {"config": {}, "build": {}}

    def test_the_lock_file_polygraphy_needs_is_created_beside_the_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Polygraphy opens ``<file>.lock`` before it writes the cache, so the export opens it first."""
        _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx")

        assert (tmp_path / "engine.cache.lock").is_file()

    def test_a_dry_run_touches_nothing(self, tmp_path: Path) -> None:
        """A dry run builds nothing, so it creates neither the cache directory nor a lock file."""
        cache = tmp_path / "not" / "yet" / "engine.cache"

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=cache)).build_engine("model.onnx", dry_run=True)

        assert not (tmp_path / "not").exists()

    def test_a_bad_location_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_convert`` checks the location first, so a doomed request does not wait for the ONNX stage."""
        _patch_polygraphy_chain_recording(monkeypatch)
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )

        with pytest.raises(ValueError, match="directory"):
            TensorRTExporter(TensorRTConfig(timing_cache=tmp_path))._convert(_minimal_export_graph())

        assert onnx_calls == []

    def test_the_home_directory_is_expanded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A shell expands ``~`` but Python does not; left alone it would create a directory named ``~``."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # Should the expansion break, the relative "~/..." must land in the test's folder, not the repository.
        monkeypatch.chdir(tmp_path)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="~/engine.cache")).build_engine("model.onnx")

        assert Path(captured["build"]["save_timing_cache"]) == tmp_path / "engine.cache"

    def test_a_relative_path_is_relative_to_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The most natural call, ``trt_timing_cache="rfdetr.cache"``, puts the cache where the command runs."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.chdir(tmp_path)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="engine.cache")).build_engine("out/model.onnx")

        assert (Path(captured["build"]["save_timing_cache"]), (tmp_path / "engine.cache.lock").is_file()) == (
            tmp_path / "engine.cache",
            True,
        )


def _tensorrt_with_hardware_levels(version: str, **levels: str) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module whose ``HardwareCompatibilityLevel`` has exactly the given members.

    Each member is a distinct string, so a wrong mapping from the ``export`` keyword to the enum member fails an
    equality check instead of passing by coincidence.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        **levels: Enum member name to the sentinel value it carries.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> module = _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        >>> module.HardwareCompatibilityLevel.AMPERE_PLUS
        'ampere-plus-level'
        >>> hasattr(module.HardwareCompatibilityLevel, "SAME_COMPUTE_CAPABILITY")
        False
    """
    module = _fake_tensorrt(version, has_fp16_flag=True)
    module.HardwareCompatibilityLevel = types.SimpleNamespace(**levels)
    return module


def _literal_spellings(annotation: object) -> set[str]:
    """Return the string values of a ``Literal[...] | None`` annotation, ignoring the ``None``.

    Args:
        annotation: A resolved annotation such as ``Literal["a", "b"] | None``.

    Returns:
        The values of the ``Literal`` member.

    Examples:
        >>> from typing import Literal
        >>> sorted(_literal_spellings(Literal["a", "b"] | None))
        ['a', 'b']
    """
    return {value for member in get_args(annotation) if member is not type(None) for value in get_args(member)}


def _cannot_load(name: str) -> None:
    """Stand in for the dynamic loader failing to find the library *name*.

    Args:
        name: The library file name the loader was asked for.

    Raises:
        OSError: Always, naming the library.

    Examples:
        >>> _cannot_load("libnvinfer_lean.so.11")
        Traceback (most recent call last):
        ...
        OSError: libnvinfer_lean.so.11: cannot open shared object file
    """
    raise OSError(f"{name}: cannot open shared object file")


class TestPortableEngines:
    """The compatibility settings reach Polygraphy's config, and nothing is passed unless one is set."""

    @pytest.fixture(autouse=True)
    def lean_runtime(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Pretend the lean runtime library loads, and record the library names TensorRT was asked for.

        The pip package is replaced too: importing the real one while ``CDLL`` is faked would leave it in
        ``sys.modules`` without its libraries, and a real version-compatible build later in the same session would then
        be refused.
        """
        loaded: list[str] = []
        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", loaded.append)
        monkeypatch.setitem(sys.modules, "tensorrt_lean_libs", types.ModuleType("tensorrt_lean_libs"))
        return loaded

    def test_no_compatibility_keyword_is_passed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without either setting the build calls Polygraphy exactly as it always did."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False}

    @pytest.mark.parametrize(
        ("level", "expected"), [("ampere_plus", "ampere-plus-level"), ("same_compute_capability", "same-cc-level")]
    )
    def test_hardware_compatibility_selects_the_matching_enum_member(
        self, monkeypatch: pytest.MonkeyPatch, level: str, expected: str
    ) -> None:
        """Each keyword value maps to its own ``HardwareCompatibilityLevel`` member."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "tensorrt",
            _tensorrt_with_hardware_levels(
                "11.3.0.99", AMPERE_PLUS="ampere-plus-level", SAME_COMPUTE_CAPABILITY="same-cc-level"
            ),
        )

        TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility=level)).build_engine("model.onnx")

        assert captured["config"]["hardware_compatibility_level"] == expected

    def test_version_compatible_reaches_the_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``version_compatible=True`` is passed to ``CreateConfig``; the default passes no such keyword."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert captured["config"] == {"fp16": False, "version_compatible": True}

    @_requires_tensorrt
    def test_the_installed_polygraphy_create_config_accepts_the_portability_keywords(self) -> None:
        """The real ``CreateConfig`` still takes the two keywords the exporter passes for portable engines.

        The unit tests above record whatever keyword the exporter passes, so a Polygraphy release that renamed one would
        go unnoticed there; this reads the installed signature instead.
        """
        from polygraphy.backend.trt import CreateConfig

        parameters = inspect.signature(CreateConfig).parameters

        assert {"version_compatible", "hardware_compatibility_level"} <= parameters.keys()

    @pytest.mark.parametrize("dynamic_batch", [False, True])
    def test_both_settings_reach_the_builder_next_to_the_profile(
        self, monkeypatch: pytest.MonkeyPatch, dynamic_batch: bool
    ) -> None:
        """The compatibility keywords travel with the batch profile, for a static and a dynamic build alike."""
        shape = (-1, 3, 384, 384) if dynamic_batch else (1, 3, 384, 384)
        captured = _patch_polygraphy_chain_recording(monkeypatch, _FakeNetwork(_FakeNetworkInput("input", shape)))
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        config = TensorRTConfig(
            fp16=False,
            dynamic_batch=dynamic_batch,
            max_batch_size=4,
            hardware_compatibility="ampere_plus",
            version_compatible=True,
        )

        TensorRTExporter(config).build_engine("model.onnx")

        assert captured["config"]["hardware_compatibility_level"] == "ampere-plus-level"
        assert captured["config"]["version_compatible"] is True
        assert ("profiles" in captured["config"]) is dynamic_batch

    def test_a_tensorrt_without_the_level_refuses_before_building(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A TensorRT that lacks the requested level names it and its own version, and builds nothing."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        )
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility="same_compute_capability"))

        with pytest.raises(ValueError, match=r"same_compute_capability.*TensorRT 10\.16\.1\.11"):
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}

    def test_ampere_plus_is_refused_on_a_pre_ampere_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A GPU older than Ampere cannot build ``ampere_plus``; the refusal names its compute capability.

        TensorRT itself fails only inside the build, after the forward pass and the ONNX export; the check reads the
        current CUDA device first, so a Turing card (7.5) is turned away before any of that.
        """
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

        with pytest.raises(ValueError, match=r"ampere_plus.*compute capability 7\.5"):
            TensorRTExporter(TensorRTConfig(hardware_compatibility="ampere_plus")).check_environment()

    @pytest.mark.parametrize(
        ("level", "capability"),
        [
            pytest.param("ampere_plus", (8, 0), id="ampere_plus-on-ampere"),
            pytest.param("ampere_plus", (12, 0), id="ampere_plus-on-blackwell"),
            pytest.param("same_compute_capability", (7, 5), id="same_compute_capability-on-turing"),
        ],
    )
    def test_the_gpu_check_passes_where_the_level_can_be_built(
        self, monkeypatch: pytest.MonkeyPatch, level: str, capability: tuple[int, int]
    ) -> None:
        """Ampere and newer pass ``ampere_plus``, and ``same_compute_capability`` has no minimum GPU at all."""
        monkeypatch.setitem(
            sys.modules,
            "tensorrt",
            _tensorrt_with_hardware_levels(
                "11.3.0.99", AMPERE_PLUS="ampere-plus-level", SAME_COMPUTE_CAPABILITY="same-cc-level"
            ),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)

        TensorRTExporter(TensorRTConfig(hardware_compatibility=level)).check_environment()

    def test_ampere_plus_is_not_checked_against_a_gpu_on_a_host_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a visible CUDA device there is no capability to compare, so the request is let through."""
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "get_device_capability", _cannot_load)

        TensorRTExporter(TensorRTConfig(hardware_compatibility="ampere_plus")).check_environment()

    @pytest.mark.parametrize("level", [pytest.param("", id="empty"), "AMPERE_PLUS", "ampere", "none", 3, True])
    def test_an_unknown_hardware_level_is_refused(self, level: object) -> None:
        """Only the documented spellings are accepted, and the refusal lists them, before any work on the model."""
        with pytest.raises(ValueError, match=r"hardware_compatibility.*ampere_plus.*same_compute_capability"):
            TensorRTExporter(TensorRTConfig(hardware_compatibility=level))

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_version_compatible_must_be_a_bool(self, value: object) -> None:
        """A truthy non-``bool`` would enable the mode by accident, so it is refused."""
        with pytest.raises(ValueError, match="version_compatible"):
            TensorRTExporter(TensorRTConfig(version_compatible=value))

    @pytest.mark.parametrize(
        ("major", "platform", "expected"),
        [
            (10, "linux", "libnvinfer_lean.so.10"),
            (10, "win32", "nvinfer_lean_10.dll"),
            (8, "win32", "nvinfer_lean.dll"),
        ],
    )
    def test_the_lean_library_name_follows_the_platform_and_major_version(
        self, major: int, platform: str, expected: str
    ) -> None:
        """The runtime library is named by TensorRT's major version, and differently on Windows (without it on 8.6)."""
        assert tensorrt_export._lean_library_name(major, platform) == expected

    def test_on_windows_the_lean_library_is_looked_up_on_path(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """A bare DLL name is not searched on PATH, where a TensorRT zip install puts the library; the full path is."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setattr(tensorrt_export.sys, "platform", "win32")
        found = "C:/TensorRT/lib/nvinfer_lean_10.dll"
        monkeypatch.setattr(tensorrt_export.ctypes.util, "find_library", lambda name: found)

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert lean_runtime == [found]

    @pytest.mark.parametrize("level", [None, "ampere_plus"])
    def test_the_lean_runtime_is_not_probed_unless_version_compatible_is_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str], level: str | None
    ) -> None:
        """Only a version-compatible build embeds the lean runtime; a hardware-compatible one must not need it."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("11.3.0.99", AMPERE_PLUS="ampere-plus-level")
        )

        TensorRTExporter(TensorRTConfig(fp16=False, hardware_compatibility=level)).build_engine("model.onnx")

        assert lean_runtime == []

    def test_an_unparseable_tensorrt_version_skips_the_lean_probe(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """The library is named by the major version; without one there is no name to look for, and the build goes
        on."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("unknown", has_fp16_flag=True))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        assert lean_runtime == []

    def test_version_compatibility_warns_before_the_forward_pass_where_it_was_not_seen_to_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A TensorRT 10 request is warned about from ``check_environment``, which runs before the model does.

        It was verified between TensorRT 11 releases only; warning from the build instead would reach the user only
        after the forward pass and the ONNX export had already been paid for.
        """
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))

        with pytest.warns(UserWarning, match=r"verified between TensorRT 11.*TensorRT 10\.16\.1\.11"):
            TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).check_environment()

    def test_version_compatibility_does_not_warn_on_tensorrt_11(
        self, monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Between TensorRT 11 releases the mode was seen to work, so the request goes through without a warning."""
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.3.0.99", has_fp16_flag=True))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).check_environment()

        assert [str(w.message) for w in recwarn if "verified between TensorRT 11" in str(w.message)] == []

    def test_the_lean_runtime_package_is_imported_before_the_library_is_probed(
        self, monkeypatch: pytest.MonkeyPatch, lean_runtime: list[str]
    ) -> None:
        """The pip package loads its libraries when imported, so it has to come first or the probe would miss it."""
        _patch_polygraphy_chain_recording(monkeypatch)
        events: list[str] = []
        real_import = tensorrt_export.importlib.import_module

        def _import(name: str, *args: object, **kwargs: object) -> object:
            events.append(f"import {name}")
            return real_import(name, *args, **kwargs) if name != "tensorrt_lean_libs" else types.ModuleType(name)

        monkeypatch.setattr(tensorrt_export.importlib, "import_module", _import)
        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", lambda name: events.append(f"load {name}"))

        TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True)).build_engine("model.onnx")

        expected = tensorrt_export._lean_library_name(10, sys.platform)
        # Other imports may pass through the patched ``import_module``, so only the two relevant events are compared.
        relevant = [event for event in events if event in ("import tensorrt_lean_libs", f"load {expected}")]
        assert relevant == ["import tensorrt_lean_libs", f"load {expected}"]

    def test_a_missing_lean_runtime_is_refused_with_the_install_hint_and_builds_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the library TensorRT fails with a bare "Invalid Engine"; the refusal names the package instead."""
        captured = _patch_polygraphy_chain_recording(monkeypatch)

        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", _cannot_load)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, version_compatible=True))

        with pytest.raises(ImportError, match=r"lean runtime.*tensorrt-lean-cu\*-libs") as refusal:
            exporter.build_engine("model.onnx")

        assert captured == {"config": {}, "build": {}}
        assert isinstance(refusal.value.__cause__, OSError), "the loader's own error stays attached"

    def test_a_missing_lean_runtime_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``Exporter.__call__`` checks first, so the user is not refused only after the ONNX stage has run."""
        _patch_polygraphy_chain_recording(monkeypatch)
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )

        monkeypatch.setattr(tensorrt_export.ctypes, "CDLL", _cannot_load)

        with pytest.raises(ImportError, match="lean runtime"):
            TensorRTExporter(TensorRTConfig(version_compatible=True))(_minimal_export_graph())

        assert onnx_calls == []

    def test_a_missing_hardware_level_is_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A TensorRT without the requested level is found out first, not after the ONNX stage has run."""
        _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "tensorrt", _tensorrt_with_hardware_levels("10.16.1.11", AMPERE_PLUS="ampere-plus-level")
        )
        onnx_calls: list[str] = []
        monkeypatch.setattr(
            "rfdetr.export._onnx.exporter.OnnxExporter._convert",
            lambda self, graph: onnx_calls.append("called") or str(tmp_path / "model.onnx"),
        )
        config = TensorRTConfig(hardware_compatibility="same_compute_capability")

        with pytest.raises(ValueError, match="same_compute_capability"):
            TensorRTExporter(config)(_minimal_export_graph())

        assert onnx_calls == []

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(lambda exporter: exporter(_minimal_export_graph()), id="call"),
            pytest.param(lambda exporter: exporter.build_engine("model.onnx"), id="build_engine"),
        ],
    )
    def test_a_host_without_tensorrt_is_named_before_the_portability_check(
        self, monkeypatch: pytest.MonkeyPatch, run: Callable[[TensorRTExporter], object]
    ) -> None:
        """The packages are checked before the configuration, whose lean probe would import the missing TensorRT."""
        monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "tensorrt", None)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            run(TensorRTExporter(TensorRTConfig(version_compatible=True)))

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            pytest.param({}, "out/model_fp16.trt", id="default"),
            pytest.param({"hardware_compatibility": "ampere_plus"}, "out/model_fp16_ampere_plus.trt", id="ampere-plus"),
            pytest.param(
                {"hardware_compatibility": "same_compute_capability"},
                "out/model_fp16_same_compute_capability.trt",
                id="same-cc",
            ),
            pytest.param(
                {"version_compatible": True}, "out/model_fp16_version_compatible.trt", id="version-compatible"
            ),
            pytest.param(
                {"hardware_compatibility": "ampere_plus", "version_compatible": True},
                "out/model_fp16_ampere_plus_version_compatible.trt",
                id="both",
            ),
            pytest.param(
                {"fp16": False, "hardware_compatibility": "ampere_plus"}, "out/model_fp32_ampere_plus.trt", id="fp32"
            ),
            pytest.param({"version_compatible": True, "output_name": "mine"}, "out/mine.trt", id="custom-output-name"),
        ],
    )
    def test_a_portable_engine_gets_its_own_file_name(self, options: dict, expected: str) -> None:
        """A portable engine must not overwrite (or be overwritten by) a default one built in the same directory.

        The default name stays exactly what it was; each option that is on adds a detail to it, following the precision
        whichever it is. A custom ``output_name`` already says what the file is, so no detail is appended to it.
        """
        path = TensorRTExporter(TensorRTConfig(**{"fp16": True, **options})).build_engine(
            "out/model.onnx", dry_run=True
        )

        assert path == expected

    def test_a_missing_level_is_refused_before_the_fp16_graph_is_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``build_engine`` checks the level first, so a doomed FP16 build on a strongly typed TensorRT casts
        nothing."""
        _patch_polygraphy_chain_recording(monkeypatch)
        strongly_typed = _fake_tensorrt("11.3.0.99", has_fp16_flag=False)
        strongly_typed.HardwareCompatibilityLevel = types.SimpleNamespace(AMPERE_PLUS="ampere-plus-level")
        monkeypatch.setitem(sys.modules, "tensorrt", strongly_typed)
        casts: list[str] = []
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path, **_: casts.append(path) or path)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, hardware_compatibility="same_compute_capability"))

        with pytest.raises(ValueError, match="same_compute_capability"):
            exporter.build_engine("model.onnx")

        assert casts == []

    def test_the_export_keyword_lists_the_levels_the_validator_accepts(self) -> None:
        """``RFDETR.export`` types the levels by hand; they must not drift from the spellings the exporter accepts."""
        spellings = set(tensorrt_export._HARDWARE_COMPATIBILITY_LEVELS)

        assert _literal_spellings(get_type_hints(RFDETR.export)["trt_hardware_compatibility"]) == spellings

    def test_the_settings_are_read_from_the_export_keywords(self) -> None:
        """``RFDETR.export``'s ``trt_``-prefixed keywords reach the configuration."""
        config = TensorRTExporter.build_config(trt_hardware_compatibility="ampere_plus", trt_version_compatible=True)

        assert (config.hardware_compatibility, config.version_compatible) == ("ampere_plus", True)

    def test_the_settings_are_off_by_default(self) -> None:
        """Without the keywords the configuration asks for no portability."""
        config = TensorRTExporter.build_config()

        assert (config.hardware_compatibility, config.version_compatible) == (None, False)


class TestTimingCacheSafety:
    """The timing cache Polygraphy is handed is one well-defined regular file, and a bad one is refused up front."""

    @_requires_symlink_privilege
    def test_a_link_to_an_existing_cache_hands_polygraphy_its_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A link and the file it points to load, save and lock the same cache file.

        Polygraphy locks ``<path>.lock`` beside the path it is given. Handed the link itself, an export through the link
        and another through the target would each lock their own file and race the same cache's read-merge-write.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        cache = tmp_path / "engine.cache"
        cache.write_bytes(b"previous timings")
        link = tmp_path / "link.cache"
        link.symlink_to(cache)

        TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=link)).build_engine("model.onnx")

        assert (captured["config"]["load_timing_cache"], captured["build"], sorted(os.listdir(tmp_path))) == (
            str(cache),
            {"save_timing_cache": str(cache)},
            ["engine.cache", "engine.cache.lock", "link.cache"],
        )

    @_requires_mkfifo
    def test_a_named_pipe_is_refused_before_anything_is_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A FIFO cannot hold the cache, and it is refused before the build without ever being opened.

        ``os.path.isfile`` is false for a FIFO, so it used to pass as a cache that does not exist yet; Polygraphy's save
        into it would then block, with no reader, after the whole build.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        fifo = tmp_path / "engine.cache"
        os.mkfifo(fifo)
        exporter = TensorRTExporter(TensorRTConfig(fp16=False, timing_cache=fifo))

        with pytest.raises(ValueError, match="trt_timing_cache must name a regular file"):
            exporter.build_engine("model.onnx")

        assert (captured, sorted(os.listdir(tmp_path))) == ({"config": {}, "build": {}}, ["engine.cache"])

    def test_a_path_with_a_nul_byte_is_refused_when_the_exporter_is_built(self) -> None:
        """No file can carry a NUL byte in its name, so the value is refused before any work on the model.

        Left to the filesystem, the NUL byte surfaced as a bare ``ValueError('embedded null byte')`` that named no
        setting.
        """
        with pytest.raises(ValueError, match="trt_timing_cache must not contain a NUL byte"):
            TensorRTExporter(TensorRTConfig(fp16=False, timing_cache="engine\x00.cache"))

    @pytest.mark.parametrize(
        ("cache_name", "has_fp16_flag"),
        [
            pytest.param("model.onnx", True, id="onnx-model"),
            pytest.param("model_fp16.trt", True, id="fp16-engine"),
            pytest.param("model_fp32.trt", False, id="fp32-fallback-engine"),
        ],
    )
    def test_a_cache_that_is_the_model_or_the_engine_is_refused_before_anything_is_created(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache_name: str, has_fp16_flag: bool
    ) -> None:
        """A cache aliasing the build's ONNX model or its engine is refused before the build, and before its lock.

        Saved over the ONNX model, the cache would destroy the export's own input; saved where the engine goes, it would
        be overwritten by the engine on every run. A TensorRT without the FP16 builder flag falls back to an FP32
        engine, so the comparison has to use the engine's final name, not the requested one.
        """
        captured = _patch_polygraphy_chain_recording(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=has_fp16_flag))
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"exported model")
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, timing_cache=tmp_path / cache_name))

        with pytest.raises(ValueError, match="trt_timing_cache .* is the same file as"):
            exporter.build_engine(str(onnx_path))

        assert (captured, sorted(os.listdir(tmp_path))) == ({"config": {}, "build": {}}, ["model.onnx"])
