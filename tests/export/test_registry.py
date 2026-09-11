# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the export format registry and the per-format configuration it is paired with.

The registry is the only thing that knows a format exists before that format's optional dependency is imported, so these
tests cover both what it answers without importing (aliases, capabilities, unknown formats) and that those answers still
agree with the exporter classes once they are imported.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from rfdetr.export import registry as registry_module
from rfdetr.export._coreml.exporter import CoreMLConfig
from rfdetr.export._executorch.exporter import ExecuTorchExporter
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export._openvino.exporter import OpenVINOConfig
from rfdetr.export._tensorrt.exporter import TensorRTConfig
from rfdetr.export._tflite.exporter import TFLiteConfig
from rfdetr.export.base import ExportConfig, Exporter, reject_unsupported_dynamic_batch
from rfdetr.export.registry import ALIASES, REGISTRY, normalize_format, resolve_exporter


def _resolve_or_skip(format: str) -> type[Exporter[ExportConfig]]:
    """Return *format*'s exporter class, skipping the test when its optional dependency is absent.

    Args:
        format: Canonical format name.

    Returns:
        The exporter class registered for *format*.

    Examples:
        >>> _resolve_or_skip("onnx").format
        'onnx'
    """
    try:
        return resolve_exporter(format)
    except ImportError:
        pytest.skip(f"optional dependencies for format={format!r} are not installed")


class TestNormalizeFormat:
    """``normalize_format`` maps the short spellings users type onto canonical format names."""

    @pytest.mark.parametrize("alias, canonical", sorted(ALIASES.items()))
    def test_alias_resolves_to_canonical_name(self, alias: str, canonical: str) -> None:
        """Every registered alias resolves to a format the registry actually knows.

        An alias pointing at a name absent from ``REGISTRY`` would surface as an "unsupported format" error for a
        spelling the public docstring advertises.
        """
        assert normalize_format(alias) in REGISTRY

    def test_canonical_name_passes_through(self) -> None:
        """A name that is already canonical is returned unchanged."""
        assert normalize_format("onnx") == "onnx"

    def test_unknown_name_passes_through_for_the_resolver_to_reject(self) -> None:
        """An unknown spelling is not rewritten, so the resolver can name it back to the user verbatim."""
        assert normalize_format("nonesuch") == "nonesuch"


class TestResolveExporter:
    """``resolve_exporter`` turns a format name into its exporter class, importing nothing else."""

    def test_unknown_format_raises_value_error_listing_the_known_ones(self) -> None:
        """An unrecognized format must fail with the list of accepted names, not an ImportError or a KeyError."""
        with pytest.raises(ValueError, match="Unsupported export format 'nonesuch'"):
            resolve_exporter("nonesuch")

    def test_missing_optional_dependency_reports_the_install_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A format whose module cannot be imported re-raises, after naming the extra that installs it.

        Every optional dependency is imported lazily, so this is the one place a user finds out which ``pip install
        rfdetr[...]`` they are missing; swallowing the hint would leave them with a bare ImportError naming a third-
        party module they never asked for.
        """
        entry = REGISTRY["coreml"]
        monkeypatch.setitem(REGISTRY, "coreml", replace(entry, module="rfdetr.export._nonexistent_module"))
        errors: list[str] = []
        monkeypatch.setattr(registry_module.logger, "error", lambda message, *_a, **_kw: errors.append(message))

        with pytest.raises(ImportError):
            resolve_exporter("coreml")

        assert "pip install rfdetr[coreml]" in "".join(errors)

    @pytest.mark.parametrize("format", sorted(REGISTRY))
    def test_registry_capabilities_match_the_exporter_class(self, format: str) -> None:
        """The registry's pre-import copy of a format's capabilities must agree with the class it names.

        The duplication exists so an impossible request can be refused — in the format's own words — before importing a
        heavy optional dependency. It is only safe while the two stay in sync, and nothing but this test enforces that.
        """
        entry = REGISTRY[format]
        try:
            exporter_class = resolve_exporter(format)
        except ImportError:
            pytest.skip(f"optional dependencies for format={format!r} are not installed")

        assert (
            exporter_class.format,
            exporter_class.display_name,
            exporter_class.supports_dynamic_batch,
            exporter_class.dynamic_batch_reason,
        ) == (format, entry.label, entry.supports_dynamic_batch, entry.dynamic_batch_reason)


class TestRejectUnsupportedDynamicBatch:
    """The pre-import guard that refuses ``dynamic_batch`` for formats which bake a fixed input shape."""

    @pytest.mark.parametrize("format", sorted(f for f, e in REGISTRY.items() if not e.supports_dynamic_batch))
    def test_fixed_shape_formats_are_refused(self, format: str) -> None:
        """A format that bakes a fixed shape must refuse, naming the format and what to do instead.

        Parametrized from the registry rather than a hand-written list so a newly registered fixed-batch format is
        covered the moment it is added — including the one mistake this guard exists to prevent, a format declaring
        ``supports_dynamic_batch=False`` without saying why.
        """
        with pytest.raises(NotImplementedError, match="dynamic_batch") as refusal:
            reject_unsupported_dynamic_batch(format, dynamic_batch=True)

        assert REGISTRY[format].label in str(refusal.value)
        assert REGISTRY[format].dynamic_batch_reason in str(refusal.value)
        assert REGISTRY[format].dynamic_batch_reason, "a fixed-batch format must explain what to do instead"

    @pytest.mark.parametrize("format", sorted(f for f, e in REGISTRY.items() if e.supports_dynamic_batch))
    def test_dynamic_capable_formats_are_allowed(self, format: str) -> None:
        """A format that can carry a dynamic batch dimension passes through silently."""
        reject_unsupported_dynamic_batch(format, dynamic_batch=True)

    def test_static_request_is_never_refused(self) -> None:
        """Without ``dynamic_batch`` the guard does nothing, even for a format that could not honour it."""
        reject_unsupported_dynamic_batch("coreml", dynamic_batch=False)

    def test_unknown_format_raises_value_error(self) -> None:
        """An unknown format is reported as such rather than silently accepted."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            reject_unsupported_dynamic_batch("nonesuch", dynamic_batch=True)


class TestBuildConfig:
    """``Exporter.build_config`` narrows ``RFDETR.export()``'s flat keyword arguments to one format's settings."""

    @pytest.mark.parametrize(
        "format, expected_type",
        [
            pytest.param("onnx", OnnxConfig, id="onnx"),
            pytest.param("openvino", OpenVINOConfig, id="openvino"),
            pytest.param("coreml", CoreMLConfig, id="coreml"),
            pytest.param("tflite", TFLiteConfig, id="tflite"),
            pytest.param("tensorrt", TensorRTConfig, id="tensorrt"),
        ],
    )
    def test_builds_the_configuration_class_the_exporter_declares(self, format: str, expected_type: type) -> None:
        """Each format gets its own configuration type, so a knob belonging to another format cannot be set."""
        exporter_class = _resolve_or_skip(format)
        assert isinstance(exporter_class.build_config(output_dir=Path("out")), expected_type)

    def test_executorch_requires_a_backend(self) -> None:
        """``format="executorch"`` without a resolved backend is an error, not a silent xnnpack default.

        The backend decides what hardware can load the ``.pte``, so defaulting quietly would hand back an artifact for
        the wrong target instead of failing.
        """
        with pytest.raises(ValueError, match="requires a backend"):
            ExecuTorchExporter.build_config(output_dir=Path("out"))

    def test_executorch_keeps_the_resolved_backend_and_soc(self) -> None:
        """A resolved backend and SoC reach the configuration unchanged."""
        config = ExecuTorchExporter.build_config(output_dir=Path("out"), backend="qnn", soc="SM8650")
        assert (config.backend, config.soc) == ("qnn", "SM8650")

    @pytest.mark.parametrize(
        "format, kwargs, attribute, expected",
        [
            pytest.param("onnx", {"opset_version": 18}, "opset_version", 18, id="onnx_opset"),
            pytest.param("openvino", {"openvino_precision": "float32"}, "precision", "float32", id="openvino"),
            pytest.param("coreml", {"coreml_precision": "float16"}, "compute_precision", "float16", id="coreml"),
            pytest.param("tflite", {"quantization": "int8"}, "quantization", "int8", id="tflite"),
            pytest.param("tensorrt", {"fp16": False}, "fp16", False, id="tensorrt"),
        ],
    )
    def test_format_specific_setting_reaches_its_configuration(
        self, format: str, kwargs: dict, attribute: str, expected: object
    ) -> None:
        """The knob belonging to a format lands on that format's configuration under its own name."""
        config = _resolve_or_skip(format).build_config(output_dir=Path("out"), **kwargs)
        assert getattr(config, attribute) == expected

    def test_settings_belonging_to_other_formats_are_dropped(self) -> None:
        """A keyword another format reads must not reach this one, which has no field to put it in.

        ``RFDETR.export()`` has one signature covering every format, so each exporter is handed the union of all their
        keywords and has to take only its own.
        """
        config = OnnxExporter.build_config(output_dir=Path("out"), opset_version=18, fp16=False, quantization="int8")
        assert config.opset_version == 18
        assert not hasattr(config, "fp16")


class TestOnnxStageHandoff:
    """The intermediate ONNX configuration the two-stage formats export through."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                TFLiteConfig(
                    output_dir=Path("out"),
                    output_name="my-model",
                    variant_name="rfdetr-small",
                    backbone_only=True,
                    dynamic_batch=True,
                    verbose=False,
                    notes="provenance",
                    opset_version=18,
                ),
                id="tflite",
            ),
            pytest.param(
                TensorRTConfig(
                    output_dir=Path("out"),
                    output_name="my-model",
                    variant_name="rfdetr-small",
                    backbone_only=True,
                    dynamic_batch=True,
                    verbose=False,
                    notes="provenance",
                    opset_version=18,
                ),
                id="tensorrt",
            ),
        ],
    )
    def test_every_shared_setting_survives_the_handoff(self, config: TFLiteConfig | TensorRTConfig) -> None:
        """The ONNX stage inherits naming, shape, verbosity and notes from the format that runs it.

        The intermediate ``.onnx`` is what the second stage reads and names its own artifact after, so a setting dropped
        here changes the final filename or silently strips the user's embedded provenance metadata.
        """
        stage = config.onnx_stage()

        assert (
            stage.output_dir,
            stage.output_name,
            stage.variant_name,
            stage.backbone_only,
            stage.dynamic_batch,
            stage.verbose,
            stage.notes,
            stage.opset_version,
        ) == (Path("out"), "my-model", "rfdetr-small", True, True, False, "provenance", 18)


class _CapabilityProbe(Exporter[ExportConfig]):
    """Exporter subclass declaring no capabilities, used to exercise the shared checks in isolation.

    Examples:
        >>> _CapabilityProbe.supports_notes
        False
    """

    format = "probe"
    display_name = "Probe"

    def _convert(self, graph: object) -> Path:
        """Return a fixed path; these tests never reach the conversion itself."""
        return Path("probe")


class TestExporterCapabilityChecks:
    """The shared capability checks every exporter inherits, exercised without any optional dependency.

    Each format used to carry its own copy of these guards. They now live once on the base class, so a mistake here
    reaches all six formats — and the branches are cheapest to pin on a subclass that declares nothing.
    """

    def test_unsupported_dynamic_batch_is_rejected_at_construction(self) -> None:
        """A format that cannot carry a dynamic batch dimension refuses before any model work is done."""
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            _CapabilityProbe(ExportConfig(output_dir=Path("out"), dynamic_batch=True))

    def test_notes_are_dropped_with_a_warning_when_the_artifact_has_no_metadata_slot(self) -> None:
        """Supplying notes to a format that cannot embed them warns and names the format, rather than failing."""
        with pytest.warns(UserWarning, match=r"`notes` is not forwarded to format='probe'"):
            _CapabilityProbe(ExportConfig(output_dir=Path("out"), notes="provenance"))

    def test_no_warning_when_notes_are_absent(self) -> None:
        """A format without a metadata slot stays silent when the caller supplied no notes."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _CapabilityProbe(ExportConfig(output_dir=Path("out")))

    def test_experimental_formats_warn_on_construction(self) -> None:
        """A format marked experimental says so, including the note explaining what is unstable about it.

        The warning is the only signal a user gets that an artifact comes from a work-in-progress path; it moved to the
        shared base class in this refactor, so nothing per-format covers it any more.
        """

        class _ExperimentalProbe(_CapabilityProbe):
            experimental = True
            experimental_note = "Upstream dependencies may affect results."

        with pytest.warns(UserWarning, match="Probe export is experimental and work-in-progress"):
            _ExperimentalProbe(ExportConfig(output_dir=Path("out")))
