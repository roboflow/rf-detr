# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for ``rfdetr.export._naming.resolve_export_stem``.

``resolve_export_stem`` is the single shared precedence/sanitization helper the ONNX, CoreML, ExecuTorch, and TensorRT
backends all call before naming their output artifact. It was previously exercised only indirectly, through each
backend's own filename-assertion tests (e.g. ``tests/export/test_export.py::TestExportOnnxVariantNaming``) — this
module covers the helper directly, once, so its precedence/sanitization contract does not need re-verifying per
backend.
"""

from __future__ import annotations

import pytest

from rfdetr.export._naming import resolve_export_stem


class TestResolveExportStemPrecedence:
    """Precedence contract: ``output_name`` > ``variant_name`` > ``default``, and ``is_custom`` only for
    ``output_name``."""

    @pytest.mark.parametrize(
        ("variant_name", "output_name", "kwargs", "expected"),
        [
            pytest.param("rfdetr-small", "my-model", {}, ("my-model", True), id="output_name_wins_over_variant_name"),
            pytest.param("rfdetr-nano", "my-model", {}, ("my-model", True), id="output_name_with_other_variant_name"),
            pytest.param(None, "my-model", {}, ("my-model", True), id="output_name_alone"),
            pytest.param("rfdetr-small", None, {}, ("rfdetr-small", False), id="variant_name_when_no_output_name"),
            pytest.param("rfdetr-nano", None, {}, ("rfdetr-nano", False), id="variant_name_alone"),
            pytest.param(None, None, {}, ("inference_model", False), id="generic_default_when_both_none"),
            pytest.param(None, None, {"default": "backbone_model"}, ("backbone_model", False), id="custom_default"),
            pytest.param(
                "rfdetr-nano",
                None,
                {"default": "backbone_model"},
                ("rfdetr-nano", False),
                id="default_ignored_for_variant",
            ),
            pytest.param(
                None, "my-model", {"default": "backbone_model"}, ("my-model", True), id="default_ignored_for_output"
            ),
        ],
    )
    def test_stem_and_is_custom_follow_precedence(
        self, variant_name: str | None, output_name: str | None, kwargs: dict[str, str], expected: tuple[str, bool]
    ) -> None:
        """The stem comes from ``output_name``, else ``variant_name``, else ``default``; a custom ``default`` only fires
        when both names are unset, and ``is_custom`` is ``True`` if and only if ``output_name`` supplied the stem."""
        assert resolve_export_stem(variant_name, output_name, **kwargs) == expected


class TestResolveExportStemCrossPlatformPathComponents:
    """``_sanitize()`` reduces a path-like input to a bare basename stem, extension stripped.

    ``_sanitize`` splits on both ``/`` and ``\\`` unconditionally (``re.split(r"[\\\\/]", name)``), so directory
    components using either separator style are stripped identically on every host platform — a POSIX host handling a
    Windows-style path (or vice versa) never leaks a foreign separator into the resolved stem.
    """

    @pytest.mark.parametrize(
        ("output_name", "expected"),
        [
            ("sub/dir/rfdetr-nano.onnx", "rfdetr-nano"),
            ("../../etc/passwd", "passwd"),
            ("/absolute/unix/path/model", "model"),
            (r"..\..\evil.onnx", "evil"),
            (r"C:\Users\name\model.onnx", "model"),
            ("mixed/dir\\model.onnx", "model"),
        ],
    )
    def test_directory_components_are_stripped_on_every_platform(self, output_name: str, expected: str) -> None:
        """Forward-slash and backslash directory components, plus the extension, are always stripped — on every OS,
        regardless of which separator style the input uses."""
        assert resolve_export_stem(None, output_name) == (expected, True)


class TestResolveExportStemDotsInName:
    """Only an export artifact extension is stripped; any other dot is part of the name and survives."""

    @pytest.mark.parametrize(
        "output_name",
        ["rfdetr-nano-v1.2", "detector_2026.10.17", "model.final", "rfdetr-nano-v1.2-backbone"],
    )
    def test_dot_that_is_not_an_artifact_extension_is_kept(self, output_name: str) -> None:
        """A version, date or other dotted name is used verbatim, as ``RFDETR.export(output_name=...)`` documents."""
        assert resolve_export_stem(None, output_name) == (output_name, True)

    @pytest.mark.parametrize(
        ("output_name", "expected"),
        [
            ("rfdetr-nano-v1.2.onnx", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.trt", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.engine", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.tflite", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.mlpackage", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.pte", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.xml", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.aimodel", "rfdetr-nano-v1.2"),
            ("rfdetr-nano-v1.2.ONNX", "rfdetr-nano-v1.2"),
        ],
    )
    def test_artifact_extension_is_stripped_from_a_dotted_name(self, output_name: str, expected: str) -> None:
        """An artifact extension the caller appended is still removed, leaving the dots before it alone."""
        assert resolve_export_stem(None, output_name) == (expected, True)

    def test_dotted_variant_name_is_kept(self) -> None:
        """``variant_name`` goes through the same sanitizer as ``output_name``."""
        assert resolve_export_stem("rfdetr-nano-v1.2", None) == ("rfdetr-nano-v1.2", False)


class TestResolveExportStemEmptyAndWhitespace:
    """Empty / whitespace-only / all-separator ``output_name`` values."""

    @pytest.mark.parametrize(
        ("variant_name", "expected_stem"),
        [
            (None, "inference_model"),
            ("rfdetr-nano", "rfdetr-nano"),
        ],
    )
    def test_empty_string_falls_back_to_variant_name_or_default(
        self, variant_name: str | None, expected_stem: str
    ) -> None:
        """An empty ``output_name`` is falsy, so it is treated like ``None`` and never shadows ``variant_name``."""
        assert resolve_export_stem(variant_name, "") == (expected_stem, False)

    def test_whitespace_only_output_name_is_treated_as_a_custom_literal_stem(self) -> None:
        """Unlike the empty string, a whitespace-only ``output_name`` is truthy in Python — it is accepted as the custom
        stem verbatim (whitespace is not stripped by ``_sanitize()``, which only splits on path separators and the
        extension), so it is NOT rejected/rewritten to fall back to default."""
        assert resolve_export_stem(None, "   ") == ("   ", True)

    @pytest.mark.parametrize("all_separator_output_name", ["///", "/"])
    def test_all_separator_output_name_raises_value_error(self, all_separator_output_name: str) -> None:
        """A truthy ``output_name`` made entirely of path separators sanitizes to an empty stem, which
        ``resolve_export_stem`` now rejects with ``ValueError`` (PR #1250 item 4) rather than silently returning an
        empty, still-``is_custom=True`` stem that would previously have produced a bare-dotfile output filename."""
        with pytest.raises(ValueError, match="non-empty filename stem"):
            resolve_export_stem(None, all_separator_output_name)


class TestResolveExportStemUnicodeAndLength:
    """Unicode characters and very long ``output_name`` values."""

    def test_unicode_characters_are_preserved_verbatim(self) -> None:
        """Non-ASCII characters are valid on modern filesystems (APFS/most Linux filesystems are UTF-8 native, and
        NTFS/exFAT support Unicode filenames) and are not touched by ``_sanitize()``, which only strips directory
        components and the extension."""
        assert resolve_export_stem(None, "rfdetr-nano-日本語-\U0001f600") == (
            "rfdetr-nano-日本語-\U0001f600",
            True,
        )

    def test_unicode_extension_is_still_stripped(self) -> None:
        """``os.path.splitext`` is codepoint-based, not ASCII-only, so a unicode stem with a plain extension still has
        that extension removed."""
        assert resolve_export_stem(None, "ééé.onnx") == ("ééé", True)

    def test_long_output_name_over_255_chars_is_preserved(self) -> None:
        """``resolve_export_stem`` performs no length validation — a stem far past the traditional 255-byte filesystem
        filename limit is passed through unchanged.

        Whether the eventual write succeeds is a filesystem concern the caller (not this helper) is responsible for.
        """
        long_name = "a" * 300
        assert resolve_export_stem(None, long_name) == (long_name, True)


class TestResolveExportStemReservedNamesCharacterization:
    """Characterization tests: ``_sanitize()`` performs no validation against OS-reserved filename characters
    (``<>:"|?*``) or Windows-reserved device names (``CON``/``PRN``/``NUL``/etc.) — it only strips directory components
    and the extension.

    These document current behavior; they intentionally assert the pass-through, not a rejection. Per PR #1250 item
    11: not exploitable today (a Windows-reserved device name or invalid character only fails at the eventual
    filesystem write, on Windows, for callers who explicitly opt in via ``output_name``/``variant_name`` — this
    helper never touches the filesystem itself, so adding validation here is out of scope for this test suite).
    """

    @pytest.mark.parametrize("reserved_device_name", ["CON", "PRN", "NUL", "AUX", "COM1", "LPT1"])
    def test_windows_reserved_device_names_pass_through_unchanged(self, reserved_device_name: str) -> None:
        """A Windows-reserved device name is returned verbatim as the custom stem -- no rejection or renaming."""
        assert resolve_export_stem(None, reserved_device_name) == (reserved_device_name, True)

    @pytest.mark.parametrize(
        ("output_name", "expected_stem"),
        [
            ('model<>:"|?*.onnx', 'model<>:"|?*'),
            ("a<b>c", "a<b>c"),
            ("model:stream", "model:stream"),
            ("wild*card?name", "wild*card?name"),
        ],
    )
    def test_os_reserved_characters_pass_through_unchanged(self, output_name: str, expected_stem: str) -> None:
        """OS-reserved characters (``<>:"|?*``, invalid in Windows filenames) are not stripped or escaped -- only the
        directory components and the (dot) extension are removed, same as any other input."""
        assert resolve_export_stem(None, output_name) == (expected_stem, True)
