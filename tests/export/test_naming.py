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
    """Precedence contract: ``output_name`` > ``variant_name`` > ``default``."""

    def test_output_name_wins_over_variant_name(self) -> None:
        """Documented example: both given, output_name wins."""
        assert resolve_export_stem("rfdetr-small", "my-model") == ("my-model", True)

    def test_variant_name_used_when_no_output_name(self) -> None:
        """Documented example: only variant_name given."""
        assert resolve_export_stem("rfdetr-small", None) == ("rfdetr-small", False)

    def test_default_used_when_both_none(self) -> None:
        """Documented example: neither given, generic default stem."""
        assert resolve_export_stem(None, None) == ("inference_model", False)

    def test_default_parameter_overrides_generic_fallback(self) -> None:
        """``default=`` lets callers pick a non-generic fallback stem (e.g. backbone-only exports)."""
        assert resolve_export_stem(None, None, default="backbone_model") == ("backbone_model", False)

    def test_default_parameter_ignored_when_variant_name_given(self) -> None:
        """A custom ``default=`` must not override a real ``variant_name`` — default only fires when both variant_name
        and output_name are unset."""
        assert resolve_export_stem("rfdetr-nano", None, default="backbone_model") == ("rfdetr-nano", False)

    def test_default_parameter_ignored_when_output_name_given(self) -> None:
        """A custom ``default=`` must not override a real ``output_name``."""
        assert resolve_export_stem(None, "my-model", default="backbone_model") == ("my-model", True)


class TestResolveExportStemIsCustomFlag:
    """``is_custom`` must be ``True`` if and only if ``output_name`` was the source of the stem."""

    @pytest.mark.parametrize(
        ("variant_name", "output_name", "expected_is_custom"),
        [
            pytest.param("rfdetr-nano", "my-model", True, id="output_name_given_with_variant"),
            pytest.param(None, "my-model", True, id="output_name_given_alone"),
            pytest.param("rfdetr-nano", None, False, id="variant_name_only"),
            pytest.param(None, None, False, id="neither_given"),
        ],
    )
    def test_is_custom_matches_output_name_branch(
        self, variant_name: str | None, output_name: str | None, expected_is_custom: bool
    ) -> None:
        _, is_custom = resolve_export_stem(variant_name, output_name)
        assert is_custom is expected_is_custom


class TestResolveExportStemCrossPlatformPathComponents:
    """``_sanitize()`` reduces a path-like input to a bare basename stem, extension stripped.

    ``_sanitize`` splits on both ``/`` and ``\\`` unconditionally (``re.split(r"[\\\\/]", name)``), so directory
    components using either separator style are stripped identically on every host platform — a POSIX host handling a
    Windows-style path (or vice versa) never leaks a foreign separator into the resolved stem.
    """

    @pytest.mark.parametrize(
        ("output_name", "expected"),
        [
            pytest.param("sub/dir/rfdetr-nano.onnx", "rfdetr-nano", id="forward_slash_nested"),
            pytest.param("../../etc/passwd", "passwd", id="forward_slash_traversal"),
            pytest.param("/absolute/unix/path/model", "model", id="forward_slash_absolute"),
            pytest.param(r"..\..\evil.onnx", "evil", id="backslash_traversal"),
            pytest.param(r"C:\Users\name\model.onnx", "model", id="windows_absolute"),
            pytest.param("mixed/dir\\model.onnx", "model", id="mixed_separators_last_component_wins"),
        ],
    )
    def test_directory_components_are_stripped_on_every_platform(self, output_name: str, expected: str) -> None:
        """Forward-slash and backslash directory components, plus the extension, are always stripped — on every OS,
        regardless of which separator style the input uses."""
        assert resolve_export_stem(None, output_name) == (expected, True)


class TestResolveExportStemEmptyAndWhitespace:
    """Empty / whitespace-only / all-separator ``output_name`` values."""

    def test_empty_string_falls_back_to_default(self) -> None:
        """An empty ``output_name`` is falsy in Python, so ``resolve_export_stem`` treats it exactly like ``None`` — it
        falls through to ``variant_name``/``default`` instead of being sanitized as a custom name."""
        assert resolve_export_stem(None, "") == ("inference_model", False)

    def test_empty_string_output_name_falls_back_even_with_variant_name_present(self) -> None:
        """An empty ``output_name`` must not shadow a real ``variant_name`` — precedence still resolves to
        ``variant_name`` since the empty string is falsy."""
        assert resolve_export_stem("rfdetr-nano", "") == ("rfdetr-nano", False)

    def test_whitespace_only_output_name_is_treated_as_a_custom_literal_stem(self) -> None:
        """Unlike the empty string, a whitespace-only ``output_name`` is truthy in Python — it is accepted as the custom
        stem verbatim (whitespace is not stripped by ``_sanitize()``, which only splits on path separators and the
        extension), so it is NOT rejected/rewritten to fall back to default."""
        assert resolve_export_stem(None, "   ") == ("   ", True)

    @pytest.mark.parametrize(
        "all_separator_output_name",
        [
            pytest.param("///", id="multiple_forward_slashes"),
            pytest.param("/", id="single_forward_slash"),
        ],
    )
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

    @pytest.mark.parametrize(
        "reserved_device_name",
        [
            pytest.param("CON", id="con"),
            pytest.param("PRN", id="prn"),
            pytest.param("NUL", id="nul"),
            pytest.param("AUX", id="aux"),
            pytest.param("COM1", id="com1"),
            pytest.param("LPT1", id="lpt1"),
        ],
    )
    def test_windows_reserved_device_names_pass_through_unchanged(self, reserved_device_name: str) -> None:
        """A Windows-reserved device name is returned verbatim as the custom stem -- no rejection or renaming."""
        assert resolve_export_stem(None, reserved_device_name) == (reserved_device_name, True)

    @pytest.mark.parametrize(
        ("output_name", "expected_stem"),
        [
            pytest.param('model<>:"|?*.onnx', 'model<>:"|?*', id="all_reserved_chars_extension_stripped"),
            pytest.param("a<b>c", "a<b>c", id="angle_brackets_no_extension"),
            pytest.param("model:stream", "model:stream", id="colon_ads_style"),
            pytest.param("wild*card?name", "wild*card?name", id="glob_metacharacters"),
        ],
    )
    def test_os_reserved_characters_pass_through_unchanged(self, output_name: str, expected_stem: str) -> None:
        """OS-reserved characters (``<>:"|?*``, invalid in Windows filenames) are not stripped or escaped -- only the
        directory components and the (dot) extension are removed, same as any other input."""
        assert resolve_export_stem(None, output_name) == (expected_stem, True)
