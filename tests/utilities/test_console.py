# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.console — Rich console helpers for callbacks."""

from unittest.mock import MagicMock, patch

import pytest

from rfdetr.utilities.console import (
    _IS_RICH_AVAILABLE,
    _build_summary_renderable,
    _get_rich_console,
    _has_progress_bar,
    _render_overall_merged,
    _render_summary_tables,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(callbacks: list[object] | None = None) -> MagicMock:
    """Return a minimal mock Trainer."""
    trainer = MagicMock(name="trainer")
    trainer.callbacks = callbacks or []
    return trainer


def _minimal_overall(max_dets: int = 500) -> dict:
    """Return an overall dict with the minimal keys _render_overall_merged expects."""
    return {
        "mAP 50:95": 0.4,
        "mAP 50": 0.6,
        "mAP 75": 0.3,
        f"mAR @{max_dets}": 0.5,
        "F1": 0.55,
        "Precision": 0.6,
        "Recall": 0.5,
    }


# ---------------------------------------------------------------------------
# _render_overall_merged
# ---------------------------------------------------------------------------


class TestRenderOverallMerged:
    """_render_overall_merged renders a multi-line ASCII table string."""

    def test_returns_string(self) -> None:
        """Result is a non-empty multi-line string."""
        result = _render_overall_merged("Val", _minimal_overall(), 500)
        assert isinstance(result, str)
        assert "\n" in result

    def test_title_prefix_in_output(self) -> None:
        """Title prefix appears in the rendered string."""
        result = _render_overall_merged("Test", _minimal_overall(), 500)
        assert "Test" in result

    def test_metric_values_in_output(self) -> None:
        """Formatted metric values appear in the output."""
        result = _render_overall_merged("Val", _minimal_overall(500), 500)
        assert "0.4000" in result

    def test_nan_renders_as_em_dash(self) -> None:
        """NaN values render as '—' (em-dash)."""
        overall = _minimal_overall()
        overall["mAP 50:95"] = float("nan")
        result = _render_overall_merged("Val", overall, 500)
        assert "—" in result

    def test_negative_sentinel_renders_as_em_dash(self) -> None:
        """Pycocotools sentinel -1 renders as '—'."""
        overall = _minimal_overall()
        overall["mAP 50"] = -1.0
        result = _render_overall_merged("Val", overall, 500)
        assert "—" in result

    def test_segm_group_present_when_key_exists(self) -> None:
        """Segm mAP group rendered when segm keys present."""
        overall = _minimal_overall()
        overall["segm mAP 50:95"] = 0.3
        overall["segm mAP 50"] = 0.5
        result = _render_overall_merged("Val", overall, 500)
        assert "segm mAP" in result

    def test_segm_group_absent_when_key_missing(self) -> None:
        """Segm mAP group not rendered when keys absent."""
        result = _render_overall_merged("Val", _minimal_overall(), 500)
        assert "segm mAP" not in result

    def test_mar_label_uses_max_dets(self) -> None:
        """MAR column label contains the max_dets value."""
        result = _render_overall_merged("Val", _minimal_overall(100), 100)
        assert "@100" in result


# ---------------------------------------------------------------------------
# _has_progress_bar
# ---------------------------------------------------------------------------


class TestHasProgressBar:
    """_has_progress_bar detects any callback whose class name ends with ProgressBar."""

    def test_returns_false_with_no_callbacks(self) -> None:
        """Returns False when trainer has no callbacks."""
        assert not _has_progress_bar(_make_trainer())

    def test_returns_true_for_tqdm_progress_bar(self) -> None:
        """Returns True when a TQDMProgressBar callback present."""
        tqdm_bar_cls = type("TQDMProgressBar", (), {})
        trainer = _make_trainer(callbacks=[tqdm_bar_cls()])
        assert _has_progress_bar(trainer)

    def test_returns_true_for_rich_progress_bar(self) -> None:
        """Returns True when a RichProgressBar callback present."""
        rich_bar_cls = type("RichProgressBar", (), {})
        trainer = _make_trainer(callbacks=[rich_bar_cls()])
        assert _has_progress_bar(trainer)

    def test_returns_false_for_non_progress_bar_callback(self) -> None:
        """Returns False when callbacks don't end with ProgressBar."""
        trainer = _make_trainer(callbacks=[MagicMock(name="SomeOtherCallback")])
        assert not _has_progress_bar(trainer)


# ---------------------------------------------------------------------------
# _get_rich_console
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _IS_RICH_AVAILABLE, reason="Rich not installed")
class TestGetRichConsole:
    """_get_rich_console returns the PTL RichProgressBar console or a fresh one."""

    def test_returns_fresh_console_with_no_callbacks(self) -> None:
        """Returns a Console instance when no RichProgressBar present."""
        from rich.console import Console

        result = _get_rich_console(_make_trainer())
        assert isinstance(result, Console)

    def test_returns_rich_progress_bar_console_when_active(self) -> None:
        """Returns _console from RichProgressBar when callback present and _console set."""
        rich_bar_cls = type("RichProgressBar", (), {})
        expected_console = MagicMock(name="expected_console")
        cb = rich_bar_cls()
        cb._console = expected_console  # type: ignore[attr-defined]
        trainer = _make_trainer(callbacks=[cb])

        result = _get_rich_console(trainer)

        assert result is expected_console

    def test_falls_back_when_console_attribute_is_none(self) -> None:
        """Falls back to fresh Console when _console is None (outside active stage)."""
        from rich.console import Console

        rich_bar_cls = type("RichProgressBar", (), {})
        cb = rich_bar_cls()
        cb._console = None  # type: ignore[attr-defined]
        trainer = _make_trainer(callbacks=[cb])

        result = _get_rich_console(trainer)

        assert isinstance(result, Console)

    def test_mro_subclass_detected(self) -> None:
        """Subclass of RichProgressBar is detected via MRO name check."""
        rich_bar_cls = type("RichProgressBar", (), {})
        themed_bar_cls = type("ThemedProgressBar", (rich_bar_cls,), {})
        expected_console = MagicMock(name="expected_console")
        cb = themed_bar_cls()
        cb._console = expected_console  # type: ignore[attr-defined]
        trainer = _make_trainer(callbacks=[cb])

        result = _get_rich_console(trainer)

        assert result is expected_console


# ---------------------------------------------------------------------------
# _build_summary_renderable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _IS_RICH_AVAILABLE, reason="Rich not installed")
class TestBuildSummaryRenderable:
    """_build_summary_renderable returns a Rich Group for console.print()."""

    def test_returns_group_without_per_class(self) -> None:
        """Returns a Group renderable when per_class is empty."""
        from rich.console import Group

        result = _build_summary_renderable("Val", "overall-text", [])
        assert isinstance(result, Group)

    def test_returns_group_with_per_class(self) -> None:
        """Returns a Group renderable when per_class rows present."""
        from rich.console import Group

        per_class = [{"name": "cat", "ap": 0.5, "ar": 0.6, "f1": 0.55, "precision": 0.6, "recall": 0.5}]
        result = _build_summary_renderable("Val", "overall-text", per_class)
        assert isinstance(result, Group)

    def test_nan_per_class_renders_as_em_dash(self) -> None:
        """NaN in per-class metric renders without error."""
        from rich.console import Console, Group

        per_class = [{"name": "cat", "ap": float("nan"), "ar": -1.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}]
        result = _build_summary_renderable("Val", "overall-text", per_class)
        assert isinstance(result, Group)
        console = Console(force_terminal=True)
        with console.capture() as capture:
            console.print(result)
        assert "—" in capture.get()


# ---------------------------------------------------------------------------
# _render_summary_tables
# ---------------------------------------------------------------------------


class TestRenderSummaryTables:
    """_render_summary_tables delegates to console.print when Rich available."""

    @pytest.mark.skipif(not _IS_RICH_AVAILABLE, reason="Rich not installed")
    def test_calls_console_print_once(self) -> None:
        """console.print called exactly once with the Group renderable."""
        console = MagicMock(name="console")
        _render_summary_tables(console, "Val", "overall-text", [])
        console.print.assert_called_once()

    def test_no_op_when_rich_unavailable(self) -> None:
        """No call made when Rich not installed."""
        console = MagicMock(name="console")
        with patch("rfdetr.utilities.console._IS_RICH_AVAILABLE", False):
            _render_summary_tables(console, "Val", "overall-text", [])
        console.print.assert_not_called()
