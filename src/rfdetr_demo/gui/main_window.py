#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tkinter GUI for RF-DETR video demo with local or Vast.ai GPU execution."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from rfdetr_demo.gui.panels.compute import ComputePanelMixin
from rfdetr_demo.gui.panels.io_task import IoTaskPanelMixin
from rfdetr_demo.gui.panels.job_runner import JobRunnerMixin
from rfdetr_demo.gui.panels.preview import PreviewPanelMixin
from rfdetr_demo.gui.state.job_state import TuneJobState
from rfdetr_demo.gui.theme import PAD_SECTION, apply_theme, status_style_for_phase
from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.inference.uncertainty_viz import DEFAULT_UNCERTAINTY_MAX_AXIS_PX
from rfdetr_demo.paths import resolve_default_source
from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS
from rfdetr_demo.vast.runner import VastGpuOffer
from rfdetr_demo.vast.safety import install_emergency_handlers

logger = logging.getLogger(__name__)


class VideoDemoGuiApp(
    IoTaskPanelMixin,
    ComputePanelMixin,
    PreviewPanelMixin,
    JobRunnerMixin,
):
    """Desktop GUI wrapper around local ``run_demo`` or remote Vast.ai execution."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RF-DETR 動画デモ")
        self.root.geometry("1440x820")
        self.root.minsize(1180, 680)
        apply_theme(self.root)

        self._cancel_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._job_started_at: float | None = None
        self._vast_offers: list[VastGpuOffer] = []
        self._vast_elapsed_started_at: float | None = None
        self._vast_elapsed_after_id: str | None = None
        self._preflight_labels: list[ttk.Label] = []
        self._progress_total: int = 0
        self._preview_pending: tuple[object, int, int] | None = None
        self._preview_flush_scheduled = False
        self._status_phase = "idle"
        self._form_widgets: list[tk.Widget | ttk.Widget] = []
        self._vast_visible = False

        self._person_only_check: ttk.Checkbutton | None = None

        self.source_var = tk.StringVar(value=str(resolve_default_source()))
        self.output_var = tk.StringVar(value="")
        self.compute_var = tk.StringVar(value="local")
        self.task_var = tk.StringVar(value="keypoint")
        self.model_var = tk.StringVar(value="nano")
        self.threshold_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.threshold)
        self.person_only_var = tk.BooleanVar(value=False)
        self.frame_stride_var = tk.IntVar(value=1)
        self.max_frames_var = tk.StringVar(value="")
        self.keypoint_uncertainty_var = tk.BooleanVar(value=True)
        self.uncertainty_style_var = tk.StringVar(value="heatmap")
        self.ellipse_sigma_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.ellipse_sigma)
        self.max_ellipse_axis_var = tk.DoubleVar(value=DEFAULT_UNCERTAINTY_MAX_AXIS_PX)
        self.heatmap_opacity_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.heatmap_opacity)
        self.heatmap_decay_var = tk.DoubleVar(value=0.38)
        self.vertex_radius_var = tk.IntVar(value=4)
        self.keypoint_threshold_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.keypoint_threshold)
        self.motion_filter_var = tk.BooleanVar(value=DEFAULT_PARAMETERS.motion_filter_enabled)
        self.motion_max_speed_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.motion_max_speed_fraction)
        self.motion_ema_alpha_var = tk.DoubleVar(value=DEFAULT_PARAMETERS.motion_ema_alpha)
        self.motion_oscillation_var = tk.BooleanVar(value=DEFAULT_PARAMETERS.motion_oscillation_enabled)
        self._tune_video_fps: float = 30.0
        self._tune_frame_stride: int = 1
        self.vast_api_key_var = tk.StringVar(value="")
        self.vast_api_key_source_var = tk.StringVar(value="未読み込み")
        self.vast_max_dph_var = tk.DoubleVar(value=0.80)
        self.vast_gpu_filter_var = tk.StringVar(value="任意")
        self.vast_destroy_var = tk.BooleanVar(value=True)
        self.vast_offer_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="待機中")
        self.metrics_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="")
        self.insight_status_var = tk.StringVar(value="待機中")
        self.insight_frames_var = tk.StringVar(value="—")
        self.insight_fps_var = tk.StringVar(value="—")
        self.insight_detections_var = tk.StringVar(value="—")
        self.insight_eta_var = tk.StringVar(value="—")
        self.preview_enabled_var = tk.BooleanVar(value=True)
        self.tune_mode_var = tk.BooleanVar(value=True)
        self.tune_preview_seconds_var = tk.DoubleVar(value=2.0)
        self.auto_tune_var = tk.BooleanVar(value=True)
        self._tune_job_state = TuneJobState.IDLE
        self._tune_mode_widgets: list[tk.Widget | ttk.Widget] = []
        self._tune_cache = TunePreviewCache(task="keypoint")
        self._tune_live_after_id: str | None = None
        self._tune_live_preview_var = tk.BooleanVar(value=True)

        self._build_layout()
        self._bind_events()
        self._set_status_phase("idle", "待機中")
        install_emergency_handlers()
        self._load_vast_api_key_from_env(silent=True)
        self._append_startup_info()
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        threading.Thread(target=self._startup_vast_orphan_cleanup, daemon=True).start()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=6)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        self._build_action_bar(outer)

        content = ttk.Frame(outer)
        content.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        content.columnconfigure(0, weight=0, minsize=272)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=0, minsize=300)
        content.rowconfigure(0, weight=1)

        self._col_left = ttk.Frame(content)
        self._col_center = ttk.Frame(content)
        self._col_right = ttk.Frame(content)
        self._col_left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._col_center.grid(row=0, column=1, sticky="nsew", padx=4)
        self._col_right.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        self._col_left.columnconfigure(0, weight=1)
        self._col_left.rowconfigure(0, weight=1)
        self._col_center.columnconfigure(0, weight=1)
        self._col_center.rowconfigure(0, weight=1)
        self._col_right.columnconfigure(0, weight=1)
        self._col_right.rowconfigure(0, weight=1)

        settings_scroll = self._make_scrollable_frame(self._col_left)
        self._build_left_column(settings_scroll)
        self._build_run_panel(settings_scroll)

        self._build_preview_area(self._col_center)
        self._build_insight_column(self._col_right)

    def _build_action_bar(self, parent: ttk.Frame) -> None:
        """Top bar: title, status pill, primary actions."""
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        ttk.Label(bar, text="RF-DETR 動画デモ", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        status_frame = ttk.Frame(bar)
        status_frame.grid(row=0, column=1, sticky="ew", padx=(PAD_SECTION, 0))
        status_frame.columnconfigure(1, weight=1)
        self._status_pill = ttk.Label(status_frame, textvariable=self.status_var, style="StatusIdle.TLabel")
        self._status_pill.grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(status_frame, textvariable=self.metrics_var, style="Metrics.TLabel").grid(
            row=0,
            column=1,
            sticky="w",
        )

        actions = ttk.Frame(bar)
        actions.grid(row=0, column=2, sticky="e")
        self.start_button = ttk.Button(actions, text="開始", style="Primary.TButton", command=self._start_job)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.cancel_button = ttk.Button(
            actions,
            text="キャンセル",
            style="Secondary.TButton",
            command=self._cancel_job,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1)

    def _make_scrollable_frame(self, parent: ttk.Frame) -> ttk.Frame:
        """Return an inner frame inside a vertical scroll area."""
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _register_form_widget(self, widget: tk.Widget | ttk.Widget) -> None:
        """Track widgets that should be disabled while a job runs."""
        if widget not in self._form_widgets:
            self._form_widgets.append(widget)

    def _set_status_phase(self, phase: str, message: str, metrics: str = "") -> None:
        """Update top-bar status pill and metrics."""
        self._status_phase = phase
        self.status_var.set(message)
        self.metrics_var.set(metrics)
        self.insight_status_var.set(message)
        self._status_pill.configure(style=status_style_for_phase(phase))

    def _bind_events(self) -> None:
        self.task_var.trace_add("write", lambda *_: self._update_task_controls())
        self.keypoint_uncertainty_var.trace_add("write", lambda *_: self._update_task_controls())
        self.compute_var.trace_add("write", lambda *_: self._update_compute_controls())
        self.tune_mode_var.trace_add("write", lambda *_: self._on_tune_mode_changed())
        self.vast_offer_var.trace_add("write", lambda *_: self._refresh_vast_preflight())
        self._bind_tune_live_traces()
        self._update_task_controls()
        self._update_compute_controls()
        self._refresh_vast_preflight()

    def _on_window_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askokcancel(
                "終了",
                "解析実行中です。キャンセルして終了しますか？\n"
                "（Vast.ai インスタンスは自動破棄を試みます）",
            ):
                return
            self._cancel_event.set()
        self.root.destroy()


def _center_and_raise_window(root: tk.Tk) -> None:
    """Place the main window on screen and bring it to the foreground."""
    root.update_idletasks()
    width = max(root.winfo_width(), 1024)
    height = max(root.winfo_height(), 640)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = max(0, (screen_w - width) // 2)
    y = max(0, (screen_h - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    with contextlib.suppress(tk.TclError):
        root.state("normal")
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(250, lambda: root.attributes("-topmost", False))
    root.focus_force()


def main() -> int:
    """Launch the Tkinter GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    root = tk.Tk()
    VideoDemoGuiApp(root)
    _center_and_raise_window(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
