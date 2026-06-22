# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixin — see ``VideoDemoGuiApp`` in ``main_window``."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from rfdetr_demo.gui.flipbook import FlipbookPreviewPanel
from rfdetr_demo.gui.theme import FONT_MONO, PAD_ROW, PAD_SECTION, format_eta_seconds


class PreviewPanelMixin:
    """Mixin for preview and insight panels."""

    def _build_preview_area(self, parent: ttk.Frame) -> None:
        """Center column: full-height live flipbook preview."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.flipbook_panel = FlipbookPreviewPanel(parent, thumb_width=56, history_count=8)
        self.flipbook_panel.grid(row=0, column=0, sticky="nsew")

    def _build_insight_column(self, parent: ttk.Frame) -> None:
        """Right column: progress, analysis metrics, and log."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        status_frame = ttk.LabelFrame(parent, text="解析状況", padding=6)
        status_frame.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        padding = {"padx": 4, "pady": PAD_ROW // 2}

        ttk.Checkbutton(
            status_frame,
            text="解析中プレビュー",
            variable=self.preview_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", **padding)

        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100.0)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4))

        ttk.Label(status_frame, text="状態", style="Caption.TLabel").grid(row=2, column=0, sticky="w", **padding)
        ttk.Label(status_frame, textvariable=self.insight_status_var, style="Metrics.TLabel").grid(
            row=2,
            column=1,
            sticky="w",
            **padding,
        )

        metrics = (
            ("進捗", self.insight_frames_var),
            ("FPS", self.insight_fps_var),
            ("検出数", self.insight_detections_var),
            ("残り時間", self.insight_eta_var),
        )
        for row_offset, (label, var) in enumerate(metrics):
            row = 3 + row_offset
            ttk.Label(status_frame, text=label, style="Caption.TLabel").grid(row=row, column=0, sticky="w", **padding)
            ttk.Label(status_frame, textvariable=var, style="Metrics.TLabel").grid(
                row=row,
                column=1,
                sticky="w",
                **padding,
            )

        ttk.Label(
            status_frame,
            textvariable=self.progress_text_var,
            style="Caption.TLabel",
            wraplength=260,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        log_frame = ttk.LabelFrame(parent, text="ログ", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(PAD_SECTION, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        log_header = ttk.Frame(log_frame)
        log_header.grid(row=0, column=0, sticky="ew")
        log_header.columnconfigure(0, weight=1)
        ttk.Button(log_header, text="クリア", style="Small.TButton", command=self._clear_log).grid(
            row=0,
            column=1,
            sticky="e",
        )

        log_body = ttk.Frame(log_frame)
        log_body.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        log_body.columnconfigure(0, weight=1)
        log_body.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_body, height=12, wrap="word", state="disabled", font=FONT_MONO)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_body, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self._configure_log_tags()

    def _configure_log_tags(self) -> None:
        """Configure colored log tags."""
        self.log_text.tag_configure("error", foreground="#b00020")
        self.log_text.tag_configure("warn", foreground="#c77700")
        self.log_text.tag_configure("vast", foreground="#4a6785")
        self.log_text.tag_configure("info", foreground="#1a1a1a")

    def _clear_log(self) -> None:
        """Clear the log text widget."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str, *, level: str = "info") -> None:
        tag = level if level in {"error", "warn", "vast", "info"} else "info"
        if message.startswith("[Vast:") or "Vast.ai" in message or "Vast API" in message:
            tag = "vast"
        elif "エラー" in message or "error" in message.lower():
            tag = "error"
        elif "警告" in message or "未検出" in message or "不可" in message:
            tag = "warn"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reset_insight_panel(self, *, status: str = "待機中") -> None:
        self.progress_var.set(0.0)
        self.progress_text_var.set("")
        self.insight_status_var.set(status)
        self.insight_frames_var.set("—")
        self.insight_fps_var.set("—")
        self.insight_detections_var.set("—")
        self.insight_eta_var.set("—")

    def _on_progress_from_worker(self, current: int, total: int, stats: dict[str, int]) -> None:
        self.root.after(0, lambda: self._update_progress_ui(current, total, stats))

    def _on_preview_from_worker(self, frame_bgr: np.ndarray, frame_index: int, processed: int) -> None:
        self._preview_pending = (frame_bgr, frame_index, processed)
        if not self._preview_flush_scheduled:
            self._preview_flush_scheduled = True
            self.root.after(0, self._flush_preview)

    def _flush_preview(self) -> None:
        self._preview_flush_scheduled = False
        pending = self._preview_pending
        if pending is None:
            return
        self._preview_pending = None
        frame_bgr, frame_index, processed = pending
        if not self.preview_enabled_var.get():
            return
        total = self._progress_total if self._progress_total > 0 else None
        self.flipbook_panel.update_frame(
            frame_bgr,
            frame_index=frame_index,
            processed_count=processed,
            total_count=total,
        )

    def _update_progress_ui(self, current: int, total: int, stats: dict[str, int]) -> None:
        if total > 0:
            self._progress_total = total
            percent = min(100.0, 100.0 * current / total)
            self.progress_var.set(percent)
        else:
            percent = 0.0

        elapsed = 0.0 if self._job_started_at is None else time.perf_counter() - self._job_started_at
        fps = current / elapsed if elapsed > 0 else 0.0
        active_tracks = stats.get("frame_live_tracks")
        if active_tracks is None:
            active_tracks = stats.get("frame_active_tracks")
        raw_frame = stats.get("frame_raw_detections")
        ghost_tracks = int(stats.get("frame_ghost_tracks", 0))
        if active_tracks is not None and stats.get("processed_frames", 0) > 0:
            if ghost_tracks > 0:
                detections_text = f"{active_tracks} (+{ghost_tracks} hold)"
            elif raw_frame is not None and raw_frame != active_tracks:
                detections_text = f"{active_tracks} (raw {raw_frame})"
            else:
                detections_text = str(active_tracks)
        else:
            detections_text = str(stats.get("total_detections", 0))

        eta_text = "—"
        if total > 0 and fps > 0 and current < total:
            eta_text = format_eta_seconds((total - current) / fps)

        percent_display = f" ({percent:.1f}%)" if total > 0 else ""
        self.insight_status_var.set("解析中")
        self.insight_frames_var.set(f"{current} / {total or '?'}{percent_display}")
        self.insight_fps_var.set(f"{fps:.2f}")
        self.insight_detections_var.set(detections_text)
        self.insight_eta_var.set(eta_text)

        eta_suffix = "" if eta_text == "—" else f"  ·  残り約 {eta_text}"
        self.progress_text_var.set(
            f"{current} / {total or '?'}  ·  {fps:.2f} FPS  ·  検出 {detections_text}{eta_suffix}",
        )
        metrics = f"{current}/{total or '?'} · {fps:.2f} FPS · 検出 {detections_text}"
        self._set_status_phase("running", "解析中", metrics)
