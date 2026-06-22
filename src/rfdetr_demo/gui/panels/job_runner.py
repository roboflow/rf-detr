# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GUI panel mixin — see ``VideoDemoGuiApp`` in ``main_window``."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any

from rfdetr_demo.gui.controllers.run_controller import (
    LocalJobCallbacks,
    RunController,
    VastJobCallbacks,
    frame_audit_gui_lines,
)
from rfdetr_demo.gui.controllers.tune_controller import TuneController
from rfdetr_demo.gui.controllers.vast_controller import VastController
from rfdetr_demo.gui.panels._common import logger
from rfdetr_demo.gui.state.job_state import StartJobError, TuneJobState
from rfdetr_demo.gui.state.tune_parameters import build_tune_parameters
from rfdetr_demo.gui.state.ui_bindings import build_run_config
from rfdetr_demo.inference.types import VideoProcessingCancelledError
from rfdetr_demo.inference.video_io import probe_video_size
from rfdetr_demo.media.frame_audit import DEFAULT_FRAME_AUDIT_COUNT
from rfdetr_demo.vast.runner import (
    VastRunnerCancelledError,
    VastRunnerError,
)
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate


class JobRunnerMixin:
    """Mixin for job runner panel."""

    def _start_job(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("実行中", "処理が既に実行中です。")
            return

        try:
            run_config = build_run_config(self)
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        plan = RunController.prepare_start(run_config, tune_state=self._tune_job_state)
        if isinstance(plan, StartJobError):
            messagebox.showerror(plan.title, plan.message)
            return

        config = plan.config
        output_path = plan.output_path
        compute_backend = config.compute_backend

        if plan.is_full_run_after_tune:
            self._tune_job_state = self._tune_job_state.transition_full_start()
            self._update_start_button_label()

        offer = self._selected_vast_offer() if compute_backend == "vast" else None
        if compute_backend == "vast":
            if not self._confirm_vast_transfer(config.source_path):
                return
            start_error = VastController.validate_job_start(
                api_key=config.vast_api_key,
                offer_selected=config.vast_offer_id is not None,
            )
            if start_error is not None:
                messagebox.showerror(start_error.title, start_error.message)
                return

        self._cancel_event.clear()
        self._reset_insight_panel(status="準備中…")
        self._progress_total = 0
        self._preview_pending = None
        self._preview_flush_scheduled = False
        if plan.is_tune_preview_run:
            self._tune_cache.clear()
            self._tune_cache.task = config.task
            self._tune_cache.person_only = config.person_only
        try:
            _frame_w, _frame_h, video_fps = probe_video_size(config.source_path)
        except RuntimeError:
            video_fps = 30.0
        self._tune_video_fps = video_fps
        self._tune_frame_stride = config.frame_stride
        self._tune_cache.fps = video_fps
        self._tune_cache.frame_stride = config.frame_stride
        if not plan.is_full_run_after_tune:
            self.flipbook_panel.reset()
        self.vast_progress_panel.reset()
        self._set_status_phase("running", "準備中…")
        self._append_log(f"開始 ({compute_backend}): {config.source_path.name} → {output_path.name}")
        if compute_backend == "local":
            self._append_log(
                f"機密監査: 先頭 {DEFAULT_FRAME_AUDIT_COUNT} フレームの解析画像を confidential/audit/ に記録します",
                level="info",
            )
        if plan.is_tune_preview_run and plan.max_source_seconds is not None:
            self._append_log(f"試走モード: 先頭 {plan.max_source_seconds:g} 秒を処理します")
        elif plan.is_full_run_after_tune:
            self._append_log("本番実行: 調整後パラメータで全編を処理します")
        if plan.is_tune_preview_run:
            self._tune_job_state = self._tune_job_state.transition_tune_start()
        self._set_running(True)
        self._job_started_at = time.perf_counter()

        if compute_backend == "vast":
            assert offer is not None
            self.flipbook_panel.show_message(
                "外部 GPU で解析中（プレビューはローカル実行時のみ表示）",
            )
            self.vast_progress_panel.apply_update(
                VastProgressUpdate(
                    phase=VastJobPhase.REQUESTING,
                    message="外部 GPU ジョブを開始します…",
                    percent=5.0,
                ),
            )
            self._start_vast_elapsed_timer()
            vast_config = RunController.build_vast_config(plan)
            self._worker = threading.Thread(
                target=self._run_vast_job,
                kwargs={"plan": plan, "vast_config": vast_config},
                daemon=True,
            )
        else:
            local_callbacks = LocalJobCallbacks(
                progress_callback=self._on_progress_from_worker,
                preview_callback=self._on_preview_from_worker if config.preview_enabled else None,
                cancel_event=self._cancel_event,
                frame_audit_log_callback=lambda line: self.root.after(
                    0,
                    lambda msg=line: self._append_log(msg, level="info"),
                ),
            )
            job_kwargs = RunController.build_local_job_kwargs(
                plan,
                callbacks=local_callbacks,
                tune_cache=self._tune_cache,
            )
            self._worker = threading.Thread(
                target=self._run_local_job,
                kwargs={"job_kwargs": job_kwargs},
                daemon=True,
            )
        self._worker.start()

    def _run_local_job(self, *, job_kwargs: dict[str, Any]) -> None:
        try:
            summary = RunController.run_local(**job_kwargs)
        except VideoProcessingCancelledError:
            self.root.after(0, self._on_cancelled)
            return
        except Exception as error:
            logger.exception("Video demo failed")
            self.root.after(0, lambda e=error: self._on_error(e))
            return
        self.root.after(0, lambda: self._on_complete(summary))

    def _log_frame_audit_summary(self, summary: dict[str, Any]) -> None:
        for line in frame_audit_gui_lines(summary):
            self._append_log(line, level="info")

    def _run_vast_job(self, *, plan: object, vast_config: object) -> None:
        del plan
        try:
            summary = RunController.run_vast(
                vast_config,
                callbacks=VastJobCallbacks(
                    cancel_event=self._cancel_event,
                    log_callback=lambda msg: self.root.after(0, lambda m=msg: self._append_log(m)),
                    phase_callback=self._on_vast_progress,
                ),
            )
        except VastRunnerCancelledError:
            self.root.after(0, self._on_cancelled)
            return
        except VastRunnerError as error:
            self.root.after(0, lambda e=error: self._on_vast_error(e, str(e)))
            return
        except Exception as error:
            logger.exception("Vast.ai video demo failed")
            hint = str(error)
            self.root.after(
                0,
                lambda e=error, h=hint: self._on_vast_error(e, h),
            )
            return
        self.root.after(0, lambda: self._on_complete(summary))

    def _on_vast_error(self, error: Exception, hint: str) -> None:
        self._stop_vast_elapsed_timer()
        self.vast_progress_panel.show_failed(str(error), hint=hint)
        self._on_error(error)

    def _cancel_job(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            return
        self._cancel_event.set()
        self._set_status_phase("running", "キャンセル要求中…")
        self._append_log("キャンセルを要求しました。", level="warn")

    def _on_cancelled(self) -> None:
        self._stop_vast_elapsed_timer()
        self._reset_tune_state(clear_checkbox=False)
        self._set_running(False)
        self._status_phase = "cancelled"
        self._reset_insight_panel(status="キャンセルしました")
        self._set_status_phase("cancelled", "キャンセルしました")
        self.flipbook_panel.show_message("キャンセルしました")
        self._append_log("処理をキャンセルしました。", level="warn")

    def _on_error(self, error: Exception) -> None:
        self._reset_tune_state(clear_checkbox=False)
        self._set_running(False)
        self._status_phase = "error"
        self._set_status_phase("error", "エラー")
        self._append_log(f"エラー: {error}", level="error")
        messagebox.showerror("エラー", str(error))

    def _on_tune_preview_complete(self, summary: dict[str, Any]) -> None:
        """Pause after a tune preview segment so the user can adjust parameters."""
        self._stop_vast_elapsed_timer()
        self._set_running(False)
        try:
            self._tune_job_state = self._tune_job_state.transition_tune_complete()
        except ValueError:
            logger.warning("Tune complete in unexpected state: %s", self._tune_job_state.value)
            self._tune_job_state = TuneJobState.TUNE_PAUSED
        self.tune_retry_button.configure(state="normal")
        self.auto_tune_button.configure(state="normal")
        self._status_phase = "idle"
        self.progress_var.set(100.0)
        processed = summary.get("processed_frames", "?")
        cached = len(self._tune_cache.entries)
        self.insight_frames_var.set(f"{processed} フレーム")
        self.insight_eta_var.set("—")

        parameters = build_tune_parameters(self)
        plan = TuneController.plan_tune_preview_complete(
            summary,
            cache_count=cached,
            live_preview_enabled=bool(self._tune_live_preview_var.get()),
            auto_tune_enabled=bool(self.auto_tune_var.get()),
            task=parameters.task,
            parameters=parameters,
        )
        self.progress_text_var.set(plan.progress_text)
        self._set_status_phase("idle", plan.status_message, plan.status_metrics)
        self._update_start_button_label()
        for line in plan.log_lines:
            self._append_log(line.message, level=line.level)
        self._log_frame_audit_summary(summary)
        if plan.run_auto_tune:
            self._run_auto_tune(apply=True)
        if plan.refresh_live_preview:
            self._refresh_tune_live_preview()

    def _on_complete(self, summary: dict[str, Any]) -> None:
        if summary.get("tune_preview"):
            self._on_tune_preview_complete(summary)
            return

        if self._tune_job_state == TuneJobState.FULL_RUNNING:
            self._tune_job_state = self._tune_job_state.transition_done()

        self._reset_tune_state(clear_checkbox=False)
        self._stop_vast_elapsed_timer()
        self._set_running(False)
        self._status_phase = "done"
        self.progress_var.set(100.0)
        self.insight_eta_var.set("—")
        if summary.get("compute") == "vast.ai":
            self.vast_progress_panel.grid(**self._vast_progress_grid)
            self.vast_progress_panel.apply_update(
                VastProgressUpdate(
                    phase=VastJobPhase.DONE,
                    message="外部 GPU ジョブ完了",
                    percent=100.0,
                ),
            )
            self.progress_text_var.set("100%  ·  外部 GPU ジョブ完了")
        else:
            processed = summary.get("processed_frames", "?")
            self.insight_frames_var.set(f"{processed} フレーム")
            self.progress_text_var.set(f"100%  ·  {processed} フレーム完了")

        compute = summary.get("compute", "local")
        self._set_status_phase("done", "完了", f"{summary.get('elapsed_sec', '?')} 秒 ({compute})")
        self._append_log(
            "完了: "
            f"{summary.get('processed_frames', '?')} フレーム, "
            f"{summary.get('total_detections', '?')} インスタンス, "
            f"{summary.get('elapsed_sec', '?')} 秒 ({compute})",
        )
        self._append_log(f"出力: {summary['target']}")
        self._log_frame_audit_summary(summary)
        messagebox.showinfo("完了", f"出力しました:\n{summary['target']}")

    def _set_form_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._form_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        if enabled:
            self._update_task_controls()
            self._update_compute_controls()
        else:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")

    def _set_running(self, running: bool) -> None:
        self._set_form_enabled(not running)
        if running:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
        else:
            self.cancel_button.configure(state="disabled")
            self._update_compute_controls()
