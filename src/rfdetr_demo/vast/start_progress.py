# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tkinter step progress panel (FlashFind GpuPodStartProgress equivalent)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from rfdetr_demo.vast.start_phases import (
    VAST_JOB_STEPS,
    VastJobPhase,
    VastProgressUpdate,
    completed_step_index,
    format_elapsed,
    phase_label,
)


class VastStartProgressPanel(ttk.LabelFrame):
    """Step-by-step external GPU job progress display."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="外部 GPU 起動・実行シーケンス", padding=8)
        self._phase = VastJobPhase.IDLE
        self._elapsed_sec = 0
        self._vast_status: str | None = None
        self._error: str | None = None
        self._error_hint: str | None = None
        self._step_labels: list[ttk.Label] = []
        self._step_icons: list[ttk.Label] = []

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self._title_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self._title_var, font=("Segoe UI", 10, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self._elapsed_var = tk.StringVar(value="")
        ttk.Label(header, textvariable=self._elapsed_var).grid(row=0, column=1, sticky="e")

        self._message_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._message_var, wraplength=300).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 2),
        )

        self._vast_status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._vast_status_var, foreground="#666666").grid(
            row=2,
            column=0,
            sticky="w",
        )

        steps_frame = ttk.Frame(self)
        steps_frame.grid(row=3, column=0, sticky="ew", pady=(6, 4))
        for index, step in enumerate(VAST_JOB_STEPS):
            icon = ttk.Label(steps_frame, text="○", width=3)
            icon.grid(row=index, column=0, sticky="w")
            label = ttk.Label(steps_frame, text=step.label)
            label.grid(row=index, column=1, sticky="w")
            self._step_icons.append(icon)
            self._step_labels.append(label)

        self._hint_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._hint_var, foreground="#666666", wraplength=300).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

        self._error_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._error_var, foreground="#b00020", wraplength=300).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

        dismiss_frame = ttk.Frame(self)
        dismiss_frame.grid(row=6, column=0, sticky="e", pady=(6, 0))
        self._dismiss_button = ttk.Button(dismiss_frame, text="閉じる", command=self.reset)
        self._dismiss_button.grid(row=0, column=0)
        self.grid_remove()

    def reset(self) -> None:
        self._phase = VastJobPhase.IDLE
        self._elapsed_sec = 0
        self._vast_status = None
        self._error = None
        self._error_hint = None
        self.grid_remove()
        self._render()

    def set_elapsed(self, elapsed_sec: int) -> None:
        self._elapsed_sec = elapsed_sec
        if self._phase not in {VastJobPhase.IDLE, VastJobPhase.DONE, VastJobPhase.FAILED}:
            self._elapsed_var.set(f"経過 {format_elapsed(elapsed_sec)}")

    def apply_update(self, update: VastProgressUpdate) -> None:
        self._phase = update.phase
        self._vast_status = update.vast_status
        if update.error:
            self._error = update.error
        if update.error_hint:
            self._error_hint = update.error_hint
        if self._phase != VastJobPhase.IDLE:
            self.grid()
        self._render(message=update.message)

    def show_failed(self, error: str, *, hint: str | None = None) -> None:
        self._phase = VastJobPhase.FAILED
        self._error = error
        self._error_hint = hint
        self.grid()
        self._render()

    def _render(self, message: str | None = None) -> None:
        phase = self._phase
        if phase == VastJobPhase.IDLE:
            return

        failed = phase == VastJobPhase.FAILED
        done = phase == VastJobPhase.DONE
        completed = completed_step_index(phase)

        if failed:
            self._title_var.set("外部 GPU ジョブ失敗")
        elif done:
            self._title_var.set("外部 GPU ジョブ完了")
        else:
            self._title_var.set("外部 GPU 起動・実行中")

        display_message = message or phase_label(phase, self._vast_status)
        self._message_var.set(display_message)

        if self._vast_status and not done and not failed:
            self._vast_status_var.set(f"Vast ステータス: {self._vast_status}")
        else:
            self._vast_status_var.set("")

        failed_step_index = max(0, completed) if failed else -1
        for index, (icon_label, _text_label) in enumerate(zip(self._step_icons, self._step_labels, strict=True)):
            step_num = index + 1
            step_failed = failed and index == failed_step_index
            step_done = not failed and completed >= step_num
            step_active = not step_done and not failed and completed == index
            if step_failed:
                icon_label.configure(text="✕", foreground="#b00020")
            elif step_done:
                icon_label.configure(text="✓", foreground="#007a3d")
            elif step_active:
                icon_label.configure(text="…", foreground="#0057b8")
            else:
                icon_label.configure(text="○", foreground="#888888")

        if not done and not failed:
            self._hint_var.set("起動には 1〜5 分かかることがあります。この画面を開いたままお待ちください。")
            self._dismiss_button.grid_remove()
        else:
            self._hint_var.set("")
            self._dismiss_button.grid()

        if self._error:
            hint_text = f"\n{self._error_hint}" if self._error_hint else ""
            self._error_var.set(f"{self._error}{hint_text}")
        else:
            self._error_var.set("")
