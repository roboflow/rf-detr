#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared ttk theme and styling for RF-DETR video demo GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

PAD_SECTION = 8
PAD_ROW = 4

COLOR_TEXT = "#1a1a1a"
COLOR_CAPTION = "#666666"
COLOR_PRIMARY = "#0078d4"
COLOR_PRIMARY_HOVER = "#106ebe"
COLOR_SUCCESS = "#007a3d"
COLOR_WARN = "#c77700"
COLOR_ERROR = "#b00020"
COLOR_RUNNING = "#0057b8"
COLOR_PREVIEW_BG = "#2b2b2b"
COLOR_ACCENT = "#0078d4"

FONT_FAMILY = "Segoe UI"
FONT_BODY = (FONT_FAMILY, 9)
FONT_TITLE = (FONT_FAMILY, 11, "bold")
FONT_CAPTION = (FONT_FAMILY, 8)
FONT_BUTTON_PRIMARY = (FONT_FAMILY, 10, "bold")
FONT_MONO = ("Consolas", 9)


def apply_theme(root: tk.Tk) -> ttk.Style:
    """Apply clam-based custom styles and return the configured Style."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=FONT_BODY)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=COLOR_TEXT)
    style.configure("Caption.TLabel", font=FONT_CAPTION, foreground=COLOR_CAPTION)
    style.configure("Metrics.TLabel", font=FONT_BODY, foreground=COLOR_TEXT)

    style.configure("StatusIdle.TLabel", font=FONT_BODY, foreground=COLOR_CAPTION)
    style.configure("StatusRunning.TLabel", font=FONT_BODY, foreground=COLOR_RUNNING)
    style.configure("StatusDone.TLabel", font=FONT_BODY, foreground=COLOR_SUCCESS)
    style.configure("StatusError.TLabel", font=FONT_BODY, foreground=COLOR_ERROR)

    style.configure(
        "Primary.TButton",
        font=FONT_BUTTON_PRIMARY,
        padding=(16, 8),
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLOR_PRIMARY_HOVER), ("!disabled", COLOR_PRIMARY)],
        foreground=[("!disabled", "white"), ("disabled", "#aaaaaa")],
    )

    style.configure("Secondary.TButton", padding=(12, 6))
    style.configure("Small.TButton", padding=(6, 2), font=FONT_CAPTION)

    style.configure("PreflightPass.TLabel", font=FONT_BODY, foreground=COLOR_SUCCESS)
    style.configure("PreflightWarn.TLabel", font=FONT_BODY, foreground=COLOR_WARN)
    style.configure("PreflightFail.TLabel", font=FONT_BODY, foreground=COLOR_ERROR)

    style.configure("TProgressbar", thickness=12)

    return style


def preflight_style_for_status(status: str) -> str:
    """Return ttk label style name for a preflight status."""
    if status == "pass":
        return "PreflightPass.TLabel"
    if status == "warn":
        return "PreflightWarn.TLabel"
    return "PreflightFail.TLabel"


def preflight_icon_for_status(status: str) -> str:
    """Return a short icon prefix for preflight rows."""
    if status == "pass":
        return "✓"
    if status == "warn":
        return "!"
    return "✕"


def status_style_for_phase(phase: str) -> str:
    """Return ttk label style for top-bar status pill."""
    mapping = {
        "idle": "StatusIdle.TLabel",
        "running": "StatusRunning.TLabel",
        "done": "StatusDone.TLabel",
        "error": "StatusError.TLabel",
        "cancelled": "StatusIdle.TLabel",
    }
    return mapping.get(phase, "StatusIdle.TLabel")


def format_eta_seconds(seconds: float) -> str:
    """Format remaining seconds as M:SS or H:MM:SS."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
