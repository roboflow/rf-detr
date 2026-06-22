#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Flipbook-style live preview panel for video demo GUI."""

from __future__ import annotations

import tkinter as tk
from collections import deque
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from rfdetr_demo.gui.theme import COLOR_ACCENT, COLOR_PREVIEW_BG, FONT_CAPTION
from rfdetr_demo.inference.preview_util import fit_bgr_for_preview, resize_bgr_for_preview

DEFAULT_PREVIEW_MAX_WIDTH = 960
DEFAULT_PREVIEW_MAX_HEIGHT = 720
DEFAULT_STRIP_THUMB_WIDTH = 72
DEFAULT_HISTORY_COUNT = 6
DEFAULT_MIN_INTERVAL_SEC = 0.12
STRIP_HEIGHT_PX = 52


def bgr_to_photo_image(frame_bgr: np.ndarray) -> ImageTk.PhotoImage:
    """Convert a BGR numpy array to a Tkinter-compatible photo."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(image=pil_image)


class FlipbookPreviewPanel(ttk.Frame):
    """Parapara-manga style preview: main frame plus recent thumbnail strip."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
        max_height: int = DEFAULT_PREVIEW_MAX_HEIGHT,
        thumb_width: int = DEFAULT_STRIP_THUMB_WIDTH,
        history_count: int = DEFAULT_HISTORY_COUNT,
    ) -> None:
        super().__init__(master, padding=0)
        self._max_width = max_width
        self._max_height = max_height
        self._thumb_width = thumb_width
        self._history_count = max(1, history_count)
        self._history: deque[tuple[ImageTk.PhotoImage, int]] = deque(maxlen=self._history_count)
        self._main_photo: ImageTk.PhotoImage | None = None
        self._strip_photos: list[ImageTk.PhotoImage] = []
        self._last_frame_bgr: np.ndarray | None = None
        self._canvas_image_id: int | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        self._preview_canvas = tk.Canvas(
            self,
            bg=COLOR_PREVIEW_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        self._preview_canvas.grid(row=0, column=0, sticky="nsew")
        self._preview_canvas.bind("<Configure>", self._on_preview_area_configure)

        footer = ttk.Frame(self)
        footer.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        footer.columnconfigure(0, weight=1)
        self._caption_var = tk.StringVar(value="待機中")
        ttk.Label(footer, textvariable=self._caption_var, style="Caption.TLabel").grid(row=0, column=0, sticky="w")

        strip_outer = ttk.Frame(self)
        strip_outer.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        strip_outer.columnconfigure(0, weight=1)

        self._strip_canvas = tk.Canvas(strip_outer, height=STRIP_HEIGHT_PX, highlightthickness=0)
        self._strip_scrollbar = ttk.Scrollbar(strip_outer, orient="horizontal", command=self._strip_canvas.xview)
        self._strip_frame = tk.Frame(self._strip_canvas, bg="#f0f0f0")
        self._strip_window = self._strip_canvas.create_window((0, 0), window=self._strip_frame, anchor="nw")
        self._strip_canvas.configure(xscrollcommand=self._strip_scrollbar.set)
        self._strip_canvas.grid(row=0, column=0, sticky="ew")
        self._strip_scrollbar.grid(row=1, column=0, sticky="ew")
        self._strip_frame.bind("<Configure>", self._on_strip_configure)
        self._strip_canvas.bind("<Configure>", self._on_strip_canvas_configure)

        self._show_placeholder()

    def _preview_dimensions(self) -> tuple[int, int]:
        width = max(160, self._preview_canvas.winfo_width())
        height = max(120, self._preview_canvas.winfo_height())
        return width, height

    def _on_preview_area_configure(self, _event: tk.Event) -> None:
        """Resize preview to use the full canvas area."""
        width, height = self._preview_dimensions()
        if width == self._max_width and height == self._max_height:
            return
        self._max_width = width
        self._max_height = height
        if self._last_frame_bgr is not None:
            self._redraw_main_image()
        else:
            self._show_placeholder()

    def _on_strip_configure(self, _event: tk.Event) -> None:
        self._strip_canvas.configure(scrollregion=self._strip_canvas.bbox("all"))

    def _on_strip_canvas_configure(self, event: tk.Event) -> None:
        self._strip_canvas.itemconfigure(self._strip_window, height=event.height)

    def _build_placeholder(self, width: int, height: int) -> ImageTk.PhotoImage:
        """Return a neutral placeholder image shown before the first frame."""
        canvas = np.full((height, width, 3), 43, dtype=np.uint8)
        cv2.putText(
            canvas,
            "Preview",
            (max(8, width // 2 - 48), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (160, 160, 160),
            2,
            cv2.LINE_AA,
        )
        return bgr_to_photo_image(canvas)

    def _show_placeholder(self) -> None:
        self._placeholder = self._build_placeholder(self._max_width, self._max_height)
        self._main_photo = self._placeholder
        self._draw_main_photo(self._main_photo)

    def _draw_main_photo(self, photo: ImageTk.PhotoImage) -> None:
        width, height = self._preview_dimensions()
        center_x = width // 2
        center_y = height // 2
        if self._canvas_image_id is None:
            self._canvas_image_id = self._preview_canvas.create_image(
                center_x,
                center_y,
                image=photo,
                anchor="center",
            )
        else:
            self._preview_canvas.coords(self._canvas_image_id, center_x, center_y)
            self._preview_canvas.itemconfig(self._canvas_image_id, image=photo)
        self._preview_canvas.image = photo

    def reset(self) -> None:
        """Clear preview history and restore the placeholder."""
        self._history.clear()
        self._strip_photos.clear()
        self._last_frame_bgr = None
        for child in self._strip_frame.winfo_children():
            child.destroy()
        self._show_placeholder()
        self._caption_var.set("待機中")
        self._strip_canvas.configure(scrollregion=(0, 0, 0, 0))

    def _redraw_main_image(self) -> None:
        if self._last_frame_bgr is None:
            self._show_placeholder()
            return
        fitted = fit_bgr_for_preview(self._last_frame_bgr, self._max_width, self._max_height)
        self._main_photo = bgr_to_photo_image(fitted)
        self._draw_main_photo(self._main_photo)

    def _render_main_frame(self, frame_bgr: np.ndarray) -> None:
        self._last_frame_bgr = frame_bgr
        self._redraw_main_image()

    def show_message(self, message: str) -> None:
        """Show a status message without changing thumbnails."""
        self._caption_var.set(message)

    def refresh_live_preview(
        self,
        frames: list[tuple[np.ndarray, int, int]],
        *,
        selected_index: int | None = None,
    ) -> None:
        """Replace preview content without appending to history (live tune mode)."""
        if not frames:
            return

        display_index = len(frames) - 1 if selected_index is None else selected_index
        display_index = max(0, min(display_index, len(frames) - 1))
        main_bgr, frame_index, processed_count = frames[display_index]

        self._render_main_frame(main_bgr)

        self._history.clear()
        self._strip_photos.clear()
        for child in self._strip_frame.winfo_children():
            child.destroy()

        for frame_bgr, f_index, _processed in frames[-self._history_count :]:
            thumb_bgr = resize_bgr_for_preview(frame_bgr, self._thumb_width)
            thumb_photo = bgr_to_photo_image(thumb_bgr)
            self._history.append((thumb_photo, f_index))

        self._rebuild_strip()
        self._caption_var.set(
            f"ライブ調整  ·  フレーム #{frame_index}  ·  推論 {processed_count}  ·  {len(frames)} キャッシュ",
        )

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        processed_count: int,
        total_count: int | None = None,
    ) -> None:
        """Append a new annotated frame to the flipbook preview."""
        self._render_main_frame(frame_bgr)
        thumb_bgr = resize_bgr_for_preview(frame_bgr, self._thumb_width)
        thumb_photo = bgr_to_photo_image(thumb_bgr)

        self._history.append((thumb_photo, frame_index))
        self._rebuild_strip()

        percent_text = ""
        if total_count is not None and total_count > 0:
            percent = 100.0 * processed_count / total_count
            percent_text = f"  ({percent:.1f}%)"
            caption = f"フレーム #{frame_index}  |  推論 {processed_count}/{total_count}{percent_text}"
        else:
            caption = f"フレーム #{frame_index}  |  推論 {processed_count}"
        self._caption_var.set(caption)

    def _rebuild_strip(self) -> None:
        """Render comic-strip style thumbnails for recent frames."""
        for child in self._strip_frame.winfo_children():
            child.destroy()
        self._strip_photos.clear()

        items = list(self._history)
        for column, (photo, frame_index) in enumerate(items):
            is_latest = column == len(items) - 1
            border_color = COLOR_ACCENT if is_latest else "#cccccc"
            border_width = 2 if is_latest else 1
            cell = tk.Frame(
                self._strip_frame,
                bg="#ffffff",
                highlightbackground=border_color,
                highlightthickness=border_width,
                padx=1,
                pady=1,
            )
            cell.grid(row=0, column=column, padx=(0, 3))
            label = tk.Label(cell, image=photo, bg="#ffffff", borderwidth=0)
            label.pack()
            tk.Label(
                cell,
                text=f"#{frame_index}",
                bg="#ffffff",
                fg="#666666" if not is_latest else COLOR_ACCENT,
                font=FONT_CAPTION,
            ).pack()
            self._strip_photos.append(photo)

        self._strip_canvas.update_idletasks()
        self._strip_canvas.configure(scrollregion=self._strip_canvas.bbox("all"))
        if items:
            self._strip_canvas.xview_moveto(1.0)
