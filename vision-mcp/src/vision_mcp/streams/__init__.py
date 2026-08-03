"""Live capture and per-stream inference components.

Import concrete classes from their defining modules so lightweight queue consumers do
not import OpenCV, Supervision, and RF-DETR as a side effect.
"""

from __future__ import annotations
