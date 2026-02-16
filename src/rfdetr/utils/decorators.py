# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecation utilities and decorators."""

import warnings


class _DeprecatedDict(dict):
    """Dictionary wrapper that emits deprecation warnings when accessing values.

    Args:
        *args: Positional arguments passed to dict constructor
        deprecated_name: Name of the deprecated object (e.g., "OPEN_SOURCE_MODELS")
        replacement: What to use instead (e.g., "ModelWeights enum from 'rfdetr.assets.model_weights'")
        **kwargs: Keyword arguments passed to dict constructor
    """

    def __init__(self, *args, deprecated_name: str = "this dictionary",
                 replacement: str = "the new API", **kwargs):
        super().__init__(*args, **kwargs)
        self._warning_shown = False
        self._deprecated_name = deprecated_name
        self._replacement = replacement

    def _show_warning(self):
        if not self._warning_shown:
            warnings.warn(
                f"{self._deprecated_name} is deprecated and will be removed in a future version."
                f" Use {self._replacement} instead.",
                DeprecationWarning,
                stacklevel=3
            )
            self._warning_shown = True

    def __getitem__(self, key):
        self._show_warning()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._show_warning()
        return super().get(key, default)
