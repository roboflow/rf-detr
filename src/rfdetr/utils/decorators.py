# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecation utilities and decorators."""

import warnings


class _DeprecatedDict(dict):
    """Dictionary wrapper that emits deprecation warnings when accessing values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warning_shown = False

    def _show_warning(self):
        if not self._warning_shown:
            warnings.warn(
                "OPEN_SOURCE_MODELS is deprecated and will be removed in a future version. "
                "Use 'ModelWeights' enum from 'rfdetr.assets.model_weights' instead.",
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
