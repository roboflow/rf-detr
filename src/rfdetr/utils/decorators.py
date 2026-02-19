# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecation utilities and decorators."""

import warnings
from typing import Any, ItemsView, Iterator, KeysView, ValuesView


class _DeprecatedDict(dict):
    """Dictionary wrapper that emits deprecation warnings when accessing values.

    This class wraps a regular dictionary and emits a DeprecationWarning the first
    time any read operation is performed (e.g., getting values, keys, items, or
    checking membership). The warning is only shown once per instance to avoid
    cluttering the user's output.

    Note:
        This class only overrides read operations. Mutation operations like
        __setitem__ are not overridden since modifications are not an expected
        use case for deprecated dictionaries.

    Args:
        *args: Positional arguments passed to dict constructor.
        deprecated_name: Name of the deprecated object (e.g., "OPEN_SOURCE_MODELS").
        replacement: What to use instead (e.g., "`ModelWeights` enum from `rfdetr.assets.model_weights`").
        **kwargs: Keyword arguments passed to dict constructor.

    Example:
        >>> legacy_dict = _DeprecatedDict(
        ...     {"model_a.pt": "url_a", "model_b.pt": "url_b"},
        ...     deprecated_name="LEGACY_MODELS",
        ...     replacement="`ModelRegistry` class"
        ... )
        >>> # First access triggers warning
        >>> url = legacy_dict["model_a.pt"]  # DeprecationWarning emitted
        >>> # Subsequent accesses don't trigger warning
        >>> url = legacy_dict["model_b.pt"]  # No warning
    """

    def __init__(
        self, *args: Any, deprecated_name: str = "this dictionary", replacement: str = "the new API", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._warning_shown = False
        self._deprecated_name = deprecated_name
        self._replacement = replacement

    def _show_warning(self) -> None:
        """Emit deprecation warning once per instance."""
        if not self._warning_shown:
            warnings.warn(
                f"{self._deprecated_name} is deprecated and will be removed in a future version."
                f" Use {self._replacement} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._warning_shown = True

    def __getitem__(self, key: Any) -> Any:
        """Get item by key and emit deprecation warning."""
        self._show_warning()
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Get item by key with default and emit deprecation warning."""
        self._show_warning()
        return super().get(key, default)

    def keys(self) -> KeysView:
        """Return dictionary keys view and emit deprecation warning."""
        self._show_warning()
        return super().keys()

    def values(self) -> ValuesView:
        """Return dictionary values view and emit deprecation warning."""
        self._show_warning()
        return super().values()

    def items(self) -> ItemsView:
        """Return dictionary items view and emit deprecation warning."""
        self._show_warning()
        return super().items()

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in dictionary and emit deprecation warning."""
        self._show_warning()
        return super().__contains__(key)

    def __iter__(self) -> Iterator[Any]:
        """Return iterator over dictionary keys and emit deprecation warning."""
        self._show_warning()
        return super().__iter__()
