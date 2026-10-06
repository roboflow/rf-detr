# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import warnings

from rfdetr.utilities.package import is_installed

_INSTALL_MSG = (
    "The {name} requires the 'plus' extras for the 'rfdetr' package."
    " Install it with `pip install rfdetr[plus]` (or `pip install rfdetr_plus` if supported)."
)

#: Model classes only ``rfdetr_plus`` provides; ``rfdetr``, ``rfdetr.platform.models`` and ``rfdetr.detr`` read it.
_PLUS_EXPORTS = frozenset({"RFDETR2XLarge", "RFDETRXLarge", "RFDETRAtto", "RFDETRFemto", "RFDETRPico"})

_IS_RFDETR_PLUS_AVAILABLE = is_installed("rfdetr_plus")
if not _IS_RFDETR_PLUS_AVAILABLE:
    warnings.warn(
        _INSTALL_MSG.format(name="platform model downloads"),
        ImportWarning,
        stacklevel=2,
    )
