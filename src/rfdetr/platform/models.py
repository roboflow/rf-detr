# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any

from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE, _PLUS_EXPORTS

__all__: list[str] = []

_UPGRADE_MSG = (
    "{name} is not available in the installed rfdetr_plus package, which predates it."
    " Upgrade it with `pip install -U rfdetr_plus`."
)

# Why the PE-model import below failed; chained onto the upgrade hint so a broken dependency isn't mistaken for an
# outdated rfdetr_plus. A name bound by `except ... as` is deleted when its block ends, hence the module attribute.
_PLUS_PE_IMPORT_ERROR: ImportError | None = None

if _IS_RFDETR_PLUS_AVAILABLE:
    from rfdetr_plus.models import (
        RFDETR2XLarge,
        RFDETRXLarge,
    )

    __all__ += [
        "RFDETR2XLarge",
        "RFDETRXLarge",
    ]

    # A separate block, not part of the import above: the `rfdetr[plus]` floor (`rfdetr_plus>=1.1.0` in
    # pyproject.toml) admits rfdetr_plus releases that ship the XLarge models but predate the PE models, so this
    # import can fail even though rfdetr_plus is installed, and that must not hide the XLarge models.
    try:
        from rfdetr_plus.models import (
            RFDETRAtto,
            RFDETRFemto,
            RFDETRPico,
        )
    except ImportError as ex:
        # rfdetr_plus releases before the PE-Core-T models; __getattr__ raises an upgrade hint on access.
        _PLUS_PE_IMPORT_ERROR = ex
    else:
        __all__ += [
            "RFDETRAtto",
            "RFDETRFemto",
            "RFDETRPico",
        ]


def __getattr__(name: str) -> Any:
    """Lazy failure for missing plus exports: warn on import, raise on access."""
    # Only intercept plus-only symbols; an installed rfdetr_plus that resolves them never reaches this hook.
    if name in _PLUS_EXPORTS:
        if not _IS_RFDETR_PLUS_AVAILABLE:
            from rfdetr.platform import _INSTALL_MSG

            # Surface a clear install hint when someone explicitly requests a plus symbol.
            raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))
        # The installed rfdetr_plus is older than this symbol.
        raise ImportError(_UPGRADE_MSG.format(name=name)) from _PLUS_PE_IMPORT_ERROR

    # Fall back to the normal attribute lookup error for everything else.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
