try:
    from rfdetr_plus.models import (
        RFDETR2XLarge,
        RFDETR2XLargeConfig,
        RFDETRXLarge,
        RFDETRXLargeConfig,
    )
except ImportError:

    _INSTALL_MSG = (
        "The {name} model requires the 'rfdetr_plus' package. "
        "Install it with: pip install rfdetr[plus]"
    )

    class RFDETRXLargeConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name="RFDETRXLargeConfig"))

    class RFDETR2XLargeConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name="RFDETR2XLargeConfig"))

    class RFDETRXLarge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name="RFDETRXLarge"))

    class RFDETR2XLarge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name="RFDETR2XLarge"))
