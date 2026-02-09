try:
    from rfdetr_plus.models import (
        RFDETR2XLarge,
        RFDETR2XLargeConfig,
        RFDETRXLarge,
        RFDETRXLargeConfig,
    )

    __all__ = [
        "RFDETR2XLarge",
        "RFDETR2XLargeConfig",
        "RFDETRXLarge",
        "RFDETRXLargeConfig",
    ]
except ModuleNotFoundError:

    __all__: list[str] = []

    _INSTALL_MSG = (
        "The {name} model requires the 'rfdetr_plus' package. "
        "Install it with `pip install rfdetr_plus`"
    )

    class RFDETRXLargeConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name=type(self).__name__))

    class RFDETR2XLargeConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name=type(self).__name__))

    class RFDETRXLarge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name=type(self).__name__))

    class RFDETR2XLarge:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG.format(name=type(self).__name__))
