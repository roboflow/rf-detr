try:
    from rfdetr_plus.platform_downloads import PLATFORM_MODELS
except ModuleNotFoundError as e:
    if e.name in ("rfdetr_plus", "rfdetr_plus.platform_downloads"):
        PLATFORM_MODELS = {}
    else:
        raise
