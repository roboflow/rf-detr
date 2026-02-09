try:
    from rfdetr_plus.platform_downloads import PLATFORM_MODELS
except ModuleNotFoundError as ex:
    if ex.name in ("rfdetr_plus", "rfdetr_plus.platform_downloads"):
        import warnings

        from rfdetr.platform import INSTALL_MSG

        warnings.warn(
            INSTALL_MSG.format(name="platform model downloads"),
            ImportWarning,
            stacklevel=2,
        )
        PLATFORM_MODELS = {}
    else:
        raise
