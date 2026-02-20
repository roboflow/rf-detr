# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from importlib.metadata import PackageNotFoundError, version
from typing import Optional


def get_version(package_name: str = "rfdetr") -> Optional[str]:
    """Get the current version of the specified package.

    Args:
        package_name (str): The name of the package to get the version for.
            Defaults to 'rfdetr'.

    Returns:
        str or None: The version string of the specified package.
            Returns None if version cannot be determined.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None
