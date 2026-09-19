# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import importlib.util

# Exclude MLX modules from --doctest-plus collection only where MLX is absent: they import
# mlx.core unconditionally at module level, and the mlx wheel resolves solely on macOS/Darwin.
# Gate on the package instead of the platform so the exclusion applies exactly where the import
# would fail: no CI workflow installs the `mlx` extra, so every runner keeps skipping them, while
# an Apple Silicon checkout with `rfdetr[mlx]` imports and collects them like any other module.
# `find_spec` only looks the package up - it imports neither mlx nor rfdetr at collection time.
if importlib.util.find_spec("mlx") is None:
    collect_ignore_glob = ["src/rfdetr/mlx/*.py"]
