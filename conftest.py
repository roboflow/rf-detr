# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Exclude MLX modules from --doctest-modules collection: they import mlx.core
# unconditionally at module level, which is only available on macOS/Darwin.
collect_ignore_glob = ["src/rfdetr/mlx/*.py"]
