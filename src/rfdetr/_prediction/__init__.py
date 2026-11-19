# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private prediction pipeline shared by ``RFDETR.predict`` and ``RFDETRInference.predict``.

``core`` prepares images, runs batches and decodes results; ``sources`` expands paths, manifests, videos and live inputs
into image batches; ``streams`` owns concurrent live captures; ``types`` holds the annotation aliases.
"""
