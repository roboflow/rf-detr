# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Runtime helpers shared by the per-format export inference modules.

Preprocessing and detection decoding are identical across exported formats -- only the session/interpreter API and the
tensor layout differ. Keeping one implementation here stops the per-format copies from drifting apart, which matters
because each copy has to stay bit-compatible with :meth:`rfdetr.detr.RFDETR.predict` for the parity suite to mean
anything.

Import the submodules directly; this package intentionally re-exports nothing.
"""
