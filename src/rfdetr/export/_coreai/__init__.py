# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Apple Core AI export availability.

Import the exporter from its submodule, not this package root.
"""

import importlib.util

# find_spec rather than an import: coreai-torch imports torch._dynamo machinery and the Core AI compiler bindings,
# which is too much to pay for on `import rfdetr`.
_IS_COREAI_TORCH_AVAILABLE: bool = importlib.util.find_spec("coreai_torch") is not None
