# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Apple Core AI export availability.

Import the exporter from its submodule, not this package root.
"""

from rfdetr.utilities.package import is_installed

# A metadata probe rather than an import: coreai-torch imports torch._dynamo machinery and the Core AI compiler
# bindings, which is too much to pay for on `import rfdetr`.
_IS_COREAI_TORCH_AVAILABLE: bool = is_installed("coreai_torch")
