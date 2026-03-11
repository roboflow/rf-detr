# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""Deprecated: most symbols have moved to ``rfdetr.utilities``.

``accuracy``, ``inverse_sigmoid``, and ``interpolate`` now live in
``rfdetr.models.math`` and are re-exported here for backward compatibility.
"""

from rfdetr.utilities.decorators import _warn_deprecated_module

_warn_deprecated_module("rfdetr.util.misc", "rfdetr.utilities")

# Re-export symbols that have moved to utilities/.
# Re-export math functions from their canonical location in rfdetr.models.math.
from rfdetr.models.math import accuracy, interpolate, inverse_sigmoid  # noqa: F401, E402
from rfdetr.utilities.distributed import (  # noqa: F401, E402
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from rfdetr.utilities.package import get_sha  # noqa: F401, E402
from rfdetr.utilities.state_dict import strip_checkpoint  # noqa: F401, E402
from rfdetr.utilities.tensors import (  # noqa: E402, F401
    NestedTensor,
    collate_fn,
    nested_tensor_from_tensor_list,
)
