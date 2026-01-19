# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Pytest configuration for RF-DETR tests.
"""
import sys
from pathlib import Path

# Add the project root to the Python path so rfdetr can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
