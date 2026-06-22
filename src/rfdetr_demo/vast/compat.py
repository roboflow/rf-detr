# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatible private aliases for Vast.ai modules.

Deprecated: import public names from ``vast.cli``, ``vast.instance``, etc.
"""

from __future__ import annotations

from rfdetr_demo.vast.cli import parse_json_output, run_vast_cli
from rfdetr_demo.vast.instance import (
    create_instance,
    destroy_instance,
    execute,
    instance_ssh_info,
    make_instance_label,
    show_instance,
    wait_until_running,
)
from rfdetr_demo.vast.remote_io import (
    build_remote_command,
    read_remote_progress,
    vast_copy,
    vast_copy_from_remote,
)

_run_vast_cli = run_vast_cli
_parse_json_output = parse_json_output
_instance_ssh_info = instance_ssh_info
_create_instance = create_instance
_show_instance = show_instance
_wait_until_running = wait_until_running
_execute = execute
_destroy_instance = destroy_instance
_make_instance_label = make_instance_label
_vast_copy = vast_copy
_vast_copy_from_remote = vast_copy_from_remote
_build_remote_command = build_remote_command
_read_remote_progress = read_remote_progress

__all__ = [
    "_build_remote_command",
    "_create_instance",
    "_destroy_instance",
    "_execute",
    "_instance_ssh_info",
    "_make_instance_label",
    "_parse_json_output",
    "_read_remote_progress",
    "_run_vast_cli",
    "_show_instance",
    "_vast_copy",
    "_vast_copy_from_remote",
    "_wait_until_running",
]
