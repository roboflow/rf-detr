# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared network-reachability probe for tests that download real assets."""

from __future__ import annotations

import socket


def is_online(host: str, port: int, timeout_s: float = 3.0) -> bool:
    """Report whether *host* accepts a TCP connection on *port* within *timeout_s*.

    Args:
        host: Hostname or IP address to probe.
        port: TCP port to attempt a connection on.
        timeout_s: Connection timeout in seconds.

    Returns:
        ``True`` if a TCP connection succeeds, ``False`` on any ``OSError``.

    Examples:
        Skip a test when an asset host is unreachable:

        ```python
        if not is_online("media.roboflow.com", 443):
            pytest.skip("offline")
        ```
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False
