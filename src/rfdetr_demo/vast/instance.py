# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai instance lifecycle helpers."""

from __future__ import annotations

import logging
import time
from threading import Event
from typing import Any

from rfdetr_demo.vast.cli import parse_json_output, run_vast_cli
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate, phase_from_instance_status, phase_label
from rfdetr_demo.vast.types import VastLogCallback, VastPhaseCallback, VastRunnerCancelledError, VastRunnerError

logger = logging.getLogger(__name__)


def instance_ssh_info(info: dict[str, Any]) -> tuple[str | None, int | None]:
    """Extract SSH connection info from a Vast instance payload."""
    ssh_port_raw = info.get("ssh_port")
    ssh_port = int(ssh_port_raw) if ssh_port_raw is not None else None
    ssh_host = info.get("public_ipaddr") or info.get("ssh_host")
    host = str(ssh_host) if ssh_host else None
    return host, ssh_port


def create_instance(
    offer_id: int,
    *,
    api_key: str,
    disk_gb: int,
    docker_image: str,
    label: str,
) -> int:
    """Create a Vast.ai instance and return the new contract ID."""
    completed = run_vast_cli(
        [
            "create",
            "instance",
            str(offer_id),
            "--image",
            docker_image,
            "--disk",
            str(disk_gb),
            "--label",
            label,
            "--ssh",
            "--direct",
            "--onstart-cmd",
            "mkdir -p /workspace/rfdetr_job && touch /tmp/rfdetr_booted",
            "--raw",
        ],
        api_key=api_key,
    )
    payload = parse_json_output(completed.stdout)
    if not isinstance(payload, dict):
        raise VastRunnerError(f"Unexpected create_instance payload: {payload!r}")
    if not payload.get("success", True):
        raise VastRunnerError(f"create_instance failed: {payload}")
    instance_id = payload.get("new_contract") or payload.get("id")
    if instance_id is None:
        raise VastRunnerError(f"create_instance response missing instance id: {payload}")
    return int(instance_id)


def show_instance(instance_id: int, *, api_key: str) -> dict[str, Any]:
    """Return raw instance metadata from Vast.ai."""
    completed = run_vast_cli(
        ["show", "instance", str(instance_id), "--raw"],
        api_key=api_key,
    )
    payload = parse_json_output(completed.stdout)
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    if isinstance(payload, dict):
        return payload
    raise VastRunnerError(f"Unexpected show instance payload: {payload!r}")


def wait_until_running(
    instance_id: int,
    *,
    api_key: str,
    cancel_event: Event | None,
    log_callback: VastLogCallback | None,
    phase_callback: VastPhaseCallback | None,
    timeout_sec: float = 900.0,
) -> dict[str, Any]:
    """Poll instance status until it reaches ``running``."""
    started = time.perf_counter()
    request_sent = True
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise VastRunnerCancelledError("Cancelled while waiting for instance boot.")
        info = show_instance(instance_id, api_key=api_key)
        status = str(info.get("actual_status") or info.get("status") or "unknown")
        ssh_host, ssh_port = instance_ssh_info(info)
        dph_raw = info.get("dph_total")
        dph_total = float(dph_raw) if dph_raw is not None else None

        if log_callback is not None:
            log_callback(f"Instance {instance_id} status: {status}")

        job_phase = phase_from_instance_status(status, ssh_port=ssh_port, request_sent=request_sent)
        if phase_callback is not None:
            phase_callback(
                VastProgressUpdate(
                    phase=job_phase,
                    message=phase_label(job_phase, status),
                    percent=25.0 if job_phase == VastJobPhase.BOOTING else 30.0,
                    vast_status=status,
                    instance_id=instance_id,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    dph_total=dph_total,
                ),
            )

        if status == "running":
            return info
        if status in {"exited", "offline", "error", "failed"}:
            raise VastRunnerError(f"Instance {instance_id} failed to start (status={status}).")
        if time.perf_counter() - started > timeout_sec:
            raise VastRunnerError(f"Timed out waiting for instance {instance_id} to start.")
        time.sleep(10.0)


def execute(instance_id: int, command: str, *, api_key: str) -> str:
    """Run a shell command on a remote instance."""
    completed = run_vast_cli(
        ["execute", str(instance_id), command],
        api_key=api_key,
    )
    return completed.stdout.strip()


def destroy_instance(instance_id: int, *, api_key: str) -> None:
    """Destroy a Vast.ai instance (raises ``VastRunnerError`` on failure)."""
    run_vast_cli(["destroy", "instance", str(instance_id)], api_key=api_key)


def make_instance_label(prefix: str) -> str:
    """Build a traceable instance label for orphan detection."""
    return f"{prefix}-{int(time.time())}"
