# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Orchestrate a remote RF-DETR video demo on Vast.ai."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from threading import Event
from typing import Any

from rfdetr_demo.media.guard import assert_vast_transfer_allowed, log_transfer_audit
from rfdetr_demo.paths import REPO_ROOT
from rfdetr_demo.vast.api_config import resolve_vast_api_key
from rfdetr_demo.vast.instance import (
    create_instance,
    execute,
    instance_ssh_info,
    make_instance_label,
    wait_until_running,
)
from rfdetr_demo.vast.remote_io import (
    REMOTE_PACKAGE_DIR,
    REMOTE_RUNNER_PATH,
    build_remote_command,
    read_remote_progress,
    vast_copy,
    vast_copy_from_remote,
)
from rfdetr_demo.vast.start_phases import VastJobPhase, VastProgressUpdate
from rfdetr_demo.vast.types import (
    REMOTE_JOB_DIR,
    REMOTE_OUTPUT_NAME,
    VastLogCallback,
    VastPhase,
    VastPhaseCallback,
    VastRunnerCancelledError,
    VastRunnerError,
    VastVideoJobConfig,
)

logger = logging.getLogger(__name__)


def run_video_demo_on_vast(
    config: VastVideoJobConfig,
    *,
    cancel_event: Event | None = None,
    log_callback: VastLogCallback | None = None,
    phase_callback: VastPhaseCallback | None = None,
) -> dict[str, Any]:
    """Provision a Vast.ai GPU, run the video demo remotely, and download the result."""
    from rfdetr_demo.vast.safety import VastJobGuard, VastSafetySettings

    try:
        api_key = resolve_vast_api_key(config.api_key)
    except ValueError as error:
        raise VastRunnerError(str(error)) from error
    safety = VastSafetySettings.from_env()
    guard = VastJobGuard(
        api_key=api_key,
        settings=safety,
        destroy_on_finish=config.destroy_on_finish,
    )
    instance_id: int | None = None
    instance_label = make_instance_label(safety.instance_label_prefix)

    def emit_phase(
        runner_phase: VastPhase,
        message: str,
        percent: float,
        *,
        vast_status: str | None = None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        dph_total: float | None = None,
    ) -> None:
        job_phase_map = {
            VastPhase.CREATING: VastJobPhase.REQUESTING,
            VastPhase.BOOTING: VastJobPhase.BOOTING,
            VastPhase.UPLOADING: VastJobPhase.UPLOADING,
            VastPhase.RUNNING: VastJobPhase.RUNNING,
            VastPhase.DOWNLOADING: VastJobPhase.DOWNLOADING,
            VastPhase.CLEANUP: VastJobPhase.CLEANUP,
            VastPhase.DONE: VastJobPhase.DONE,
        }
        job_phase = job_phase_map.get(runner_phase, VastJobPhase.BOOTING)
        update = VastProgressUpdate(
            phase=job_phase,
            message=message,
            percent=percent,
            vast_status=vast_status,
            instance_id=instance_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            dph_total=dph_total,
        )
        if phase_callback is not None:
            phase_callback(update)
        if log_callback is not None:
            log_callback(message)

    try:
        emit_phase(
            VastPhase.CREATING,
            f"GPU インスタンスを作成中 (offer {config.offer_id})…",
            12.0,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise VastRunnerCancelledError("Cancelled before instance creation.")
        instance_id = create_instance(
            config.offer_id,
            api_key=api_key,
            disk_gb=config.disk_gb,
            docker_image=config.docker_image,
            label=instance_label,
        )
        guard.attach_instance(instance_id, label=instance_label)
        emit_phase(
            VastPhase.BOOTING,
            f"インスタンス {instance_id} ({instance_label}) の起動を待機中…",
            18.0,
            vast_status="creating",
        )
        boot_info = wait_until_running(
            instance_id,
            api_key=api_key,
            cancel_event=cancel_event,
            log_callback=log_callback,
            phase_callback=phase_callback,
            timeout_sec=safety.boot_timeout_sec,
        )
        ssh_host, ssh_port = instance_ssh_info(boot_info)
        dph_raw = boot_info.get("dph_total")
        emit_phase(
            VastPhase.BOOTING,
            "Pod 稼働開始 — SSH 準備完了",
            32.0,
            vast_status="running",
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            dph_total=float(dph_raw) if dph_raw is not None else None,
        )
        execute(instance_id, f"mkdir -p {REMOTE_PACKAGE_DIR}", api_key=api_key)

        assert_vast_transfer_allowed(config.source_path, user_acknowledged=config.user_acknowledged)
        emit_phase(VastPhase.UPLOADING, "入力動画と rfdetr_demo パッケージをアップロード中…", 35.0)
        remote_input = f"{instance_id}:{REMOTE_JOB_DIR}/input{config.source_path.suffix}"
        remote_package = f"{instance_id}:{REMOTE_PACKAGE_DIR}/"
        remote_runner = f"{instance_id}:{REMOTE_RUNNER_PATH}"
        demo_package_src = REPO_ROOT / "src" / "rfdetr_demo"
        runner_src = REPO_ROOT / "src" / "rfdetr_demo" / "vast" / "remote_runner.py"
        log_transfer_audit(
            "upload_start",
            config.source_path,
            remote_input,
            instance_id=instance_id,
        )
        vast_copy(config.source_path, remote_input, api_key=api_key)
        vast_copy(demo_package_src, remote_package, api_key=api_key)
        vast_copy(runner_src, remote_runner, api_key=api_key)

        emit_phase(VastPhase.RUNNING, "リモート GPU で解析を実行中…", 45.0)
        remote_command = build_remote_command(config)
        execute_started = time.perf_counter()
        poll_thread_started = time.perf_counter()

        with subprocess.Popen(
            ["vastai", "execute", str(instance_id), remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "VAST_API_KEY": api_key},
        ) as process:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    process.kill()
                    raise VastRunnerCancelledError("Cancelled during remote execution.")
                limit_hit, limit_reason = guard.check_runtime_limits(execute_started)
                if limit_hit:
                    process.kill()
                    raise VastRunnerError(
                        f"安全上限に達したためリモート実行を停止しました ({limit_reason}). "
                        f"RFDETR_VAST_MAX_EXECUTE_HOURS / RFDETR_VAST_MAX_SESSION_HOURS を確認してください。",
                    )
                progress = read_remote_progress(instance_id, api_key=api_key)
                guard.heartbeat()
                if progress is not None:
                    current = int(progress.get("current", 0))
                    total = int(progress.get("total", 0) or 0)
                    if total > 0:
                        frame_percent = 45.0 + min(50.0, 50.0 * current / total)
                        if phase_callback is not None:
                            phase_callback(
                                VastProgressUpdate(
                                    phase=VastJobPhase.RUNNING,
                                    message=f"解析中 {current}/{total} フレーム…",
                                    percent=frame_percent,
                                    vast_status="running",
                                    instance_id=instance_id,
                                    ssh_host=ssh_host,
                                    ssh_port=ssh_port,
                                ),
                            )
                elif time.perf_counter() - poll_thread_started > 30.0:
                    emit_phase(
                        VastPhase.RUNNING,
                        "リモートセットアップ / モデル読み込み中…",
                        45.0,
                        vast_status="running",
                    )
                time.sleep(5.0)
            stdout = process.stdout.read() if process.stdout is not None else ""
            if process.returncode != 0:
                raise VastRunnerError(
                    f"Remote execution failed (exit {process.returncode}). Output tail:\n{stdout[-2000:]}",
                )
            if log_callback is not None and stdout.strip():
                log_callback(stdout.strip()[-1000:])

        elapsed_remote = time.perf_counter() - execute_started
        emit_phase(VastPhase.DOWNLOADING, "結果動画をダウンロード中…", 96.0)
        config.target_path.parent.mkdir(parents=True, exist_ok=True)
        vast_copy_from_remote(
            f"{instance_id}:{REMOTE_JOB_DIR}/{REMOTE_OUTPUT_NAME}",
            config.target_path,
            api_key=api_key,
        )
        if not config.target_path.is_file() or config.target_path.stat().st_size == 0:
            raise VastRunnerError(f"Downloaded output is missing or empty: {config.target_path}")

        log_transfer_audit(
            "download_complete",
            config.source_path,
            str(config.target_path.resolve()),
            instance_id=instance_id,
        )

        progress = read_remote_progress(instance_id, api_key=api_key) or {}
        summary: dict[str, Any] = {
            "source": str(config.source_path.resolve()),
            "target": str(config.target_path.resolve()),
            "task": config.task,
            "compute": "vast.ai",
            "vast_instance_id": instance_id,
            "vast_offer_id": config.offer_id,
            "vast_instance_label": instance_label,
            "processed_frames": int(progress.get("current", 0)),
            "total_detections": int((progress.get("stats") or {}).get("total_detections", 0)),
            "elapsed_sec": round(elapsed_remote, 2),
            "vast_safety": guard.snapshot(),
        }
        emit_phase(VastPhase.DONE, "Vast.ai 解析が完了しました。", 100.0)
        return summary
    except VastRunnerCancelledError:
        raise
    finally:
        if instance_id is not None and config.destroy_on_finish:
            emit_phase(VastPhase.CLEANUP, f"インスタンス {instance_id} を停止中…", 99.0)
            destroyed = guard.destroy_if_needed(reason="job_finished")
            if not destroyed:
                logger.critical(
                    "Instance %s may still be running — run: scripts\\vast_cleanup_orphans.cmd",
                    instance_id,
                )
