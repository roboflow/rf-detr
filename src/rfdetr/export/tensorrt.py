# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
TensorRT export helpers: trtexec invocation and output parsing.
"""

import os
import re
import subprocess

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def run_command_shell(command, dry_run: bool = False) -> subprocess.CompletedProcess:
    if dry_run:
        logger.info(f"\nCUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')} {command}\n")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        raise


def trtexec(onnx_dir: str, args) -> None:
    engine_dir = onnx_dir.replace(".onnx", ".engine")

    # Base trtexec command
    trt_command = " ".join(
        [
            "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            "--memPoolSize=workspace:4096 --fp16",
            "--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10",
            f"{'--verbose' if args.verbose else ''}",
        ]
    )

    if args.profile:
        profile_dir = onnx_dir.replace(".onnx", ".nsys-rep")
        # Wrap with nsys profile command
        command = " ".join(
            ["nsys profile", f"--output={profile_dir}", "--trace=cuda,nvtx", "--force-overwrite true", trt_command]
        )
        logger.info(f"Profile data will be saved to: {profile_dir}")
    else:
        command = trt_command

    output = run_command_shell(command, args.dry_run)
    parse_trtexec_output(output.stdout)


def parse_trtexec_output(output_text):
    logger.info(output_text)
    # Common patterns in trtexec output
    gpu_compute_pattern = (
        r"GPU Compute Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms, median = (\d+\.\d+) ms"
    )
    h2d_pattern = r"Host to Device Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    d2h_pattern = r"Device to Host Transfer Time: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    latency_pattern = r"Latency: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms"
    throughput_pattern = r"Throughput: (\d+\.\d+) qps"

    stats = {}

    # Extract compute times
    if match := re.search(gpu_compute_pattern, output_text):
        stats.update(
            {
                "compute_min_ms": float(match.group(1)),
                "compute_max_ms": float(match.group(2)),
                "compute_mean_ms": float(match.group(3)),
                "compute_median_ms": float(match.group(4)),
            }
        )

    # Extract H2D times
    if match := re.search(h2d_pattern, output_text):
        stats.update(
            {
                "h2d_min_ms": float(match.group(1)),
                "h2d_max_ms": float(match.group(2)),
                "h2d_mean_ms": float(match.group(3)),
            }
        )

    # Extract D2H times
    if match := re.search(d2h_pattern, output_text):
        stats.update(
            {
                "d2h_min_ms": float(match.group(1)),
                "d2h_max_ms": float(match.group(2)),
                "d2h_mean_ms": float(match.group(3)),
            }
        )

    if match := re.search(latency_pattern, output_text):
        stats.update(
            {
                "latency_min_ms": float(match.group(1)),
                "latency_max_ms": float(match.group(2)),
                "latency_mean_ms": float(match.group(3)),
            }
        )

    # Extract throughput
    if match := re.search(throughput_pattern, output_text):
        stats["throughput_qps"] = float(match.group(1))

    return stats
