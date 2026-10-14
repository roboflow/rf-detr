# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compute compatible shared-model cache keys for CI jobs.

Purpose:
    Let parallel jobs share platform-independent weights while isolating changed sources and Plus releases.
Scope:
    Standard-library setup only; model validation, artifact transfer and backbone downloads live in the CI jobs.
Usage:
    Run ``python .github/scripts/model_cache_key.py`` after installing the job dependencies through setup-uv.
Outputs:
    Append the cache key to GITHUB_OUTPUT. Publishers include their run/attempt suffix; readers use a compatible
    prefix to find the newest shared archive. The composite action sets shared model directories with shell echo.
Failure:
    Missing runner variables or committed Git inputs raise instead of selecting an unrelated cache.
Used by:
    .github/actions/cache-model-weights for both readers and the develop smoke-matrix publisher.
"""

import hashlib
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main() -> None:
    """Export the compatible cache key for the current CI job."""
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    sources = [
        "src/rfdetr/assets/model_weights.py",
        "src/rfdetr/models/backbone",
        "pyproject.toml",
        "tests/conftest.py",
        "tests/run_smoke_all_models.py",
    ]
    objects = subprocess.run(
        ["git", "rev-parse", *[f"HEAD:{path}" for path in sources]],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    key = f"rfdetr-model-weights-v4-{hashlib.sha256(objects.encode()).hexdigest()}-plus-"
    try:
        key += hashlib.sha256(version("rfdetr_plus").encode()).hexdigest() + "-"
    except PackageNotFoundError:
        pass
    if os.environ["MODEL_CACHE_SAVE"] == "true":
        key += f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8", newline="\n") as output:
        output.write(f"key={key}\n")


if __name__ == "__main__":
    main()
