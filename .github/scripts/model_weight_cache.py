# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Configure and validate the shared pretrained-model cache used by CI.

Purpose:
    Share fresh smoke-tested checkpoints across parallel model-test jobs without letting readers publish caches.
Scope:
    Configure needs only Python and Git, including in historical release environments. Manifest/verify run with
    current rfdetr+[plus]; verify also prepares and checks the Hugging Face snapshot used by the test fixture.
Usage:
    Run ``python .github/scripts/model_weight_cache.py configure``, ``manifest`` or ``verify``. The local composite
    action configures readers; one successful smoke leg writes the manifest; the post-matrix publisher verifies it.
Outputs:
    Configure exports workspace-root RF_HOME, a common HF Hub directory and a source/Plus-scoped cache key. Manifest
    records the exact checksummed smoke files and Plus version. Verify checks every transferred file and Hub blob.
Failure:
    Missing Git inputs, manifests or model files, changed Plus versions and checksum/download failures propagate.
    The publisher must finish verify before saving; an incomplete or corrupt handoff is never published.
Used by:
    The cache-model-weights composite action and smoke/cache-publisher jobs in ci-integrations.yml.
"""

import argparse
import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MANIFEST = ".model-cache-manifest.json"


def configure_cache() -> None:
    """Export common directories and a key selecting only compatible master archives."""
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    objects = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "rev-parse",
            "HEAD:src/rfdetr/assets/model_weights.py",
            "HEAD:src/rfdetr/models/backbone",
            "HEAD:pyproject.toml",
            "HEAD:tests/conftest.py",
            "HEAD:tests/run_smoke_all_models.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_digest = hashlib.sha256(objects.encode("utf-8")).hexdigest()
    prefix = f"rfdetr-model-weights-v4-{source_digest}-plus-"
    try:
        plus_version = version("rfdetr_plus")
    except PackageNotFoundError:
        plus_version = None
    if plus_version is not None:
        # Fixed-length digests keep native primary-key prefix matching from confusing 1.2 with 1.2.1.
        prefix += hashlib.sha256(plus_version.encode("utf-8")).hexdigest() + "-"
    key = prefix
    if os.environ.get("MODEL_CACHE_SAVE") == "true":
        key += f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"key={key}\n")
    hub_home = workspace / ".huggingface"
    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8", newline="\n") as environment:
        environment.write(f"RF_HOME={workspace.as_posix()}\n")
        environment.write(f"HF_HOME={hub_home.as_posix()}\nHF_HUB_CACHE={(hub_home / 'hub').as_posix()}\n")
    print(f"Model cache key: {key}; Plus version: {plus_version or 'not installed'}")


def checkpoint_manifest(*, verify: bool) -> None:
    """Record or verify the exact registered checkpoint set handed off by the successful smoke leg.

    Args:
        verify: Validate a transferred manifest instead of creating the smoke handoff manifest.
    """
    # Reader configuration must work without current rfdetr or Plus; only smoke and publisher import model assets.
    from rfdetr_plus.assets import ModelWeights as PlusModelWeights

    from rfdetr.assets.model_weights import ModelWeights
    from rfdetr.utilities.files import _validate_file_md5

    workspace = Path(os.environ["RF_HOME"])
    assets = {
        member.value.filename: member.value for registry in (ModelWeights, PlusModelWeights) for member in registry
    }
    manifest_path = workspace / MANIFEST
    plus_version = version("rfdetr_plus")
    if verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["plus_version"] != plus_version:
            raise ValueError("Smoke and publisher Plus versions differ; refusing to label mismatched weights.")
        filenames = manifest["files"]
    else:
        filenames = sorted(name for name in assets if (workspace / name).is_file())
    if not filenames:
        raise ValueError("The smoke-model cache is empty.")
    for name in filenames:
        asset = assets[name]  # Registry lookup also rejects unknown filenames and paths in the manifest.
        if not asset.md5_hash or not _validate_file_md5(str(workspace / name), asset.md5_hash):
            raise ValueError(f"Missing or corrupt smoke checkpoint: {name}")
    if not verify:
        manifest_path.write_text(
            json.dumps({"plus_version": plus_version, "files": filenames}) + "\n", encoding="utf-8"
        )
    print(f"Validated {len(filenames)} smoke-tested model checkpoints.")


def prepare_master_cache() -> None:
    """Verify transferred checkpoints and populate a checksummed test-backbone snapshot before saving."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

    checkpoint_manifest(verify=True)
    repo = "facebook/dinov2-with-registers-base"
    snapshot = Path(snapshot_download(repo, ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"]))
    files = [path for path in snapshot.rglob("*") if not path.is_dir()]
    if not files:
        raise ValueError("The Hugging Face backbone snapshot is empty.")
    for path in files:
        # ETags are SHA256 for LFS weights and Git blob SHA1 for configs; metadata works without symlinks.
        url = hf_hub_url(repo, path.relative_to(snapshot).as_posix(), revision=snapshot.name)
        expected = get_hf_file_metadata(url).etag
        if expected is None or len(expected) not in (40, 64):
            raise ValueError(f"No supported checksum for backbone file: {path.name}")
        digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
        if len(expected) == 40:
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Corrupt backbone cache file: {path.name}")
    print(f"Validated {len(files)} Hugging Face snapshot files; master cache is ready to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("configure", "manifest", "verify"))
    operation = parser.parse_args().operation
    if operation == "configure":
        configure_cache()
    elif operation == "manifest":
        checkpoint_manifest(verify=False)
    else:
        prepare_master_cache()
