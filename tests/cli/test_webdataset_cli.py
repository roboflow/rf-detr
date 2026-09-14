# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exercise new and compatibility WebDataset packing commands as subprocesses."""

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from rfdetr.datasets.webdataset.index import read_shard_index


class TestWebDatasetCLI:
    """The WebDataset command preserves packing and argument validation."""

    def test_help(self) -> None:
        """Help succeeds without required packing arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "rfdetr.cli.webdataset", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--image-dir" in result.stdout
        assert "--category-ids {remap,raw}" in result.stdout

    def test_missing_arguments(self) -> None:
        """Missing required paths produce argparse's usage error."""
        result = subprocess.run(
            [sys.executable, "-m", "rfdetr.cli.webdataset"], capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode == 2
        assert "the following arguments are required" in result.stderr
        assert "--annotations" in result.stderr

    def test_help_without_training_extra(self) -> None:
        """Packing must not import the optional Lightning training dependency."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; sys.modules['pytorch_lightning'] = None; "
                "module = sys.argv.pop(1); runpy.run_module(module, run_name='__main__')",
                "rfdetr.cli.webdataset",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--image-dir" in result.stdout

    @pytest.mark.parametrize("category_ids", ["remap", "raw"])
    def test_pack(self, category_ids: str, tmp_path: Path) -> None:
        """Packing preserves image bytes, category policy and output summary."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_path = image_dir / "cat.png"
        Image.new("RGB", (8, 8)).save(image_path)
        annotations = tmp_path / "annotations.json"
        annotations.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "cat.png", "width": 8, "height": 8}],
                    "categories": [{"id": 9, "name": "cat"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 9, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}
                    ],
                }
            )
        )
        output = tmp_path / "shards"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rfdetr.cli.webdataset",
                "--image-dir",
                str(image_dir),
                "--annotations",
                str(annotations),
                "--output-dir",
                str(output),
                "--split",
                "val",
                "--max-shard-mb",
                "1",
                "--category-ids",
                category_ids,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        index = read_shard_index(output, "val")
        assert index.num_samples == 1
        assert index.category_ids == category_ids
        assert index.categories == ({"id": 9, "name": "cat"},)
        assert index.annotated_category_ids == (9,)
        assert len(index.shards) == 1
        assert result.stdout.splitlines()[-1] == f"1 samples -> 1 shard(s) in {output} (val-index.json)"
        with tarfile.open(output / index.shards[0]) as shard:
            members = shard.getmembers()
            image_member = next(member for member in members if member.name.endswith(".png"))
            image_file = shard.extractfile(image_member)
            assert image_file is not None
            assert image_file.read() == image_path.read_bytes()
