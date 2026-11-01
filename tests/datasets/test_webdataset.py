# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the WebDataset shard pack and streaming-load package.

Cover the packer (standard library only), the shard index contract, epoch planning arithmetic, and — behind a ``skipif``
decorator on the optional ``data`` extra — streaming, sizing and parity against the loose-file
:class:`~rfdetr.datasets.coco.CocoDetection` the shards were packed from.
"""

from __future__ import annotations

import json
import tarfile
import types
import warnings
from collections.abc import Iterable
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets import build_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms
from rfdetr.datasets.webdataset import index, load, pack
from rfdetr.datasets.webdataset.index import (
    DEFAULT_MAX_SHARD_BYTES,
    INDEX_VERSION,
    ShardIndex,
    WebDatasetSplitUnavailableError,
    _validate_split_name,
    index_name,
    read_shard_index,
    resolve_within,
)
from rfdetr.datasets.webdataset.load import (
    SHARD_SKEW_RAISE_FRACTION,
    SHARD_SKEW_WARN_FRACTION,
    WebDatasetDetection,
    _shard_url,
    build_webdataset,
    build_webdataset_loader,
    plan_samples_per_worker,
)
from rfdetr.datasets.webdataset.pack import _pack_generation, pack_coco_to_shards, tar_member_bytes
from rfdetr.utilities.imports import _IS_MATPLOTLIB_INSTALLED, _IS_PYTORCH_LIGHTNING_INSTALLED, _IS_WEBDATASET_INSTALLED
from rfdetr.utilities.tensors import make_collate_fn
from tests.datasets._memory import peak_traced_bytes

webdataset_only = pytest.mark.skipif(
    not _IS_WEBDATASET_INSTALLED, reason="webdataset not installed; skip streaming tests (optional `data` extra)"
)
pytorch_lightning_only = pytest.mark.skipif(
    not _IS_PYTORCH_LIGHTNING_INSTALLED, reason="pytorch_lightning not installed; skip DataModule tests"
)
matplotlib_only = pytest.mark.skipif(not _IS_MATPLOTLIB_INSTALLED, reason="matplotlib not installed; skip sample grid")

_CATEGORIES = [{"id": 3, "name": "cat"}, {"id": 9, "name": "dog"}]


def test_package_modules_own_the_webdataset_contract() -> None:
    """Index, pack and load modules expose their owned public APIs."""
    assert callable(index.read_shard_index)
    assert callable(index.resolve_within)
    assert callable(pack.pack_coco_to_shards)
    assert callable(load.build_webdataset)


def _build_coco_split(
    tmp_path: Path,
    *,
    count: int = 8,
    categories: list[dict[str, Any]] | None = None,
    extension: str = "jpg",
    empty_from: int | None = None,
    subdir: str = "split",
    segmentation: bool = False,
) -> tuple[Path, Path]:
    """Write a synthetic COCO split to disk and return its ``(image_dir, annotations_path)``.

    Args:
        tmp_path: Root temporary directory for this test.
        count: Number of images to generate.
        categories: COCO ``categories`` entries; defaults to :data:`_CATEGORIES`.
        extension: Image file extension.
        empty_from: Index from which images carry no annotation; ``None`` annotates every image.
        subdir: Sub-directory of *tmp_path* to write the split into, so a test that packs more than one split can
            keep them apart.
        segmentation: Attach an alternating pair of rectangular polygons to each annotation.

    Returns:
        The image directory and the annotation file path.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     image_dir, annotations = _build_coco_split(Path(tmp), count=2)
        ...     sorted(p.name for p in image_dir.iterdir())
        ['img_0000.jpg', 'img_0001.jpg']
    """
    image_dir = tmp_path / subdir / "images"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    polygons = [[4.0, 4.0, 20.0, 4.0, 20.0, 16.0, 4.0, 16.0], [10.0, 6.0, 30.0, 6.0, 30.0, 20.0, 10.0, 20.0]]
    payload: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": list(_CATEGORIES if categories is None else categories),
    }
    for i in range(count):
        name = f"img_{i:04d}.{extension}"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(image_dir / name)
        payload["images"].append({"id": 1000 + i, "file_name": name, "height": 48, "width": 64})
        if empty_from is not None and i >= empty_from:
            continue
        annotation: dict[str, Any] = {
            "id": i,
            "image_id": 1000 + i,
            "category_id": _CATEGORIES[i % 2]["id"],
            "bbox": [4.0, 4.0, 16.0, 12.0],
            "area": 192.0,
            "iscrowd": 0,
        }
        if segmentation:
            annotation["segmentation"] = [polygons[i % 2]]
        payload["annotations"].append(annotation)
    annotations_path = tmp_path / subdir / "annotations.json"
    annotations_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_dir, annotations_path


class TestPackCocoToShards:
    """Packing a COCO split into tar shards uses only the standard library."""

    def test_index_reports_every_sample(self, tmp_path: Path) -> None:
        """Packing reports the sample count and writes the split's index file."""
        image_dir, annotations = _build_coco_split(tmp_path, count=8)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        assert index.num_samples == 8
        assert (tmp_path / "shards" / index_name("train")).exists()

    @pytest.mark.parametrize(
        ("max_shard_bytes", "expected_single_shard"),
        [
            pytest.param(DEFAULT_MAX_SHARD_BYTES, True, id="default-limit-one-shard"),
            pytest.param(1, False, id="tiny-limit-rolls-over"),
        ],
    )
    def test_shard_rollover_follows_the_size_limit(
        self, tmp_path: Path, max_shard_bytes: int, expected_single_shard: bool
    ) -> None:
        """The default limit keeps one shard, a tiny limit rolls over, and shard files match the index."""
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        index = pack_coco_to_shards(
            image_dir, annotations, tmp_path / "shards", split="train", max_shard_bytes=max_shard_bytes
        )
        assert (len(index.shards) == 1) is expected_single_shard
        assert sum(1 for _ in (tmp_path / "shards").glob("train-*.tar")) == len(index.shards)

    def test_image_bytes_are_copied_verbatim(self, tmp_path: Path) -> None:
        """Image bytes inside the shard equal the source file bytes, with no re-encoding."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            member = tar.extractfile("00000000.jpg")
            assert member is not None
            assert member.read() == (image_dir / "img_0000.jpg").read_bytes()

    def test_images_without_annotations_get_an_empty_list(self, tmp_path: Path) -> None:
        """An image with no annotations still gets a sidecar JSON holding an empty list."""
        image_dir, annotations = _build_coco_split(tmp_path, count=4, empty_from=2)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            member = tar.extractfile("00000003.json")
            assert member is not None
            assert json.loads(member.read())["annotations"] == []

    def test_packing_twice_produces_byte_identical_shards(self, tmp_path: Path) -> None:
        """Packing the same split twice yields the same shard names and identical bytes."""
        image_dir, annotations = _build_coco_split(tmp_path, count=4)
        first = pack_coco_to_shards(image_dir, annotations, tmp_path / "a", split="train")
        second = pack_coco_to_shards(image_dir, annotations, tmp_path / "b", split="train")
        assert first.shards == second.shards
        for shard in first.shards:
            assert (tmp_path / "a" / shard).read_bytes() == (tmp_path / "b" / shard).read_bytes()

    def test_png_split_keeps_its_own_member_extension(self, tmp_path: Path) -> None:
        """A PNG source split packs its images as ``.png`` shard members."""
        image_dir, annotations = _build_coco_split(tmp_path, count=2, extension="png")
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            assert "00000000.png" in tar.getnames()

    def test_packer_records_samples_per_shard(self, tmp_path: Path) -> None:
        """The index's per-shard sample counts match the shard list and add up to the total."""
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train", max_shard_bytes=1)
        assert len(index.samples_per_shard) == len(index.shards)
        assert sum(index.samples_per_shard) == index.num_samples
        assert all(count == 1 for count in index.samples_per_shard)


class TestTarMemberBytes:
    """Shard-size accounting must match a member's real tar footprint, not its raw content length."""

    @pytest.mark.parametrize(
        ("payload_len", "expected"),
        [
            pytest.param(0, 512, id="empty"),
            pytest.param(1, 1024, id="one-byte-still-costs-a-full-data-block"),
            pytest.param(511, 1024, id="just-under-a-block"),
            pytest.param(512, 1024, id="exactly-one-block"),
            pytest.param(513, 1536, id="just-over-a-block-needs-a-second"),
        ],
    )
    def test_accounts_for_the_header_and_padding_a_raw_byte_count_misses(self, payload_len: int, expected: int) -> None:
        """Regression test: shard-size accounting used to sum raw payload lengths alone.

        That undercounted every member by its 512-byte tar header plus padding to the next 512-byte boundary -- two
        members per sample here (image, JSON sidecar) -- so ``max_shard_bytes`` under-shot the real shard size on disk,
        worse the smaller the average sample.
        """
        assert tar_member_bytes(payload_len) == expected


class TestPackCocoToShardsFailures:
    """The packer fails closed rather than dropping data silently, and never touches a valid prior pack."""

    def test_missing_image_is_fatal(self, tmp_path: Path) -> None:
        """An annotated image missing from disk aborts packing instead of being dropped silently."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        (image_dir / "img_0001.jpg").unlink()
        with pytest.raises(FileNotFoundError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")
        assert "silently dropping" in str(exc_info.value)

    def test_missing_annotation_file_is_fatal(self, tmp_path: Path) -> None:
        """A nonexistent annotation file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            pack_coco_to_shards(tmp_path, tmp_path / "absent.json", tmp_path / "shards")
        assert "does not exist" in str(exc_info.value)

    def test_annotation_file_without_images_is_rejected(self, tmp_path: Path) -> None:
        """An annotation file that lists no images is rejected."""
        annotations = tmp_path / "empty.json"
        annotations.write_text(json.dumps({"images": [], "categories": []}), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(tmp_path, annotations, tmp_path / "shards")
        assert "lists no images" in str(exc_info.value)

    def test_unsupported_image_extension_fails_at_pack_time(self, tmp_path: Path) -> None:
        """An undecodable image format (``.tiff``) is rejected when packing, not when reading."""
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        (image_dir / "img_0000.jpg").rename(image_dir / "img_0000.tiff")
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["images"][0]["file_name"] = "img_0000.tiff"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")
        assert "cannot decode" in str(exc_info.value)

    def test_annotations_that_match_no_image_are_rejected(self, tmp_path: Path) -> None:
        """Annotations whose image ids match no image (string versus int ids) are rejected."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        # A COCO export whose image ids are strings while annotations keep ints, or vice versa.
        for entry in payload["images"]:
            entry["id"] = str(entry["id"])
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")
        assert "matching no image" in str(exc_info.value)

    def test_partially_orphaned_annotations_are_rejected(self, tmp_path: Path) -> None:
        """A minority of mismatched annotations must not be dropped silently either."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["annotations"][0]["image_id"] = "no-such-image"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")
        assert "matching no image" in str(exc_info.value)

    @pytest.mark.parametrize(
        "category_ids",
        ["contiguous", pytest.param("", id="empty")],
    )
    def test_unknown_category_policy_is_rejected_before_anything_is_written(
        self, tmp_path: Path, category_ids: str
    ) -> None:
        """The CLI's choices= does not protect a library caller; without this the pack only fails at read time."""
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        shard_dir = tmp_path / "shards"
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", category_ids=category_ids)
        assert "category_ids" in str(exc_info.value)
        assert not shard_dir.exists() or not list(shard_dir.iterdir())

    @pytest.mark.parametrize(
        "split",
        ["*", "tr?in", "tr[ai]n"],
    )
    def test_glob_metacharacters_in_split_are_rejected(self, split: str) -> None:
        """A split used as a glob would match, and then delete, other splits' shards."""
        with pytest.raises(ValueError) as exc_info:
            _validate_split_name(split)
        assert any(text in str(exc_info.value) for text in ("glob metacharacter", "must not contain"))

    def test_repacking_one_split_leaves_another_splits_shards_alone(self, tmp_path: Path) -> None:
        """Re-packing ``train`` leaves the shards and index of a packed ``val`` split untouched."""
        shard_dir = tmp_path / "shards"
        val_images, val_annotations = _build_coco_split(tmp_path, count=4, subdir="other")
        val_index = pack_coco_to_shards(val_images, val_annotations, shard_dir, split="val")
        val_bytes = {name: (shard_dir / name).read_bytes() for name in val_index.shards}

        train_images, train_annotations = _build_coco_split(tmp_path, count=6, subdir="tr")
        pack_coco_to_shards(train_images, train_annotations, shard_dir, split="train")
        pack_coco_to_shards(train_images, train_annotations, shard_dir, split="train", max_shard_bytes=4096)

        assert read_shard_index(shard_dir, "val").shards == val_index.shards
        for name, payload in val_bytes.items():
            assert (shard_dir / name).read_bytes() == payload

    def test_repack_never_leaves_index_pointing_at_missing_shard(self, tmp_path: Path) -> None:
        """Publication order must keep the index and the shards it names consistent at every step."""
        shard_dir = tmp_path / "shards"
        first_images, first_annotations = _build_coco_split(tmp_path, count=8, subdir="first")
        pack_coco_to_shards(first_images, first_annotations, shard_dir, split="train", max_shard_bytes=4096)

        second_images, second_annotations = _build_coco_split(tmp_path, count=3, subdir="second")
        pack_coco_to_shards(second_images, second_annotations, shard_dir, split="train", max_shard_bytes=4096)

        index = read_shard_index(shard_dir, "train")
        assert index.num_samples == 3
        for name in index.shards:
            assert (shard_dir / name).exists()
        # Nothing from the first generation is left behind unindexed.
        on_disk = {path.name for path in shard_dir.glob("train-*.tar")}
        assert on_disk == set(index.shards)

    @pytest.mark.parametrize(
        ("left", "right", "expect_equal"),
        [
            pytest.param(({"x": 1}, ["aa"]), ({"x": 1}, ["aa"]), True, id="same-inputs"),
            pytest.param(({"x": 1}, ["aa"]), ({"x": 2}, ["aa"]), False, id="manifest-change"),
            pytest.param(({"x": 1}, ["aaaa"]), ({"x": 1}, ["bbbb"]), False, id="shard-content-change-same-size"),
        ],
    )
    def test_pack_generation_follows_manifest_and_shard_content(
        self, left: tuple[dict[str, int], list[str]], right: tuple[dict[str, int], list[str]], expect_equal: bool
    ) -> None:
        """The generation token is content-derived: equal inputs repeat it and any manifest or shard change moves it.

        Regression test (shard-content case): the token used to hash only the shard byte *size*, not its content, so two
        structurally identical packs with different bytes (same sample/category counts, same per-shard sizes) could
        reuse the live generation name and publication could overwrite a shard the current index still points at, mid-
        epoch.
        """
        assert (_pack_generation(*left) == _pack_generation(*right)) is expect_equal

    def test_republish_detects_a_same_length_image_content_change(self, tmp_path: Path) -> None:
        """Re-packing after an image's bytes change, at an unchanged file length, still writes a fresh shard.

        End-to-end regression test for the same bug at the ``pack_coco_to_shards`` level: flipping one interior byte of
        the source image changes its content without changing its length, so the old size-only generation hash would
        have reused the live generation name and republished over the shard the current index still points at.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=1)
        shard_dir = tmp_path / "shards"
        first = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        first_bytes = (shard_dir / first.shards[0]).read_bytes()

        image_path = next(image_dir.iterdir())
        data = bytearray(image_path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        image_path.write_bytes(bytes(data))

        second = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        assert first.shards != second.shards
        assert (shard_dir / second.shards[0]).read_bytes() != first_bytes

    @pytest.mark.parametrize("max_shard_bytes", [0, -1])
    def test_non_positive_shard_size_is_rejected(self, tmp_path: Path, max_shard_bytes: int) -> None:
        """A zero or negative ``max_shard_bytes`` is rejected."""
        image_dir, annotations = _build_coco_split(tmp_path, count=1)
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", max_shard_bytes=max_shard_bytes)
        assert "max_shard_bytes" in str(exc_info.value)

    def test_file_name_outside_image_dir_is_rejected(self, tmp_path: Path) -> None:
        """A ``file_name`` that traverses out of the image directory is rejected."""
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["images"][0]["file_name"] = "../escape.jpg"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        (image_dir.parent / "escape.jpg").write_bytes(b"not a real image")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")
        assert "escapes" in str(exc_info.value)

    def test_symlinked_image_directory_is_accepted(self, tmp_path: Path) -> None:
        """Packing an image directory reached through a symlink succeeds.

        Per-file or per-subdirectory symlinks into a shared pool is the standard way a large COCO-scale dataset is
        shared on a cluster without duplicating storage, and the loose-file loader opens these files today. A prior
        implementation resolved ``file_name`` through the symlink before checking containment, so a legitimately
        symlinked tree failed to pack with a security-flavored error; the traversal threat is a crafted ``file_name``
        string, which is lexical, not a symlink target.
        """
        real_dir, annotations = _build_coco_split(tmp_path, count=2)
        symlinked_dir = tmp_path / "images_via_symlink"
        symlinked_dir.symlink_to(real_dir, target_is_directory=True)
        index = pack_coco_to_shards(symlinked_dir, annotations, tmp_path / "shards")
        assert index.num_samples == 2

    def test_split_with_path_separator_is_rejected(self, tmp_path: Path) -> None:
        """A split name containing a path separator is rejected when packing."""
        image_dir, annotations = _build_coco_split(tmp_path, count=1)
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="../escape")
        assert "path separator" in str(exc_info.value)

    def test_failed_repack_does_not_corrupt_a_previous_valid_pack(self, tmp_path: Path) -> None:
        """A re-pack that fails partway through must leave a previously-packed split untouched.

        The first image is overwritten with different bytes before the failing re-pack, so a packer that writes shards
        in place (rather than staging them and publishing only on success) would leave shard 0 changed even though the
        whole pack failed — the earlier, in-place implementation passed a byte-identical check here only because the
        synthetic fixture regenerates the same pixels on every call; changing image 0's content between the two packs is
        what makes an in-place overwrite observable.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        shard_dir = tmp_path / "shards"
        first = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=1)
        before_shards = {name: (shard_dir / name).read_bytes() for name in first.shards}
        before_index = (shard_dir / index_name("train")).read_bytes()

        rng = np.random.default_rng(1)
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(image_dir / "img_0000.jpg")
        (image_dir / "img_0003.jpg").unlink()
        with pytest.raises(FileNotFoundError):
            pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=1)

        assert (shard_dir / index_name("train")).read_bytes() == before_index
        assert sorted(p.name for p in shard_dir.glob("train-*.tar")) == sorted(before_shards)
        for name, payload in before_shards.items():
            assert (shard_dir / name).read_bytes() == payload
        assert not any(shard_dir.glob(".train-pack-*"))

    def test_staged_index_publish_failure_leaves_no_stray_dotfile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure between staging and publishing the index leaves nothing behind in destination.

        Regression test: the staged index dotfile used to be written directly into ``destination`` rather than
        the run's ``work_dir``, so a failure in ``Path.replace()`` after ``write_bytes()`` had already succeeded
        left a stray ``.train-index.json.<generation>`` file in ``destination`` that the ``finally`` block's
        ``rmtree(work_dir)`` never reached, since it only ever cleaned ``work_dir`` itself.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        shard_dir = tmp_path / "shards"
        real_replace = Path.replace
        staged_prefix = f".{index_name('train')}."

        def _replace(self: Path, target: Any) -> Path:
            if self.name.startswith(staged_prefix):
                raise OSError("simulated failure publishing the index")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", _replace)
        with pytest.raises(OSError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        assert "simulated failure" in str(exc_info.value)
        assert not any(shard_dir.glob(f"{staged_prefix}*"))
        assert not any(shard_dir.glob(".train-pack-*"))

    @pytest.mark.parametrize("previous_pack", ["none", "identical", "changed"])
    @pytest.mark.parametrize("failure_stage", ["second-shard", "index-write", "index-replace"])
    def test_publication_failure_restores_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous_pack: str, failure_stage: str
    ) -> None:
        """A failed publication removes only its new shards and preserves every previously published byte."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        if previous_pack != "none":
            pack_coco_to_shards(image_dir, annotations, shard_dir, max_shard_bytes=1)
        before = {path.name: path.read_bytes() for path in shard_dir.iterdir()}
        if previous_pack == "changed":
            Image.new("RGB", (64, 48), color="red").save(image_dir / "img_0000.jpg")
        real_replace, real_write = Path.replace, Path.write_bytes
        moved = 0

        def fail_replace(path: Path, target: Path) -> Path:
            """Inject a filesystem rename failure at the selected publication boundary.

            Examples:
                >>> fail_replace(Path("shard"), Path("target"))  # doctest: +SKIP
                Requires the enclosing pytest scenario and its rename counter.
            """
            nonlocal moved
            if target.suffix == ".tar":
                moved += 1
            if (failure_stage == "second-shard" and moved == 2) or (
                failure_stage == "index-replace" and target.name == index_name("train")
            ):
                raise OSError("injected publication failure")
            return real_replace(path, target)

        def fail_write(path: Path, payload: bytes) -> int:
            """Inject failure while writing the staged index.

            Examples:
                >>> fail_write(Path("index"), b"{}")  # doctest: +SKIP
                Requires the enclosing pytest scenario.
            """
            if failure_stage == "index-write" and path.name.startswith(".train-index.json."):
                raise OSError("injected publication failure")
            return real_write(path, payload)

        monkeypatch.setattr(Path, "replace", fail_replace)
        monkeypatch.setattr(Path, "write_bytes", fail_write)
        with pytest.raises(OSError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, shard_dir, max_shard_bytes=1)
        assert "injected publication failure" in str(exc_info.value)
        assert {path.name: path.read_bytes() for path in shard_dir.iterdir()} == before


class TestShardIndex:
    """The index carries the label space so a reader never parses the source annotation file."""

    def test_json_roundtrip_preserves_every_field(self) -> None:
        """``to_json`` followed by ``from_json`` returns an equal index."""
        index = ShardIndex("train", ("train-000000.tar",), 5, tuple(_CATEGORIES), (3, 9), "remap", (5,))
        assert ShardIndex.from_json(index.to_json()) == index

    def test_unsupported_schema_version_is_rejected(self) -> None:
        """An index written with a newer schema version is rejected."""
        payload = ShardIndex("train", (), 0, (), (), "remap").to_json()
        payload["version"] = INDEX_VERSION + 1
        with pytest.raises(ValueError) as exc_info:
            ShardIndex.from_json(payload)
        assert "schema version" in str(exc_info.value)

    def test_unknown_category_policy_is_rejected(self) -> None:
        """An index declaring an unknown ``category_ids`` policy is rejected."""
        payload = ShardIndex("train", (), 0, (), (), "remap").to_json()
        payload["category_ids"] = "contiguous"
        with pytest.raises(ValueError) as exc_info:
            ShardIndex.from_json(payload)
        assert "category_ids policy" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            pytest.param("remap", {3: 0, 9: 1}, id="remap-assigns-contiguous-labels"),
            pytest.param("raw", None, id="raw-keeps-source-ids"),
        ],
    )
    def test_cat2label_follows_the_declared_policy(self, policy: str, expected: dict[int, int] | None) -> None:
        """``remap`` maps raw ids to contiguous labels while ``raw`` yields no mapping."""
        index = ShardIndex("train", (), 0, tuple(_CATEGORIES), (3, 9), policy)  # type: ignore[arg-type]
        assert index.cat2label() == expected

    def test_unannotated_grouping_category_consumes_no_label_slot(self) -> None:
        """A grouping category with no annotations takes no label slot under ``remap``."""
        categories = (
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 1, "name": "cat", "supercategory": "root"},
            {"id": 2, "name": "dog", "supercategory": "root"},
        )
        assert ShardIndex("train", (), 0, categories, (1, 2), "remap").cat2label() == {1: 0, 2: 1}

    def test_packed_policy_reaches_the_reader(self, tmp_path: Path) -> None:
        """The ``raw`` policy chosen at pack time is what the reader's index reports."""
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train", category_ids="raw")
        assert read_shard_index(tmp_path / "shards", "train").cat2label() is None

    def test_missing_index_names_the_packing_command(self, tmp_path: Path) -> None:
        """A missing split index raises an error that names the packing CLI."""
        with pytest.raises(WebDatasetSplitUnavailableError) as exc_info:
            read_shard_index(tmp_path, "train")
        assert "rfdetr.cli.webdataset" in str(exc_info.value)

    def test_missing_split_error_is_still_a_file_not_found(self, tmp_path: Path) -> None:
        """Callers that only care the split is absent keep catching FileNotFoundError."""
        assert issubclass(WebDatasetSplitUnavailableError, FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            read_shard_index(tmp_path, "train")

    def test_split_with_path_separator_is_rejected_on_read(self, tmp_path: Path) -> None:
        """Reading a split name containing a path separator is rejected."""
        with pytest.raises(ValueError) as exc_info:
            read_shard_index(tmp_path, "../escape")
        assert "path separator" in str(exc_info.value)

    def test_index_whose_own_split_field_disagrees_with_the_filename_is_rejected(self, tmp_path: Path) -> None:
        """The index's own recorded ``split`` field is checked, not just the filename it was read from.

        Regression test: a val-index.json copied or renamed to train-index.json (or vice versa) used to be
        accepted silently and read as if it were the requested split, since only the filename was ever checked.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="val")
        (shard_dir / index_name("val")).rename(shard_dir / index_name("train"))
        with pytest.raises(ValueError) as exc_info:
            read_shard_index(shard_dir, "train")
        assert "recorded split" in str(exc_info.value)


class TestShardPathValidation:
    """A shard index is on-disk JSON, not a trusted value; every shard entry is validated before use.

    Covers both consumers that previously joined an index entry onto a directory with no check: opening a shard
    for reading (:meth:`WebDatasetDetection._shard_urls`) and unlinking a stale shard during republish
    (:func:`pack_coco_to_shards`'s cleanup loop).
    """

    def test_resolve_within_accepts_a_shard_under_base(self, tmp_path: Path) -> None:
        """An ordinary shard file name resolves to the expected path under the base directory."""
        assert resolve_within(tmp_path, "train-000000.tar") == (tmp_path / "train-000000.tar").resolve()

    @pytest.mark.parametrize(
        "entry",
        ["../../etc/passwd", "/etc/passwd"],
    )
    def test_resolve_within_rejects_an_escaping_entry(self, tmp_path: Path, entry: str) -> None:
        """A shard entry that resolves outside the base directory is rejected rather than followed.

        An index entry is untrusted on-disk JSON: a tampered or corrupted index could name a path anywhere on
        the filesystem, and every caller that joins an entry onto a base directory must reject that instead of
        opening or deleting whatever it points at.
        """
        with pytest.raises(ValueError) as exc_info:
            resolve_within(tmp_path, entry)
        assert "resolves outside" in str(exc_info.value)

    def test_shard_urls_rejects_a_malicious_index_entry(self, tmp_path: Path) -> None:
        """Reading shard URLs rejects an index whose ``shards`` list was tampered with a traversal entry.

        A prior implementation joined ``index.shards`` entries onto ``shard_dir`` with no validation, so a
        corrupted index could hand ``webdataset`` a ``file:`` URL to any file the training process can read.
        """
        shard_dir = _pack(tmp_path, count=2)
        payload = json.loads((shard_dir / index_name("train")).read_text(encoding="utf-8"))
        payload["shards"] = ["../../escape.tar"]
        (shard_dir / index_name("train")).write_text(json.dumps(payload), encoding="utf-8")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        with pytest.raises(ValueError) as exc_info:
            dataset._shard_urls()
        assert "resolves outside" in str(exc_info.value)

    def test_republish_cleanup_rejects_a_malicious_previous_index_entry(self, tmp_path: Path) -> None:
        """Re-packing a split refuses to delete a stale-shard entry that escapes the shard directory.

        A prior implementation read the previously published index's ``shards`` list with no validation and unlinked
        whatever path each entry named; a tampered on-disk index could therefore make the next re-pack of that split
        silently delete an arbitrary file.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        index_path = shard_dir / index_name("train")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        payload["shards"] = [*payload["shards"], "../escape.tar"]
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        assert "resolves outside" in str(exc_info.value)


def _pack(tmp_path: Path, **kwargs: Any) -> Path:
    """Pack a synthetic split into *tmp_path* / ``"shards"`` and return that directory.

    Examples:
        >>> import tempfile
        >>> from unittest.mock import patch
        >>> with tempfile.TemporaryDirectory() as tmp, patch("rfdetr.datasets.webdataset.pack.logger.info"):
        ...     shard_dir = _pack(Path(tmp), count=1)
        ...     sample_count = read_shard_index(shard_dir, "train").num_samples
        >>> sample_count
        1
    """
    image_dir, annotations = _build_coco_split(tmp_path, **kwargs)
    shard_dir = tmp_path / "shards"
    pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=4096)
    return shard_dir


@webdataset_only
class TestWebDatasetDetection:
    """Streaming a packed split reproduces the loose-file dataset it was packed from."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            pytest.param(
                PureWindowsPath(r"C:\Temp\shards\train-000000.tar"),
                "C:/Temp/shards/train-000000.tar",
                id="windows-drive",
            ),
            pytest.param(
                PurePosixPath("/tmp/shard dir/train-000000.tar"),
                "/tmp/shard dir/train-000000.tar",
                id="posix-with-space",
            ),
        ],
    )
    def test_shard_url_names_a_path_open_accepts(self, path: PurePosixPath | PureWindowsPath, expected: str) -> None:
        """The shard URL path equals the posix form of both Windows and POSIX shard paths."""
        assert urlparse(_shard_url(path)).path == expected

    def test_webdataset_opens_every_shard_url_this_dataset_emits(self, tmp_path: Path) -> None:
        """Webdataset's own opener streams every emitted shard URL, even from a directory with a space."""
        from webdataset.cache import StreamingOpen

        shard_dir = _pack(tmp_path / "shard dir", count=4)
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        opener = StreamingOpen()
        opened = [source["stream"].read() for source in opener(dataset._shard_urls())]
        assert opened == [(shard_dir / name).read_bytes() for name in dataset.index.shards]

    def test_every_sample_is_streamed_once(self, tmp_path: Path) -> None:
        """Iterating the dataset yields each packed sample exactly once."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=12), "train", transforms=None)
        image_ids = [int(target["image_id"]) for _, target in dataset]
        assert sorted(image_ids) == list(range(1000, 1012))

    @pytest.mark.parametrize("extension", ["jpg", "png"])
    def test_output_matches_the_loose_file_dataset(self, tmp_path: Path, extension: str) -> None:
        """Streamed images, boxes and labels equal the loose-file dataset's for the same split."""
        image_dir, annotations = _build_coco_split(tmp_path, count=6, extension=extension)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=4096)
        transforms = make_coco_transforms("val", 224)
        streamed = list(WebDatasetDetection(shard_dir, "train", transforms=transforms, cat2label={3: 0, 9: 1}))
        loose = CocoDetection(image_dir, annotations, transforms=transforms, remap_category_ids=True)
        # Shards are packed in annotation-file order while CocoDetection sorts by image id, so pair by id.
        reference = {int(target["image_id"]): (image, target) for image, target in loose}
        assert len(streamed) == len(reference)
        for image, target in streamed:
            reference_image, reference_target = reference[int(target["image_id"])]
            assert torch.equal(image, reference_image)
            assert torch.equal(target["boxes"], reference_target["boxes"])
            assert torch.equal(target["labels"], reference_target["labels"])

    def test_draft_size_matches_the_loose_file_decode_and_rescales_annotations(self, tmp_path: Path) -> None:
        """A ``draft_size``-reduced decode matches ``CocoDetection``'s training decode path, boxes included.

        Regression test: the reader always fully decoded JPEGs regardless of ``draft_size``, so it did not
        reproduce the actual loose-file training decode (``CocoDetection._decode_image``'s ``PIL.Image.draft``
        plus annotation rescale) the PR's parity claim rested on. The existing parity test above uses
        ``make_coco_transforms("val", ...)``, where drafting is intentionally disabled, so it could not catch
        this; this test forces a real draft reduction with a large source image and a small ``draft_size``.
        """
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)).save(
            image_dir / "img_0000.jpg", quality=90
        )
        payload = {
            "images": [{"id": 1000, "file_name": "img_0000.jpg", "height": 512, "width": 512}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 1000,
                    "category_id": 3,
                    "bbox": [40.0, 40.0, 160.0, 120.0],
                    "area": 19200.0,
                    "iscrowd": 0,
                }
            ],
            "categories": list(_CATEGORIES),
        }
        annotations = tmp_path / "annotations.json"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")

        draft_size = 128
        transforms = make_coco_transforms("val", 224)
        streamed = list(
            WebDatasetDetection(
                shard_dir, "train", transforms=transforms, cat2label={3: 0, 9: 1}, draft_size=draft_size
            )
        )
        loose = CocoDetection(
            image_dir,
            annotations,
            transforms=transforms,
            remap_category_ids=True,
            cat2label={3: 0, 9: 1},
            draft_size=draft_size,
        )
        assert len(streamed) == len(loose) == 1
        streamed_image, streamed_target = streamed[0]
        loose_image, loose_target = loose[0]
        assert torch.equal(streamed_image, loose_image)
        assert torch.equal(streamed_target["boxes"], loose_target["boxes"])

    @pytest.mark.parametrize("extension", ["png", "bmp"])
    def test_png_or_bmp_member_is_not_copied_through_numpy(self, tmp_path: Path, extension: str) -> None:
        """A PNG or BMP shard member reaches the transforms without a frame-sized NumPy copy (#1544).

        PNG's encoded bytes, which the tar reader holds in memory, stay far below the 1.44 MB frame because the gradient
        compresses well; BMP has no compression, so its encoded bytes are already about one frame plus whatever Pillow's
        decode buffer and mandatory RGB ``convert()`` copy add on top. The bound below is the on-disk size plus two
        frames, wide enough for that legitimate BMP overhead while still catching the further frame-sized copy a round
        trip through ``np.array`` and ``Image.fromarray`` would add.
        """
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        rows, columns = np.mgrid[0:600, 0:800]
        gradient = np.stack([columns % 256, rows % 256, (rows + columns) % 256], axis=-1).astype(np.uint8)
        file_name = f"img_0000.{extension}"
        image_path = image_dir / file_name
        Image.fromarray(gradient).save(image_path)
        encoded_size = image_path.stat().st_size
        annotations = tmp_path / "annotations.json"
        annotations.write_text(
            json.dumps(
                {
                    "images": [{"id": 1000, "file_name": file_name, "height": 600, "width": 800}],
                    "annotations": [],
                    "categories": list(_CATEGORIES),
                }
            ),
            encoding="utf-8",
        )
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)

        assert peak_traced_bytes(lambda: next(iter(dataset))) < encoded_size + 2 * (800 * 600 * 3)

    def test_segmentation_masks_match_the_loose_file_dataset(self, tmp_path: Path) -> None:
        """Streamed instance masks equal the loose-file dataset's masks for the same polygons."""
        image_dir, annotations = _build_coco_split(tmp_path, count=4, segmentation=True)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=4096)
        transforms = make_coco_transforms("val", 224)
        streamed = list(
            WebDatasetDetection(shard_dir, "train", transforms=transforms, cat2label={3: 0, 9: 1}, include_masks=True)
        )
        loose = CocoDetection(
            image_dir, annotations, transforms=transforms, include_masks=True, remap_category_ids=True
        )
        reference = {int(target["image_id"]): target for _, target in loose}
        assert len(streamed) == len(reference)
        for _, target in streamed:
            reference_target = reference[int(target["image_id"])]
            assert torch.equal(target["masks"], reference_target["masks"])

    @pytest.mark.parametrize(
        ("category_ids", "expected"),
        [
            # The grouping root carries no annotation, so only the remapped label space drops it.
            pytest.param("remap", ["cat", "dog"], id="remap-drops-the-unannotated-parent"),
            # ids 0, 3, 9: index-aligned by raw id, with an empty string at every skipped index.
            pytest.param(
                "raw", ["root", "", "", "cat", "", "", "", "", "", "dog"], id="raw-keeps-every-category-by-id"
            ),
        ],
    )
    def test_class_names_follow_the_label_space(self, tmp_path: Path, category_ids: str, expected: list[str]) -> None:
        """``remap`` drops the unannotated parent while ``raw`` keeps every category, indexed by raw id."""
        grouped_categories = [
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 3, "name": "cat", "supercategory": "root"},
            {"id": 9, "name": "dog", "supercategory": "root"},
        ]
        image_dir, annotations = _build_coco_split(tmp_path, count=4, categories=grouped_categories)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", category_ids=category_ids)
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        assert dataset.class_names == expected

    def test_class_names_leave_a_gap_for_an_unnamed_label_slot(self, tmp_path: Path) -> None:
        """A label slot with no category gets an empty class name between named ones."""
        image_dir, annotations = _build_coco_split(tmp_path, count=4)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None, cat2label={3: 0, 9: 2})
        assert dataset.class_names == ["cat", "", "dog"]

    def test_length_is_a_type_error_until_an_epoch_is_planned(self, tmp_path: Path) -> None:
        """``len`` raises TypeError until ``configure_epoch`` plans the epoch, then reports its size."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        with pytest.raises(TypeError) as exc_info:
            len(dataset)
        assert "no planned epoch length" in str(exc_info.value)
        dataset.configure_epoch(samples_per_worker=4, num_workers=2)
        assert len(dataset) == 8

    @pytest.mark.parametrize(
        ("samples_per_worker", "num_workers"),
        [pytest.param(0, 1, id="zero-samples"), pytest.param(4, 0, id="zero-workers")],
    )
    def test_configure_epoch_rejects_degenerate_plans(
        self, tmp_path: Path, samples_per_worker: int, num_workers: int
    ) -> None:
        """A plan with zero samples or zero workers is rejected."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        with pytest.raises(ValueError) as exc_info:
            dataset.configure_epoch(samples_per_worker=samples_per_worker, num_workers=num_workers)
        assert "must be >= 1" in str(exc_info.value)

    def test_planned_epoch_bounds_the_sample_count(self, tmp_path: Path) -> None:
        """A planned epoch stops iteration at ``samples_per_worker`` samples."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=12), "train", transforms=None)
        dataset.configure_epoch(samples_per_worker=5, num_workers=1)
        assert len(list(dataset)) == 5

    @pytest.mark.parametrize(
        ("drop_member", "expected_message"),
        [
            pytest.param("jpg", "no image member", id="no-image"),
            pytest.param("json", "annotation sidecar", id="no-json"),
        ],
    )
    def test_malformed_sample_names_the_missing_member(
        self, tmp_path: Path, drop_member: str, expected_message: str
    ) -> None:
        """A shard sample missing its image or JSON member raises a KeyError naming the gap."""
        shard_dir = _pack(tmp_path, count=2)
        index = read_shard_index(shard_dir, "train")
        _rewrite_shard_without(shard_dir / index.shards[0], drop_member)
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        with pytest.raises(KeyError) as exc_info:
            list(dataset)
        assert expected_message in str(exc_info.value)


def _rewrite_shard_without(shard: Path, extension: str) -> None:
    """Rewrite *shard* in place, dropping every member with the given extension.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as temporary:
        ...     shard = Path(temporary) / "sample.tar"
        ...     with tarfile.open(shard, "w") as archive:
        ...         archive.addfile(tarfile.TarInfo("sample.jpg"), BytesIO())
        ...         archive.addfile(tarfile.TarInfo("sample.json"), BytesIO())
        ...     _rewrite_shard_without(shard, "json")
        ...     with tarfile.open(shard) as archive:
        ...         archive.getnames()
        ['sample.jpg']
    """
    with tarfile.open(shard) as tar:
        kept = [(member, tar.extractfile(member).read()) for member in tar.getmembers()]  # type: ignore[union-attr]
    with tarfile.open(shard, "w") as tar:
        for member, payload in kept:
            if member.name.endswith(f".{extension}"):
                continue
            tar.addfile(member, BytesIO(payload))


@webdataset_only
class TestStreamingShuffle:
    """Shard-order shuffling has to stay a partition within an epoch and change between epochs."""

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_rank_partition_survives_distinct_rngs(self, tmp_path: Path, persistent_workers: bool) -> None:
        """Real workers must share a global shard permutation despite rank-local RNG state."""
        shard_dir = _pack(tmp_path, count=24)
        loaders = []
        for rank in range(2):
            torch.manual_seed(42 + rank)
            dataset = WebDatasetDetection(shard_dir, "train", None, shard_shuffle=100, seed=42)
            loaders.append(
                build_webdataset_loader(
                    dataset,
                    batch_size=2,
                    collate_fn=_id_collate,
                    num_workers=1,
                    persistent_workers=persistent_workers,
                    world_size=2,
                    rank=rank,
                )
            )
        epochs = []
        for epoch in range(2):
            rank_ids = []
            for rank, loader in enumerate(loaders):
                torch.manual_seed(100 + 10 * epoch + rank)
                rank_ids.append([image_id for batch in loader for image_id in batch])
            assert len(rank_ids[0]) == len(rank_ids[1]) == 12
            assert not set(rank_ids[0]) & set(rank_ids[1])
            assert sorted(rank_ids[0] + rank_ids[1]) == list(range(1000, 1024))
            epochs.append(rank_ids)
        assert set(epochs[0][0]) != set(epochs[1][0])

    @pytest.mark.parametrize(
        ("num_workers", "persistent_workers"),
        [
            pytest.param(0, False, id="main-process"),
            pytest.param(2, False, id="two-workers"),
            # A persistent worker keeps its process (and torch seed) across epochs; it is the DataModule's own default
            # whenever num_workers > 0 (RFDETRDataModule._persistent_workers), so it has to reshuffle too.
            pytest.param(2, True, id="persistent-workers"),
        ],
    )
    def test_sample_order_changes_between_epochs(
        self, tmp_path: Path, num_workers: int, persistent_workers: bool
    ) -> None:
        """With shuffling on, each epoch covers the whole split in a new order, persistent workers included."""
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=24), "train", transforms=None, shuffle_buffer=8, shard_shuffle=4
        )
        loader = build_webdataset_loader(
            dataset,
            batch_size=2,
            collate_fn=_id_collate,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            fixed_epoch=False,
            world_size=1,
        )
        first = [image_id for batch in loader for image_id in batch]
        second = [image_id for batch in loader for image_id in batch]
        assert sorted(first) == sorted(second) == list(range(1000, 1024))
        assert first != second

    def test_shuffled_epoch_stays_a_partition_across_workers(self, tmp_path: Path) -> None:
        """Four workers together stream every sample once per shuffled epoch."""
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=24), "train", transforms=None, shuffle_buffer=8, shard_shuffle=4
        )
        loader = build_webdataset_loader(
            dataset, batch_size=2, collate_fn=_id_collate, num_workers=4, fixed_epoch=False, world_size=1
        )
        streamed = [image_id for batch in loader for image_id in batch]
        assert sorted(streamed) == list(range(1000, 1024))

    def test_shard_to_worker_assignment_rotates_between_epochs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker's shard subset changes epoch to epoch instead of being frozen for the whole run.

        Regression test for the root cause behind the PR's own reported 2-3x seed-to-seed accuracy variance:
        ``webdataset``'s ``nodesplitter``/``workersplitter`` run on the shard-URL list's existing order, *before* its
        own shard-order shuffler (verified against the pinned ``webdataset==1.0.2`` source), so leaving that list
        unshuffled gave every worker the same fixed shard subset every epoch — only the order *within* that fixed subset
        varied. The test above stays a partition either way (it would pass against the pre-fix code too), so it cannot
        tell the two apart; this one checks the actual set one worker sees, which the pre-fix code could not change.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=24), "train", transforms=None, shard_shuffle=4)
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda: types.SimpleNamespace(id=0, num_workers=4, seed=42),
        )
        first_epoch = frozenset(target["image_id"] for _, target in dataset)
        second_epoch = frozenset(target["image_id"] for _, target in dataset)
        assert first_epoch != second_epoch

    def test_streaming_emits_no_webdataset_warning(self, tmp_path: Path) -> None:
        """Streaming a shuffled split emits no UserWarning from webdataset."""
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=8), "train", transforms=None, shuffle_buffer=4, shard_shuffle=4
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert len(list(dataset)) == 8


class TestPlanSamplesPerWorker:
    """Epoch planning floors to a whole number of accumulation windows so the reported length is exact."""

    @pytest.mark.parametrize(
        ("total", "batch_size", "num_workers", "world_size", "grad_accum_steps", "expected"),
        [
            pytest.param(1000, 4, 2, 1, 1, 500, id="exact-division"),
            pytest.param(1000, 16, 3, 1, 1, 320, id="floors-to-batch-multiple"),
            pytest.param(1000, 4, 2, 2, 1, 248, id="splits-across-ranks"),
            pytest.param(1000, 4, 0, 1, 1, 1000, id="zero-workers-counts-as-one"),
            pytest.param(1000, 4, 2, 1, 8, 496, id="floors-to-accumulation-window"),
            pytest.param(8, 4, 2, 1, 2, 4, id="workers-share-window"),
            pytest.param(48, 4, 3, 1, 4, 16, id="coprime-worker-window"),
            pytest.param(64, 4, 4, 1, 6, 12, id="common-divisor-window"),
        ],
    )
    def test_plan_is_a_whole_number_of_batches(
        self, total: int, batch_size: int, num_workers: int, world_size: int, grad_accum_steps: int, expected: int
    ) -> None:
        """The plan equals the expected size and is a whole number of batches and accumulation windows."""
        planned = plan_samples_per_worker(
            total,
            batch_size=batch_size,
            num_workers=num_workers,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
        )
        assert planned == expected
        assert planned % batch_size == 0
        assert (planned // batch_size * max(1, num_workers)) % grad_accum_steps == 0

    def test_split_too_small_for_one_batch_per_worker_is_rejected(self) -> None:
        """A split too small to give each worker one batch is rejected."""
        with pytest.raises(ValueError) as exc_info:
            plan_samples_per_worker(10, batch_size=4, num_workers=8)
        assert "cannot fill one accumulation window" in str(exc_info.value)

    def test_split_too_small_for_one_accumulation_window_is_rejected(self) -> None:
        """A split too small for one accumulation window is rejected."""
        with pytest.raises(ValueError) as exc_info:
            plan_samples_per_worker(79, batch_size=4, num_workers=2, grad_accum_steps=20)
        assert "cannot fill one accumulation window" in str(exc_info.value)


@webdataset_only
class TestBuildWebdatasetLoader:
    """Training plans a fixed epoch; evaluation passes over every sample exactly once."""

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_training_length_matches_the_batches_produced(self, tmp_path: Path, num_workers: int) -> None:
        """The training loader's ``len`` equals the batches it yields, all of full size."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=24), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=4, collate_fn=_count_collate, num_workers=num_workers, world_size=1
        )
        batches = list(loader)
        assert len(loader) == len(batches)
        assert set(batches) == {4}

    def test_training_epoch_is_a_whole_number_of_accumulation_windows(self, tmp_path: Path) -> None:
        """The training loader length is a multiple of ``grad_accum_steps`` and matches its batches."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=64), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=2, collate_fn=_count_collate, num_workers=1, world_size=1, grad_accum_steps=3
        )
        assert len(loader) % 3 == 0
        assert len(loader) == len(list(loader))

    def test_evaluation_sees_every_sample_exactly_once(self, tmp_path: Path) -> None:
        """The evaluation loader streams every sample exactly once, none dropped or repeated."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=14), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=4, collate_fn=_id_collate, num_workers=2, fixed_epoch=False, world_size=1
        )
        streamed = [image_id for batch in loader for image_id in batch]
        assert sorted(streamed) == list(range(1000, 1014))

    def test_evaluation_loader_reports_no_length(self, tmp_path: Path) -> None:
        """The evaluation loader is unsized: ``len`` raises TypeError."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=4, collate_fn=_count_collate, num_workers=0, fixed_epoch=False, world_size=1
        )
        with pytest.raises(TypeError):
            len(loader)

    @pytest.mark.parametrize("fixed_epoch", [True, False])
    def test_fewer_shards_than_ranks_is_rejected_on_both_paths(self, tmp_path: Path, fixed_epoch: bool) -> None:
        """A rank with no shard never reaches the step function while the others do, which deadlocks DDP.

        Evaluation used to skip this check, so only training was protected.
        """
        shard_dir = _pack(tmp_path, count=8)
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        shards = len(dataset.index.shards)
        with pytest.raises(ValueError) as exc_info:
            build_webdataset_loader(
                dataset,
                batch_size=2,
                collate_fn=_count_collate,
                num_workers=1,
                fixed_epoch=fixed_epoch,
                world_size=shards + 1,
            )
        assert "rank" in str(exc_info.value)

    def test_explicit_world_size_and_rank_apply_without_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The node split uses the rank/world_size the loader resolved once, not each worker's own env guess.

        Regression test: ``webdataset.split_by_node`` re-resolves rank and world size inside whichever process
        calls it, preferring ``RANK``/``WORLD_SIZE`` environment variables and falling back to
        ``torch.distributed`` only if those are unset. With neither set and no process group initialised (as in
        this test, and as a ``spawn``-started worker with a launcher that exports neither would see it), that
        resolution used to silently give ``world_size=1`` regardless of what the loader was told, so every rank
        streamed the entire split, duplicated.
        """
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        shard_dir = _pack(tmp_path, count=24)
        loader0 = build_webdataset_loader(
            WebDatasetDetection(shard_dir, "train", transforms=None),
            batch_size=2,
            collate_fn=_id_collate,
            num_workers=0,
            fixed_epoch=False,
            world_size=2,
            rank=0,
        )
        loader1 = build_webdataset_loader(
            WebDatasetDetection(shard_dir, "train", transforms=None),
            batch_size=2,
            collate_fn=_id_collate,
            num_workers=0,
            fixed_epoch=False,
            world_size=2,
            rank=1,
        )
        ids0 = {image_id for batch in loader0 for image_id in batch}
        ids1 = {image_id for batch in loader1 for image_id in batch}
        assert ids0 and ids1
        assert ids0.isdisjoint(ids1)
        assert ids0 | ids1 == set(range(1000, 1024))

    def test_rank_and_worker_splits_compose_into_one_partition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixed nodesplitter and webdataset's own workersplitter don't overlap when combined.

        The single-rank test above (``num_workers=4``, ``world_size=1``) exercises only the worker split, and
        ``test_explicit_world_size_and_rank_apply_without_env_vars`` above exercises only the rank
        split (``num_workers=0``). Neither proves the two compose cleanly: a bug that made the workersplitter
        ignore which rank's shard subset it was handed (e.g. re-deriving shards from the full index instead of
        the node-split iterator) would pass both of those and still duplicate or drop samples once both splits
        are active together, which is the actual training configuration. Drives ``configure_distribution`` and a
        monkeypatched ``get_worker_info`` directly against all four (rank, worker) cells in one process, rather
        than real ``num_workers`` DataLoader subprocesses per rank: two live multi-worker ``DataLoader``\\ s
        constructed back to back in one test process take noticeably longer here and are harder to reason about
        (each parent blocks on its own worker subprocesses, so its own CPU time is not a reliable progress signal
        while waiting on it), so this drives the same split composition the loader relies on directly and fast,
        without going through DataLoader at all.
        """
        shard_dir = _pack(tmp_path, count=32)

        def _cell(rank: int, worker_id: int) -> frozenset[int]:
            dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
            dataset.configure_distribution(rank=rank, world_size=2)
            monkeypatch.setattr(
                torch.utils.data,
                "get_worker_info",
                lambda: types.SimpleNamespace(id=worker_id, num_workers=2, seed=42 + worker_id),
            )
            return frozenset(int(target["image_id"]) for _, target in dataset)

        rank0_worker0 = _cell(rank=0, worker_id=0)
        rank0_worker1 = _cell(rank=0, worker_id=1)
        rank1_worker0 = _cell(rank=1, worker_id=0)
        rank1_worker1 = _cell(rank=1, worker_id=1)
        assert rank0_worker0 and rank0_worker1 and rank1_worker0 and rank1_worker1
        assert rank0_worker0.isdisjoint(rank0_worker1)
        assert rank0_worker0.isdisjoint(rank1_worker0)
        assert rank0_worker0.isdisjoint(rank1_worker1)
        assert rank0_worker1.isdisjoint(rank1_worker0)
        assert rank0_worker1.isdisjoint(rank1_worker1)
        assert rank1_worker0.isdisjoint(rank1_worker1)
        assert rank0_worker0 | rank0_worker1 | rank1_worker0 | rank1_worker1 == frozenset(range(1000, 1032))

    def test_more_workers_than_shards_is_rejected_for_training(self, tmp_path: Path) -> None:
        """Training with more workers than shards is rejected, since some worker would get none."""
        image_dir, annotations = _build_coco_split(tmp_path, count=16)
        shard_dir = tmp_path / "shards"
        index = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        with pytest.raises(ValueError) as exc_info:
            build_webdataset_loader(
                dataset, batch_size=2, collate_fn=_count_collate, num_workers=len(index.shards) + 1, world_size=1
            )
        assert "cannot cover" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("num_workers", "expect_warning"),
        [
            # 48 shards over 7 workers is 7x6 + 6: the short worker is 12.5% under the average.
            pytest.param(7, True, id="uneven-split-warns"),
            # 48 shards over 8 workers is 6 each: nothing is short.
            pytest.param(8, False, id="even-split-is-quiet"),
        ],
    )
    def test_uneven_shard_split_is_reported(
        self, tmp_path: Path, capsys: Any, num_workers: int, expect_warning: bool
    ) -> None:
        """A shard split leaving a worker short of the average warns; an even split stays quiet."""
        image_dir, annotations = _build_coco_split(tmp_path, count=48)
        shard_dir = tmp_path / "shards"
        # One sample per shard, so the shard count is exactly 48 whatever the images weigh.
        index = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=1)
        assert len(index.shards) == 48
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=2, collate_fn=_count_collate, num_workers=num_workers, world_size=1)
        assert ("fewer samples than the epoch asks" in capsys.readouterr().err) is expect_warning

    @pytest.mark.parametrize(
        ("batch_size", "expect_warning"),
        [
            # 24 samples / 2 workers = 12, floored to a multiple of 5 -> 10: 4 samples (16.7%) never seen.
            pytest.param(5, True, id="floors-away-more-than-the-threshold-warns"),
            # window=1 never has a remainder to floor away.
            pytest.param(1, False, id="no-remainder-is-quiet"),
        ],
    )
    def test_epoch_floor_loss_is_reported(
        self, tmp_path: Path, capsys: Any, batch_size: int, expect_warning: bool
    ) -> None:
        """Flooring the fixed epoch to a whole accumulation window can silently drop a chunk of the split.

        Regression test: unlike the map-style loader (which pads to the same boundary),
        plan_samples_per_worker floors, so a split not already an exact multiple of
        batch_size * grad_accum_steps * slots is never seen in full, every epoch, with no warning before this.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=24, subdir="ep"), "train", transforms=None)
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=batch_size, collate_fn=_count_collate, num_workers=2, world_size=1)
        assert ("never seen this epoch" in capsys.readouterr().err) is expect_warning

    def test_skew_warning_uses_real_per_shard_counts_not_shard_count_alone(self, tmp_path: Path, capsys: Any) -> None:
        """Regression test: shards are cut by byte size, so a count-only estimate can miss a real skew entirely.

        Two shards split evenly by *count* (one per worker) would score 0% deficit under a shard-count-only estimate,
        yet here one shard carries 60% of the samples (20% worse than average): the real per-shard counts the packer now
        records have to be what drives the warning. Deficit stays below SHARD_SKEW_RAISE_FRACTION on purpose, so this
        exercises the warning path, not the raise path below.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None)
        dataset.index = replace(
            dataset.index, shards=("a.tar", "b.tar"), num_samples=1000, samples_per_shard=(600, 400)
        )
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=2, world_size=1)
        assert "fewer samples than the epoch asks" in capsys.readouterr().err

    def test_extreme_shard_skew_raises_instead_of_warning(self, tmp_path: Path) -> None:
        """A skew far past the warning threshold raises instead of logging a line nobody watches live.

        Every degradation mode this loader can measure is otherwise silent past a warning in a training log; a 98%/2%
        split is not a tuning nuisance worth a log line, it is close enough to one worker seeing almost none of its
        assigned share that failing fast is cheaper than discovering it from a degraded metric hours into a run.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None)
        dataset.index = replace(dataset.index, shards=("a.tar", "b.tar"), num_samples=1000, samples_per_shard=(980, 20))
        with pytest.raises(ValueError) as exc_info:
            build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=2, world_size=1)
        assert "no longer a tuning nuisance" in str(exc_info.value)

    def test_shuffled_skew_is_checked_again_on_each_epoch(self, tmp_path: Path) -> None:
        """A balanced first permutation must not mask an unsafe assignment in a later epoch."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None, shard_shuffle=4, seed=0)
        dataset.index = replace(dataset.index, num_samples=200, samples_per_shard=(90, 90, 10, 10))
        build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=0, world_size=2, rank=0)
        # Seed 0 assigns (100, 100); seed 1 assigns (20, 180). Validate before opening any shard.
        iter(dataset)
        with pytest.raises(ValueError) as exc_info:
            iter(dataset)
        assert "no longer a tuning nuisance" in str(exc_info.value)

    def test_shuffled_balanced_assignment_ignores_packing_order_skew(self, tmp_path: Path) -> None:
        """An unsafe packing order must not reject a safe shuffled assignment."""
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None, shard_shuffle=4, seed=1)
        dataset.index = replace(dataset.index, num_samples=200, samples_per_shard=(90, 10, 90, 10))
        build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=0, world_size=2, rank=0)
        # Seed 1 assigns (100, 100), although packing-order totals are (180, 20).
        iter(dataset)

    def test_epoch_plan_is_logged_unconditionally(self, tmp_path: Path, capsys: Any) -> None:
        """The planned-vs-total sample count is always logged at INFO, not only when something looks wrong.

        Regression test: every degradation mode was otherwise silent unless it happened to cross a warning
        threshold, so a run that stays under every threshold still left no record of what the fixed-epoch plan
        actually was.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=24), "train", transforms=None)
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=1, world_size=1)
        assert "fixed training epoch plans" in capsys.readouterr().out

    def test_skew_threshold_is_a_fraction(self) -> None:
        """The warn fraction is positive and below the raise fraction, which is below one."""
        assert 0.0 < SHARD_SKEW_WARN_FRACTION < SHARD_SKEW_RAISE_FRACTION < 1.0

    @pytest.mark.parametrize(
        ("samples_per_shard", "expect_warning"),
        [
            pytest.param((980, 20), True, id="uneven-split-warns"),
            pytest.param((500, 500), False, id="even-split-is-quiet"),
        ],
    )
    def test_uneven_eval_split_across_ranks_is_reported(
        self, tmp_path: Path, capsys: Any, samples_per_shard: tuple[int, int], expect_warning: bool
    ) -> None:
        """An uneven per-rank evaluation split is surfaced; every rank has a shard but runs a different batch count.

        That mismatch is the follow-on deadlock risk the fewer-shards-than-ranks check alone does not cover.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None)
        dataset.index = replace(
            dataset.index,
            shards=("a.tar", "b.tar"),
            num_samples=sum(samples_per_shard),
            samples_per_shard=samples_per_shard,
        )
        capsys.readouterr()
        build_webdataset_loader(
            dataset, batch_size=1, collate_fn=_count_collate, num_workers=1, fixed_epoch=False, world_size=2
        )
        assert ("different numbers of evaluation batches" in capsys.readouterr().err) is expect_warning

    def test_batches_collate_into_the_model_input_contract(self, tmp_path: Path) -> None:
        """Batches collate into padded nested tensors with one target per sample."""
        transforms = make_coco_transforms("val", 224)
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=transforms)
        loader = build_webdataset_loader(
            dataset, batch_size=2, collate_fn=make_collate_fn(block_size=64), num_workers=0, world_size=1
        )
        samples, targets = next(iter(loader))
        assert samples.tensors.shape[0] == 2
        assert samples.tensors.shape[-1] % 64 == 0
        assert len(targets) == 2


def _count_collate(batch: list[tuple[Any, Any]]) -> int:
    """Collate to the batch size alone, for tests that only count samples.

    Examples:
        >>> _count_collate([(None, {}), (None, {})])
        2
    """
    return len(batch)


def _id_collate(batch: list[tuple[Any, Any]]) -> list[int]:
    """Collate to the batch's image ids, for tests that check coverage.

    Examples:
        >>> import torch
        >>> _id_collate([(None, {"image_id": torch.tensor([7])})])
        [7]
    """
    return [int(target["image_id"]) for _, target in batch]


@webdataset_only
class TestBuildWebdataset:
    """The dataset builder mirrors the loose-file builders' conventions."""

    @staticmethod
    def _namespace(dataset_dir: Path, **overrides: Any) -> types.SimpleNamespace:
        """Build the merged model/train namespace :func:`~rfdetr.datasets.webdataset.load.build_webdataset` expects.

        Args:
            dataset_dir: Directory holding the packed shards.
            **overrides: Fields to override on top of the defaults.

        Returns:
            A namespace with every attribute :func:`build_webdataset` reads.

        Examples:
            >>> TestBuildWebdataset._namespace(Path("/data/shards")).dataset_file
            'webdataset'
            >>> TestBuildWebdataset._namespace(Path("/data/shards"), segmentation_head=True).segmentation_head
            True
        """
        defaults: dict[str, Any] = {
            "dataset_dir": str(dataset_dir),
            "dataset_file": "webdataset",
            "multi_scale": False,
            "expanded_scales": False,
            "patch_size": 16,
            "num_windows": 4,
            "square_resize_div_64": False,
            "segmentation_head": False,
            "use_grouppose_keypoints": False,
            "aug_config": {},
            "scale_jitter": False,
            "augmentation_backend": "cpu",
            "seed": 0,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_missing_shard_directory_is_reported(self, tmp_path: Path) -> None:
        """A nonexistent shard directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            build_webdataset("train", self._namespace(tmp_path / "absent"), 224)
        assert "does not exist" in str(exc_info.value)

    def test_missing_train_index_names_itself_not_the_requested_split(self, tmp_path: Path) -> None:
        """Evaluating a shard directory with no packed 'train' split names the real gap, not 'val'.

        Regression test: adopting the train split's label space for any non-train split unconditionally reads
        the 'train' index, so a shard directory that only ever packed 'val' used to fail with "No WebDataset
        index for split 'train'" while the caller asked to evaluate 'val' -- reading as though 'train' were the
        requested split rather than a prerequisite for evaluating the one that was.
        """
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="val")
        with pytest.raises(WebDatasetSplitUnavailableError) as exc_info:
            build_webdataset("val", self._namespace(shard_dir), 224)
        assert "needs a packed 'train' index" in str(exc_info.value)

    def test_keypoint_training_is_rejected_with_a_pointer_to_the_other_formats(self, tmp_path: Path) -> None:
        """Keypoint training on shards raises NotImplementedError mentioning keypoints."""
        shard_dir = _pack(tmp_path, count=2)
        namespace = self._namespace(shard_dir, use_grouppose_keypoints=True)
        with pytest.raises(NotImplementedError) as exc_info:
            build_webdataset("train", namespace, 224)
        assert "keypoint" in str(exc_info.value)

    @pytest.mark.parametrize("split", ["val", "test"])
    @pytest.mark.parametrize(
        ("category_ids", "expected_mapping", "expected_names", "expected_label"),
        [
            pytest.param("remap", {3: 0, 9: 1}, ["cat", "dog"], 1, id="remapped"),
            pytest.param("raw", None, ["", "", "", "cat", "", "", "", "", "", "dog"], 9, id="raw"),
        ],
    )
    def test_non_train_split_adopts_the_train_label_space(
        self,
        tmp_path: Path,
        split: str,
        category_ids: str,
        expected_mapping: dict[int, int] | None,
        expected_names: list[str],
        expected_label: int,
    ) -> None:
        """Evaluation keeps train-owned labels and names even when its category list is incomplete."""
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", category_ids=category_ids)
        eval_images, eval_annotations = _build_coco_split(tmp_path, count=2, categories=[_CATEGORIES[1]], subdir=split)
        payload = json.loads(eval_annotations.read_text())
        for annotation in payload["annotations"]:
            annotation["category_id"] = 9
        eval_annotations.write_text(json.dumps(payload))
        pack_coco_to_shards(eval_images, eval_annotations, shard_dir, split=split, category_ids=category_ids)
        dataset = build_webdataset(split, self._namespace(shard_dir), 224)
        assert dataset.cat2label == expected_mapping
        assert dataset.class_names == expected_names
        assert dataset.index.categories == (_CATEGORIES[1],)
        assert dataset.total_samples == 2
        assert [int(target["labels"][0]) for _, target in dataset] == [expected_label, expected_label]

    @pytest.mark.parametrize("square_resize_div_64", [False, True])
    def test_both_resize_pipelines_produce_model_ready_tensors(
        self, tmp_path: Path, square_resize_div_64: bool
    ) -> None:
        """Both resize pipelines yield a 3-D image and four-column boxes."""
        shard_dir = _pack(tmp_path, count=4)
        namespace = self._namespace(shard_dir, square_resize_div_64=square_resize_div_64)
        dataset = build_webdataset("train", namespace, 224)
        image, target = next(iter(dataset))
        assert image.ndim == 3
        assert target["boxes"].shape[-1] == 4

    def test_category_policy_mismatch_between_splits_is_rejected(self, tmp_path: Path) -> None:
        """Train packed raw beside val packed remap would score two different label spaces against each other."""
        shard_dir = tmp_path / "shards"
        train_images, train_annotations = _build_coco_split(tmp_path, count=6, subdir="tr")
        pack_coco_to_shards(train_images, train_annotations, shard_dir, split="train", category_ids="raw")
        val_images, val_annotations = _build_coco_split(tmp_path, count=4, subdir="va")
        pack_coco_to_shards(val_images, val_annotations, shard_dir, split="val", category_ids="remap")
        with pytest.raises(ValueError) as exc_info:
            build_webdataset("val", self._namespace(shard_dir), 224)
        assert "category_ids" in str(exc_info.value)

    def test_matching_raw_policy_across_splits_is_accepted(self, tmp_path: Path) -> None:
        """Train and val both packed with ``raw`` build fine and keep raw category ids."""
        shard_dir = tmp_path / "shards"
        train_images, train_annotations = _build_coco_split(tmp_path, count=6, subdir="tr2")
        pack_coco_to_shards(train_images, train_annotations, shard_dir, split="train", category_ids="raw")
        val_images, val_annotations = _build_coco_split(tmp_path, count=4, subdir="va2")
        pack_coco_to_shards(val_images, val_annotations, shard_dir, split="val", category_ids="raw")
        dataset = build_webdataset("val", self._namespace(shard_dir), 224)
        assert dataset.cat2label is None

    def test_build_dataset_routes_the_webdataset_format(self, tmp_path: Path) -> None:
        """``build_dataset`` returns a WebDatasetDetection for ``dataset_file='webdataset'``."""
        shard_dir = _pack(tmp_path, count=4)
        dataset = build_dataset("train", self._namespace(shard_dir), 224)
        assert isinstance(dataset, WebDatasetDetection)


def _batched_sample_count(loader: Iterable[tuple[Any, Any]]) -> int:
    """Count the samples across batches whose ``samples.tensors`` stack along dimension 0.

    Examples:
        >>> batch = (types.SimpleNamespace(tensors=torch.zeros(2, 3)), None)
        >>> _batched_sample_count([batch, batch])
        4
    """
    return sum(int(samples.tensors.shape[0]) for samples, _ in loader)


def _setup_datamodule(tmp_path: Path, split_counts: dict[str, int], stage: str = "fit", **train_overrides: Any) -> Any:
    """Pack *split_counts* under *tmp_path* and return an ``RFDETRDataModule`` set up for *stage*.

    Skipped as a doctest: it needs the ``data`` and ``train`` extras and a real ``tmp_path``.

    Args:
        tmp_path: Root temporary directory; shards go to ``tmp_path / "shards"``.
        split_counts: Number of synthetic images to pack for each split name.
        stage: Lightning stage passed to ``setup``.
        **train_overrides: ``TrainConfig`` fields overriding the streaming-friendly defaults.

    Returns:
        The set-up data module.

    Examples:
        >>> _setup_datamodule(tmp_path, {"train": 64, "val": 32}).class_names  # doctest: +SKIP
        ['cat', 'dog']
    """
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.training.module_data import RFDETRDataModule

    shard_dir = tmp_path / "shards"
    for split, count in split_counts.items():
        image_dir, annotations = _build_coco_split(tmp_path, count=count, subdir=split)
        pack_coco_to_shards(image_dir, annotations, shard_dir, split=split, max_shard_bytes=8192)
    settings: dict[str, Any] = {
        "dataset_dir": str(shard_dir),
        "dataset_file": "webdataset",
        "batch_size": 4,
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "multi_scale": False,
        "expanded_scales": False,
        "tensorboard": False,
        "output_dir": str(tmp_path / "out"),
    }
    settings.update(train_overrides)
    module = RFDETRDataModule(RFDETRSmallConfig(pretrain_weights=None), TrainConfig(**settings))
    module.setup(stage)
    return module


@webdataset_only
@pytorch_lightning_only
class TestDataModuleStreaming:
    """The DataModule routes a streaming split away from the sampler-based loaders."""

    @pytest.fixture
    def datamodule(self, tmp_path: Path) -> Any:
        """Data module over freshly packed train (64) and val (32) splits, set up for fit."""
        return _setup_datamodule(tmp_path, {"train": 64, "val": 32}, grad_accum_steps=1)

    def test_training_loader_reports_the_batches_it_yields(self, datamodule: Any) -> None:
        """The training loader's ``len`` equals the number of batches it yields."""
        loader = datamodule.train_dataloader()
        assert len(loader) == len(list(loader))

    def test_training_batches_are_all_full(self, datamodule: Any) -> None:
        """Every training batch holds exactly ``batch_size`` samples, with no short tail batch."""
        loader = datamodule.train_dataloader()
        assert {int(samples.tensors.shape[0]) for samples, _ in loader} == {4}

    def test_training_loader_aligns_to_grad_accum_steps(self, tmp_path: Path) -> None:
        """The training loader length is a multiple of the configured ``grad_accum_steps``."""
        module = _setup_datamodule(tmp_path, {"train": 96, "val": 16}, batch_size=2, grad_accum_steps=3, num_workers=1)
        assert len(module.train_dataloader()) % 3 == 0

    def test_training_loader_aligns_to_trainer_accumulation_override(self, datamodule: Any) -> None:
        """Automatic optimization aligns the stream to the Trainer's effective accumulation value."""
        trainer_grad_accum_steps = 3
        datamodule.trainer = types.SimpleNamespace(
            world_size=1, global_rank=0, accumulate_grad_batches=trainer_grad_accum_steps
        )

        loader = datamodule.train_dataloader()

        assert len(loader) % trainer_grad_accum_steps == 0

    def test_validation_loader_is_unsized_and_covers_the_split(self, datamodule: Any) -> None:
        """The validation loader has no length and streams all 32 validation samples."""
        loader = datamodule.val_dataloader()
        with pytest.raises(TypeError):
            len(loader)
        assert _batched_sample_count(loader) == 32

    @pytest.mark.parametrize("stage", ["test", "predict"])
    def test_other_eval_loaders_also_stream_the_split(self, datamodule: Any, stage: str) -> None:
        """The test and predict loaders are unsized and cover the split; no test split is packed, so val is used."""
        datamodule.setup(stage)
        loader = getattr(datamodule, f"{stage}_dataloader")()
        with pytest.raises(TypeError):
            len(loader)
        assert _batched_sample_count(loader) == 32

    def test_packed_test_split_is_used_instead_of_val(self, tmp_path: Path) -> None:
        """When a test split is packed, the test loader streams it (16 samples) rather than val."""
        module = _setup_datamodule(tmp_path, {"train": 64, "val": 32, "test": 16}, stage="test")
        assert _batched_sample_count(module.test_dataloader()) == 16

    def test_absent_test_split_falls_back_to_val_and_says_so(self, datamodule: Any, capsys: Any) -> None:
        """Without a packed test split, the test loader streams val and logs that it did."""
        datamodule.setup("test")
        assert "evaluating the 'val' split instead" in capsys.readouterr().err
        assert _batched_sample_count(datamodule.test_dataloader()) == 32

    def test_class_names_come_from_the_shard_index(self, datamodule: Any) -> None:
        """The data module reports the class names recorded in the shard index."""
        assert datamodule.class_names == ["cat", "dog"]

    @matplotlib_only
    def test_sample_grid_rejects_a_streaming_split(self, datamodule: Any) -> None:
        """Drawing a sample grid needs a map-style dataset, so a streaming split raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            datamodule._show_samples(2, split="train")
        assert "map-style dataset" in str(exc_info.value)
