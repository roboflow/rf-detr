# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the WebDataset sequential-I/O pipeline.

Cover the packer (standard library only), the shard index contract, epoch planning arithmetic, and — behind an
``importorskip`` on the optional ``webdataset`` extra — streaming, sizing and parity against the loose-file
:class:`~rfdetr.datasets.coco.CocoDetection` the shards were packed from.
"""

from __future__ import annotations

import io
import json
import tarfile
import types
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets import build_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms
from rfdetr.datasets.webdataset_io import (
    DEFAULT_MAX_SHARD_BYTES,
    INDEX_VERSION,
    SHARD_SKEW_WARN_FRACTION,
    ShardIndex,
    WebDatasetDetection,
    WebDatasetSplitUnavailableError,
    build_webdataset,
    build_webdataset_loader,
    index_name,
    pack_coco_to_shards,
    plan_samples_per_worker,
    read_shard_index,
)
from rfdetr.utilities.tensors import make_collate_fn

_CATEGORIES = [{"id": 3, "name": "cat"}, {"id": 9, "name": "dog"}]


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
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        index = pack_coco_to_shards(
            image_dir, annotations, tmp_path / "shards", split="train", max_shard_bytes=max_shard_bytes
        )
        assert (len(index.shards) == 1) is expected_single_shard
        assert sum(1 for _ in (tmp_path / "shards").glob("train-*.tar")) == len(index.shards)

    def test_image_bytes_are_copied_verbatim(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            member = tar.extractfile("00000000.jpg")
            assert member is not None
            assert member.read() == (image_dir / "img_0000.jpg").read_bytes()

    def test_images_without_annotations_get_an_empty_list(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=4, empty_from=2)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            member = tar.extractfile("00000003.json")
            assert member is not None
            assert json.loads(member.read())["annotations"] == []

    def test_packing_twice_produces_byte_identical_shards(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=4)
        first = pack_coco_to_shards(image_dir, annotations, tmp_path / "a", split="train")
        second = pack_coco_to_shards(image_dir, annotations, tmp_path / "b", split="train")
        assert first.shards == second.shards
        for shard in first.shards:
            assert (tmp_path / "a" / shard).read_bytes() == (tmp_path / "b" / shard).read_bytes()

    def test_png_split_keeps_its_own_member_extension(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=2, extension="png")
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train")
        with tarfile.open(tmp_path / "shards" / index.shards[0]) as tar:
            assert "00000000.png" in tar.getnames()

    def test_packer_records_samples_per_shard(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        index = pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train", max_shard_bytes=1)
        assert len(index.samples_per_shard) == len(index.shards)
        assert sum(index.samples_per_shard) == index.num_samples
        assert all(count == 1 for count in index.samples_per_shard)


class TestPackCocoToShardsFailures:
    """The packer fails closed rather than dropping data silently, and never touches a valid prior pack."""

    def test_missing_image_is_fatal(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        (image_dir / "img_0001.jpg").unlink()
        with pytest.raises(FileNotFoundError, match="silently dropping"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")

    def test_missing_annotation_file_is_fatal(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pack_coco_to_shards(tmp_path, tmp_path / "absent.json", tmp_path / "shards")

    def test_annotation_file_without_images_is_rejected(self, tmp_path: Path) -> None:
        annotations = tmp_path / "empty.json"
        annotations.write_text(json.dumps({"images": [], "categories": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="lists no images"):
            pack_coco_to_shards(tmp_path, annotations, tmp_path / "shards")

    def test_unsupported_image_extension_fails_at_pack_time(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        (image_dir / "img_0000.jpg").rename(image_dir / "img_0000.tiff")
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["images"][0]["file_name"] = "img_0000.tiff"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot decode"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")

    def test_annotations_that_match_no_image_are_rejected(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        # A COCO export whose image ids are strings while annotations keep ints, or vice versa.
        for entry in payload["images"]:
            entry["id"] = str(entry["id"])
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="matching no image"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")

    def test_partially_orphaned_annotations_are_rejected(self, tmp_path: Path) -> None:
        """A minority of mismatched annotations must not be dropped silently either."""
        image_dir, annotations = _build_coco_split(tmp_path, count=3)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["annotations"][0]["image_id"] = "no-such-image"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="matching no image"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")

    @pytest.mark.parametrize("max_shard_bytes", [pytest.param(0, id="zero"), pytest.param(-1, id="negative")])
    def test_non_positive_shard_size_is_rejected(self, tmp_path: Path, max_shard_bytes: int) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=1)
        with pytest.raises(ValueError, match="max_shard_bytes"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", max_shard_bytes=max_shard_bytes)

    def test_file_name_outside_image_dir_is_rejected(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        payload["images"][0]["file_name"] = "../escape.jpg"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        (image_dir.parent / "escape.jpg").write_bytes(b"not a real image")
        with pytest.raises(ValueError, match="resolves outside"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards")

    def test_split_with_path_separator_is_rejected(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=1)
        with pytest.raises(ValueError, match="path separator"):
            pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="../escape")

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


class TestShardIndex:
    """The index carries the label space so a reader never parses the source annotation file."""

    def test_json_roundtrip_preserves_every_field(self) -> None:
        index = ShardIndex("train", ("train-000000.tar",), 5, tuple(_CATEGORIES), (3, 9), "remap", (5,))
        assert ShardIndex.from_json(index.to_json()) == index

    def test_unsupported_schema_version_is_rejected(self) -> None:
        payload = ShardIndex("train", (), 0, (), (), "remap").to_json()
        payload["version"] = INDEX_VERSION + 1
        with pytest.raises(ValueError, match="schema version"):
            ShardIndex.from_json(payload)

    def test_unknown_category_policy_is_rejected(self) -> None:
        payload = ShardIndex("train", (), 0, (), (), "remap").to_json()
        payload["category_ids"] = "contiguous"
        with pytest.raises(ValueError, match="category_ids policy"):
            ShardIndex.from_json(payload)

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            pytest.param("remap", {3: 0, 9: 1}, id="remap-assigns-contiguous-labels"),
            pytest.param("raw", None, id="raw-keeps-source-ids"),
        ],
    )
    def test_cat2label_follows_the_declared_policy(self, policy: str, expected: dict[int, int] | None) -> None:
        index = ShardIndex("train", (), 0, tuple(_CATEGORIES), (3, 9), policy)  # type: ignore[arg-type]
        assert index.cat2label() == expected

    def test_unannotated_grouping_category_consumes_no_label_slot(self) -> None:
        categories = (
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 1, "name": "cat", "supercategory": "root"},
            {"id": 2, "name": "dog", "supercategory": "root"},
        )
        assert ShardIndex("train", (), 0, categories, (1, 2), "remap").cat2label() == {1: 0, 2: 1}

    def test_packed_policy_reaches_the_reader(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=2)
        pack_coco_to_shards(image_dir, annotations, tmp_path / "shards", split="train", category_ids="raw")
        assert read_shard_index(tmp_path / "shards", "train").cat2label() is None

    def test_missing_index_names_the_packing_command(self, tmp_path: Path) -> None:
        with pytest.raises(WebDatasetSplitUnavailableError, match="webdataset_io"):
            read_shard_index(tmp_path, "train")

    def test_missing_split_error_is_still_a_file_not_found(self, tmp_path: Path) -> None:
        """Callers that only care the split is absent keep catching FileNotFoundError."""
        assert issubclass(WebDatasetSplitUnavailableError, FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            read_shard_index(tmp_path, "train")

    def test_split_with_path_separator_is_rejected_on_read(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path separator"):
            read_shard_index(tmp_path, "../escape")


def _pack(tmp_path: Path, **kwargs: Any) -> Path:
    """Pack a synthetic split into *tmp_path* / ``"shards"`` and return that directory.

    Examples:
        >>> _pack  # doctest: +SKIP
        Needs a real tmp_path to write files into, so it cannot run standalone.
    """
    image_dir, annotations = _build_coco_split(tmp_path, **kwargs)
    shard_dir = tmp_path / "shards"
    pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=4096)
    return shard_dir


class TestWebDatasetDetection:
    """Streaming a packed split reproduces the loose-file dataset it was packed from."""

    @pytest.fixture(autouse=True)
    def _require_webdataset(self) -> None:
        """Skip every test in this class when the optional ``webdataset`` extra is not installed.

        Examples:
            >>> pass  # doctest: +SKIP
            A pytest fixture, only runnable through pytest's fixture injection.
        """
        pytest.importorskip("webdataset")

    def test_every_sample_is_streamed_once(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=12), "train", transforms=None)
        image_ids = [int(target["image_id"]) for _, target in dataset]
        assert sorted(image_ids) == list(range(1000, 1012))

    def test_output_matches_the_loose_file_dataset(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
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

    def test_segmentation_masks_match_the_loose_file_dataset(self, tmp_path: Path) -> None:
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
        image_dir, annotations = _build_coco_split(tmp_path, count=4)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None, cat2label={3: 0, 9: 2})
        assert dataset.class_names == ["cat", "", "dog"]

    def test_length_is_a_type_error_until_an_epoch_is_planned(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        with pytest.raises(TypeError, match="no planned epoch length"):
            len(dataset)
        dataset.configure_epoch(samples_per_worker=4, num_workers=2)
        assert len(dataset) == 8

    @pytest.mark.parametrize(
        ("samples_per_worker", "num_workers"),
        [pytest.param(0, 1, id="zero-samples"), pytest.param(4, 0, id="zero-workers")],
    )
    def test_configure_epoch_rejects_degenerate_plans(
        self, tmp_path: Path, samples_per_worker: int, num_workers: int
    ) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        with pytest.raises(ValueError, match="must be >= 1"):
            dataset.configure_epoch(samples_per_worker=samples_per_worker, num_workers=num_workers)

    def test_planned_epoch_bounds_the_sample_count(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=12), "train", transforms=None)
        dataset.configure_epoch(samples_per_worker=5, num_workers=1)
        assert len(list(dataset)) == 5

    @pytest.mark.parametrize(
        ("drop_member", "match"),
        [
            pytest.param("jpg", "no image member", id="no-image"),
            pytest.param("json", "annotation sidecar", id="no-json"),
        ],
    )
    def test_malformed_sample_names_the_missing_member(self, tmp_path: Path, drop_member: str, match: str) -> None:
        shard_dir = _pack(tmp_path, count=2)
        index = read_shard_index(shard_dir, "train")
        _rewrite_shard_without(shard_dir / index.shards[0], drop_member)
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        with pytest.raises(KeyError, match=match):
            list(dataset)


def _rewrite_shard_without(shard: Path, extension: str) -> None:
    """Rewrite *shard* in place, dropping every member with the given extension.

    Examples:
        >>> _rewrite_shard_without  # doctest: +SKIP
        Operates on a packed shard produced by a fixture, so it cannot run standalone.
    """
    with tarfile.open(shard) as tar:
        kept = [(member, tar.extractfile(member).read()) for member in tar.getmembers()]  # type: ignore[union-attr]
    with tarfile.open(shard, "w") as tar:
        for member, payload in kept:
            if member.name.endswith(f".{extension}"):
                continue
            tar.addfile(member, io.BytesIO(payload))


class TestStreamingShuffle:
    """Shard-order shuffling has to stay a partition within an epoch and change between epochs."""

    @pytest.fixture(autouse=True)
    def _require_webdataset(self) -> None:
        """Skip every test in this class when the optional ``webdataset`` extra is not installed.

        Examples:
            >>> pass  # doctest: +SKIP
            A pytest fixture, only runnable through pytest's fixture injection.
        """
        pytest.importorskip("webdataset")

    @pytest.mark.parametrize("num_workers", [pytest.param(0, id="main-process"), pytest.param(2, id="two-workers")])
    def test_sample_order_changes_between_epochs(self, tmp_path: Path, num_workers: int) -> None:
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=24), "train", transforms=None, shuffle_buffer=8, shard_shuffle=4
        )
        loader = build_webdataset_loader(
            dataset, batch_size=2, collate_fn=_id_collate, num_workers=num_workers, fixed_epoch=False, world_size=1
        )
        first = [image_id for batch in loader for image_id in batch]
        second = [image_id for batch in loader for image_id in batch]
        assert sorted(first) == sorted(second) == list(range(1000, 1024))
        assert first != second

    def test_sample_order_changes_between_epochs_with_persistent_workers(self, tmp_path: Path) -> None:
        """Regression test: a persistent worker keeps its process (and its torch seed) across epochs.

        ``persistent_workers=True`` is the DataModule's own default whenever ``num_workers > 0``
        (:attr:`~rfdetr.training.module_data.RFDETRDataModule._persistent_workers`), so this has to reshuffle too,
        not just the non-persistent case above.
        """
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=24), "train", transforms=None, shuffle_buffer=8, shard_shuffle=4
        )
        loader = build_webdataset_loader(
            dataset,
            batch_size=2,
            collate_fn=_id_collate,
            num_workers=2,
            persistent_workers=True,
            fixed_epoch=False,
            world_size=1,
        )
        first = [image_id for batch in loader for image_id in batch]
        second = [image_id for batch in loader for image_id in batch]
        assert sorted(first) == sorted(second) == list(range(1000, 1024))
        assert first != second

    def test_shuffled_epoch_stays_a_partition_across_workers(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(
            _pack(tmp_path, count=24), "train", transforms=None, shuffle_buffer=8, shard_shuffle=4
        )
        loader = build_webdataset_loader(
            dataset, batch_size=2, collate_fn=_id_collate, num_workers=4, fixed_epoch=False, world_size=1
        )
        streamed = [image_id for batch in loader for image_id in batch]
        assert sorted(streamed) == list(range(1000, 1024))

    def test_streaming_emits_no_webdataset_warning(self, tmp_path: Path) -> None:
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
            pytest.param(1000, 4, 2, 1, 8, 480, id="floors-to-accumulation-window"),
        ],
    )
    def test_plan_is_a_whole_number_of_batches(
        self, total: int, batch_size: int, num_workers: int, world_size: int, grad_accum_steps: int, expected: int
    ) -> None:
        planned = plan_samples_per_worker(
            total,
            batch_size=batch_size,
            num_workers=num_workers,
            world_size=world_size,
            grad_accum_steps=grad_accum_steps,
        )
        assert planned == expected
        assert planned % (batch_size * grad_accum_steps) == 0

    def test_split_too_small_for_one_batch_per_worker_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot fill one accumulation window"):
            plan_samples_per_worker(10, batch_size=4, num_workers=8)

    def test_split_too_small_for_one_accumulation_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot fill one accumulation window"):
            plan_samples_per_worker(100, batch_size=4, num_workers=2, grad_accum_steps=20)


class TestBuildWebdatasetLoader:
    """Training plans a fixed epoch; evaluation passes over every sample exactly once."""

    @pytest.fixture(autouse=True)
    def _require_webdataset(self) -> None:
        """Skip every test in this class when the optional ``webdataset`` extra is not installed.

        Examples:
            >>> pass  # doctest: +SKIP
            A pytest fixture, only runnable through pytest's fixture injection.
        """
        pytest.importorskip("webdataset")

    @pytest.mark.parametrize("num_workers", [pytest.param(0, id="main-process"), pytest.param(2, id="two-workers")])
    def test_training_length_matches_the_batches_produced(self, tmp_path: Path, num_workers: int) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=24), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=4, collate_fn=_count_collate, num_workers=num_workers, world_size=1
        )
        batches = list(loader)
        assert len(loader) == len(batches)
        assert set(batches) == {4}

    def test_training_epoch_is_a_whole_number_of_accumulation_windows(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=64), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset,
            batch_size=2,
            collate_fn=_count_collate,
            num_workers=1,
            world_size=1,
            grad_accum_steps=3,
        )
        assert len(loader) % 3 == 0
        assert len(loader) == len(list(loader))

    def test_evaluation_sees_every_sample_exactly_once(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=14), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset,
            batch_size=4,
            collate_fn=_id_collate,
            num_workers=2,
            fixed_epoch=False,
            world_size=1,
        )
        streamed = [image_id for batch in loader for image_id in batch]
        assert sorted(streamed) == list(range(1000, 1014))

    def test_evaluation_loader_reports_no_length(self, tmp_path: Path) -> None:
        dataset = WebDatasetDetection(_pack(tmp_path, count=8), "train", transforms=None)
        loader = build_webdataset_loader(
            dataset, batch_size=4, collate_fn=_count_collate, num_workers=0, fixed_epoch=False, world_size=1
        )
        with pytest.raises(TypeError):
            len(loader)

    def test_more_workers_than_shards_is_rejected_for_training(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=16)
        shard_dir = tmp_path / "shards"
        index = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        with pytest.raises(ValueError, match="cannot cover"):
            build_webdataset_loader(
                dataset,
                batch_size=2,
                collate_fn=_count_collate,
                num_workers=len(index.shards) + 1,
                world_size=1,
            )

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
        image_dir, annotations = _build_coco_split(tmp_path, count=48)
        shard_dir = tmp_path / "shards"
        # One sample per shard, so the shard count is exactly 48 whatever the images weigh.
        index = pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=1)
        assert len(index.shards) == 48
        dataset = WebDatasetDetection(shard_dir, "train", transforms=None)
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=2, collate_fn=_count_collate, num_workers=num_workers, world_size=1)
        assert ("fewer samples than the epoch asks" in capsys.readouterr().err) is expect_warning

    def test_skew_warning_uses_real_per_shard_counts_not_shard_count_alone(self, tmp_path: Path, capsys: Any) -> None:
        """Regression test: shards are cut by byte size, so a count-only estimate can miss a real skew entirely.

        Two shards split evenly by *count* (one per worker) would score 0% deficit under a shard-count-only estimate,
        yet here one shard carries 98% of the samples: the real per-shard counts the packer now records have to be what
        drives the warning.
        """
        dataset = WebDatasetDetection(_pack(tmp_path, count=4), "train", transforms=None)
        dataset.index = replace(dataset.index, shards=("a.tar", "b.tar"), num_samples=1000, samples_per_shard=(980, 20))
        capsys.readouterr()
        build_webdataset_loader(dataset, batch_size=1, collate_fn=_count_collate, num_workers=2, world_size=1)
        assert "fewer samples than the epoch asks" in capsys.readouterr().err

    def test_skew_threshold_is_a_fraction(self) -> None:
        assert 0.0 < SHARD_SKEW_WARN_FRACTION < 1.0

    def test_batches_collate_into_the_model_input_contract(self, tmp_path: Path) -> None:
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


class TestBuildWebdataset:
    """The dataset builder mirrors the loose-file builders' conventions."""

    @pytest.fixture(autouse=True)
    def _require_webdataset(self) -> None:
        """Skip every test in this class when the optional ``webdataset`` extra is not installed.

        Examples:
            >>> pass  # doctest: +SKIP
            A pytest fixture, only runnable through pytest's fixture injection.
        """
        pytest.importorskip("webdataset")

    @staticmethod
    def _namespace(dataset_dir: Path, **overrides: Any) -> types.SimpleNamespace:
        """Build the merged model/train namespace :func:`~rfdetr.datasets.webdataset_io.build_webdataset` expects.

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
            "do_random_resize_via_padding": False,
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
        with pytest.raises(FileNotFoundError, match="does not exist"):
            build_webdataset("train", self._namespace(tmp_path / "absent"), 224)

    def test_keypoint_training_is_rejected_with_a_pointer_to_the_other_formats(self, tmp_path: Path) -> None:
        shard_dir = _pack(tmp_path, count=2)
        namespace = self._namespace(shard_dir, use_grouppose_keypoints=True)
        with pytest.raises(NotImplementedError, match="keypoint"):
            build_webdataset("train", namespace, 224)

    def test_non_train_split_adopts_the_train_label_space(self, tmp_path: Path) -> None:
        image_dir, annotations = _build_coco_split(tmp_path, count=6)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train")
        # The val split only annotates category 9, so a split-local mapping would send it to label 0.
        val_images, val_annotations = _build_coco_split(tmp_path, count=2, categories=[_CATEGORIES[1]], subdir="val")
        pack_coco_to_shards(val_images, val_annotations, shard_dir, split="val")
        dataset = build_webdataset("val", self._namespace(shard_dir), 224)
        assert dataset.cat2label == {3: 0, 9: 1}

    @pytest.mark.parametrize(
        "square_resize_div_64",
        [pytest.param(False, id="aspect-preserving-resize"), pytest.param(True, id="square-div-64-resize")],
    )
    def test_both_resize_pipelines_produce_model_ready_tensors(
        self, tmp_path: Path, square_resize_div_64: bool
    ) -> None:
        shard_dir = _pack(tmp_path, count=4)
        namespace = self._namespace(shard_dir, square_resize_div_64=square_resize_div_64)
        dataset = build_webdataset("train", namespace, 224)
        image, target = next(iter(dataset))
        assert image.ndim == 3
        assert target["boxes"].shape[-1] == 4

    def test_build_dataset_routes_the_webdataset_format(self, tmp_path: Path) -> None:
        shard_dir = _pack(tmp_path, count=4)
        dataset = build_dataset("train", self._namespace(shard_dir), 224)
        assert isinstance(dataset, WebDatasetDetection)


class TestDataModuleStreaming:
    """The DataModule routes a streaming split away from the sampler-based loaders."""

    @pytest.fixture(autouse=True)
    def _require_webdataset(self) -> None:
        """Skip every test in this class when the optional ``webdataset``/``pytorch_lightning`` extras are absent.

        Examples:
            >>> pass  # doctest: +SKIP
            A pytest fixture, only runnable through pytest's fixture injection.
        """
        pytest.importorskip("webdataset")
        pytest.importorskip("pytorch_lightning")

    @pytest.fixture
    def datamodule(self, tmp_path: Path) -> Any:
        """Build an :class:`RFDETRDataModule` over a freshly packed train/val split.

        Examples:
            >>> pass  # doctest: +SKIP
            Needs a real tmp_path and the webdataset/pytorch_lightning extras, so it cannot run standalone.
        """
        from rfdetr.config import RFDETRSmallConfig, TrainConfig
        from rfdetr.training.module_data import RFDETRDataModule

        shard_dir = tmp_path / "shards"
        for split, count in (("train", 64), ("val", 32)):
            image_dir, annotations = _build_coco_split(tmp_path, count=count, subdir=split)
            pack_coco_to_shards(image_dir, annotations, shard_dir, split=split, max_shard_bytes=8192)
        train_config = TrainConfig(
            dataset_dir=str(shard_dir),
            dataset_file="webdataset",
            batch_size=4,
            grad_accum_steps=1,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            multi_scale=False,
            expanded_scales=False,
            tensorboard=False,
            output_dir=str(tmp_path / "out"),
        )
        module = RFDETRDataModule(RFDETRSmallConfig(pretrain_weights=None), train_config)
        module.setup("fit")
        return module

    def test_training_loader_reports_the_batches_it_yields(self, datamodule: Any) -> None:
        loader = datamodule.train_dataloader()
        assert len(loader) == len(list(loader))

    def test_training_batches_are_all_full(self, datamodule: Any) -> None:
        loader = datamodule.train_dataloader()
        assert {int(samples.tensors.shape[0]) for samples, _ in loader} == {4}

    def test_training_loader_aligns_to_grad_accum_steps(self, tmp_path: Path) -> None:
        from rfdetr.config import RFDETRSmallConfig, TrainConfig
        from rfdetr.training.module_data import RFDETRDataModule

        shard_dir = tmp_path / "shards"
        image_dir, annotations = _build_coco_split(tmp_path, count=96, subdir="train")
        pack_coco_to_shards(image_dir, annotations, shard_dir, split="train", max_shard_bytes=8192)
        val_image_dir, val_annotations = _build_coco_split(tmp_path, count=16, subdir="val")
        pack_coco_to_shards(val_image_dir, val_annotations, shard_dir, split="val", max_shard_bytes=8192)
        train_config = TrainConfig(
            dataset_dir=str(shard_dir),
            dataset_file="webdataset",
            batch_size=2,
            grad_accum_steps=3,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            multi_scale=False,
            expanded_scales=False,
            tensorboard=False,
            output_dir=str(tmp_path / "out"),
        )
        module = RFDETRDataModule(RFDETRSmallConfig(pretrain_weights=None), train_config)
        module.setup("fit")
        loader = module.train_dataloader()
        assert len(loader) % train_config.grad_accum_steps == 0

    def test_validation_loader_is_unsized_and_covers_the_split(self, datamodule: Any) -> None:
        loader = datamodule.val_dataloader()
        with pytest.raises(TypeError):
            len(loader)
        assert sum(int(samples.tensors.shape[0]) for samples, _ in loader) == 32

    @pytest.mark.parametrize(
        "stage",
        [pytest.param("test", id="test-loader"), pytest.param("predict", id="predict-loader")],
    )
    def test_other_eval_loaders_also_stream_the_split(self, datamodule: Any, stage: str) -> None:
        # This fixture packs no test split, so "test" resolves to the 32-sample val shards.
        datamodule.setup(stage)
        loader = datamodule.test_dataloader() if stage == "test" else datamodule.predict_dataloader()
        with pytest.raises(TypeError):
            len(loader)
        assert sum(int(samples.tensors.shape[0]) for samples, _ in loader) == 32

    def test_packed_test_split_is_used_instead_of_val(self, tmp_path: Path) -> None:
        from rfdetr.config import RFDETRSmallConfig, TrainConfig
        from rfdetr.training.module_data import RFDETRDataModule

        shard_dir = tmp_path / "with-test"
        for split, count in (("train", 64), ("val", 32), ("test", 16)):
            image_dir, annotations = _build_coco_split(tmp_path, count=count, subdir=f"wt-{split}")
            pack_coco_to_shards(image_dir, annotations, shard_dir, split=split, max_shard_bytes=8192)
        train_config = TrainConfig(
            dataset_dir=str(shard_dir),
            dataset_file="webdataset",
            batch_size=4,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            multi_scale=False,
            expanded_scales=False,
            tensorboard=False,
            output_dir=str(tmp_path / "out2"),
        )
        module = RFDETRDataModule(RFDETRSmallConfig(pretrain_weights=None), train_config)
        module.setup("test")
        assert sum(int(s.tensors.shape[0]) for s, _ in module.test_dataloader()) == 16

    def test_absent_test_split_falls_back_to_val_and_says_so(self, datamodule: Any, capsys: Any) -> None:
        datamodule.setup("test")
        assert "evaluating the 'val' split instead" in capsys.readouterr().err
        assert sum(int(s.tensors.shape[0]) for s, _ in datamodule.test_dataloader()) == 32

    def test_datamodule_reports_the_shard_index_class_names(self, datamodule: Any) -> None:
        assert datamodule.class_names == ["cat", "dog"]

    def test_sample_grid_rejects_a_streaming_split(self, datamodule: Any) -> None:
        pytest.importorskip("matplotlib")
        with pytest.raises(TypeError, match="map-style dataset"):
            datamodule._show_samples(2, split="train")
