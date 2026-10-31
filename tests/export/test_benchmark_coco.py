# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the COCO val2017 accuracy helpers the export cookbooks score each exported model with.

Everything runs on a tiny synthetic COCO file written to ``tmp_path``; no network access and no real model.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import ANY, Mock

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.export._benchmark import (
    CocoValSubset,
    _download,
    evaluate_coco_map,
    fetch_coco_val2017,
    select_coco_val_ids,
)

#: Ground truth per image: one box in pixel ``xywh`` and its sparse COCO category ID.
_GT_BOXES = {1: ([8.0, 8.0, 32.0, 24.0], 18), 2: ([20.0, 4.0, 16.0, 40.0], 90), 3: ([0.0, 0.0, 30.0, 30.0], 1)}
#: Each image gets its own ``(width, height)``, so a fake runtime can tell which image it was handed.
_IMAGE_SIZES = {1: (64, 64), 2: (64, 80), 3: (80, 64)}


def _write_coco(root: Path) -> CocoValSubset:
    """Write a three-image COCO ``instances_val2017.json`` plus blank JPEGs under *root*.

    Args:
        root: Directory that receives ``annotations/`` and ``val2017/``.

    Returns:
        A subset over all three images.

    Examples:
        >>> import tempfile
        >>> len(_write_coco(Path(tempfile.mkdtemp())).image_ids)
        3
    """
    images_dir, annotations_dir = root / "val2017", root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    images, annotations = [], []
    for image_id, (bbox, category_id) in _GT_BOXES.items():
        file_name = f"{image_id:012d}.jpg"
        width, height = _IMAGE_SIZES[image_id]
        Image.new("RGB", (width, height)).save(images_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "coco_url": f"http://images.cocodataset.org/val2017/{file_name}",
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "category_id": category_id,
                "iscrowd": 0,
            }
        )
    categories = [{"id": category_id, "name": str(category_id)} for category_id in (1, 18, 90)]
    annotations_path = annotations_dir / "instances_val2017.json"
    annotations_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return CocoValSubset(images_dir=images_dir, annotations_path=annotations_path, image_ids=(1, 2, 3))


def _set_image_field(annotations_path: Path, image_id: int, key: str, value: str) -> None:
    """Rewrite one field of one image record in a COCO annotation file, as a tampered download would.

    Examples:
        >>> import tempfile
        >>> subset = _write_coco(Path(tempfile.mkdtemp()))
        >>> _set_image_field(subset.annotations_path, 1, "file_name", "../evil.jpg")
        >>> json.loads(subset.annotations_path.read_text())["images"][0]["file_name"]
        '../evil.jpg'
    """
    data = json.loads(annotations_path.read_text())
    next(image for image in data["images"] if image["id"] == image_id)[key] = value
    annotations_path.write_text(json.dumps(data))


def _write_ids_file(path: Path, n_images: int) -> Path:
    """Write a minimal annotation file that lists *n_images* image IDs, ``1`` to *n_images*.

    Examples:
        >>> import tempfile
        >>> json.loads(_write_ids_file(Path(tempfile.mkdtemp()) / "a.json", 2).read_text())
        {'images': [{'id': 1}, {'id': 2}]}
    """
    path.write_text(json.dumps({"images": [{"id": image_id} for image_id in range(1, n_images + 1)]}))
    return path


class _FakeResponse(io.BytesIO):
    """An ``urlopen`` response stand-in: a byte stream whose ``Content-Length`` may promise more than it holds.

    Args:
        body: Bytes the response actually delivers.
        content_length: Size the ``Content-Length`` header announces.

    Examples:
        >>> response = _FakeResponse(b"abc", content_length=100)
        >>> response.read(), response.headers["Content-Length"]
        (b'abc', '100')
    """

    def __init__(self, body: bytes, content_length: int) -> None:
        """Store *body* as the stream and announce *content_length* in the headers."""
        super().__init__(body)
        self.headers = {"Content-Length": str(content_length)}


def _image_id(image: Image.Image) -> int:
    """Recover the COCO image ID from the image's unique size.

    Examples:
        >>> _image_id(Image.new("RGB", (64, 80)))
        2
    """
    return next(image_id for image_id, size in _IMAGE_SIZES.items() if size == image.size)


def _gt_detections(image: Image.Image) -> sv.Detections:
    """Return the ground-truth box of *image* as already-decoded detections.

    Examples:
        >>> _gt_detections(Image.new("RGB", (64, 64))).xyxy.tolist()
        [[8.0, 8.0, 40.0, 32.0]]
    """
    (x, y, w, h), category_id = _GT_BOXES[_image_id(image)]
    return sv.Detections(
        xyxy=np.array([[x, y, x + w, y + h]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([category_id]),
    )


def _gt_raw_outputs(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``dets``/``labels`` arrays (batch of one) whose top query decodes to the ground-truth box.

    Boxes are normalized ``cxcywh`` and the 91-slot logits put the only confident score at the category's own slot,
    the way an official sparse-ID COCO checkpoint does.

    Examples:
        >>> boxes, logits = _gt_raw_outputs(Image.new("RGB", (64, 64)))
        >>> boxes.shape, logits.shape
        ((1, 4, 4), (1, 4, 91))
    """
    (x, y, w, h), category_id = _GT_BOXES[_image_id(image)]
    width, height = image.size
    boxes = np.full((1, 4, 4), 0.25, dtype=np.float32)
    boxes[0, 0] = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
    logits = np.full((1, 4, 91), -10.0, dtype=np.float32)
    logits[0, 0, category_id] = 5.0
    return boxes, logits


def _gt_raw_outputs_without_batch(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_gt_raw_outputs`` with the leading batch axis dropped, as a runtime that squeezes it would.

    Examples:
        >>> boxes, logits = _gt_raw_outputs_without_batch(Image.new("RGB", (64, 64)))
        >>> boxes.shape, logits.shape
        ((4, 4), (4, 91))
    """
    boxes, logits = _gt_raw_outputs(image)
    return boxes[0], logits[0]


@pytest.fixture
def download(mocker: MockerFixture) -> Mock:
    """Replace the image downloader with a recording mock, so no test touches the network.

    Examples:
        A pytest fixture -- it needs ``mocker`` to install the mock, so the example is not run:

        >>> download(mocker)  # doctest: +SKIP
        <MagicMock name='_download' id=...>
    """
    return mocker.patch("rfdetr.export._benchmark._download")


@pytest.fixture
def logger(mocker: MockerFixture) -> Mock:
    """Replace the benchmark module's logger with a recording mock.

    Examples:
        A pytest fixture -- it needs ``mocker`` to install the mock, so the example is not run:

        >>> logger(mocker)  # doctest: +SKIP
        <MagicMock name='logger' id=...>
    """
    return mocker.patch("rfdetr.export._benchmark.logger")


class TestSelectCocoValIds:
    """``select_coco_val_ids`` picks a seeded, reproducible subset of image IDs."""

    def test_same_seed_gives_same_ids(self, tmp_path: Path) -> None:
        """The same seed selects the same images, so every cookbook scores the identical subset.

        Cross-cookbook mAP comparisons only mean something when the evaluated images are the same everywhere.
        """
        path = _write_ids_file(tmp_path / "instances.json", 50)
        assert select_coco_val_ids(path, 10, seed=3) == select_coco_val_ids(path, 10, seed=3)

    def test_different_seeds_select_different_subsets(self, tmp_path: Path) -> None:
        """A different seed shuffles differently; without this a constant selector would pass the seed test above."""
        path = _write_ids_file(tmp_path / "instances.json", 50)
        assert select_coco_val_ids(path, 10, seed=0) != select_coco_val_ids(path, 10, seed=1)

    def test_subset_has_requested_size_and_distinct_members_of_the_split(self, tmp_path: Path) -> None:
        """The subset has exactly *n_images* IDs, none repeated, all from the split."""
        path = _write_ids_file(tmp_path / "instances.json", 50)
        ids = select_coco_val_ids(path, 10)
        assert len(ids) == len(set(ids)) == 10
        assert set(ids) <= set(range(1, 51))

    @pytest.mark.parametrize("n_images", [2.5, True, "3"])
    def test_non_integer_count_raises(self, tmp_path: Path, n_images: object) -> None:
        """A float, bool or string count is a type error up front, not a ``TypeError`` from deep inside NumPy."""
        subset = _write_coco(tmp_path)
        with pytest.raises(TypeError) as exc_info:
            select_coco_val_ids(subset.annotations_path, n_images)  # type: ignore[arg-type]

        assert "must be an int" in str(exc_info.value)

    @pytest.mark.parametrize("n_images", [0, -1])
    def test_fewer_than_one_image_raises(self, tmp_path: Path, n_images: int) -> None:
        """``0`` would score nothing and ``-1`` would silently select all but one image; both are refused."""
        subset = _write_coco(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            select_coco_val_ids(subset.annotations_path, n_images)

        assert "at least 1" in str(exc_info.value)

    def test_none_selects_every_image_sorted(self, tmp_path: Path) -> None:
        """``n_images=None`` is the full split, in image-ID order, for the ``FULL_VAL`` path."""
        subset = _write_coco(tmp_path)
        assert select_coco_val_ids(subset.annotations_path, None) == [1, 2, 3]

    def test_more_images_than_split_raises(self, tmp_path: Path) -> None:
        """Asking for more images than exist fails loudly instead of silently evaluating fewer."""
        subset = _write_coco(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            select_coco_val_ids(subset.annotations_path, 4)

        assert "only 3" in str(exc_info.value)


class TestFetchCocoVal2017:
    """``fetch_coco_val2017`` downloads only the selected images that are not on disk yet."""

    def test_downloads_only_missing_images(self, tmp_path: Path, download: Mock) -> None:
        """An image already on disk is not fetched again; a missing one is fetched from the fixed val2017 URL.

        Cookbooks rerun this cell on every session, so a second run must not re-download the subset.
        """
        _write_coco(tmp_path)
        (tmp_path / "val2017" / "000000000002.jpg").unlink()

        subset = fetch_coco_val2017(tmp_path, n_images=None)

        download.assert_called_once_with(
            "http://images.cocodataset.org/val2017/000000000002.jpg", tmp_path / "val2017" / "000000000002.jpg"
        )
        assert subset.image_ids == (1, 2, 3)

    def test_n_images_downloads_only_the_selected_subset(self, tmp_path: Path, download: Mock) -> None:
        """With ``n_images`` set, only the selected images are fetched, not the whole split."""
        subset = _write_coco(tmp_path)
        for path in (tmp_path / "val2017").iterdir():
            path.unlink()

        fetched = fetch_coco_val2017(tmp_path, n_images=2)

        assert len(fetched.image_ids) == download.call_count == 2
        assert set(fetched.image_ids) <= set(subset.image_ids)

    @pytest.mark.parametrize("n_images", [0, -1])
    def test_fewer_than_one_image_raises_before_any_download(
        self, tmp_path: Path, n_images: int, download: Mock
    ) -> None:
        """An invalid ``n_images`` is refused up front, with nothing fetched."""
        _write_coco(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            fetch_coco_val2017(tmp_path, n_images=n_images)

        assert "at least 1" in str(exc_info.value)
        download.assert_not_called()

    @pytest.mark.parametrize("file_name", ["../evil.jpg", "/abs/evil.jpg", "..", pytest.param("", id="empty")])
    def test_file_name_outside_images_dir_is_refused(self, tmp_path: Path, file_name: str, download: Mock) -> None:
        """A ``file_name`` that is not a bare name inside ``val2017/`` raises before anything is downloaded.

        The annotation file travels over plain HTTP, so its ``file_name`` is untrusted: a ``../`` component or an
        absolute path would otherwise choose where the image is written. ``..`` and the empty name pass a basename-only
        check, which is why the destination is also resolved against ``val2017/``.
        """
        subset = _write_coco(tmp_path)
        _set_image_field(subset.annotations_path, 2, "file_name", file_name)

        with pytest.raises(ValueError) as exc_info:
            fetch_coco_val2017(tmp_path, n_images=None)

        assert "Refusing COCO image file name" in str(exc_info.value)
        download.assert_not_called()

    def test_image_url_ignores_the_annotation_coco_url(self, tmp_path: Path, download: Mock) -> None:
        """Each image is fetched from the fixed val2017 host, never from the record's own ``coco_url``.

        A tampered ``coco_url`` (``file://``, another host) would otherwise turn the fetch into a local-file copy or a
        request to an attacker-chosen server.
        """
        subset = _write_coco(tmp_path)
        _set_image_field(subset.annotations_path, 2, "coco_url", "file:///etc/passwd")
        (tmp_path / "val2017" / "000000000002.jpg").unlink()

        fetch_coco_val2017(tmp_path, n_images=None)

        download.assert_called_once_with(
            "http://images.cocodataset.org/val2017/000000000002.jpg", tmp_path / "val2017" / "000000000002.jpg"
        )

    def test_missing_annotations_use_the_shared_zip_downloader(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Without ``instances_val2017.json`` the annotation archive goes through the shared retrying zip downloader.

        That downloader rejects truncated archives, guards extraction against path traversal, and serializes concurrent
        callers; ``is_complete`` lets a caller queued behind another process skip an archive that process just finished.
        """
        subset = _write_coco(tmp_path)
        annotations = subset.annotations_path.read_text()
        subset.annotations_path.unlink()
        download_and_extract = mocker.patch(
            "rfdetr.datasets._develop._download_and_extract",
            side_effect=lambda *args, **kwargs: subset.annotations_path.write_text(annotations),
        )

        fetch_coco_val2017(tmp_path, n_images=None)

        download_and_extract.assert_called_once_with(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", tmp_path, is_complete=ANY
        )
        assert download_and_extract.call_args.kwargs["is_complete"]() is True


class TestDownload:
    """``_download`` puts a file at its destination only once the whole body has arrived."""

    def test_truncated_transfer_leaves_no_file(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A body shorter than ``Content-Length`` raises and leaves neither the image nor its ``.part`` behind.

        ``fetch_coco_val2017`` treats any file at the destination as already downloaded, so a truncated image that
        survived on disk would never be re-fetched; with nothing left behind, the next run fetches it again.
        """
        dest = tmp_path / "val2017" / "000000000002.jpg"
        mocker.patch("rfdetr.export._benchmark.urlopen", return_value=_FakeResponse(b"abc", 100))

        with pytest.raises(OSError) as exc_info:
            _download("http://images.cocodataset.org/val2017/000000000002.jpg", dest)

        assert "Truncated download" in str(exc_info.value)
        assert list(dest.parent.iterdir()) == []

    def test_stale_part_file_is_replaced_by_the_complete_image(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A ``.part`` left by a killed run is overwritten, and the complete body lands at the destination.

        A process killed mid-transfer leaves only the ``.part`` sibling; the rerun must recover from it, not keep it.
        """
        dest = tmp_path / "000000000002.jpg"
        (tmp_path / "000000000002.jpg.part").write_bytes(b"stale")
        mocker.patch("rfdetr.export._benchmark.urlopen", return_value=_FakeResponse(b"jpeg-bytes", 10))

        _download("http://images.cocodataset.org/val2017/000000000002.jpg", dest)

        assert [(path.name, path.read_bytes()) for path in tmp_path.iterdir()] == [("000000000002.jpg", b"jpeg-bytes")]

    def test_request_carries_a_timeout(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Each request is opened with a socket timeout, so a stalled server fails the cell instead of hanging it."""
        urlopen = mocker.patch("rfdetr.export._benchmark.urlopen", return_value=_FakeResponse(b"jpeg-bytes", 10))

        _download("http://images.cocodataset.org/val2017/000000000002.jpg", tmp_path / "000000000002.jpg")

        assert urlopen.call_args.kwargs["timeout"] > 0


class TestEvaluateCocoMap:
    """``evaluate_coco_map`` scores one runtime's batch-1 outputs against COCO ground truth."""

    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(_gt_detections, id="decoded-detections"),
            pytest.param(_gt_raw_outputs, id="raw-outputs"),
            pytest.param(_gt_raw_outputs_without_batch, id="raw-outputs-without-batch-axis"),
        ],
    )
    def test_perfect_predictions_score_one(self, tmp_path: Path, run: Callable[..., object]) -> None:
        """Predictions equal to the ground truth reach mAP 1.0 in every form a runtime can return them.

        Decoded detections are the PyTorch ``predict()`` path. Raw ``dets``/``labels`` decode with sparse COCO IDs, so
        category 90 is kept: the default ``background_class_id=None`` matters, since the decoder's own default (``-1``)
        would drop slot 90 and score the image whose only object is category 90 as zero. Raw arrays with no batch axis
        (``(queries, 4)`` / ``(queries, classes)``) score the same.
        """
        result = evaluate_coco_map(run, _write_coco(tmp_path), progress=False)
        assert (result.map50_95, result.map50, result.n_images) == (pytest.approx(1.0), pytest.approx(1.0), 3)

    def test_background_class_id_is_forwarded_to_the_decoder(self, tmp_path: Path) -> None:
        """``background_class_id=-1`` reaches the decoder, which then drops slot 90 and misses that image's object."""
        result = evaluate_coco_map(_gt_raw_outputs, _write_coco(tmp_path), background_class_id=-1, progress=False)
        assert result.map50 < 1.0

    def test_detections_without_confidence_are_refused(self, tmp_path: Path) -> None:
        """Decoded detections need ``confidence`` and ``class_id`` to be scored; missing ones are a clear error."""

        def run(image: Image.Image) -> sv.Detections:
            return sv.Detections(xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32))

        with pytest.raises(ValueError) as exc_info:
            evaluate_coco_map(run, _write_coco(tmp_path), progress=False)

        assert "confidence" in str(exc_info.value)

    def test_image_name_outside_images_dir_is_refused_before_the_runtime_runs(self, tmp_path: Path) -> None:
        """A tampered annotation file, or a hand-built subset, cannot make the scorer open a file outside ``val2017/``.

        ``fetch_coco_val2017`` validates names when it downloads, but ``evaluate_coco_map`` reads them again from the
        annotation file, so it needs its own guard.
        """
        subset = _write_coco(tmp_path)
        _set_image_field(subset.annotations_path, 1, "file_name", "../evil.jpg")
        run = Mock(side_effect=_gt_detections)
        with pytest.raises(ValueError) as exc_info:
            evaluate_coco_map(run, subset, progress=False)

        assert "Refusing COCO image file name" in str(exc_info.value)
        run.assert_not_called()

    def test_summary_is_not_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """COCOeval's 12-line summary stays out of the notebook output; callers get the numbers from the result.

        ``faster_coco_eval`` writes the summary through ``logger.info``, which Jupyter shows, so it is that log the test
        reads.
        """
        with caplog.at_level("INFO"):
            evaluate_coco_map(_gt_detections, _write_coco(tmp_path), progress=False)
        assert "Average Precision" not in caplog.text

    def test_no_detections_score_zero(self, tmp_path: Path) -> None:
        """A runtime that returns nothing scores 0.0 instead of crashing on an empty result set."""
        result = evaluate_coco_map(lambda image: sv.Detections.empty(), _write_coco(tmp_path), progress=False)
        assert (result.map50_95, result.map50) == (0.0, 0.0)

    def test_no_detections_log_a_warning(self, tmp_path: Path, logger: Mock) -> None:
        """An all-empty run reports 0.0 but says why: that is the symptom of collapsed reduced-precision logits."""
        evaluate_coco_map(lambda image: sv.Detections.empty(), _write_coco(tmp_path), progress=False)

        logger.warning.assert_called_once()
        assert "no detections" in logger.warning.call_args.args[0]

    def test_empty_subset_does_not_blame_the_runtime(self, tmp_path: Path, logger: Mock) -> None:
        """With no images to score nothing ran, so the collapsed-logits warning would be misleading."""
        subset = _write_coco(tmp_path)
        empty = CocoValSubset(subset.images_dir, subset.annotations_path, ())

        result = evaluate_coco_map(_gt_detections, empty, progress=False)

        assert (result.map50_95, result.n_images) == (0.0, 0)
        logger.warning.assert_not_called()
