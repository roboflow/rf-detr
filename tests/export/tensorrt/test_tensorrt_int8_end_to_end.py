# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end tests for ``export(format="tensorrt", quantization="int8")``.

These classes build real FP16 and INT8 engines of the pretrained Nano, segmentation and keypoint models on a GPU and
compare their outputs on a photo. The checks that run without TensorRT are in ``test_tensorrt_int8.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.export._tensorrt.exporter import _IS_POLYGRAPHY_AVAILABLE, _IS_TENSORRT_AVAILABLE
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou

_requires_tensorrt = pytest.mark.skipif(
    not (_IS_TENSORRT_AVAILABLE and _IS_POLYGRAPHY_AVAILABLE), reason="tensorrt/polygraphy not installed"
)


def _people_photo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Download supervision's PEOPLE_WALKING photo; return a directory of its crops and flips, and the photo itself.

    Examples:
        Needs a network download, so it is documentation rather than a doctest:

        >>> calibration_dir, photo_path = _people_photo(tmp_path_factory)  # doctest: +SKIP
    """
    from PIL import Image, ImageOps
    from supervision.assets import ImageAssets, download_assets

    asset_dir = tmp_path_factory.mktemp("int8_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        photo = Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)
    calibration = tmp_path_factory.mktemp("int8_calibration")
    with Image.open(photo) as image:
        image = image.convert("RGB")
        width, height = image.size
        for index, box in enumerate(
            [(0, 0, width, height), (0, 0, width // 2, height), (width // 2, 0, width, height)]
        ):
            crop = image.crop(box)
            crop.save(calibration / f"{index}.jpg")
            ImageOps.mirror(crop).save(calibration / f"{index}_flipped.jpg")
    return calibration, photo


def _photo_input(photo: Path, resolution: int) -> np.ndarray:
    """Preprocess *photo* to a ``(1, 3, resolution, resolution)`` model input the way ``predict()`` does.

    Examples:
        >>> import tempfile
        >>> from PIL import Image
        >>> with tempfile.TemporaryDirectory() as root:
        ...     Image.new("RGB", (20, 10)).save(Path(root) / "p.jpg")
        ...     _photo_input(Path(root) / "p.jpg", 8).shape
        (1, 3, 8, 8)
    """
    from PIL import Image

    from rfdetr.export._runtime.preprocess import preprocess_to_nchw

    with Image.open(photo) as image:
        return preprocess_to_nchw(image.convert("RGB"), height=resolution, width=resolution)


#: Score above which an FP16 detection of the PEOPLE_WALKING photo by RFDETRNano counts as confident. One person there
#: scores 0.498, 0.501 and 0.503 in FP16 engines of batch 1, 2 and 4, and INT8 boxes it with an IoU of only 0.82 to 0.84
#: at every batch size (NVIDIA L4, TensorRT 11.4), so a cutoff of 0.5 checked that person at batch 2 and 4 alone. Above
#: 0.55 the lowest FP16 score is 0.57, and eleven detections remain.
_DETECTION_CONFIDENCE_THRESHOLD = 0.55


def _assert_confident_detections_match(
    fp16_engine: Path,
    int8_engine: Path,
    example: np.ndarray,
    batch: int,
    *,
    threshold: float = _DETECTION_CONFIDENCE_THRESHOLD,
) -> None:
    """Run both engines on *batch* copies of *example*; every confident FP16 detection needs an INT8 match.

    A match is an INT8 box with IoU above 0.85 and a score within 0.1 of the FP16 one. An FP16 detection is confident
    above *threshold*, and at least three must be for the check to mean anything. The INT8 engine must also give
    every row of the batch the same boxes for its confident queries.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(np.repeat(example, batch, axis=0)).cuda()}
    results = []
    for path in (fp16_engine, int8_engine):
        outputs = TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs)
        results.append((outputs["dets"].float().cpu(), outputs["labels"].float().sigmoid().max(dim=-1).values.cpu()))
    (fp16_boxes, fp16_scores), (int8_boxes, int8_scores) = results
    assert torch.isfinite(int8_boxes).all() and torch.isfinite(int8_scores).all()
    # Rows are compared on the queries the INT8 engine is confident about: the boxes of near-zero-score queries are
    # unconstrained, and a different INT8 tactic per batch row moves them by more than any useful tolerance.
    confident_int8 = int8_scores[0] > 0.5
    assert torch.allclose(
        int8_boxes[:, confident_int8], int8_boxes[:1, confident_int8].expand(batch, -1, -1), atol=2e-2
    ), "rows of one batch differ"
    for row in range(batch):
        confident = fp16_scores[row] > threshold
        assert int(confident.sum()) >= 3, "the photo must give the FP16 engine several confident people"
        # (confident FP16 boxes, all INT8 boxes); several INT8 queries can cover one person, so a match is the closest
        # score among the INT8 boxes that overlap, not the score of the single best-overlapping box.
        overlaps = box_iou(box_cxcywh_to_xyxy(fp16_boxes[row][confident]), box_cxcywh_to_xyxy(int8_boxes[row]))[0]
        gaps = (int8_scores[row][None, :] - fp16_scores[row][confident][:, None]).abs()
        gaps = gaps.masked_fill(overlaps <= 0.85, float("inf")).min(dim=1).values
        unmatched = [
            (round(float(o), 3), round(float(g), 3)) for o, g in zip(overlaps.max(dim=1).values, gaps) if g >= 0.1
        ]
        assert unmatched == [], (
            f"row {row}: confident FP16 detections without an INT8 match (best IoU, score gap): {unmatched}"
        )


#: Smallest mask IoU between an FP16 and an INT8 mask for a confident detection, and largest keypoint offset (as a
#: fraction of the image side) between them. Measured on RFDETRSegNano and RFDETRKeypointPreview (RTX 5070, TensorRT
#: 11.4): the lowest mask IoU was 0.90 and the largest offset 0.015; the bounds leave room for another GPU.
_MIN_MASK_IOU, _MAX_KEYPOINT_OFFSET = 0.8, 0.05

#: Score above which an FP16 detection of the PEOPLE_WALKING photo counts as confident, per extra output. The keypoint
#: model scores the photo's people low (0.62, 0.50, 0.48, 0.43 on RTX 5070), so 0.5 would let a rebuild flip one.
_CONFIDENCE_THRESHOLD = {"masks": 0.5, "keypoints": 0.4}


def _assert_heads_agree(fp16_engine: Path, int8_engine: Path, example: np.ndarray, head: str) -> None:
    """Run both engines on *example*; every confident FP16 detection needs an INT8 detection with the same *head*.

    A match is the INT8 query whose box has IoU above 0.85 with the FP16 box and whose score is closest within 0.1.
    For ``head="masks"`` the matched masks (logits above zero) need an IoU of at least :data:`_MIN_MASK_IOU`; for
    ``head="keypoints"`` the person keypoints' normalized x and y may differ by at most :data:`_MAX_KEYPOINT_OFFSET`.
    FP16 detections are confident above ``_CONFIDENCE_THRESHOLD[head]``, and at least three must be confident and
    matched for the check to mean anything.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_heads_agree(fp16_path, int8_path, example, head="masks")  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(example).cuda()}
    fp16, int8 = (
        {
            name: value.float().cpu()
            for name, value in TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs).items()
        }
        for path in (fp16_engine, int8_engine)
    )
    for outputs in (fp16, int8):
        assert torch.isfinite(outputs[head]).all(), f"{head} hold NaN or infinite values"
    fp16_scores = fp16["labels"][0].sigmoid().max(dim=-1).values
    int8_scores = int8["labels"][0].sigmoid().max(dim=-1).values
    confident = torch.nonzero(fp16_scores > _CONFIDENCE_THRESHOLD[head]).flatten()
    overlaps = box_iou(box_cxcywh_to_xyxy(fp16["dets"][0][confident]), box_cxcywh_to_xyxy(int8["dets"][0]))[0]
    gaps = (int8_scores[None, :] - fp16_scores[confident][:, None]).abs().masked_fill(overlaps <= 0.85, float("inf"))
    matched = [
        (int(q), int(m)) for q, m, gap in zip(confident, gaps.argmin(dim=1), gaps.min(dim=1).values) if gap < 0.1
    ]
    assert len(matched) >= 3, f"only {len(matched)} of {len(confident)} confident FP16 detections have an INT8 match"
    for query, other in matched:
        if head == "masks":
            fp16_mask, int8_mask = fp16["masks"][0, query] > 0, int8["masks"][0, other] > 0
            agreement = float((fp16_mask & int8_mask).sum() / (fp16_mask | int8_mask).sum().clamp(min=1))
            assert agreement >= _MIN_MASK_IOU, f"query {query}: mask IoU {agreement:.3f}"
        else:
            offset = float(
                (fp16["keypoints"][0, query, 17:34, :2] - int8["keypoints"][0, other, 17:34, :2]).abs().max()
            )
            assert offset <= _MAX_KEYPOINT_OFFSET, f"query {query}: keypoint offset {offset:.3f}"


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained RFDETRNano agree on a photo's confident detections."""

    @pytest.fixture(scope="class")
    def photo(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
        """Crops and flips of supervision's PEOPLE_WALKING photo for calibration, and the photo as a model input.

        Examples:
            A pytest fixture, and it needs a network download.

            >>> calibration_dir, example = photo(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, image = _people_photo(tmp_path_factory)
        return calibration, _photo_input(image, int(RFDETRNano().model.resolution))

    @pytest.fixture(scope="class")
    def engines(
        self, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray]:
        """Export FP16 and INT8 engines at batch 1, calibrating on the photo's crops and flips.

        Examples:
            A pytest fixture, and it needs a GPU and TensorRT.

            >>> fp16_engine, int8_engine, example = engines(photo, tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp("int8_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        return Path(fp16), Path(int8), example

    def test_engine_is_written_without_leftovers(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """The INT8 export leaves only the ONNX, the engine and its description."""
        _, int8, _ = engines
        assert int8.name == "rfdetr-nano_int8.trt"
        assert sorted(p.name for p in int8.parent.iterdir()) == [
            "rfdetr-nano.onnx",
            "rfdetr-nano_int8.trt",
            "rfdetr-nano_int8.trt.rfdetr.json",
        ]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16_path, int8_path, example = engines
        _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)

    @pytest.mark.parametrize("source", ["npy-path", "array"])
    def test_preprocessed_calibration_gives_a_matching_engine(
        self,
        source: str,
        photo: tuple[Path, np.ndarray],
        engines: tuple[Path, Path, np.ndarray],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Calibrating on a preprocessed .npy path or array gives an engine matching FP16."""
        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.calibration import calibration_batches

        calibration, example = photo
        fp16_path, _, _ = engines
        size = example.shape[-1]
        samples = np.concatenate(list(calibration_batches(calibration, height=size, width=size)))
        data: Any = samples
        if source == "npy-path":
            data = tmp_path_factory.mktemp("int8_samples") / "calibration.npy"
            np.save(data, samples)
        int8 = RFDETRNano().export(
            output_dir=str(tmp_path_factory.mktemp(f"int8_{source}")),
            format="tensorrt",
            quantization="int8",
            calibration_data=data,
            verbose=False,
        )
        assert Path(int8).name == "rfdetr-nano_int8.trt"
        _assert_confident_detections_match(fp16_path, Path(int8), example, batch=1)

    @pytest.mark.parametrize("batch", [2, 4])
    def test_static_batch_engine_matches_fp16_on_every_row(
        self, batch: int, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A static-batch INT8 engine matches FP16 on every row of the batch."""
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp(f"int8_batch_{batch}")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", batch_size=batch, verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            batch_size=batch,
            verbose=False,
        )
        _assert_confident_detections_match(Path(fp16), Path(int8), example, batch=batch)


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestSegmentationAndKeypointInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained segmentation and keypoint models agree on a photo."""

    @pytest.fixture(scope="class", params=["RFDETRSegNano", "RFDETRKeypointPreview"])
    def engines(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray, str]:
        """Export FP16 and INT8 engines of the model at batch 1, calibrating on the photo's crops and flips.

        Returns the two engine paths, the photo as that model's input, and the name of its extra output.

        Examples:
            A pytest fixture, and it needs a GPU, TensorRT and a network download.

            >>> fp16_engine, int8_engine, example, head = engines(request, tmp_path_factory)  # doctest: +SKIP
        """
        import rfdetr

        calibration, photo = _people_photo(tmp_path_factory)
        model = getattr(rfdetr, request.param)()
        out_dir = tmp_path_factory.mktemp("int8_head_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        head = "masks" if request.param == "RFDETRSegNano" else "keypoints"
        return Path(fp16), Path(int8), _photo_input(photo, int(model.model.resolution)), head

    def test_engine_keeps_the_extra_output(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 engine keeps the mask or keypoint output after dets and labels."""
        from rfdetr.export._tensorrt.inference import TRTInference

        _, int8, _, head = engines
        assert TRTInference(str(int8), sync_mode=True, device="cuda:0").get_output_names() == ["dets", "labels", head]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16, int8, example, head = engines
        _assert_confident_detections_match(fp16, int8, example, batch=1, threshold=_CONFIDENCE_THRESHOLD[head])

    def test_head_matches_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 mask or keypoint head agrees with FP16 on confident detections."""
        fp16, int8, example, head = engines
        _assert_heads_agree(fp16, int8, example, head)
