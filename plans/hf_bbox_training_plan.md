# HF DatasetRecordWithBBox Training Plan

## Goal

Add a dedicated RF-DETR training entry point at `src/rfdetr/train.py` that can be driven by a Pydantic YAML
configuration and can train from either an existing dataset directory or a Hugging Face dataset containing
Docling-style `DatasetRecordWithBBox` records.

The new path should reuse the current RF-DETR training stack rather than duplicating trainer internals. In practice,
`src/rfdetr/train.py` should instantiate the selected RF-DETR variant and call `model.train(...)`, while the Hugging
Face data source should be exposed through the existing `rfdetr.datasets.build_dataset(...)` boundary.

## Current Training Architecture

RF-DETR already has two relevant training APIs:

- Public wrapper: `RFDETR.train(**kwargs)` in `src/rfdetr/detr.py`.
- Lightning stack: `RFDETRModelModule`, `RFDETRDataModule`, and `build_trainer(...)` under `src/rfdetr/training/`.

The public wrapper is the safest integration point because it already handles:

- model resolution override validation;
- device mapping to PyTorch Lightning accelerator/device arguments;
- auto-batch probing;
- dataset class-count auto-detection;
- keypoint schema alignment;
- checkpoint metadata such as `model_name`;
- syncing trained weights and class names back onto the model context;
- writing `training_config.json` after training.

The `RFDETRDataModule` builds datasets through:

```python
from rfdetr.datasets import build_dataset
```

That means a new Hugging Face-backed dataset should plug into `build_dataset(...)` rather than replacing the
datamodule or trainer.

## Target User Experience

The dedicated script should run as:

```bash
uv run python -m rfdetr.train --write-default-config --config train.yaml
uv run python -m rfdetr.train --config train.yaml
```

Example Hugging Face configuration:

```yaml
variant: large

model:
  # Optional RF-DETR model-config overrides.
  # resolution: 704
  # num_classes: 2

dataset:
  source: huggingface
  hf_repo_id: your-org/your-dataset
  hf_name: null
  hf_revision: null
  hf_train_split: train
  hf_val_split: validation
  hf_test_split: test
  hf_image_column: GroundTruthPageImages
  hf_bbox_column: GroundTruthBboxOnPageImages

train:
  output_dir: ./output/rfdetr-docling
  epochs: 50
  batch_size: 4
  grad_accum_steps: 4
  num_workers: 4
  tensorboard: true
  progress_bar: tqdm
```

Example local directory configuration:

```yaml
variant: large

dataset:
  source: directory
  dataset_dir: ./dataset
  dataset_file: roboflow

train:
  output_dir: ./output/rfdetr-local
  epochs: 50
  batch_size: 4
```

## Proposed Config Model

Add config models in `src/rfdetr/train.py`:

- `TrainingVariant`: literal model names mapped to RF-DETR variant classes.
- `DirectoryDatasetConfig`: local dataset directory fields.
- `HuggingFaceBBoxDatasetConfig`: Hugging Face repo and Docling column fields.
- `RFDETRTrainingConfig`: top-level Pydantic model containing `variant`, `model`, `dataset`, and `train`.

The top-level model should provide:

- `read_yaml(path) -> RFDETRTrainingConfig`;
- `write_yaml(path) -> None`;
- `to_model_kwargs() -> dict[str, Any]`;
- `to_train_kwargs() -> dict[str, Any]`;
- `train_model() -> None`.

Nested `model` and `train` dictionaries should be validated against the selected RF-DETR variant classes:

- `variant_cls._model_config_class(**model)`;
- `variant_cls._train_config_class(**train_kwargs)`.

This keeps the dedicated YAML file simple while preserving RF-DETR's existing Pydantic validation rules.

## Dataset Integration

Add `src/rfdetr/datasets/hf_bbox.py`.

Responsibilities:

1. Load a Hugging Face dataset split with `datasets.load_dataset(...)`.
2. Flatten document/page records into image-level samples:
   - each `DatasetRecordWithBBox` row may contain multiple `GroundTruthPageImages`;
   - `GroundTruthBboxOnPageImages` maps zero-based page number to a list of bbox records;
   - each page image becomes one RF-DETR sample.
3. Convert bbox records to COCO-style annotations:
   - `label`: class name;
   - `category_id`: source category id;
   - `bbox`: `[x, y, width, height]` in COCO format;
   - optional `ltrb`: `[left, top, right, bottom]`, used only as fallback if `bbox` is missing.
4. Return `(image, target)` compatible with the current RF-DETR pipeline.
5. Expose a COCO-like API as `.coco` so validation metrics still work.

The target emitted by `__getitem__` should match `CocoDetection` after `ConvertCoco`:

```python
target = {
    "boxes": FloatTensor[N, 4],      # xyxy pixel coordinates
    "labels": LongTensor[N],        # contiguous 0-based labels
    "image_id": LongTensor[1],
    "area": FloatTensor[N],
    "iscrowd": LongTensor[N],
    "orig_size": LongTensor[2],     # [height, width]
    "size": LongTensor[2],          # [height, width]
}
```

The dataset should then apply the same RF-DETR transform builders already used by COCO/Roboflow:

- `make_coco_transforms_square_div_64(...)` when `square_resize_div_64=True`;
- `make_coco_transforms(...)` otherwise.

## COCO API Requirement

`COCOEvalCallback` uses `get_coco_api_from_dataset(...)`. The new HF dataset must be discoverable there.

Implementation approach:

- Build an in-memory `pycocotools.coco.COCO` object from flattened samples.
- Include `images`, `annotations`, and `categories`.
- Set `coco.label2cat` so prediction labels can be mapped back to original category ids.
- Update `get_coco_api_from_dataset(...)` to return `.coco` for the new dataset class.

This avoids disabling existing validation metrics.

## TrainConfig Changes

Extend `TrainConfig.dataset_file` in `src/rfdetr/config.py` from:

```python
Literal["coco", "o365", "roboflow", "yolo"]
```

to include:

```python
"hf_bbox"
```

Add optional fields:

```python
hf_repo_id: str | None = None
hf_name: str | None = None
hf_revision: str | None = None
hf_train_split: str = "train"
hf_val_split: str = "validation"
hf_test_split: str = "test"
hf_image_column: str = "GroundTruthPageImages"
hf_bbox_column: str = "GroundTruthBboxOnPageImages"
```

Validation:

- if `dataset_file == "hf_bbox"`, require `hf_repo_id`;
- validate that split names are non-empty strings;
- keep fields optional for non-HF dataset types.

## Dataset Builder Changes

Update `src/rfdetr/datasets/__init__.py`:

```python
if args.dataset_file == "hf_bbox":
    return build_hf_bbox(image_set, args, resolution)
```

Split mapping in `build_hf_bbox(...)`:

- `image_set == "train"` -> `args.hf_train_split`;
- `image_set == "val"` -> `args.hf_val_split`;
- `image_set == "test"` -> `args.hf_test_split`;
- unknown split names should raise `ValueError`.

## Class Names And Num Classes

RF-DETR's `RFDETR.train()` currently auto-detects class count for COCO/YOLO/Roboflow dataset directories. Hugging Face
datasets do not have a local directory that this method can inspect.

Recommended behavior:

- `src/rfdetr/train.py` should detect HF class names before calling `.train(...)`;
- if `model.num_classes` is not explicitly set, set it to the number of classes discovered from the HF training split;
- if `train.class_names` is not explicitly set, populate it from discovered class names.

This preserves class-name checkpoint metadata and avoids relying on directory-based class detection.

The HF class discovery should:

- scan the training split's bbox records;
- prefer category ids for stable ordering;
- map each category id to the first observed label;
- create contiguous RF-DETR labels in sorted source category-id order.

## Dependencies

Add `datasets` to the `train` optional dependency group in `pyproject.toml`.

Rationale:

- HF dataset loading is training functionality.
- Users who install `rfdetr[train]` should be able to use `dataset.source: huggingface`.

Avoid adding `docling-eval` as a dependency. The adapter should read the already-serialized HF rows directly and should
not need to import `DatasetRecordWithBBox`.

## Error Handling

The HF adapter should raise clear errors for:

- missing `hf_repo_id`;
- missing image or bbox columns;
- page images that cannot be converted to PIL images;
- bbox column values that are neither dicts nor JSON strings;
- missing page entries;
- bbox records with neither `bbox` nor `ltrb`;
- invalid bbox dimensions;
- inconsistent category ids and labels.

Rows or boxes with invalid zero-area boxes should be skipped, matching existing COCO behavior after clipping.

## Tests

Add focused tests before or alongside implementation:

1. `tests/training/test_train_script_config.py`
   - default YAML writes;
   - YAML reads;
   - local dataset config maps to existing `dataset_dir` and `dataset_file`;
   - HF config maps to `dataset_file="hf_bbox"` and HF fields.

2. `tests/datasets/test_hf_bbox.py`
   - builds flattened samples from in-memory mocked HF rows;
   - converts `bbox` to xyxy tensors;
   - falls back from `ltrb` when `bbox` is absent;
   - builds `.coco` with categories/images/annotations;
   - applies class-name ordering by category id.

3. Existing trainer smoke tests should continue to pass.

Full local verification target:

```bash
uv run --no-sync pytest tests/datasets/test_hf_bbox.py tests/training/test_train_script_config.py
uv run --no-sync pytest src/ tests/ -n 1 -m "not gpu" --ignore=tests/run_smoke_all_models.py --timeout=240 --durations=50
pre-commit run --all-files
```

## Implementation Order

1. Add tests for the training config script behavior.
2. Add `src/rfdetr/train.py`.
3. Add tests for HF bbox flattening and COCO metadata.
4. Add `src/rfdetr/datasets/hf_bbox.py`.
5. Wire `dataset_file="hf_bbox"` through `TrainConfig` and `build_dataset(...)`.
6. Add `datasets` dependency to the train extra.
7. Run focused tests.
8. Run broader CPU tests if dependency setup allows.
9. Run pre-commit.

## Open Questions

1. Should HF datasets without a validation split use the training split for validation, or should the script require
   `hf_val_split` to exist?
2. Should `GroundTruthBboxOnPageImages` entries keyed by strings (`"0"`) and integers (`0`) both be accepted? Proposed:
   accept both.
3. Should bbox category ids be preserved exactly in COCO evaluation metadata? Proposed: yes, while RF-DETR training
   labels remain contiguous 0-based labels.
4. Should pages without any bbox entries be included as negative samples? Proposed: yes, because document pages without
   target objects are valid detection negatives.
