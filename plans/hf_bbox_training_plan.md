# Parquet DatasetRecordWithBBox Training Plan

## Goal

Add a dedicated RF-DETR training entry point at `src/rfdetr/train.py` that trains directly from parquet shards
containing Docling-style `DatasetRecordWithBBox` rows.

The intended data path is:

1. Resolve a local dataset directory or Hugging Face dataset repo snapshot.
2. Find train parquet shards.
3. Lazily iterate parquet rows during training.
4. Expand each row into page-level `(image, bbox_target)` samples.
5. Collate those page samples into normal RF-DETR image batches.
6. Reuse RF-DETR's existing model, transform, Lightning trainer, checkpoint, and logging stack.

This plan intentionally avoids requiring a COCO-style dataset directory or a full preprocessing step that writes images
and annotations to disk.

## Important Schema Assumption

The reference `DatasetRecordWithBBox` model describes:

- page images in `GroundTruthPageImages`;
- bounding boxes in `GroundTruthBboxOnPageImages`;
- `GroundTruthBboxOnPageImages` as a dict from zero-based page number to bbox records.

The user message says both images and boxes are on `GroundTruthBboxOnPageImages`. The implementation should follow the
Docling model and RT-DETR reference:

- image column: `GroundTruthPageImages`;
- bbox column: `GroundTruthBboxOnPageImages`.

## Reference Implementation

Use `_reference/RTDETRv2/src/rtdetrv2/tools/train_from_datsetrecord_with_bbox.py` as the behavioral reference.

Key ideas to port and simplify for RF-DETR:

- `resolve_dataset_root(...)`
  - accepts either a local dataset directory or an HF dataset repo id;
  - uses `huggingface_hub.snapshot_download(repo_type="dataset")` for repo ids;
  - returns a local root containing parquet files.
- `detect_layout(...)`
  - supports Hugging Face-style flat layout: `<root>/data/train-*.parquet`;
  - supports split-subdirectory layout: `<root>/train/*.parquet`.
- `resolve_parquet_paths(...)`
  - for RF-DETR, only needs to return the train parquet directory and optional `train` split prefix.
- `DatasetRecordWithBBoxParquetDataset`
  - indexes parquet files from metadata only;
  - caches only the current parquet file table;
  - validates rows as `DatasetRecordWithBBox`;
  - expands each document row into page-level samples via `get_page_images_with_bboxes()`;
- retries invalid rows or rows with no usable pages according to code defaults;
  - returns a list of page samples per row.
- `FlattenPageBatchCollateFunction`
  - flattens row samples into a normal image batch.

RF-DETR's implementation should adapt these concepts to RF-DETR's target format and existing collate function.

## Current RF-DETR Training Architecture

RF-DETR already has:

- public wrapper: `RFDETR.train(**kwargs)` in `src/rfdetr/detr.py`;
- Lightning data module: `RFDETRDataModule` in `src/rfdetr/training/module_data.py`;
- dataset dispatch: `build_dataset(image_set, args, resolution)` in `src/rfdetr/datasets/__init__.py`;
- model-compatible collate: `make_collate_fn(...)` in `src/rfdetr/utilities/tensors.py`;
- transform builders in `src/rfdetr/datasets/coco.py`.

The new path should plug into `build_dataset(...)`, then let `RFDETRDataModule` build DataLoaders as it does today.
The dedicated `src/rfdetr/train.py` should still call `model.train(...)`, because that preserves:

- model-config validation;
- resolution override validation;
- device mapping;
- auto-batch support;
- checkpoint metadata;
- post-training class-name sync;
- `training_config.json` output.

## Target User Experience

Run from YAML:

```bash
uv run python -m rfdetr.train --write-default-config --config train.yaml
uv run python -m rfdetr.train --config train.yaml
```

Example Hugging Face repo-backed parquet config:

```yaml
variant: large

model:
  # Optional RF-DETR model-config overrides.
  # resolution: 704
  # num_classes: 2

dataset:
  source: parquet
  dataset_repo_id: your-org/your-dataset
  max_objects: 400

label_mapping:
  0: text
  1: table
  2: figure

train:
  output_dir: ./output/rfdetr-docling
  epochs: 50
  batch_size: 4
  grad_accum_steps: 4
  num_workers: 4
  tensorboard: true
  progress_bar: tqdm
```

Example local parquet directory config:

```yaml
variant: large

dataset:
  source: parquet
  dataset_dir: ./dataset_root

label_mapping:
  0: text
  1: table

train:
  output_dir: ./output/rfdetr-local-parquet
  epochs: 50
  batch_size: 4
```

Supported layouts:

```text
dataset_root/
  data/
    train-00000-of-00010.parquet
```

or:

```text
dataset_root/
  train/
    shard-0000.parquet
```

## `src/rfdetr/train.py`

Add a dedicated script with Pydantic YAML config.

Top-level config:

- `variant`: RF-DETR variant name.
- `model`: free-form dictionary validated by the selected variant's model config class.
- `dataset`: parquet source config.
- `label_mapping`: explicit mapping from category id to class name.
- `train`: free-form dictionary validated by the selected variant's train config class.

Dataset config fields:

```python
source: Literal["parquet"] = "parquet"
dataset_dir: str | None = None
dataset_repo_id: str | None = None
max_objects: int | None = 400
```

Validation:

- exactly one of `dataset_dir` or `dataset_repo_id` must be provided;
- `max_objects` must be `None` or positive.
- `label_mapping` must be non-empty;
- label-mapping keys must be integer category ids;
- label-mapping values must be non-empty class names.

The script should convert the YAML into `model.train(...)` kwargs:

- `dataset_file="parquet_bbox"`;
- resolved parquet root or repo metadata fields;
- explicit `label_mapping`;
- output/training hyperparameters from `train`;
- `class_names` derived from `label_mapping`.

## Dataset Implementation

Add `src/rfdetr/datasets/parquet_bbox.py`.

Core class:

```python
class DatasetRecordWithBBoxParquetDataset(torch.utils.data.Dataset):
    ...
```

Responsibilities:

1. Accept a train parquet file path, train split directory, or shared parquet directory plus the fixed `train` prefix.
2. Use `pyarrow.parquet.ParquetFile(...).metadata.num_rows` to index file offsets without loading all data.
3. Resolve global row index to `(file_idx, local_idx)`.
4. Cache the current parquet table only:
   - when the sampler moves to a different file, read that file;
   - keep only one file in memory per worker.
5. Convert row dictionaries into page-level samples.
6. Apply RF-DETR image/target transforms per page sample.
7. Return `list[tuple[Image.Image | Tensor, dict]]` from `__getitem__`.

The row parser should support two modes:

- preferred: import and validate `docling_eval.datamodels.dataset_record.DatasetRecordWithBBox` when available;
- fallback: parse `GroundTruthPageImages` and `GroundTruthBboxOnPageImages` directly from parquet row values.

The fallback keeps RF-DETR usable without forcing `docling-eval` as a hard dependency.

## Page Sample Extraction

Preferred path:

```python
record = DatasetRecordWithBBox.model_validate(row)
samples = record.get_page_images_with_bboxes()
```

Fallback path:

- read `row["GroundTruthPageImages"]` as a list of PIL-compatible page images;
- read `row["GroundTruthBboxOnPageImages"]` as a dict or JSON string;
- for each page index:
  - accept page keys as either `int` or `str`;
  - convert the image to RGB PIL;
  - get the page bbox list;
  - build target tensors.

Pages without bboxes should be dropped for training. If a row produces no usable page samples, retry within the same
parquet file or fail with a clear error after the configured retry limit in code.

## RF-DETR Target Format

For each page sample, build:

```python
target = {
    "boxes": torch.float32[N, 4],   # xyxy pixel coordinates
    "labels": torch.int64[N],       # contiguous RF-DETR label ids
    "image_id": torch.int64[1],
    "area": torch.float32[N],
    "iscrowd": torch.int64[N],
    "orig_size": torch.int64[2],    # [height, width]
    "size": torch.int64[2],         # [height, width]
}
```

Bounding box conversion:

- prefer `ltrb` if present;
- otherwise convert COCO `bbox = [x, y, width, height]` into `[left, top, right, bottom]`;
- clamp to page image bounds;
- drop zero-area or negative-area boxes after clipping.

Labels:

- source boxes must use `category_id` values present in the explicit YAML `label_mapping`;
- RF-DETR loss expects contiguous label ids;
- the recommended config is therefore to use contiguous keys starting at `0`;
- if non-contiguous category ids are supplied, build a deterministic `category_id -> label_id` mapping in sorted
  category-id order and set `class_names` from that same order.

Image IDs:

- deterministic and unique across row/page samples;
- use a large multiplier as in the reference, e.g. `row_idx * 100000 + page_index`.

## Collation Strategy

The dataset returns row-level lists of page samples. RF-DETR's existing `make_collate_fn(...)` expects a flat list of
image samples.

Add a small wrapper collate in `src/rfdetr/datasets/parquet_bbox.py`:

```python
class FlattenPageSamplesCollate:
    def __init__(self, base_collate):
        self.base_collate = base_collate

    def __call__(self, items):
        flat_items = [sample for row_samples in items for sample in row_samples]
        if not flat_items:
            raise RuntimeError("Collate received no page samples to batch.")
        return self.base_collate(flat_items)
```

Then update `RFDETRDataModule` so that when `train_config.dataset_file == "parquet_bbox"`, it wraps the existing
`self._collate_fn` with `FlattenPageSamplesCollate`.

This preserves RF-DETR's current padding, masks, and batch structure.

## Sampler Strategy

Port `ShardSequentialSampler` from the reference.

Purpose:

- iterate file-by-file to maximize cache locality;
- optionally shuffle file order and row order within each file per epoch;
- avoid thrashing between parquet files.

Training DataLoader:

- use `ShardSequentialSampler(..., shuffle=True)` for parquet training datasets;
- avoid `shuffle=True` at the DataLoader level;
- keep existing RF-DETR small-dataset replacement sampler behavior only if it can work with row-list samples.

Integration option:

- expose `file_offsets` and `file_num_rows` on the dataset;
- in `RFDETRDataModule.train_dataloader()`, detect parquet dataset and use the shard sampler.

## Explicit Labels

The script must not infer classes from parquet. Labels are part of the training contract and must be declared in YAML:

```yaml
label_mapping:
  0: text
  1: table
  2: figure
```

The script should:

- normalize mapping keys to integers;
- validate unique, non-empty names;
- derive `class_names` from mapping values ordered by category id;
- set `model.num_classes` to `len(label_mapping)` unless explicitly provided;
- set `train.class_names` from `label_mapping` unless explicitly provided;
- pass the mapping to the parquet dataset so bbox `category_id` values are remapped consistently.

## TrainConfig Changes

Extend `TrainConfig.dataset_file` in `src/rfdetr/config.py` to include:

```python
"parquet_bbox"
```

Add optional fields:

```python
dataset_repo_id: str | None = None
parquet_max_objects: int | None = 400
parquet_label_mapping: dict[int, str] | None = None
```

Use `dataset_dir` for the resolved parquet root. The dataset builder should use fixed conventions from the reference:

- `<dataset_dir>/data/train-*.parquet`, or
- `<dataset_dir>/train/*.parquet`.

The dedicated training script can resolve `dataset_repo_id` into a local snapshot path and pass that path as
`dataset_dir` before calling `model.train(...)`.

## Dependencies

Add to the training dependency set:

- `pyarrow`, for parquet row access;
- `huggingface_hub`, for snapshot download from HF dataset repos.

Do not make `docling-eval` a hard dependency. If present, use it for row validation and `get_page_images_with_bboxes()`.
If absent, parse configured columns directly.

## Error Handling

Clear errors should be raised for:

- neither `dataset_dir` nor `dataset_repo_id` provided;
- missing or empty `label_mapping`;
- no train parquet files found;
- missing `GroundTruthPageImages` or `GroundTruthBboxOnPageImages` columns;
- bbox column values that cannot be parsed as dict/JSON;
- image values that cannot be converted to PIL images;
- rows with no usable page samples when skip/retry behavior is disabled;
- too many repeated invalid rows;
- bbox category ids missing from the explicit label mapping.

Invalid boxes should be skipped after clipping, not crash training.

## Tests

Add focused tests:

1. `tests/training/test_train_script_config.py`
   - YAML writes default parquet config;
   - YAML reads repo-backed config;
   - YAML reads local directory config;
   - config maps to `dataset_file="parquet_bbox"`;
   - missing both `dataset_dir` and `dataset_repo_id` fails validation;
   - empty `label_mapping` fails validation;
   - label mapping is converted into `model.num_classes` and `train.class_names`.

2. `tests/datasets/test_parquet_bbox.py`
   - writes a tiny parquet file with two document rows;
   - verifies parquet metadata indexing does not load all rows during initialization;
   - loads one row and expands multiple page samples;
   - converts `bbox` and `ltrb` to xyxy tensors;
   - drops invalid boxes;
   - supports string and integer page keys;
   - verifies `FlattenPageSamplesCollate` delegates to RF-DETR base collate.

3. `tests/datasets/test_parquet_bbox_labels.py`
   - maps explicit category ids to contiguous RF-DETR labels;
   - rejects bbox category ids absent from `label_mapping`;
   - verifies stable class-name order by category id.

4. Existing data module tests
   - add a narrow test that `RFDETRDataModule` wraps collate and uses shard sampler for `dataset_file="parquet_bbox"`.

Verification commands:

```bash
uv run --no-sync pytest tests/datasets/test_parquet_bbox.py tests/datasets/test_parquet_bbox_labels.py tests/training/test_train_script_config.py
uv run --no-sync pytest tests/training/test_module_data.py
uv run --no-sync pytest src/ tests/ -n 1 -m "not gpu" --ignore=tests/run_smoke_all_models.py --timeout=240 --durations=50
pre-commit run --all-files
```

## Implementation Order

1. Add tests for YAML config mapping and validation.
2. Add `src/rfdetr/train.py`.
3. Add parquet dataset tests with generated temporary parquet shards.
4. Add `src/rfdetr/datasets/parquet_bbox.py`.
5. Wire `dataset_file="parquet_bbox"` through `TrainConfig` and `build_dataset(...)`.
6. Add collate wrapping and shard sampler integration in `RFDETRDataModule`.
7. Add explicit label mapping validation and `num_classes`/`class_names` injection in `src/rfdetr/train.py`.
8. Add dependencies.
9. Run focused tests.
10. Run broader CPU tests if dependency setup allows.
11. Run pre-commit.

## Open Questions

1. How should the training path bypass RF-DETR's current COCO validation callback cleanly? Dedicated evals are out of
   scope, so the implementation should avoid constructing validation/evaluation metadata.
2. Should training drop pages without boxes? Proposed: yes by default, matching the RT-DETR reference.
3. Should the row parser require `docling-eval`? Proposed: no; use it opportunistically and keep a direct parser
   fallback.
