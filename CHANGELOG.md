# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `BuilderArgs` — a `@runtime_checkable` `typing.Protocol` documenting the minimum attribute set consumed by `build_model()`, `build_backbone()`, `build_transformer()`, and `build_criterion_and_postprocessors()`. Enables static type-checker support for custom builder integrations. Exported from `rfdetr.models`.
- `build_model_from_config(model_config, train_config=None, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_model(build_namespace(mc, tc))`; accepts Pydantic config objects directly and constructs the internal namespace automatically. Exported from `rfdetr.models`.
- `build_criterion_from_config(model_config, train_config, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_criterion_and_postprocessors(build_namespace(mc, tc))`; returns a `(SetCriterion, PostProcess)` tuple. Exported from `rfdetr.models`.
- `ModelDefaults` dataclass — exposes the 35 hardcoded architectural constants previously buried inside `build_namespace()`. Pass a `dataclasses.replace(MODEL_DEFAULTS, ...)` override to the new config-native builders to customise individual constants. **Note:** fields may be promoted to `ModelConfig`/`TrainConfig` in future phases. Exported from `rfdetr.models`.
- `MODEL_DEFAULTS` — the canonical `ModelDefaults` singleton with production defaults. Exported from `rfdetr.models`.

### Deprecated

- `build_namespace(model_config, train_config)` — no longer used internally and deprecated in this release; use `build_model_from_config`, `build_criterion_from_config`, or `_namespace_from_configs` directly. It will be removed in v1.9 and currently emits a `DeprecationWarning` on use.
- `load_pretrain_weights(nn_model, model_config, train_config)` — the `train_config` positional argument is deprecated and will be removed in v1.9; it is no longer used internally. Omit it: `load_pretrain_weights(nn_model, model_config)`. Passing a non-`None` value emits a `DeprecationWarning`.

The following fields are duplicated between `ModelConfig` and `TrainConfig`; clear ownership is being established for v1.9. Each field now emits `DeprecationWarning` when set on the wrong config object. The fields continue to work as before — this is a warning-only Phase A change.

- `TrainConfig.group_detr` — architecture decision; set on `ModelConfig` instead.
- `TrainConfig.ia_bce_loss` — loss type tied to architecture family; set on `ModelConfig` instead.
- `TrainConfig.segmentation_head` — architecture flag; set on `ModelConfig` instead.
- `TrainConfig.num_select` — postprocessor count is an architecture decision; set on `ModelConfig` instead. `SegmentationTrainConfig` users: remove the `num_select` override — the model config value is always used.
- `ModelConfig.cls_loss_coef` — training hyperparameter; set on `TrainConfig` instead.

These fields will be **removed** in v1.9 after a full release cycle.

### Fixed

- Fixed `_namespace.py`: `num_select` in the builder namespace now always reads from `ModelConfig`, eliminating a regression where `TrainConfig.num_select` (default 300) silently overrode model-specific values of 100–200 for segmentation variants (`RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegPreview`). Post-processing now uses the correct top-k count for each model.
- Fixed `models/weights.py`: `load_pretrain_weights` now correctly auto-aligns the model head when the checkpoint has fewer classes than the configured default, preventing a silent mismatch when `num_classes` was not explicitly set by the caller.

---

## [1.6.0] — 2026-03-20

### Added

- PyTorch Lightning training building blocks: `RFDETRModelModule`, `RFDETRDataModule`, `build_trainer()`, and individual callbacks (`RFDETREMACallback`, `COCOEvalCallback`, `BestModelCallback`, `DropPathCallback`, `MetricsPlotCallback`) — all standard PTL components, swap/subclass/extend any piece. Level 3: `rfdetr fit --config` CLI with zero Python required. (#757, #794, closes #709)
- Multi-GPU DDP via `model.train()`: `strategy`, `devices`, and `num_nodes` added to `TrainConfig`; single-GPU behaviour unchanged when omitted. (#808, closes #803)
- `batch_size='auto'`: CUDA memory probe finds the largest safe micro-batch size, then recommends `grad_accum_steps` to reach a configurable effective batch target (default 16 via `auto_batch_target_effective`). (#814)
- `ModelContext` promoted from `_ModelContext` to a public, exported API — inspect `class_names`, `num_classes`, and related metadata via `model.context` after training. (#835)
- `backbone_lora` and `freeze_encoder` added as first-class fields in `ModelConfig`. (#829)
- `generate_coco_dataset(with_segmentation=True)` produces COCO polygon annotations alongside bounding boxes for segmentation fine-tuning with synthetic data. (#781)
- `set_attn_implementation("eager" | "sdpa")` on the DINOv2 backbone — switch attention implementation at runtime. (#760)
- `eval_max_dets`, `eval_interval`, and `log_per_class_metrics` added to `TrainConfig`.
- `python -m rfdetr` entry point alongside the `rfdetr` console script.
- `py.typed` marker — RF-DETR is now PEP 561–compliant.

### Changed

- **Breaking:** Minimum `transformers` version bumped to `>=5.1.0,<6.0.0`. The DINOv2 windowed-attention backbone now uses the transformers v5 API (`BackboneMixin._init_transformers_backbone()`, removed `head_mask` plumbing). Projects still on transformers v4 must pin `rfdetr<1.6.0`. (#760, closes #730)
- **Breaking:** PyPI install extras renamed — `rfdetr[metrics]` → `rfdetr[loggers]`, `rfdetr[onnxexport]` → `rfdetr[onnx]`.
- `draw_synthetic_shape` now returns `Tuple[np.ndarray, List[float]]` instead of `np.ndarray`. The second element is a flat COCO-style polygon list `[x1, y1, x2, y2, …]`. Any caller that previously did `img = draw_synthetic_shape(...)` must be updated to `img, polygon = draw_synthetic_shape(...)`. (#781)
- Albumentations version constraint broadened to `>=1.4.24,<3.0.0`; `RandomSizedCrop` configs using `height`/`width` kwargs are automatically adapted to the 2.x `size=(height, width)` API. (#786, closes #779)
- Current learning rate is now shown in the training progress bar alongside loss. (#809, closes #804)
- `supervision`, `pytorch_lightning`, and other heavy dependencies are now imported lazily (on first use) rather than at module load, reducing cold-import time in inference-only environments. (#801)

### Deprecated

- `rfdetr.deploy.*` — redirects to `rfdetr.export.*` with a `DeprecationWarning`. Migrate before v1.7.
- `rfdetr.util.*` — redirects to `rfdetr.utilities.*` with a `DeprecationWarning`. Migrate before v1.7.

### Fixed

- Raised a descriptive `ValueError` instead of a cryptic `RuntimeError` / tensor-size mismatch when a checkpoint is incompatible with the current model architecture — covers `segmentation_head` mismatch and `patch_size` mismatch. (#810, closes #806)
- Fixed `class_names` not reflecting dataset labels on `model.predict()` after training — class names are now synced from the dataset so inference always uses the correct label list. (#816)
- Fixed detection head reinitialization overwriting fine-tuned weights when loading a checkpoint with fewer classes than the model default. The second `reinitialize_detection_head` call now fires only in the backbone-pretrain scenario. (#815, closes #813, #509)
- Fixed `grid_sample` and bicubic interpolation silently falling back to CPU on MPS (Apple Silicon) — both now run natively on the MPS device. (#821)
- Fixed `early_stopping=False` in `TrainConfig` being silently ignored — the setting now propagates correctly. (#835)
- Fixed `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern. (closes #750)
- Added warning when `drop_path_rate > 0.0` is configured with a non-windowed DINOv2 backbone, where drop-path is silently ignored.
- Fixed `ValueError: matrix entries are not finite` in `HungarianMatcher` when the cost matrix contains NaN or Inf — non-finite entries are replaced with a finite sentinel before `linear_sum_assignment`, warning emitted at most once per matcher instance. (#787, closes #784)
- Fixed YOLO dataset validation rejecting `data.yml` — both `.yaml` and `.yml` are now accepted. (#777, closes #775)
- Silently dropped degenerate bounding boxes (zero width or height) before Albumentations validation instead of raising `ValueError`. (#825)

## [1.5.2] — 2026-03-04

### Added

- Added peak GPU memory (`max_mem` in MB) to training and evaluation progress bars on CUDA; omitted on CPU and MPS. (#773)

### Fixed

- Fixed `aug_config` being silently ignored when training on YOLO-format datasets — `build_roboflow_from_yolo` never forwarded the value, so transforms always fell back to the default. (#774)
- Fixed segmentation evaluation metrics not being written to `results_mask.json` during validation and test runs. (#772)
- Fixed `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern — `_get_backbone_encoder_layers` now returns `None` for unrecognised architectures. (#762)
- Fixed `drop_path_rate` not being forwarded to the DINOv2 model configuration; stochastic depth was never applied even when explicitly set. Added a warning when `drop_path_rate > 0.0` is used with a non-windowed backbone. (#762)
- Fixed incorrect COCO hierarchy filtering that excluded parent categories from the class list. (#759)
- Fixed evaluation metric corruption on 1-indexed Roboflow datasets caused by a flawed contiguity check in `_should_use_raw_category_ids`. (#755)

## [1.5.1] — 2026-02-27

### Added

- Added support for nested Albumentations containers (`OneOf`, `Sequential`) inside `aug_config`. (#752)

### Changed

- Migrated dataset transform pipeline to torchvision-native `Compose`, `ToImage`, and `ToDtype`; `Normalize` now defaults to ImageNet mean/std. (#745)

### Fixed

- Fixed `RFDETRMedium` missing from the public API — `__all__` contained a duplicate `RFDETRSmall` entry. (#748)
- Fixed `AR50_90` reporting an incorrect value in `MetricsMLFlowSink` due to a wrong COCO evaluation index. (#735)
- Fixed supercategory filtering in `_load_classes` for COCO datasets with flat or mixed supercategory structures. (#744)
- Fixed crash in geometric transforms when a sample contained zero-area or empty masks. (#727)
- Fixed segmentation training on Colab — `DepthwiseConvBlock` now disables cuDNN for depthwise separable convolutions. (#728)
- Pinned `onnxsim<0.6.0` to prevent `pip install` from hanging indefinitely. (#749)

## [1.5.0] — 2026-02-23

### Added

- Added custom training augmentations via `aug_config` in `model.train()` — accepts a dict of Albumentations transforms, a built-in preset (`AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, `AUG_INDUSTRIAL`), or `{}` to disable. Bounding boxes and segmentation masks are transformed automatically. (#263, #702)
- Added `save_dataset_grids=True` in `TrainConfig` to write 3×3 JPEG grids of augmented samples to `output_dir` before training begins. (#153)
- Added ClearML logger: set `clearml=True` in `TrainConfig` to stream per-epoch metrics to ClearML. (#520)
- Added MLflow logger: set `mlflow=True` in `TrainConfig` to log runs and metrics to MLflow with custom tracking URI support. (#109)
- Added live progress bar for training and validation with structured per-epoch logs. (#204)
- Added `device` field to `TrainConfig` for explicit device selection. (#687)
- `ModelConfig` now raises an error on unknown parameters, preventing silent misconfiguration. (#196)

### Changed

- Deprecated `OPEN_SOURCE_MODELS` constant in favour of `ModelWeights` enum. (#696)
- Added MD5 checksum validation for pretrained weight downloads. (#679)

### Fixed

- Fixed Albumentations bool-mask crash during segmentation training. (#706)
- Fixed `UnboundLocalError` when resuming training from a completed checkpoint. (#707)
- Prevented corruption of `checkpoint_best_total.pth` via atomic checkpoint stripping. (#708)
- Fixed PyTorch 2.9+ compatibility issue with CUDA capability detection. (#686)
- Fixed dtype mismatch error when `use_position_supervised_loss=True`. (#447)
- Fixed inconsistent return values from `build_model`. (#519)
- Fixed `positional_encoding_size` type annotation (`bool` → `int`). (#524)
- Fixed ONNX export `output_names` to include masks when exporting segmentation models. (#402)
- Fixed `num_select` not being updated correctly during segmentation model fine-tuning. (#399)
- Fixed `np.argwhere` → `np.argmax` misuse. (#536)
- Fixed COCO sparse category ID remapping for non-contiguous or offset category IDs. (#712)
- Fixed segmentation mask filtering when using aggressive augmentations. (#717)
