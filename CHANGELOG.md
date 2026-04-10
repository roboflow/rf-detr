# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `augmentation_backend` field on `TrainConfig` (`"cpu"` / `"auto"` / `"gpu"`): opt-in GPU-side augmentation via [Kornia](https://kornia.readthedocs.io) applied in `RFDETRDataModule.on_after_batch_transfer` after the batch is resident on the GPU. CPU path is unchanged and remains the default. Install with `pip install 'rfdetr[kornia]'`. Phase 1 supports detection only; segmentation mask support is planned for Phase 2.
- `BuilderArgs` — a `@runtime_checkable` `typing.Protocol` documenting the minimum attribute set consumed by `build_model()`, `build_backbone()`, `build_transformer()`, and `build_criterion_and_postprocessors()`. Enables static type-checker support for custom builder integrations. Exported from `rfdetr.models`. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- `build_model_from_config(model_config, train_config=None, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_model(build_namespace(mc, tc))`; accepts Pydantic config objects directly and constructs the internal namespace automatically. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `build_criterion_from_config(model_config, train_config, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_criterion_and_postprocessors(build_namespace(mc, tc))`; returns a `(SetCriterion, PostProcess)` tuple. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `ModelDefaults` dataclass — exposes the 35 hardcoded architectural constants previously buried inside `build_namespace()`. Pass a `dataclasses.replace(MODEL_DEFAULTS, ...)` override to the new config-native builders to customise individual constants. **Note:** fields may be promoted to `ModelConfig`/`TrainConfig` in future phases. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `MODEL_DEFAULTS` — the canonical `ModelDefaults` singleton with production defaults. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `RFDETR.predict(include_source_image=...)` — opt-out flag (default `True`) to skip storing the source image in `detections.data["source_image"]`; set to `False` to reduce memory use when the image is not needed for annotation. ([#912](https://github.com/roboflow/rf-detr/pull/912))
- `model_name` is now stored in checkpoint files during training so that `RFDETR.from_checkpoint()` can resolve the correct model class directly from the checkpoint, without requiring the caller to know or pass a class hint. `strip_checkpoint()` preserves this key. Backward-compatible: checkpoints without `model_name` continue to resolve via `pretrain_weights` filename matching. ([#895](https://github.com/roboflow/rf-detr/pull/895))
- `rfdetr_version` is now stored in checkpoint files during training for provenance tracking and compatibility hints. `strip_checkpoint()` preserves this key. The key is omitted gracefully when the package version cannot be resolved (e.g. editable install without metadata). Backward-compatible: checkpoints without `rfdetr_version` continue to load normally. ([#918](https://github.com/roboflow/rf-detr/pull/918))

### Deprecated

- `build_namespace(model_config, train_config)` — no longer used internally and deprecated in this release; use `build_model_from_config`, `build_criterion_from_config`, or `_namespace_from_configs` directly. It will be removed in v1.9 and currently emits a `DeprecationWarning` on use. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `load_pretrain_weights(nn_model, model_config, train_config)` — the `train_config` positional argument is deprecated and will be removed in v1.9; it is no longer used internally. Omit it: `load_pretrain_weights(nn_model, model_config)`. Passing a non-`None` value emits a `DeprecationWarning`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `TrainConfig.group_detr` (architecture decision → `ModelConfig`), `TrainConfig.ia_bce_loss` (loss type tied to architecture family → `ModelConfig`), `TrainConfig.segmentation_head` (architecture flag → `ModelConfig`), `TrainConfig.num_select` (postprocessor count → `ModelConfig`; `SegmentationTrainConfig` users: remove the `num_select` override — the model config value is always used), `ModelConfig.cls_loss_coef` (training hyperparameter → `TrainConfig`) — each now emits `DeprecationWarning` when set on the wrong config object and will be **removed** in v1.9. ([#841](https://github.com/roboflow/rf-detr/pull/841))

### Fixed

- `WindowedDinov2WithRegistersEmbeddings.forward()` now raises `ValueError` (instead of silently failing under `-O`) when input spatial dimensions are not divisible by `patch_size * num_windows`, with a clear message identifying the divisor and actual shape. ([#167](https://github.com/roboflow/rf-detr/pull/167))
- Fixed `_namespace.py`: `num_select` in the builder namespace now always reads from `ModelConfig`, eliminating a regression where `TrainConfig.num_select` (default 300) silently overrode model-specific values of 100–200 for segmentation variants (`RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegPreview`). Post-processing now uses the correct top-k count for each model. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- Fixed `models/weights.py`: `load_pretrain_weights` now correctly auto-aligns the model head when the checkpoint has fewer classes than the configured default, preventing a silent mismatch when `num_classes` was not explicitly set by the caller. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- Fixed YOLO segmentation training on large datasets hitting OS out-of-memory: `supervision.DetectionDataset.from_yolo(force_masks=True)` was eager-rasterising H×W boolean masks for every image at dataset construction time (≈1 GB/1 000 images at 1024 px). A new `_LazyYoloDetectionDataset` stores polygon coordinates only and defers dense mask rasterisation to `__getitem__`, keeping RAM proportional to annotation count rather than (N × H × W). ([#851](https://github.com/roboflow/rf-detr/pull/851))
- Fixed ONNX/TRT dynamic batch inference: `gen_encoder_output_proposals` and `Transformer.forward` extracted the batch size as a Python int and passed it to `torch.full`, `.view(N_, ...)`, `.expand(N_, ...)`, and `.repeat(bs, ...)`, causing the ONNX tracer to bake the training batch size (e.g. 8) as a compile-time constant. TRT engines built with `--minShapes` smaller than the trace batch would fail at inference with `Reshape: reshaping failed`. All six call sites are now replaced with ONNX-symbolic equivalents (`zeros_like`, `-1` reshapes, `expand(memory.shape[0], ...)`), keeping the batch dimension fully dynamic. ([#950](https://github.com/roboflow/rf-detr/pull/950), closes [#949](https://github.com/roboflow/rf-detr/issues/949))

---

## [1.6.4] — 2026-04-10

### Changed

- `predict()` now includes `class_name` in `detections.data`, mapping each detection's 0-indexed class ID to its human-readable name. ([#914](https://github.com/roboflow/rf-detr/pull/914))

### Fixed

- Fixed segmentation multi-GPU DDP training crash: `build_trainer()` now wraps `strategy="ddp"` with `DDPStrategy(find_unused_parameters=True)` when `segmentation_head=True`. The segmentation head's `sparse_forward()` leaves parameters unused on some forward steps; plain `"ddp"` raised `RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss`. Non-segmentation DDP and other strategies are unchanged. ([#942](https://github.com/roboflow/rf-detr/pull/942), [#947](https://github.com/roboflow/rf-detr/pull/947))
- Fixed fused AdamW crash under FP32 multi-GPU training: `configure_optimizers()` and `clip_gradients()` now gate fused AdamW on the trainer's actual precision (requiring a BF16 variant) rather than GPU capability alone. On Ampere+ hardware `torch.cuda.is_bf16_supported()` is always `True`, so the old code enabled fused AdamW even with `precision="32-true"`, causing `RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout` from DDP gradient bucket view stride mismatches. ([#942](https://github.com/roboflow/rf-detr/pull/942), [#947](https://github.com/roboflow/rf-detr/pull/947))
- Fixed multi-GPU DDP training crashing in Jupyter notebooks and Kaggle: replaced fork-based `ddp_notebook` strategy with a spawn-based DDP strategy that avoids OpenMP thread pool corruption after `fork()`. ([#928](https://github.com/roboflow/rf-detr/pull/928))
- Fixed `RFDETR.train(resolution=...)` being silently ignored — the kwarg is now applied to `model_config` before training begins, with validation that the value is divisible by `patch_size * num_windows`. ([#933](https://github.com/roboflow/rf-detr/pull/933))
- Fixed `save_dataset_grids` being silently a no-op — `DatasetGridSaver` is now wired into the training loop, saving sample grids to `{output_dir}/dataset_grids/` when enabled. Grid save failures are caught without interrupting training. ([#946](https://github.com/roboflow/rf-detr/pull/946))
- Fixed partial gradient-accumulation windows at the tail of training epochs: the training dataset is now padded to an exact multiple of `effective_batch_size * world_size`, ensuring every optimizer step uses a full gradient window. Workaround for [pytorch-lightning#19987](https://github.com/Lightning-AI/pytorch-lightning/issues/19987). ([#937](https://github.com/roboflow/rf-detr/pull/937))
- Fixed `torch.export.export` failing on the transformer decoder by threading `spatial_shapes_hw` through all decoder layers. ([#936](https://github.com/roboflow/rf-detr/pull/936))
- `download_pretrain_weights()` no longer overwrites fine-tuned checkpoints that share a filename with a registry model (e.g. `rf-detr-nano.pth`). Previously, an MD5 mismatch would fall through to `_download_file()` and silently replace the user's weights with the original COCO checkpoint. The function now returns early whenever the file exists and `redownload=False`, regardless of MD5 status — a warning is emitted when the hash differs. Pass `redownload=True` to force a fresh download. ([#935](https://github.com/roboflow/rf-detr/pull/935))

## [1.6.3] — 2026-04-02

### Changed

- `predict()` now stores the original image and its shape on returned `sv.Detections` objects — `detections.data["source_image"]` (NumPy array) and `detections.data["source_shape"]` (height, width) let you annotate results without loading the image separately. ([#892](https://github.com/roboflow/rf-detr/pull/892))
- `RFDETR.train()` auto-detects `num_classes` from the dataset directory when not explicitly set, reinitializing the detection head to the correct class count automatically. A warning is emitted when the configured value differs from the dataset count. ([#893](https://github.com/roboflow/rf-detr/pull/893))
- `optimize_for_inference()` now accepts dtype as a string name (e.g. `"float16"`) in addition to a `torch.dtype` object; invalid dtype inputs uniformly raise `TypeError`. ([#899](https://github.com/roboflow/rf-detr/pull/899))

### Fixed

- Fixed `models/lwdetr.py`: `reinitialize_detection_head` now replaces `nn.Linear` modules instead of mutating `.data` tensors in-place, ensuring `out_features` metadata stays consistent with the actual weight shape. This prevents ONNX export and `torch.jit.trace` from emitting stale (pre-fine-tuning) class counts for fine-tuned models. ([#904](https://github.com/roboflow/rf-detr/pull/904))
- Fixed `RFDETR.optimize_for_inference()` leaking a CUDA context on multi-GPU setups: the deep-copy, export, and JIT-trace steps now run inside `torch.cuda.device(device)` to pin the context to the correct device. ([#899](https://github.com/roboflow/rf-detr/pull/899))
- Fixed `optimize_for_inference()` leaving inconsistent state on failure: prior optimized state is now reset and flags are committed only after a successful build/trace; temp download files use unique per-process paths to avoid parallel worker collisions.
- Fixed `deploy_to_roboflow` failing with `FileNotFoundError` after PyTorch Lightning migration: `class_names.txt` is now written to the upload directory and `args.class_names` is populated before saving the checkpoint. ([#890](https://github.com/roboflow/rf-detr/pull/890))

## [1.6.2] — 2026-03-27

### Added

- `RFDETR.predict(shape=...)` — optional `(height, width)` tuple overrides the default square inference resolution; useful when matching a non-square ONNX export. Both dimensions must be positive integers divisible by `patch_size × num_windows` as determined by the model configuration. ([#866](https://github.com/roboflow/rf-detr/pull/866))

### Changed

- `ModelConfig.device` and `RFDETR.train(device=...)` now accept `torch.device` objects and indexed device strings such as `"cuda:0"`. Values are normalized to canonical torch-style strings. `RFDETR.train()` warns when an unmapped device type is passed to PyTorch Lightning auto-detection. ([#872](https://github.com/roboflow/rf-detr/pull/872))

### Fixed

- Fixed ONNX export ignoring an explicit `patch_size` argument: `export()` and `predict()` now resolve `patch_size` from `model_config` by default, validate it strictly (positive integer, not bool), and enforce that `(H, W)` dimensions are divisible by `patch_size × num_windows`. ([#876](https://github.com/roboflow/rf-detr/pull/876))
- Fixed ONNX export for models with dynamic batch dimensions — replaced `H_.expand(N_)` with `torch.full` for Python-int spatial dims to eliminate tracer failures. ([#871](https://github.com/roboflow/rf-detr/pull/871))

## [1.6.1] — 2026-03-25

### Deprecated

- `RFDETR.export(..., simplify=..., force=...)` — both arguments are now no-ops and emit a `DeprecationWarning`. RF-DETR no longer runs ONNX simplification automatically; remove these arguments from your calls. They will be removed in v1.8. ([#861](https://github.com/roboflow/rf-detr/pull/861))

### Fixed

- Fixed `RFDETR.train()`: a missing `rfdetr[train]` install (e.g. plain `pip install rfdetr` in Colab) now raises an `ImportError` with an actionable message — `pip install "rfdetr[train,loggers]"` — instead of a raw `ModuleNotFoundError` with no install hint. ([#858](https://github.com/roboflow/rf-detr/pull/858))
- Fixed `AUG_AGGRESSIVE` preset: `translate_percent` was `(0.1, 0.1)` — a degenerate range that forced Albumentations `Affine` to always translate right/down by exactly 10%. Corrected to `(-0.1, 0.1)` for symmetric bidirectional translation. ([#863](https://github.com/roboflow/rf-detr/pull/863))
- Fixed PTL training path: `latest.ckpt` and per-interval checkpoints (`checkpoint_interval_N.ckpt`) are now properly written and restored on resume. ([#847](https://github.com/roboflow/rf-detr/pull/847))
- Fixed `BestModelCallback` and checkpoint monitor raising `MisconfigurationException` on non-eval epochs when `eval_interval > 1` — monitor key absence is now handled gracefully. ([#848](https://github.com/roboflow/rf-detr/pull/848))
- Fixed `protobuf` version constraint in the `loggers` extra to guard against TensorBoard descriptor crash (`TypeError: Descriptors cannot be created directly`) with protobuf ≥ 4. ([#846](https://github.com/roboflow/rf-detr/pull/846))
- Fixed duplicate `ModelCheckpoint` state keys when `checkpoint_interval=1`; `last.ckpt` is omitted in that configuration to avoid collision. ([#859](https://github.com/roboflow/rf-detr/pull/859))

## [1.6.0] — 2026-03-20

### Added

- PyTorch Lightning training building blocks: `RFDETRModelModule`, `RFDETRDataModule`, `build_trainer()`, and individual callbacks (`RFDETREMACallback`, `COCOEvalCallback`, `BestModelCallback`, `DropPathCallback`, `MetricsPlotCallback`) — all standard PTL components, swap/subclass/extend any piece. Level 3: `rfdetr fit --config` CLI with zero Python required. ([#757](https://github.com/roboflow/rf-detr/pull/757), [#794](https://github.com/roboflow/rf-detr/pull/794))
- Multi-GPU DDP via `model.train()`: `strategy`, `devices`, and `num_nodes` added to `TrainConfig`; single-GPU behaviour unchanged when omitted. ([#808](https://github.com/roboflow/rf-detr/pull/808))
- `batch_size='auto'`: CUDA memory probe finds the largest safe micro-batch size, then recommends `grad_accum_steps` to reach a configurable effective batch target (default 16 via `auto_batch_target_effective`). ([#814](https://github.com/roboflow/rf-detr/pull/814))
- `ModelContext` promoted from `_ModelContext` to a public, exported API — inspect `class_names`, `num_classes`, and related metadata via `model.context` after training. ([#835](https://github.com/roboflow/rf-detr/pull/835))
- `backbone_lora` and `freeze_encoder` added as first-class fields in `ModelConfig`. ([#829](https://github.com/roboflow/rf-detr/pull/829))
- `generate_coco_dataset(with_segmentation=True)` produces COCO polygon annotations alongside bounding boxes for segmentation fine-tuning with synthetic data. ([#781](https://github.com/roboflow/rf-detr/pull/781))
- `set_attn_implementation("eager" | "sdpa")` on the DINOv2 backbone — switch attention implementation at runtime. ([#760](https://github.com/roboflow/rf-detr/pull/760))
- `eval_max_dets`, `eval_interval`, and `log_per_class_metrics` added to `TrainConfig`.
- `python -m rfdetr` entry point alongside the `rfdetr` console script.
- `py.typed` marker — RF-DETR is now PEP 561–compliant.

### Changed

- **Breaking:** Minimum `transformers` version bumped to `>=5.1.0,<6.0.0`. The DINOv2 windowed-attention backbone now uses the transformers v5 API (`BackboneMixin._init_transformers_backbone()`, removed `head_mask` plumbing). Projects still on transformers v4 must pin `rfdetr<1.6.0`. ([#760](https://github.com/roboflow/rf-detr/pull/760))
- **Breaking:** PyPI install extras renamed — `rfdetr[metrics]` → `rfdetr[loggers]`, `rfdetr[onnxexport]` → `rfdetr[onnx]`.
- `draw_synthetic_shape` now returns `Tuple[np.ndarray, List[float]]` instead of `np.ndarray`. The second element is a flat COCO-style polygon list `[x1, y1, x2, y2, …]`. Any caller that previously did `img = draw_synthetic_shape(...)` must be updated to `img, polygon = draw_synthetic_shape(...)`. ([#781](https://github.com/roboflow/rf-detr/pull/781))
- Albumentations version constraint broadened to `>=1.4.24,<3.0.0`; `RandomSizedCrop` configs using `height`/`width` kwargs are automatically adapted to the 2.x `size=(height, width)` API. ([#786](https://github.com/roboflow/rf-detr/pull/786))
- Current learning rate is now shown in the training progress bar alongside loss. ([#809](https://github.com/roboflow/rf-detr/pull/809))
- `supervision`, `pytorch_lightning`, and other heavy dependencies are now imported lazily (on first use) rather than at module load, reducing cold-import time in inference-only environments. ([#801](https://github.com/roboflow/rf-detr/pull/801))

### Deprecated

- `rfdetr.deploy.*` — redirects to `rfdetr.export.*` with a `DeprecationWarning`. Migrate before v1.7.
- `rfdetr.util.*` — redirects to `rfdetr.utilities.*` with a `DeprecationWarning`. Migrate before v1.7.

### Fixed

- Raised a descriptive `ValueError` instead of a cryptic `RuntimeError` / tensor-size mismatch when a checkpoint is incompatible with the current model architecture — covers `segmentation_head` mismatch and `patch_size` mismatch. ([#810](https://github.com/roboflow/rf-detr/pull/810))
- Fixed `class_names` not reflecting dataset labels on `model.predict()` after training — class names are now synced from the dataset so inference always uses the correct label list. ([#816](https://github.com/roboflow/rf-detr/pull/816))
- Fixed detection head reinitialization overwriting fine-tuned weights when loading a checkpoint with fewer classes than the model default. The second `reinitialize_detection_head` call now fires only in the backbone-pretrain scenario. ([#815](https://github.com/roboflow/rf-detr/pull/815), [#509](https://github.com/roboflow/rf-detr/pull/509))
- Fixed `grid_sample` and bicubic interpolation silently falling back to CPU on MPS (Apple Silicon) — both now run natively on the MPS device. ([#821](https://github.com/roboflow/rf-detr/pull/821))
- Fixed `early_stopping=False` in `TrainConfig` being silently ignored — the setting now propagates correctly. ([#835](https://github.com/roboflow/rf-detr/pull/835))
- Fixed `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern.
- Added warning when `drop_path_rate > 0.0` is configured with a non-windowed DINOv2 backbone, where drop-path is silently ignored.
- Fixed `ValueError: matrix entries are not finite` in `HungarianMatcher` when the cost matrix contains NaN or Inf — non-finite entries are replaced with a finite sentinel before `linear_sum_assignment`, warning emitted at most once per matcher instance. ([#787](https://github.com/roboflow/rf-detr/pull/787))
- Fixed YOLO dataset validation rejecting `data.yml` — both `.yaml` and `.yml` are now accepted. ([#777](https://github.com/roboflow/rf-detr/pull/777))
- Silently dropped degenerate bounding boxes (zero width or height) before Albumentations validation instead of raising `ValueError`. ([#825](https://github.com/roboflow/rf-detr/pull/825))

---

## [1.5.2] — 2026-03-04

### Added

- Added peak GPU memory (`max_mem` in MB) to training and evaluation progress bars on CUDA; omitted on CPU and MPS. ([#773](https://github.com/roboflow/rf-detr/pull/773))

### Fixed

- Fixed `aug_config` being silently ignored when training on YOLO-format datasets — `build_roboflow_from_yolo` never forwarded the value, so transforms always fell back to the default. ([#774](https://github.com/roboflow/rf-detr/pull/774))
- Fixed segmentation evaluation metrics not being written to `results_mask.json` during validation and test runs. ([#772](https://github.com/roboflow/rf-detr/pull/772))
- Fixed `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern — `_get_backbone_encoder_layers` now returns `None` for unrecognised architectures. ([#762](https://github.com/roboflow/rf-detr/pull/762))
- Fixed `drop_path_rate` not being forwarded to the DINOv2 model configuration; stochastic depth was never applied even when explicitly set. Added a warning when `drop_path_rate > 0.0` is used with a non-windowed backbone. ([#762](https://github.com/roboflow/rf-detr/pull/762))
- Fixed incorrect COCO hierarchy filtering that excluded parent categories from the class list. ([#759](https://github.com/roboflow/rf-detr/pull/759))
- Fixed evaluation metric corruption on 1-indexed Roboflow datasets caused by a flawed contiguity check in `_should_use_raw_category_ids`. ([#755](https://github.com/roboflow/rf-detr/pull/755))

## [1.5.1] — 2026-02-27

### Added

- Added support for nested Albumentations containers (`OneOf`, `Sequential`) inside `aug_config`. ([#752](https://github.com/roboflow/rf-detr/pull/752))

### Changed

- Migrated dataset transform pipeline to torchvision-native `Compose`, `ToImage`, and `ToDtype`; `Normalize` now defaults to ImageNet mean/std. ([#745](https://github.com/roboflow/rf-detr/pull/745))

### Fixed

- Fixed `RFDETRMedium` missing from the public API — `__all__` contained a duplicate `RFDETRSmall` entry. ([#748](https://github.com/roboflow/rf-detr/pull/748))
- Fixed `AR50_90` reporting an incorrect value in `MetricsMLFlowSink` due to a wrong COCO evaluation index. ([#735](https://github.com/roboflow/rf-detr/pull/735))
- Fixed supercategory filtering in `_load_classes` for COCO datasets with flat or mixed supercategory structures. ([#744](https://github.com/roboflow/rf-detr/pull/744))
- Fixed crash in geometric transforms when a sample contained zero-area or empty masks. ([#727](https://github.com/roboflow/rf-detr/pull/727))
- Fixed segmentation training on Colab — `DepthwiseConvBlock` now disables cuDNN for depthwise separable convolutions. ([#728](https://github.com/roboflow/rf-detr/pull/728))
- Pinned `onnxsim<0.6.0` to prevent `pip install` from hanging indefinitely. ([#749](https://github.com/roboflow/rf-detr/pull/749))

## [1.5.0] — 2026-02-23

### Added

- Added custom training augmentations via `aug_config` in `model.train()` — accepts a dict of Albumentations transforms, a built-in preset (`AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, `AUG_INDUSTRIAL`), or `{}` to disable. Bounding boxes and segmentation masks are transformed automatically. ([#263](https://github.com/roboflow/rf-detr/pull/263), [#702](https://github.com/roboflow/rf-detr/pull/702))
- Added `save_dataset_grids=True` in `TrainConfig` to write 3×3 JPEG grids of augmented samples to `output_dir` before training begins. ([#153](https://github.com/roboflow/rf-detr/pull/153))
- Added ClearML logger: set `clearml=True` in `TrainConfig` to stream per-epoch metrics to ClearML. ([#520](https://github.com/roboflow/rf-detr/pull/520))
- Added MLflow logger: set `mlflow=True` in `TrainConfig` to log runs and metrics to MLflow with custom tracking URI support. ([#109](https://github.com/roboflow/rf-detr/pull/109))
- Added live progress bar for training and validation with structured per-epoch logs. ([#204](https://github.com/roboflow/rf-detr/pull/204))
- Added `device` field to `TrainConfig` for explicit device selection. ([#687](https://github.com/roboflow/rf-detr/pull/687))
- `ModelConfig` now raises an error on unknown parameters, preventing silent misconfiguration. ([#196](https://github.com/roboflow/rf-detr/pull/196))

### Changed

- Deprecated `OPEN_SOURCE_MODELS` constant in favour of `ModelWeights` enum. ([#696](https://github.com/roboflow/rf-detr/pull/696))
- Added MD5 checksum validation for pretrained weight downloads. ([#679](https://github.com/roboflow/rf-detr/pull/679))

### Fixed

- Fixed Albumentations bool-mask crash during segmentation training. ([#706](https://github.com/roboflow/rf-detr/pull/706))
- Fixed `UnboundLocalError` when resuming training from a completed checkpoint. ([#707](https://github.com/roboflow/rf-detr/pull/707))
- Prevented corruption of `checkpoint_best_total.pth` via atomic checkpoint stripping. ([#708](https://github.com/roboflow/rf-detr/pull/708))
- Fixed PyTorch 2.9+ compatibility issue with CUDA capability detection. ([#686](https://github.com/roboflow/rf-detr/pull/686))
- Fixed dtype mismatch error when `use_position_supervised_loss=True`. ([#447](https://github.com/roboflow/rf-detr/pull/447))
- Fixed inconsistent return values from `build_model`. ([#519](https://github.com/roboflow/rf-detr/pull/519))
- Fixed `positional_encoding_size` type annotation (`bool` → `int`). ([#524](https://github.com/roboflow/rf-detr/pull/524))
- Fixed ONNX export `output_names` to include masks when exporting segmentation models. ([#402](https://github.com/roboflow/rf-detr/pull/402))
- Fixed `num_select` not being updated correctly during segmentation model fine-tuning. ([#399](https://github.com/roboflow/rf-detr/pull/399))
- Fixed `np.argwhere` → `np.argmax` misuse. ([#536](https://github.com/roboflow/rf-detr/pull/536))
- Fixed COCO sparse category ID remapping for non-contiguous or offset category IDs. ([#712](https://github.com/roboflow/rf-detr/pull/712))
- Fixed segmentation mask filtering when using aggressive augmentations. ([#717](https://github.com/roboflow/rf-detr/pull/717))

---

## [1.4.3] — 2026-02-16

### Changed

- Pretrained weight downloads now validate against an MD5 checksum to detect corrupted files. ([#679](https://github.com/roboflow/rf-detr/pull/679))

### Fixed

- Fixed `deploy_to_roboflow` failing for segmentation model exports. ([#578](https://github.com/roboflow/rf-detr/pull/578))
- Fixed missing `info` key in COCO export format. ([#681](https://github.com/roboflow/rf-detr/pull/681))

## [1.4.2] — 2026-02-12

### Added

- Added `generate_coco_dataset()` utility for generating synthetic COCO-format datasets with configurable class counts, split ratios, and bounding box annotations. ([#617](https://github.com/roboflow/rf-detr/pull/617))
- Added `run_test=False` to `TrainConfig` — skip test-split evaluation when your dataset has no test set. ([#628](https://github.com/roboflow/rf-detr/pull/628))

### Changed

- `model.predict()` now accepts image URLs directly — no need to download images before inference. ([#629](https://github.com/roboflow/rf-detr/pull/629))
- Plus models (`RFDETRXLarge`, `RFDETR2XLarge`) are now distributed as a separate `rfdetr_plus` package under the Roboflow Model License. ([#645](https://github.com/roboflow/rf-detr/pull/645))

### Fixed

- Fixed segmentation ONNX export failure. ([#626](https://github.com/roboflow/rf-detr/pull/626))

## [1.4.1] — 2026-01-30

### Added

- Added native YOLO dataset format support alongside COCO. ([#74](https://github.com/roboflow/rf-detr/pull/74))
- Added `--print-freq` CLI argument to control training log frequency. ([#603](https://github.com/roboflow/rf-detr/pull/603))

### Changed

- Pinned `transformers` to `<5.0.0` to prevent incompatibility with the transformers v5 API. ([#599](https://github.com/roboflow/rf-detr/pull/599))

### Fixed

- Fixed class count mismatch in `train_from_config` for Roboflow-uploaded datasets. ([#588](https://github.com/roboflow/rf-detr/pull/588))
- Improved `num_classes` mismatch warning messages to be actionable rather than misleading. ([#261](https://github.com/roboflow/rf-detr/pull/261))
- Fixed CLI crash when specifying the `device` argument. ([#246](https://github.com/roboflow/rf-detr/pull/246))

## [1.4.0] — 2026-01-22

Headline release introducing new pre-trained model sizes — L, XL, and 2XL for object detection, and the full N/S/M/L/XL/2XL range for instance segmentation. Also added YOLO format training support, simplified the dependency footprint by removing several heavy packages (`cython`, `fairscale`, `timm`, `einops`, and others), and fixed per-class precision/recall/F1 computation. Drops Python 3.9 support.
