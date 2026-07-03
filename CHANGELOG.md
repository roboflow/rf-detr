# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.3] — 2026-06-27

### Added

- `trust_checkpoint: bool = False` keyword-only parameter on `RFDETR.from_checkpoint()` to opt into full pickle deserialization for trusted checkpoint files.
- `optimize_for_inference(inplace=True)` — new keyword-only argument on `RFDETR.optimize_for_inference()`; skips the deep-copy of the base model for memory-constrained inference-only deployments (~0.5× model-weight peak memory reduction). Requires `compile=False`. After inplace optimization, `export()` raises `RuntimeError` and `remove_optimized_model()` issues a `UserWarning` and returns cleanly instead of silently clearing state. New `RFDETR.is_optimized_inplace` property returns `True` after a successful inplace optimization. ([#1089](https://github.com/roboflow/rf-detr/pull/1089))
- `CocoKeypointSchema.keypoint_flip_pairs` and `YoloKeypointSchema.keypoint_flip_pairs` fields — horizontal-flip swap pairs inferred automatically from keypoint names (left/right naming convention) for COCO schemas, and from `flip_idx` permutation for YOLO schemas. Auto-populated by `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` respectively. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` re-exported from `rfdetr.datasets` (previously only accessible from `rfdetr.datasets._keypoint_schema`). ([#1164](https://github.com/roboflow/rf-detr/pull/1164))

### Changed

- `RFDETR.from_checkpoint()` now uses safe deserialization by default (`weights_only=True`). Checkpoints containing custom Python objects beyond `argparse.Namespace` or `types.SimpleNamespace` must pass `trust_checkpoint=True`. Previously, full pickle deserialization was always used.
- Horizontal flip detection in `AlbumentationsWrapper` now uses Albumentations `ReplayCompose` replay metadata instead of heuristic bbox-center mirroring; eliminates false positives on non-flip transforms that shift box centers. Falls back to `alb.Compose` with a `UserWarning` when `albumentations <1.3` is detected. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- Keypoint schema inference now supports native COCO format (`dataset_file="coco"`) in addition to `"roboflow"` and `"yolo"`. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- `_keypoint_schema_cache` key changed from `dataset_dir` (string) to `(dataset_file, dataset_dir)` tuple to prevent cross-format cache collisions when the same directory is used with different dataset formats. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))

### Fixed

- Predicted bounding boxes are now clamped to image bounds `[0, width] × [0, height]` in `PostProcess._postprocess_boxes()`; model regression is unbounded and could previously produce negative or out-of-frame coordinates. `scale_fct` is also cast to `boxes.dtype` before multiplication to prevent dtype mismatch when boxes are `float16`. ([#1168](https://github.com/roboflow/rf-detr/pull/1168))
- `SegmentationTrainConfig.cls_loss_coef` default corrected from `5.0` to `1.0` to restore the pre-v1.7 effective classification loss weight. The `5.0` value was present in `SegmentationTrainConfig` since v1.6 but was dead code until the v1.7 TrainConfig ownership migration activated it, silently over-penalising classification relative to mask losses during segmentation fine-tuning. To reproduce pre-fix behaviour, pass `cls_loss_coef=5.0` explicitly. ([#1165](https://github.com/roboflow/rf-detr/pull/1165))
- `KeypointTrainConfig.keypoint_nll_loss_coef` default restored to `1.0` to align with the other keypoint loss terms (`keypoint_l1_loss_coef`, `keypoint_findable_loss_coef`, `keypoint_visible_loss_coef`). The previous default of `0.5` was set to dampen OKS@75 oscillation but under-weighted the NLL loss relative to other terms in practice. ([#1165](https://github.com/roboflow/rf-detr/pull/1165))

---

## [1.8.2] — 2026-06-25

### Added

- YOLO pose keypoint dataset support: load Ultralytics YOLO pose datasets (`.yaml` with `kpt_shape`) directly for keypoint fine-tuning. Schema is inferred automatically via `infer_yolo_keypoint_schema`. ([#1156](https://github.com/roboflow/rf-detr/pull/1156))
- `is_bg_first_schema`, `to_active_first`, `to_bg_first`, `schemas_semantically_equal` utilities in `rfdetr.utilities.keypoints` (and re-exported from `rfdetr.utilities`) for schema-aware keypoint processing. ([#1160](https://github.com/roboflow/rf-detr/pull/1160))
- `amp_dtype` field on `TrainConfig` (`"auto"` / `"bf16"` / `"fp16"`): pin the mixed-precision autocast dtype instead of relying on device-capability auto-detection. `"auto"` (default) preserves the historical behaviour — `bf16-mixed` on Ampere+ CUDA, `16-mixed` otherwise. Invalid values degrade gracefully to `"auto"` with a `UserWarning`. ([#1143](https://github.com/roboflow/rf-detr/pull/1143))
- Instance segmentation fine-tuning cookbook (`docs/cookbooks/fine-tune_segmentation.ipynb`) — end-to-end walkthrough using `RFDETRSegSmall` across seven diverse segmentation datasets. ([#1159](https://github.com/roboflow/rf-detr/pull/1159))
- Inference latency benchmark cookbook (`docs/cookbooks/inference-latency-benchmark.ipynb`) — benchmarks CPU/GPU throughput across model sizes with reproducible measurement methodology. ([#1152](https://github.com/roboflow/rf-detr/pull/1152))

### Changed

- Default `num_keypoints_per_class` in `RFDETRKeypointPreviewConfig` changed from `[0, 17]` (background-first) to `[17]` (active-first). Legacy bg-first checkpoints auto-align on load via `_kp_active_mask`. ([#1160](https://github.com/roboflow/rf-detr/pull/1160))

### Fixed

- `RFDETR.from_checkpoint()` now correctly infers `num_classes` and `num_keypoints_per_class` from checkpoint weights (`class_embed.weight.shape[0] - 1` and `_kp_active_mask` respectively). Previously, `num_classes` was read as `shape[0]` (i.e. `num_classes + 1` including the background class), causing `load_state_dict` shape mismatches or a silent extra output class on every load. `BestModelCallback._serialize_model_config` is also fixed to persist the correct foreground-only `num_classes`. ([#1158](https://github.com/roboflow/rf-detr/pull/1158))
- `HungarianMatcher.forward()` now uses the configured `focal_alpha` in the focal classification matching cost. Previously the value was hardcoded to `0.25`, silently ignoring any non-default `focal_alpha` passed to the constructor or `build_matcher`. This misaligned the bipartite matching cost with the focal classification loss in `criterion.py`, which correctly used `self.focal_alpha`. ([#1147](https://github.com/roboflow/rf-detr/pull/1147))
- `spatial_shapes` in `Transformer.forward()` is now built from symbolic `Shape` ops (`torch.stack` of per-level `torch._shape_as_tensor` slices) instead of `torch.empty` + in-place index assignment. The previous pattern emitted a `ScatterND` feeding a shape tensor (`level_start_index`), which TensorRT rejected with "IScatterLayer cannot be used to compute a shape tensor". This fix is required to export any RF-DETR model to a TensorRT engine. ([#1155](https://github.com/roboflow/rf-detr/pull/1155))
- Keypoint model inference now returns the correct `class_name` field in predictions. ([#1151](https://github.com/roboflow/rf-detr/pull/1151))
- `predict()` now re-asserts eval mode before each call for unoptimized models, preventing silent train-mode inference after the first prediction. ([#1146](https://github.com/roboflow/rf-detr/pull/1146))
- TFLite inference preprocessing and mask decoder aligned with PyTorch `predict()` behaviour. ([#1131](https://github.com/roboflow/rf-detr/pull/1131))
- Python version mismatch in optional-dependency version overrides resolved. ([#1137](https://github.com/roboflow/rf-detr/pull/1137))

---

## [1.8.1] — 2026-06-19

### Changed

- Config path parameters (e.g. `dataset_dir`, `output_dir`, `pretrain_weights`) now accept `pathlib.Path` objects in addition to strings. Paths are coerced to `str` automatically via the `expand_paths` validator. No API changes required; existing string usage unaffected. ([#1124](https://github.com/roboflow/rf-detr/pull/1124))
- Keypoint training now disables horizontal flip augmentation until keypoint flip-pair swapping is implemented. Previously, flipping was applied without reordering keypoint pairs, producing incorrect labels. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- Training metric plots improved with optional seaborn error bands, AP@0.75 metric grouping, and custom AP metric group configuration. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))

### Fixed

- Keypoint encoder in eval mode now routes all queries through head 0 instead of splitting across group heads. Previously, `group_detr = len(self.enc_out_keypoint_embed)` caused the encoder keypoint path to split `num_queries` queries across all group heads in eval; now uses `if self.training else 1` guard. ([#1135](https://github.com/roboflow/rf-detr/pull/1135))
- `config.use_return_dict` (deprecated in `transformers`) replaced with `config.return_dict` in DINOv2 windowed attention backbone. ([#1135](https://github.com/roboflow/rf-detr/pull/1135))
- Epoch metric tables now display correctly when a Rich progress bar callback is active. Tables are printed through the progress bar's owned Rich console, preventing cursor conflicts with active live displays. ([#1128](https://github.com/roboflow/rf-detr/pull/1128))
- Keypoint fine-tuning checkpoint selection stabilised with smoothed (EMA) best-metric comparison to avoid spurious checkpoint switches on noisy OKS metrics. Smoothing state is correctly restored on training resume. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- Group DETR train-time metric evaluation now evaluates only the primary query group, preventing crashes on non-tensor mask outputs from auxiliary decoder layers. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- `_detect_horizontal_flip` in Albumentations transform pipeline now uses `len(bboxes) == 0` instead of `not bboxes` to correctly handle Albumentations 2.x where bboxes is a NumPy array (falsy even when non-empty). ([#1126](https://github.com/roboflow/rf-detr/pull/1126))
- TensorBoard logger is now disabled gracefully when `tensorboard` is installed alongside a NumPy-2.0-incompatible `tensorflow`. Training degrades to CSV-only logging with a clear warning instead of crashing inside `_log_hyperparams`. ([#1123](https://github.com/roboflow/rf-detr/pull/1123))

---

## [1.8.0] — 2026-06-13

### Added

- `RFDETRKeypointPreview` — keypoint detection model variant with GroupPose-style head, covariance-based uncertainty (precision-Cholesky parameterization), and COCO keypoint AP evaluation. Public config classes: `KeypointTrainConfig`, `RFDETRKeypointPreviewConfig` (from `rfdetr.config`). Utility: `precision_cholesky_to_pixel_covariance` (from `rfdetr.utilities`). Schema helpers `infer_coco_keypoint_schema`, `CocoKeypointSchema`, `active_keypoint_counts` accessible via `rfdetr.datasets._keypoint_schema`. ([#1099](https://github.com/roboflow/rf-detr/pull/1099))
- `RFDETR.export_for_roboflow(output_dir)` — writes a Roboflow upload bundle (`weights.pt` + `class_names.txt`) without a network call; extracted from `deploy_to_roboflow`, which now delegates to it. ([#1086](https://github.com/roboflow/rf-detr/pull/1086))
- Keypoint fine-tuning cookbook (`docs/cookbooks/fine-tune_keypoints.ipynb`) — end-to-end walkthrough: dataset download, schema inference, `KeypointTrainConfig`, training metrics, and inference with covariance uncertainty. ([#1104](https://github.com/roboflow/rf-detr/pull/1104))
- `MetricKeypointOKS` — reusable OKS metric facade over `CocoEvaluator`, exported from `rfdetr.evaluation`. Supports arbitrary keypoint counts, per-category OKS sigma values, DDP-safe evaluation with first-rank-wins deduplication, and an `OKSKey` enum (`mAP`, `mAP@50`, `mAP@75`, `mAR`) for standardised metric keys. ([#1107](https://github.com/roboflow/rf-detr/pull/1107))

### Changed

- DDP strategy now enables `find_unused_parameters=True` for all detection, keypoint, and segmentation models when running under `strategy='ddp'` or `strategy='auto'` with a distributed launcher. Previously only enabled for segmentation. Opt out via `trainer_kwargs={"strategy": DDPStrategy(find_unused_parameters=False)}`. ([#1094](https://github.com/roboflow/rf-detr/pull/1094))
- `rfdetr.datasets.aug_config` module renamed to `rfdetr.datasets.aug_configs` (plural). Direct imports from `rfdetr.datasets.aug_config` must be updated to `rfdetr.datasets.aug_configs`; the augmentation preset constants (`AUG_AGGRESSIVE`, etc.) are unchanged. ([#1103](https://github.com/roboflow/rf-detr/pull/1103))

### Removed

- `RFDETR.export(simplify=..., force=...)` — both kwargs removed from the signature. Deprecated since v1.6.0 with `remove_in="1.8.0"`; both were no-ops during the deprecation window. Callers passing these args must remove them before upgrading. ([#1102](https://github.com/roboflow/rf-detr/pull/1102))

### Fixed

- `RFDETR.from_checkpoint()` no longer treats `num_classes` loaded from the checkpoint as a user-supplied override. Previously, fine-tuning a checkpoint model on a dataset with a different class count was silently refused — the head refused to re-initialise and trained against the stale class count. An explicit `num_classes` kwarg from the caller still wins over both the checkpoint value and the dataset. ([#1106](https://github.com/roboflow/rf-detr/pull/1106))
- Scale jitter restored in non-square training crop. `RandomCrop` in the `option_b` branch replaced with `RandomSizedCrop`, restoring the scale-augmentation behaviour lost during the Albumentations migration. ([#1088](https://github.com/roboflow/rf-detr/pull/1088))
- Multi-GPU validation deadlock in COCO mAP synchronization prevented. `_merge_metric_state_across_ranks` now safe across zero-batch ranks. ([#1085](https://github.com/roboflow/rf-detr/pull/1085))
- `import rfdetr` no longer fails on NumPy 2.x when a transitive dependency references the removed `np.complex_` alias. ([#1064](https://github.com/roboflow/rf-detr/pull/1064))
- `rfdetr_plus` module availability check corrected; false-positive hit when the package was partially installed. ([#1083](https://github.com/roboflow/rf-detr/pull/1083))
- Fixed spurious "Keypoint class-logit boost has N classes but detection head has M" warning on custom (non-Roboflow) keypoint datasets: `_align_num_classes_from_dataset` now zero-pads `num_keypoints_per_class` when auto-adjusting `num_classes` beyond the schema length ([#1113](https://github.com/roboflow/rf-detr/pull/1113))
- Loss scaling corrected for keypoint training under gradient accumulation (`accumulate_grad_batches > 1`). Keypoint models now use manual optimization to normalize losses by the accumulated box count across the effective batch; detection and segmentation remain on Lightning's automatic-optimization path. Optimizer-step scheduling, LR warmup/decay, and epoch-boundary flushing are correctly handled in both paths. ([#1117](https://github.com/roboflow/rf-detr/pull/1117))
- Device auto-detection now verifies accelerator runtime availability before selecting a device (PyTorch ≥ 2.4: `torch.accelerator.current_accelerator`; older builds: `torch.cuda.is_available()`). Previously, a machine with CUDA headers but no GPU driver could be assigned a CUDA device that fails at first use. ([#1111](https://github.com/roboflow/rf-detr/pull/1111))
- `RFDETR.from_checkpoint()` and related APIs now honor an explicit `num_classes` argument even when its value equals the model default. Previously, passing `num_classes=N` where `N` is the default (e.g. 80 for COCO) was silently treated as unset, causing fine-tuning on a different class count to be refused. ([#1109](https://github.com/roboflow/rf-detr/pull/1109))
- `RFDETR.from_checkpoint()` correctly infers the model variant from the checkpoint filename when `pretrain_weights` is absent or unset-like (empty string, `None`, whitespace). Previously, starter-like checkpoints without an explicit `pretrain_weights` entry raised an error or silently loaded the wrong model class. ([#1065](https://github.com/roboflow/rf-detr/pull/1065))

---

## [1.7.0] — 2026-04-29

### Added

- `augmentation_backend` field on `TrainConfig` (`"cpu"` / `"auto"` / `"gpu"`): opt-in GPU-side augmentation via [Kornia](https://kornia.readthedocs.io) applied in `RFDETRDataModule.on_after_batch_transfer` after the batch is resident on the GPU. CPU path is unchanged and remains the default. Install with `pip install 'rfdetr[kornia]'`. Supports detection and segmentation (see below). ([#1003](https://github.com/roboflow/rf-detr/pull/1003))
- Kornia GPU augmentation now supports instance segmentation: images, boxes, and per-instance masks are augmented in sync on the GPU. New public helper `collate_masks` packs `[N_i, H, W]` boolean masks into a `[B, N_max, H, W]` float32 tensor for Kornia; `build_kornia_pipeline` gains a `with_masks: bool = False` parameter; `unpack_boxes` gains an optional `masks_aug` tensor that re-binarises and filters masks in sync with boxes. Previously `augmentation_backend="gpu"/"auto"` was silently ignored for segmentation models; now it works identically to detection. **Note**: the mask buffer is `[B, N_max, H, W]` float32 — approximately 500 MB at `B=8, N_max=50, H=W=560`; use `augmentation_backend="cpu"` on cards with limited VRAM. ([#1003](https://github.com/roboflow/rf-detr/pull/1003), closes [#997](https://github.com/roboflow/rf-detr/issues/997))
- `BuilderArgs` — a `@runtime_checkable` `typing.Protocol` documenting the minimum attribute set consumed by `build_model()`, `build_backbone()`, `build_transformer()`, and `build_criterion_and_postprocessors()`. Enables static type-checker support for custom builder integrations. Exported from `rfdetr.models`. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- `build_model_from_config(model_config, train_config=None, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_model(build_namespace(mc, tc))`; accepts Pydantic config objects directly and constructs the internal namespace automatically. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `build_criterion_from_config(model_config, train_config, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_criterion_and_postprocessors(build_namespace(mc, tc))`; returns a `(SetCriterion, PostProcess)` tuple. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `ModelDefaults` dataclass — exposes the 35 hardcoded architectural constants previously buried inside `build_namespace()`. Pass a `dataclasses.replace(MODEL_DEFAULTS, ...)` override to the new config-native builders to customise individual constants. **Note:** fields may be promoted to `ModelConfig`/`TrainConfig` in future phases. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `MODEL_DEFAULTS` — the canonical `ModelDefaults` singleton with production defaults. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `RFDETR.predict(include_source_image=...)` — opt-out flag (default `True`) to skip storing the source image in `detections.metadata["source_image"]`; set to `False` to reduce memory use when the image is not needed for annotation. ([#912](https://github.com/roboflow/rf-detr/pull/912))
- `model_name` is now stored in checkpoint files during training so that `RFDETR.from_checkpoint()` can resolve the correct model class directly from the checkpoint, without requiring the caller to know or pass a class hint. `strip_checkpoint()` preserves this key. Backward-compatible: checkpoints without `model_name` continue to resolve via `pretrain_weights` filename matching. ([#895](https://github.com/roboflow/rf-detr/pull/895))
- `rfdetr_version` is now stored in checkpoint files during training for provenance tracking and compatibility hints. `strip_checkpoint()` preserves this key. The key is omitted gracefully when the package version cannot be resolved (e.g. editable install without metadata). Backward-compatible: checkpoints without `rfdetr_version` continue to load normally. ([#918](https://github.com/roboflow/rf-detr/pull/918))
- `notes` parameter on `RFDETR.train()` and `RFDETR.export()` — embed arbitrary JSON-serialisable provenance metadata (labeller, date, class names, etc.) into best-model `.pth` checkpoints (under `checkpoint["args"]["notes"]`) and ONNX files (under the `"rfdetr_notes"` metadata property). String values are stored verbatim; all other types are JSON-encoded. ([#1025](https://github.com/roboflow/rf-detr/pull/1025), closes [#1021](https://github.com/roboflow/rf-detr/issues/1021))
- `RF_HOME` environment variable controls where pretrained model weights are cached (default: `~/.roboflow/models`). Bare filenames passed as `pretrain_weights` (e.g. `"rf-detr-base.pth"`) are now resolved relative to this directory; paths with a directory component are used as-is with parent directories created automatically. ([#130](https://github.com/roboflow/rf-detr/pull/130))
- Grayscale and multispectral imagery support: RF-DETR models now accept inputs with any number of channels (not just 3). The pretrained DINOv2 patch-embedding weights are automatically adapted to the specified channel count at model construction time — no additional dependencies required. ([#180](https://github.com/roboflow/rf-detr/pull/180), closes [#75](https://github.com/roboflow/rf-detr/issues/75))
- Training configuration is now saved to `training_config.json` in the output directory after training completes. The file captures the full `TrainConfig`, `ModelConfig`, effective training parameters, class names, and number of classes — useful for reproducibility and debugging predictions from older checkpoints. ([#194](https://github.com/roboflow/rf-detr/pull/194))
- `dinov2_registers_windowed_small` backbone is now available as a config option in `ModelConfig.encoder`. ([#236](https://github.com/roboflow/rf-detr/pull/236))
- `rfdetr.from_checkpoint(path)` — new top-level convenience function that loads a checkpoint and infers the correct model subclass automatically, without requiring the caller to specify a class. Equivalent to `RFDETR.from_checkpoint(path)` but importable directly from the `rfdetr` package. ([#664](https://github.com/roboflow/rf-detr/pull/664))
- ONNX export filenames now include the model variant name (e.g. `rfdetr-medium.onnx`) instead of the generic `inference_model.onnx`. Exporting multiple variants to the same directory no longer overwrites previous exports. ([#910](https://github.com/roboflow/rf-detr/pull/910))
- Background images (images without a matching label file) are now included in YOLO detection datasets as empty-detection samples instead of being silently dropped. Both detection and segmentation paths now use `_LazyYoloDetectionDataset` for consistent behaviour. ([#915](https://github.com/roboflow/rf-detr/pull/915))
- TFLite export via `model.export(format="tflite")`. Converts through ONNX using `onnx2tf`; FP32 and FP16 outputs are always produced, INT8 quantization is available with a calibration image directory: `model.export(format="tflite", quantization="int8", calibration_data="path/to/images/")`. Requires `pip install 'rfdetr[onnx,tflite]'`. ([#920](https://github.com/roboflow/rf-detr/pull/920))
- PyTorch Lightning `.ckpt` files are now accepted as `pretrain_weights`. Keys are automatically normalized from PTL format (`state_dict` with `model.`-prefixed keys, `hyper_parameters` → `args`) so that `load_pretrain_weights`, class-name extraction, and compatibility checks work without manual conversion. ([#951](https://github.com/roboflow/rf-detr/pull/951))
- `skip_best_epochs` parameter for `RFDETR.train()` and `TrainConfig`: the first N epochs are excluded from best-checkpoint selection and early-stopping comparison, preventing strong pretrained weights or resumed checkpoints from locking in a suboptimal early score. ([#1000](https://github.com/roboflow/rf-detr/pull/1000), closes [#789](https://github.com/roboflow/rf-detr/issues/789))
- TFLite inference now decodes segmentation mask outputs into `sv.Detections.mask`. Mask logits are upsampled to the source image size using Pillow bilinear resampling and thresholded at zero, matching the behaviour of `PostProcess.forward`. The mask tensor is detected by output name (`"masks"` substring) with a rank-4 shape fallback. ([#1053](https://github.com/roboflow/rf-detr/pull/1053))
- `PretrainWeightsCompatibilityWarning` — new warning class emitted when a `ModelConfig` override (e.g. custom `encoder`, `num_queries`, or `num_feature_levels`) risks breaking pretrained weight loading. Importable as `from rfdetr.config import PretrainWeightsCompatibilityWarning` for targeted filtering. ([#1017](https://github.com/roboflow/rf-detr/pull/1017))

### Changed

- `peft` is no longer installed as part of the default `rfdetr` package. It has moved to the `[lora]` and `[train]` optional extras. If you use LoRA fine-tuning, install with `pip install 'rfdetr[lora]'`. ([#838](https://github.com/roboflow/rf-detr/pull/838))
- Native RLE annotation support in the COCO segmentation pipeline: `convert_coco_poly_to_mask` now explicitly detects and decodes both compressed (string counts) and uncompressed (int-list counts) RLE formats alongside existing polygon support. Malformed annotations now raise instead of being silently swallowed. ([#897](https://github.com/roboflow/rf-detr/pull/897))
- Pinned PyTorch Lightning to exclude known-compromised versions. ([#1020](https://github.com/roboflow/rf-detr/pull/1020))

### Deprecated

- `build_namespace(model_config, train_config)` — no longer used internally and deprecated in this release; use `build_model_from_config`, `build_criterion_from_config`, or `_namespace_from_configs` directly. It will be removed in v1.9 and currently emits a `DeprecationWarning` on use. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `load_pretrain_weights(nn_model, model_config, train_config)` — the `train_config` positional argument is deprecated and will be removed in v1.9; it is no longer used internally. Omit it: `load_pretrain_weights(nn_model, model_config)`. Passing a non-`None` value emits a `DeprecationWarning`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `TrainConfig.group_detr` (architecture decision → `ModelConfig`), `TrainConfig.ia_bce_loss` (loss type tied to architecture family → `ModelConfig`), `TrainConfig.segmentation_head` (architecture flag → `ModelConfig`), `TrainConfig.num_select` (postprocessor count → `ModelConfig`; `SegmentationTrainConfig` users: remove the `num_select` override — the model config value is always used), `ModelConfig.cls_loss_coef` (training hyperparameter → `TrainConfig`) — each now emits `DeprecationWarning` when set on the wrong config object and will be **removed** in v1.9. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- `RFDETRBase` — use `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, or `RFDETRLarge` instead. Emits `FutureWarning` on instantiation; scheduled for removal in v2.0. ([#900](https://github.com/roboflow/rf-detr/pull/900))
- `RFDETRSegPreview` — use `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, or `RFDETRSegLarge` instead. Emits `FutureWarning` on instantiation; scheduled for removal in v2.0. ([#900](https://github.com/roboflow/rf-detr/pull/900))
- `rfdetr.util` and `rfdetr.deploy` sub-modules are deprecated and will be removed in v1.9. A `__getattr__` hook on the `rfdetr` package now emits a clear `ImportError` with migration guidance when these legacy paths are accessed. ([#839](https://github.com/roboflow/rf-detr/pull/839))

### Fixed

- Fixed TFLite export (`format="tflite"`) producing detection scores that collapse to ~0.02 (vs ~0.62 from ONNX) on real inputs. Root cause is a long-standing onnx2tf bug ([PINTO0309/onnx2tf#274](https://github.com/PINTO0309/onnx2tf/issues/274)) where the `GridSample` lowering diverges numerically from ONNX while onnx2tf's own validator silently passes. RF-DETR's deformable cross-attention uses `F.grid_sample` once per decoder layer; drift compounds and is amplified by the attention softmax. The converter now detects onnx2tf's `GridSample → pseudo-GridSample` replacement kwarg at runtime (introspecting `convert()` via `inspect.signature`) and passes it as `True`; a warning is logged when the kwarg is absent. ([#1041](https://github.com/roboflow/rf-detr/pull/1041))

- `WindowedDinov2WithRegistersEmbeddings.forward()` now raises `ValueError` (instead of silently failing under `-O`) when input spatial dimensions are not divisible by `patch_size * num_windows`, with a clear message identifying the divisor and actual shape. ([#167](https://github.com/roboflow/rf-detr/pull/167))

- Fixed `_namespace.py`: `num_select` in the builder namespace now always reads from `ModelConfig`, eliminating a regression where `TrainConfig.num_select` (default 300) silently overrode model-specific values of 100–200 for segmentation variants (`RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegPreview`). Post-processing now uses the correct top-k count for each model. ([#841](https://github.com/roboflow/rf-detr/pull/841))

- Fixed `models/weights.py`: `load_pretrain_weights` now correctly auto-aligns the model head when the checkpoint has fewer classes than the configured default, preventing a silent mismatch when `num_classes` was not explicitly set by the caller. ([#845](https://github.com/roboflow/rf-detr/pull/845))

- Fixed `models/weights.py`: `load_pretrain_weights` now slices `refpoint_embed.weight` and `query_feat.weight` per-group when reshaping checkpoint queries, instead of taking a flat `tensor[: num_queries * group_detr]` slice. The flat slice silently scrambled groups 1+ when `num_queries` decreased and `group_detr > 1`; inference (which only reads group 0) was unaffected, but training-resume corrupted query embeddings for groups 1 onward. ([#1019](https://github.com/roboflow/rf-detr/pull/1019))

- Fixed YOLO segmentation training on large datasets hitting OS out-of-memory: `supervision.DetectionDataset.from_yolo(force_masks=True)` was eager-rasterising H×W boolean masks for every image at dataset construction time (≈1 GB/1 000 images at 1024 px). A new `_LazyYoloDetectionDataset` stores polygon coordinates only and defers dense mask rasterisation to `__getitem__`, keeping RAM proportional to annotation count rather than (N × H × W). ([#851](https://github.com/roboflow/rf-detr/pull/851))

- Fixed ONNX/TRT dynamic batch inference: `gen_encoder_output_proposals` and `Transformer.forward` extracted the batch size as a Python int and passed it to `torch.full`, `.view(N_, ...)`, `.expand(N_, ...)`, and `.repeat(bs, ...)`, causing the ONNX tracer to bake the training batch size (e.g. 8) as a compile-time constant. TRT engines built with `--minShapes` smaller than the trace batch would fail at inference with `Reshape: reshaping failed`. All six call sites are now replaced with ONNX-symbolic equivalents (`zeros_like`, `-1` reshapes, `expand(memory.shape[0], ...)`), keeping the batch dimension fully dynamic. ([#950](https://github.com/roboflow/rf-detr/pull/950), closes [#949](https://github.com/roboflow/rf-detr/issues/949))

- Fixed training failure when `square_resize_div_64=False`: the non-square resize pipeline (`SmallestMaxSize` + `LongestMaxSize`) did not guarantee output dimensions divisible by `patch_size * num_windows`, causing `WindowedDinov2WithRegistersEmbeddings.forward` to raise `ValueError`. A `PadIfNeeded` step (with `pad_height_divisor` and `pad_width_divisor` set to `patch_size * num_windows`) is now appended after the resize pair in both the train and val/test pipelines. ([#991](https://github.com/roboflow/rf-detr/pull/991), closes [#983](https://github.com/roboflow/rf-detr/issues/983))

- Fixed non-square batch padding correctness: batch-level `block_size` rounding is now applied in the DataLoader collator (`nested_tensor_from_tensor_list` via `make_collate_fn`) in addition to the transform-level `PadIfNeeded`, ensuring divisibility by `patch_size * num_windows` survives any `Compose` reordering and applies uniformly to custom evaluation harnesses. ([#992](https://github.com/roboflow/rf-detr/pull/992))

- Fixed `RFDETRModelModule.on_load_checkpoint` crashing with `RuntimeError` when resuming training from a checkpoint saved at a different image resolution: DINOv2 positional embeddings in the checkpoint are now bicubic-interpolated to match `model_config.positional_encoding_size` before PyTorch Lightning applies the state dict. ([#1002](https://github.com/roboflow/rf-detr/pull/1002), closes [#998](https://github.com/roboflow/rf-detr/issues/998))

- Fixed `RFDETRLarge` initialization showing two conflicting `ValueError`s (for `patch_size=14` and `patch_size=16`) when the deprecated-config fallback retry also fails. The fallback now re-raises the original error without chained context, so users see a single deterministic message. ([#975](https://github.com/roboflow/rf-detr/pull/975))

- Fixed `RFDETRModelModule.__init__` crashing with `RuntimeError: size mismatch for backbone.0.encoder.encoder.embeddings.position_embeddings` when training segmentation models at a custom resolution (e.g. `RFDETRSegLarge(resolution=1008).train(...)`). The training entry path now delegates to the canonical `load_pretrain_weights` helper, which bicubic-interpolates the DINOv2 positional embeddings before `load_state_dict`. ([#1040](https://github.com/roboflow/rf-detr/pull/1040), closes [#1038](https://github.com/roboflow/rf-detr/issues/1038), [#1023](https://github.com/roboflow/rf-detr/issues/1023))

- Fixed TFLite detection scores collapsing for all queries (scores ~0.02 vs ~0.62 from ONNX) when `GridSample` was used as an onnx2tf pseudo-operator. The `GridSample` ONNX node is now rewritten to `Gather`-based integer-index arithmetic before conversion, eliminating all numerical drift from attention position sampling. This supersedes the pseudo-`GridSample` runtime-kwarg approach added in [#1041](https://github.com/roboflow/rf-detr/pull/1041). ([#1054](https://github.com/roboflow/rf-detr/pull/1054))

- Fixed `class_name` lookup for pretrained COCO models: COCO category IDs are sparse (1–90 with gaps for 80 classes), so flat 0-based indexing returned the wrong name (e.g. `class_id=18` ("dog") incorrectly returned `class_names[18]` instead of `class_names[16]`). Detection now uses a `coco_id → class_name` mapping built from the canonical `COCO_CLASSES` list so every COCO category resolves to its correct label. Fine-tuned models continue to use direct 0-based indexing unchanged. ([#1051](https://github.com/roboflow/rf-detr/pull/1051))

---

## [1.6.5] — 2026-04-22

### Breaking Changes

- `predict()` now stores the source image in `detections.metadata["source_image"]` instead of `detections.data["source_image"]`. supervision indexes every value in `data` by the detection mask; `source_image` is per-image, not per-detection, so boolean/integer indexing raised `IndexError`. Moving it to `metadata` (passed through unchanged) fixes the issue. Update any code that reads `detections.data["source_image"]`. ([#972](https://github.com/roboflow/rf-detr/pull/972), [#968](https://github.com/roboflow/rf-detr/issues/968))

### Fixed

- Fixed segmentation training crash on T4 and P100 GPUs: cuDNN engine selection fails for depthwise convolution backward on some CUDA stacks (Kaggle, Colab). A custom `autograd.Function` now disables cuDNN in both forward and backward passes. ([#967](https://github.com/roboflow/rf-detr/pull/967))
- Fixed `ema_segm_mAP_50_95` and `ema_segm_mAP_50` being computed from the base (non-EMA) metric accumulator instead of the EMA accumulator, producing misleading validation scores for segmentation models. ([#980](https://github.com/roboflow/rf-detr/pull/980))
- Fixed `BestModelCallback` losing the best EMA score on training resume because `_best_ema` was not persisted in `state_dict()`. ([#973](https://github.com/roboflow/rf-detr/pull/973))
- Fixed `positional_encoding_size` not updating when `resolution` is set at construction time (e.g. `RFDETRLarge(resolution=640)`), causing shape mismatches during forward. A model validator now auto-syncs PE size. ([#956](https://github.com/roboflow/rf-detr/pull/956))
- Fixed pretrained weight loading crash with custom resolution: DINOv2 positional embeddings are now bicubic-interpolated to match the target grid before `load_state_dict`. ([#964](https://github.com/roboflow/rf-detr/pull/964))
- Fixed `validate_checkpoint_compatibility` producing a cryptic `RuntimeError` on `patch_size` mismatch when checkpoint lacks explicit `args.patch_size`. The function now infers `patch_size` from the DINOv2 projection weight shape and raises a descriptive `ValueError`. ([#971](https://github.com/roboflow/rf-detr/pull/971))
- Fixed `predict()` storing `detections.data["source_shape"]` as a Python `tuple`, which caused `TypeError` whenever `sv.Detections` was iterated. The value is now an `np.ndarray` of shape `(N, 2)` and dtype `int64`. ([#966](https://github.com/roboflow/rf-detr/pull/966), [#963](https://github.com/roboflow/rf-detr/issues/963))
- Fixed `predict()` emitting a misleading "class_id out of range" warning for the background/no-object class (class index `num_classes`). Background-class detections now map `data["class_name"]` to `"__background__"` without any warning. ([#970](https://github.com/roboflow/rf-detr/issues/970))

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

- `predict()` now stores the original image and its shape on returned `sv.Detections` objects — `detections.data["source_image"]` (NumPy array) and `detections.data["source_shape"]` (NumPy array of shape `(N, 2)` where each row is `[height, width]`) let you annotate results without loading the image separately. ([#892](https://github.com/roboflow/rf-detr/pull/892))
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
