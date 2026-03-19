# RF-DETR PyTorch Lightning Migration Plan

## 🎯 Motivation

### What maintainers gain

| Pain point today | After PTL migration |
|---|---|
| ~800 lines of boilerplate in `engine.py` + `main.py` (GradScaler, optimizer step, gradient clipping, AMP, DDP init, sampler logic) | All replaced by `Trainer` config — delete the code |
| Custom `defaultdict(list)` callback system in `RFDETR.train()` | PTL's typed, lifecycle-aware callback hooks |
| Manual multi-GPU setup (`init_distributed_mode`, custom samplers, `all_gather`, `reduce_dict`, `save_on_master`) | PTL handles entirely via `strategy="auto"` |
| Resume from checkpoint requires manual `start_epoch` tracking | `trainer.fit(..., ckpt_path=...)` — PTL resumes seamlessly |
| Two parallel config tiers: Pydantic (`config.py`) + argparse Namespace (`populate_args()`) | Pydantic only — `populate_args()` deleted |
| Testing training loop requires full run | PTL's `Trainer(fast_dev_run=1, limit_train_batches=...)` |
| `CocoEvaluator` wraps pycocotools internals; distributed sync is manual; `evalImgs` surgery for F1 sweep | Torchmetrics `MeanAveragePrecision` — distributed-aware, no internals access needed |
| `coco_eval.py` hand-rolls distributed gather + pycocotools patching | Deleted entirely |

### What users gain

| Capability | How |
|---|---|
| **Multi-GPU out of the box** | `rfdetr fit --trainer.devices 4` |
| **YAML experiments** | `rfdetr fit --config my_run.yaml` |
| **Resume training** | `rfdetr fit --ckpt_path output/last.ckpt` |
| **Logger integration** | `--trainer.logger WandbLogger` or YAML |
| **HP tuning** | Optuna, Ray Tune compatible |
| **Evaluate / predict** | `rfdetr validate/predict --ckpt_path best.ckpt` |
| **Versioned checkpoints** | PTL `ModelCheckpoint` auto-saves `last.ckpt` + `epoch=N.ckpt` |

---

## 🔒 Public API (backward-compatibility surface)

Everything below must remain importable and functionally equivalent after the migration, or have a `@deprecated` shim. This is what backward-compatible means in this context.

```
rfdetr/
  __init__.py
    ├─ RFDETRNano
    ├─ RFDETRSmall
    ├─ RFDETRMedium
    ├─ RFDETRBase
    ├─ RFDETRLarge
    ├─ RFDETRLargeNew
    ├─ RFDETRLargeDeprecated
    ├─ RFDETRSegPreview
    ├─ RFDETRSegNano
    ├─ RFDETRSegSmall
    ├─ RFDETRSegMedium
    ├─ RFDETRSegLarge
    ├─ RFDETRSegXLarge
    ├─ RFDETRSeg2XLarge
    ├─ RFDETR2XLarge       (lazy, platform-only)
    └─ RFDETRXLarge        (lazy, platform-only)

  detr.py
    └─ RFDETR              (base class for all of the above)
         ├─ __init__(**kwargs)
         ├─ train(**kwargs)
         ├─ predict(images, threshold, **kwargs) → sv.Detections | List[sv.Detections]
         ├─ export(**kwargs)
         ├─ optimize_for_inference(compile, batch_size, dtype)
         ├─ remove_optimized_model()
         ├─ deploy_to_roboflow(workspace, project_id, version, api_key, size)
         └─ .class_names    (property)

  config.py
    ├─ TrainConfig          (Pydantic; used by .train(**kwargs))
    ├─ SegmentationTrainConfig
    ├─ ModelConfig
    ├─ RFDETRBaseConfig
    ├─ RFDETRNanoConfig
    ├─ RFDETRSmallConfig
    ├─ RFDETRMediumConfig
    ├─ RFDETRLargeConfig
    ├─ RFDETRLargeDeprecatedConfig
    ├─ RFDETRSegPreviewConfig
    ├─ RFDETRSegNanoConfig
    ├─ RFDETRSegSmallConfig
    ├─ RFDETRSegMediumConfig
    ├─ RFDETRSegLargeConfig
    ├─ RFDETRSegXLargeConfig
    └─ RFDETRSeg2XLargeConfig

  datasets/aug_config.py              (merged, public)
    ├─ AUG_CONFIG
    ├─ AUG_CONSERVATIVE
    ├─ AUG_AGGRESSIVE
    ├─ AUG_AERIAL
    └─ AUG_INDUSTRIAL
```

**Breaking changes accepted in this migration:**

| Item | Reason |
|---|---|
| `TrainConfig.device` field **dropped** | PTL handles device placement; passing `device=` to `.train()` was always misleading since the field had no effect in distributed runs |
| CLI argument structure | LightningCLI replaces argparse; no backward compat on CLI flags |
| `callbacks` dict parameter in `.train()` | The `defaultdict(list)` hook system is replaced by PTL callbacks; `@deprecated` warning emitted if non-empty |

**Not public API** (can change freely):

```
engine.py, main.py, cli/main.py   — internal; fully replaced
lit/                               — new PTL implementation package
datasets/coco_eval.py              — deleted (replaced by torchmetrics)
models/                            — stable architecture; not a public API contract
```

## 🧪 Test & Benchmark Constraints (must hold during migration)

- **`tests/benchmarks/test_coco_inference.py`** — asserts `stats["results_json"]`/`stats["results_json_masks"]` keys and hard mAP/F1 thresholds; keep compatibility wrapper until Chapter 6
- **`tests/benchmarks/test_synthetic_convergence.py`** — uses `populate_args()`, `evaluate()`, legacy `.train()`; migrate in Chapter 6, not as implicit fallout
- **`tests/util/test_metrics.py`** — validates `coco_extended_metrics` F1 exactly; re-implement via new matching helpers with parity tests first
- **`src/rfdetr/main.py` (`do_benchmark`)** — preserve benchmark semantics in PTL path before removing
- **`rfdetr.util.*` imports** — do not hard-break import paths in the same PR as PTL migration

---

## 📊 Torchmetrics Evaluation Migration

### Why torchmetrics

`torchmetrics.detection.MeanAveragePrecision` replaces the entire `CocoEvaluator` stack:

| Current | Replacement |
|---|---|
| `pycocotools.cocoeval.COCOeval` | `torchmetrics.detection.MeanAveragePrecision` |
| `CocoEvaluator.synchronize_between_processes()` + `all_gather()` | torchmetrics handles distributed sync internally |
| `patched_pycocotools_summarize()` | Removed; torchmetrics returns a clean dict |
| `CocoEvaluator.prepare_for_coco_detection()` — format conversion | Removed; torchmetrics accepts `boxes` (xyxy float) + `labels` (int) directly |
| `pycocotools.mask` RLE encoding in `prepare_for_coco_segmentation()` | Removed; torchmetrics accepts boolean mask tensors directly |
| `coco_eval.py` distributed merge helpers (`merge()`, `create_common_coco_eval()`) | Removed; handled by torchmetrics |

The `faster-coco-eval` standalone migration PR is **superseded** — torchmetrics uses it as a backend for segmentation mask IoU (transitive dependency via `torchmetrics[detection]`).

### What happens to `coco_extended_metrics()`

`coco_extended_metrics()` digs into `COCOeval.evalImgs` internals. Rewritten as two pieces:

**Part 1 — per-class AP:** `MeanAveragePrecision(class_metrics=True)` returns `map_per_class`, `mar_{max_dets}_per_class`.

**Part 2 — F1 sweep:** New helper `build_matching_data(preds_list, targets_list, iou_threshold=0.5, iou_type="bbox"|"segm")` replaces `evalImgs` surgery. Greedy highest-score-first matching at IoU=0.5 via `torchvision.ops.box_iou` (bbox) or boolean-mask IoU (segm). Returns compact per-class `{scores, matches, ignore, total_gt}` arrays for DDP merging and confidence sweeping. See Notes for Implementers for details.

`sweep_confidence_thresholds()` remains the public entry point, consuming the output of this helper.

**Key contracts:** `coco_extended_metrics()` is rewritten not deleted; `sweep_confidence_thresholds()` signature stays stable; no `COCOeval.evalImgs` dependency.

**Distributed F1 (required):** `MeanAveragePrecision` handles its own DDP reduction; custom F1 does not. Aggregate compact matching data (`scores/matches/ignore/total_gt`) across ranks via `all_gather_object` at epoch end, compute a single global confidence sweep, then log `val/F1`, `val/precision`, `val/recall`. For segmentation, use mask IoU (not bbox IoU proxy).

**Dependency summary:** `pycocotools.cocoeval` and `pycocotools.mask` are deleted from the eval path; `pycocotools.coco.COCO` is kept for dataset loading only. `faster-coco-eval` becomes a torchmetrics backend (transitive). `torchmetrics[detection]>=1.2` is the new core eval dependency.

**Format:** `PostProcess` returns xyxy absolute coords. Targets need CxCyWH→xyxy conversion in `COCOEvalCallback._convert_targets()`.

---

## 🏗️ Target Architecture
```
rfdetr.__init__  →  detr.py (RFDETR, RFDETRBase, RFDETRNano, ...) — primary API preserved
                         └─ .train() delegates to ↓

RFDETRModelModule (LightningModule) — CANONICAL
       ├─ training_step() / validation_step() / test_step()
       ├─ configure_optimizers()        — AdamW + LambdaLR
       ├─ on_train_batch_start()        — drop path/dropout scheduling, multi-scale
       └─ forward()                     — delegates to lwdetr model

RFDETRDataModule (LightningDataModule)
  ├─ prepare_data()      — Roboflow download (rank-0 only)
  ├─ setup()             — build_dataset (COCO / YOLO / Roboflow)
  ├─ train_dataloader()  — with multi-scale collate
  ├─ val_dataloader()
  └─ test_dataloader()

Lightning Callbacks
  ├─ RFDETREMACallback     — PTL `WeightAveraging`-based, custom avg_fn for tau warmup
  ├─ COCOEvalCallback      — MeanAveragePrecision (torchmetrics) + F1 sweep
  ├─ DropPathCallback      — drop path / dropout scheduling
  ├─ MetricsPlotCallback   — replaces MetricsPlotSink
  └─ BestModelCallback     — best regular/EMA selection

CLI (LightningCLI)
  ├─ fit / validate / test / predict   (auto-generated)
  └─ export / deploy / benchmark       (custom)
```

---

## 🔧 Phase 0: Preparation & Infrastructure

### 0.1 Dependencies

**File:** `pyproject.toml`

```toml
[project.dependencies]
lightning = ">=2.6,<3"        # LightningCLI + callbacks + jsonargparse (transitive)
torchmetrics = {version = ">=1.2", extras = ["detection"]}  # >=1.2 required for backend param
pyDeprecate = ">=0.3,<1"      # @deprecated decorator for public API shims
```

`jsonargparse` and `faster-coco-eval` are transitive (via `lightning` and `torchmetrics[detection]`). Do not list explicitly to avoid version conflicts.

Remove from the torchmetrics extras: `pycocotools.cocoeval` is no longer a direct dependency of our code.

### 0.2 `util/` package — keep as-is during migration

`util/` has 14 modules; `utilities/` has only `decorators.py`. Renaming `util/` → `utilities/` adds an entire shim layer for zero user benefit and should **not** be bundled with the PTL migration. Keep `util/` as the canonical package throughout this migration. A rename, if desired, is a separate follow-up PR.

### 0.3 Add `@deprecated` decorator (via pyDeprecate)

**File:** `src/rfdetr/utilities/decorators.py`

Add a `deprecated` function/class decorator alongside `_DeprecatedDict` using the `pyDeprecate` package (added in 0.1):

```python
from pyDeprecate import deprecated, void

# Deprecate an entire function or class:
@deprecated("Use `new_function` instead.", ver=2.0, alternative="new_function")
def old_function(*args, **kwargs):
    ...

# Deprecate specific keyword arguments:
@deprecated(args={"old_kwarg": "Use `new_kwarg` instead."}, ver=2.0)
def my_func(new_kwarg, old_kwarg=None):
    ...
```

`pyDeprecate` handles `DeprecationWarning` with correct `stacklevel` and optional `FutureWarning` upgrade automatically. Re-export `deprecated` and `void` from `utilities/decorators.py` so all shims import from a single internal location.

### 0.4 Create package structure

The package lives under `lit/` — a short, neutral name for the transitional layer. After the migration is complete and the PTL types are distributed into their natural homes, this package can be dissolved.

```
src/rfdetr/
├─ lit/
│  ├─ __init__.py          # build_trainer() factory + re-exports
│  ├─ module.py            # RFDETRModelModule (LightningModule)
│  ├─ datamodule.py        # RFDETRDataModule (LightningDataModule)
│  ├─ callbacks/
│  │  ├─ __init__.py
│  │  ├─ ema.py            # RFDETREMACallback
│  │  ├─ coco_eval.py      # COCOEvalCallback (uses torchmetrics MAP)
│  │  ├─ metrics.py        # MetricsPlotCallback
│  │  ├─ best_model.py     # BestModelCallback
│  │  └─ drop_schedule.py  # DropPathCallback
│  ├─ cli.py               # LightningCLI subclass
│  ├─ checkpoint.py        # Legacy checkpoint converter
│  └─ compat.py            # Legacy evaluate()/stats-schema adapters for migration period
```

---

## ⚡ Phase 1: LightningModule — `RFDETRModelModule`

**File:** `src/rfdetr/lit/module.py`

### 1.1 Core structure

```python
class RFDETRModelModule(L.LightningModule):
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.train_config = train_config
        # build_model(), load pretrain weights, PostProcess, optional LoRA
```

**What moves in:**

| Current location | New location | Notes |
|---|---|---|
| `Model.__init__()` — `build_model(args)` | `RFDETRModelModule.__init__()` | Model construction, weight loading |
| `Model.__init__()` — LoRA setup | `RFDETRModelModule.__init__()` | Conditional `get_peft_model` |
| `Model.__init__()` — `PostProcess` | `RFDETRModelModule.__init__()` | Kept as attribute |
| `engine.train_one_epoch()` — forward + loss | `RFDETRModelModule.training_step()` | PTL handles grad accum, AMP, clipping |
| `engine.evaluate()` — forward | `RFDETRModelModule.validation_step()` | Returns preds + targets for COCOEvalCallback |
| `Model.train()` — optimizer setup | `RFDETRModelModule.configure_optimizers()` | AdamW + LambdaLR |
| `engine.train_one_epoch()` — drop path/dropout | `RFDETRModelModule.on_train_batch_start()` | Uses `self.trainer.global_step` |
| `engine.train_one_epoch()` — multi-scale resize | `RFDETRModelModule.on_train_batch_start()` | Deterministic via `random.seed(global_step)` |
| `Model.reinitialize_detection_head()` | `RFDETRModelModule.reinitialize_detection_head()` | Preserved |

### 1.2 `training_step()`

```python
def training_step(self, batch, batch_idx):
    samples, targets = batch
    outputs = self.model(samples, targets)
    loss_dict = self.criterion(outputs, targets)
    weight_dict = self.criterion.weight_dict
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, sync_dist=True)
    self.log("train/loss", loss, prog_bar=True, sync_dist=True)
    return loss
```

- **Gradient accumulation:** `Trainer(accumulate_grad_batches=config.grad_accum_steps)`. **CRITICAL:** Current code scales each sub-batch loss by `1/grad_accum_steps`; PTL does NOT do this. The DataLoader must return `batch_size` (not `effective_batch_size`) and `accumulate_grad_batches` handles the rest. The loss in `training_step` is NOT scaled — PTL's accumulated gradients are equivalent to the mean because each step processes `1/N` of the data.
- **AMP:** `Trainer(precision="bf16-mixed")` — manual `GradScaler` + `autocast` deleted.
- **Gradient clipping:** `Trainer(gradient_clip_val=0.1)` (current `clip_max_norm`).

> **`num_boxes` normalization (preserved):** `SetCriterion` calls `all_reduce(num_boxes)` + `/ get_world_size()` before every loss — PTL's `sync_dist=True` does not replace this. See Decision #13.

### 1.3 `validation_step()`

```python
def validation_step(self, batch, batch_idx):
    samples, targets = batch
    outputs = self.model(samples)
    loss_dict = self.criterion(outputs, targets)
    orig_sizes = torch.stack([t["orig_size"] for t in targets])
    results = self.postprocess(outputs, orig_sizes)
    # Log validation loss
    loss = sum(loss_dict[k] * self.criterion.weight_dict[k]
               for k in loss_dict if k in self.criterion.weight_dict)
    self.log("val/loss", loss, sync_dist=True)
    return {"results": results, "targets": targets}
```

The `COCOEvalCallback` collects these outputs via `on_validation_batch_end()`.

### 1.4 `configure_optimizers()`

- Migrate `get_param_dict()` from `util/get_param_dicts.py` (layer-wise LR decay, component decay)
- Migrate `lr_lambda()` (cosine annealing with warmup, step decay)
- Return `{"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}`

### 1.5 `transfer_batch_to_device()`

`NestedTensor` is not a standard tensor — PTL's default device transfer iterates tuple elements and calls `.to(device)`. Override explicitly to be safe:

```python
def transfer_batch_to_device(self, batch, device, dataloader_idx):
    samples, targets = batch
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets
```

### 1.6 `gradient_checkpointing`

`ModelConfig.gradient_checkpointing: bool` is passed through `build_model(args)` → backbone constructor → `DinoVisionTransformerWithWindowedAttention`. It is a **constructor argument**, not a runtime toggle — there is no `set_gradient_checkpointing()` method. Ensure the flag flows through the args dict passed to `build_model()` in `RFDETRModelModule.__init__()`.

### 1.7 `predict_step()`

Wraps `RFDETR.predict()` preprocessing + forward + postprocessing. Returns `sv.Detections`.

---

## 📦 Phase 2: LightningDataModule — `RFDETRDataModule`

**File:** `src/rfdetr/lit/datamodule.py`

### 2.1 Core structure

```python
class RFDETRDataModule(L.LightningDataModule):
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        super().__init__()
        self.config = train_config
        self.model_config = model_config
```

### 2.2 What moves in

| Current location | New location | Notes |
|---|---|---|
| `Model.train()` — `build_dataset(...)` | `RFDETRDataModule.setup()` | Train, val, test datasets |
| `Model.train()` — DataLoader construction | `train_dataloader()`, `val_dataloader()` | Including `collate_fn` |
| `Model.train()` — small dataset uniform sampler | `train_dataloader()` | `replacement=True` sampler |
| `Model.train()` — `num_workers` spawn guard | `__init__()` | Platform detection |
| `RFDETR._load_classes()` | `setup()` | Class name loading |
| `RFDETR.train_from_config()` — Roboflow download | `prepare_data()` | PTL ensures rank-0 only |
| `TrainConfig.aug_config` | Passed to `build_dataset()` in `setup()` | Already in config |

### 2.3 Segmentation support

Both detection and segmentation models use this same `RFDETRDataModule`. The `SegmentationTrainConfig` extends `TrainConfig`, so the DataModule accepts either config type. The `segmentation_head` flag is passed to `build_dataset()` which routes to the correct dataset builder.

### 2.4 Multi-scale training

Keep in `RFDETRModelModule.on_train_batch_start()`: resize samples tensor based on `random.seed(global_step)`. Equivalent to the current `engine.py` approach without any distributed coordination issues.

### 2.5 What stays unchanged

- `datasets/coco.py`, `datasets/yolo.py`, `datasets/transforms.py`, `datasets/aug_config.py`
- `datasets/__init__.py` — `build_dataset()`
- `util/misc.py` — `collate_fn()`, `NestedTensor`, `nested_tensor_from_tensor_list()`

---

## 🪝 Phase 3: Lightning Callbacks

### 3.1 `RFDETREMACallback`

**File:** `src/rfdetr/lit/callbacks/ema.py`

PTL 2.6+ includes built-in weight averaging callbacks (`WeightAveraging` and `EMAWeightAveraging`), and they are explicitly customizable (`avg_fn`/`multi_avg_fn`, plus subclass overrides such as `should_update`). For RF-DETR, use these extension points to preserve existing EMA behavior.

Recommended implementation:
- Base on `WeightAveraging` with a custom `avg_fn` that reproduces current `ModelEma` tau warmup formula.
- Keep default validation on regular weights; run EMA validation explicitly so both regular and EMA metrics exist every epoch.
- If strict tau parity is not required, `EMAWeightAveraging(decay=...)` is acceptable as a simplified mode.

```python
class RFDETREMACallback(WeightAveraging):
    def __init__(self, decay: float = 0.993, tau: int = 100):
        self._decay = decay
        self._tau = tau
        # avg_fn passed via **kwargs to AveragedModel (not a named parameter)
        super().__init__(use_buffers=True, avg_fn=self._avg_fn)

    def _avg_fn(self, averaged_param, model_param, num_averaged):
        updates = num_averaged + 1
        effective_decay = self._decay * (1 - math.exp(-updates / self._tau)) if self._tau > 0 else self._decay
        return averaged_param * effective_decay + model_param * (1 - effective_decay)

    def should_update(self, step_idx=None, epoch_idx=None):
        # Override method (not a constructor param) — update on every step
        return step_idx is not None or epoch_idx is not None
```

**Metric policy (compat with current behavior):**
- Log regular-model validation metrics every epoch.
- If `use_ema=True`, also evaluate EMA weights every epoch and log `ema_*` metrics.
- Preserve `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, and `checkpoint_best_total.pth`.
- Keep callback interactions order-independent: `COCOEvalCallback` writes metrics, and `BestModelCallback` reads `trainer.callback_metrics` in `on_validation_end()` (not `on_validation_epoch_end()`).

### 3.2 `COCOEvalCallback`

**File:** `src/rfdetr/lit/callbacks/coco_eval.py`

Uses `torchmetrics.detection.MeanAveragePrecision`:

```python
from torchmetrics.detection import MeanAveragePrecision

class COCOEvalCallback(L.Callback):
    def __init__(self, max_dets: int = 500, segmentation: bool = False):
        self._max_dets = max_dets
        self._segmentation = segmentation
        self._class_names: list[str] = []
        self._f1_local = init_matching_accumulator()  # compact stats, not raw preds

    def setup(self, trainer, pl_module, stage):
        # Create metric here so it lands on the correct device after DDP setup
        iou_type = ["bbox", "segm"] if self._segmentation else "bbox"
        kwargs = dict(class_metrics=True, max_detection_thresholds=[1, 10, self._max_dets])
        if self._segmentation:
            kwargs["backend"] = "faster_coco_eval"
        self.map_metric = MeanAveragePrecision(iou_type=iou_type, **kwargs)

    def on_fit_start(self, trainer, pl_module):
        # Pull class names from DataModule once the dataset is set up
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "class_names"):
            self._class_names = dm.class_names

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        preds   = outputs["results"]   # list of dicts: boxes(xyxy float), scores, labels
        targets = self._convert_targets(outputs["targets"])
        self.map_metric.update(preds, targets)
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type="bbox")
        self._f1_local = merge_matching_data(self._f1_local, batch_matching)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = self.map_metric.compute()
        # When iou_type is a tuple, torchmetrics prefixes ALL keys (e.g., "bbox_map", "segm_map")
        # When iou_type is a single string, keys have no prefix (e.g., "map")
        pfx = "bbox_" if self._segmentation else ""
        mar_key = f"{pfx}mar_{self._max_dets}"  # dynamic: mar_500, not mar_100
        pl_module.log("val/mAP_50_95", metrics[f"{pfx}map"])
        pl_module.log("val/mAP_50",    metrics[f"{pfx}map_50"])
        pl_module.log("val/mAP_75",    metrics[f"{pfx}map_75"])
        pl_module.log("val/mAR",       metrics[mar_key])
        if self._segmentation:
            pl_module.log("val/segm_mAP_50_95", metrics["segm_map"])
            pl_module.log("val/segm_mAP_50",    metrics["segm_map_50"])
        # per-class AP via returned class IDs (safe: class_id maps to class_names by value)
        pc_key = f"{pfx}map_per_class"
        if pc_key in metrics and "classes" in metrics:
            for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
                idx = int(class_id)
                name = self._class_names[idx] if idx < len(self._class_names) else str(idx)
                pl_module.log(f"val/AP/{name}", ap)

        # F1 sweep — gather compact matching state across ranks before sweep
        per_class_data = distributed_merge_matching_data(self._f1_local, trainer)
        f1_results = sweep_confidence_thresholds(per_class_data,
                                                 np.linspace(0, 1, 101), ...)
        best = max(f1_results, key=lambda x: x["macro_f1"])
        pl_module.log("val/F1",        best["macro_f1"])
        pl_module.log("val/precision", best["macro_precision"])
        pl_module.log("val/recall",    best["macro_recall"])

        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()

    def _convert_targets(self, targets):
        """Convert targets from normalised CxCyWH to absolute xyxy; pass iscrowd through."""
        out = []
        for t in targets:
            h, w = t["orig_size"].tolist()
            boxes = box_cxcywh_to_xyxy(t["boxes"]) * t["boxes"].new_tensor([w, h, w, h])
            entry = {"boxes": boxes, "labels": t["labels"]}
            if "masks" in t:
                entry["masks"] = t["masks"].bool()
            if "iscrowd" in t:
                entry["iscrowd"] = t["iscrowd"]
            out.append(entry)
        return out
```

Key properties:
- **Distributed:** `MeanAveragePrecision` accumulates correctly across ranks — no manual sync needed
- **F1 sweep:** compact per-batch matching stats merged across ranks; no raw epoch buffers
- **Segmentation:** `iou_type=["bbox", "segm"]` + `backend="faster_coco_eval"` for mask IoU; segmentation F1 uses mask-IoU matching
- **EMA eval:** regular and EMA evaluations both supported to preserve current checkpoint/early-stopping semantics
- **Class names:** populated in `on_fit_start()` from `trainer.datamodule.class_names`
- **Crowd handling:** `iscrowd` passed from COCO targets into `build_matching_data()`; defaults to absent (all non-crowd) for YOLO/Roboflow datasets
- **Order safety:** `BestModelCallback` consumes finalized `trainer.callback_metrics` in `on_validation_end()` to avoid depending on callback execution order.

### 3.3 `DropPathCallback`

**File:** `src/rfdetr/lit/callbacks/drop_schedule.py`

- `on_train_batch_start()`: reads rates from `drop_scheduler()` and calls `model.update_drop_path()` / `model.update_dropout()`
- `util/drop_scheduler.py` stays unchanged

### 3.4 Logging Callbacks

**File:** `src/rfdetr/lit/callbacks/metrics.py`

Replace the current dict-sink pattern with PTL's native logger system:

| Current | PTL replacement |
|---|---|
| `MetricsTensorBoardSink` | `TensorBoardLogger` (built-in) |
| `MetricsWandBSink` | `WandbLogger` (built-in) |
| `MetricsMLFlowSink` | `MLFlowLogger` (built-in) |
| `MetricsClearMLSink` | Custom callback or community `ClearMLLogger` |
| `MetricsPlotSink` | `MetricsPlotCallback` (custom; no PTL equivalent) |

`MetricsPlotSink` is rewritten as a PTL callback (`MetricsPlotCallback`) since PTL has no built-in metrics plot.

### 3.5 `BestModelCallback`

**File:** `src/rfdetr/lit/callbacks/best_model.py`

- Track both `val/regular_mAP_50_95` and `val/ema_mAP_50_95` when EMA is enabled
- `on_validation_epoch_end()`: update best regular/EMA checkpoints independently
- `on_fit_end()`:
  - Write `checkpoint_best_regular.pth` and `checkpoint_best_ema.pth` (if EMA enabled)
  - Choose best of the two and copy to `checkpoint_best_total.pth`
  - Strip optimizer state via `util/misc.py:strip_checkpoint()`
  - If `run_test=True`, call `trainer.test(pl_module, datamodule=trainer.datamodule, ckpt_path="best")`
- Extends PTL's `ModelCheckpoint`

> Compatibility note: this preserves existing artifact semantics documented in `docs/learn/train/*` and current training outputs.

### 3.6 Early stopping

- **Delete:** `util/early_stopping.py`
- **Replace with:** PTL callback implementation that preserves current behavior:
  - `early_stopping_use_ema=True` → monitor EMA metric
  - `early_stopping_use_ema=False` → monitor `max(regular, EMA)` when EMA exists, otherwise regular metric

---

## 🚂 Phase 4: Trainer Assembly

### 4.1 Factory function

**File:** `src/rfdetr/lit/__init__.py`

```python
def build_trainer(config: TrainConfig, model_config: ModelConfig) -> L.Trainer:
    def _resolve_precision() -> str:
        if not model_config.amp: return "32-true"
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return "bf16-mixed"
        return "16-mixed" if torch.cuda.is_available() else "32-true"

    sharded = any(s in str(getattr(config, "strategy", "auto")).lower() for s in ("fsdp", "deepspeed"))
    enable_ema = bool(config.use_ema) and not sharded
    if config.use_ema and sharded:
        warnings.warn("EMA disabled for sharded strategies.", UserWarning, stacklevel=2)

    # Build callbacks: EMA (optional), COCOEval, DropPath, BestModel, MetricsPlot, ModelCheckpoint,
    # EarlyStopping (optional). Loggers: TensorBoard / WandB / MLflow / ClearML — conditional on config flags.

    return L.Trainer(
        max_epochs=config.epochs, accelerator="auto", devices="auto",
        strategy=getattr(config, "strategy", "auto"), precision=_resolve_precision(),
        accumulate_grad_batches=config.grad_accum_steps, gradient_clip_val=config.clip_max_norm,
        sync_batchnorm=config.sync_bn, callbacks=callbacks,
        logger=loggers if loggers else False,
        default_root_dir=config.output_dir, log_every_n_steps=50, deterministic=False,
    )
```

**EMA + sharded strategies:** PTL 2.6+ `WeightAveraging` is not strategy-aware for FSDP/DeepSpeed. Disable EMA under sharded strategies and warn.

### 4.2 `RFDETR.train()` — refactored to delegate to PTL

```python
def train(self, **kwargs):
    if kwargs.get("callbacks") and any(kwargs["callbacks"].values()):
        warnings.warn(
            "Custom callbacks dict is not forwarded to PTL. "
            "Use PTL Callback objects instead.",
            DeprecationWarning, stacklevel=2
        )
    kwargs.pop("callbacks", None)
    kwargs.pop("device", None)   # TrainConfig.device dropped
    run_benchmark = bool(kwargs.pop("do_benchmark", False))

    config   = self.get_train_config(**kwargs)
    module   = RFDETRModelModule(self.model_config, config)
    datamodule = RFDETRDataModule(config, self.model_config)
    trainer  = build_trainer(config, self.model_config)
    trainer.fit(module, datamodule, ckpt_path=config.resume or None)
    if run_benchmark:
        warnings.warn(
            "`do_benchmark` in `.train()` is deprecated; use `rfdetr benchmark`.",
            DeprecationWarning,
            stacklevel=2,
        )
        run_compat_benchmark(module, datamodule, output_dir=config.output_dir)
    self.model.model = module.model  # sync back for predict()
```

### 4.3 Configuration mapping

| Current field | PTL equivalent | Action |
|---|---|---|
| `batch_size` | `DataModule(batch_size=...)` | Pass through |
| `grad_accum_steps` | `Trainer(accumulate_grad_batches=...)` | Pass through |
| `amp` (`ModelConfig`) | `Trainer(precision=auto bf16/16/32)` | Map with CUDA capability fallback |
| `epochs` | `Trainer(max_epochs=...)` | Pass through |
| `TrainConfig.device` | — | **Dropped** — PTL auto-detects |
| `distributed` / `world_size` / `dist_url` | `Trainer(devices="auto", strategy="auto")` | **Deleted** from populate_args |
| `sync_bn` | `Trainer(sync_batchnorm=True)` | Promote or always-on default |
| `clip_max_norm` | `Trainer(gradient_clip_val=...)` | Pass through (no hardcoded 0.1) |
| `seed` | `L.seed_everything(seed)` | Promote to `TrainConfig` |
| `resume` | `trainer.fit(..., ckpt_path=config.resume)` | Already in `TrainConfig` |
| `start_epoch` | — | **Deleted** — PTL resumes automatically |
| `eval` flag | `rfdetr validate` CLI subcommand | **Deleted** from populate_args |
| `do_benchmark` | `rfdetr benchmark` subcommand / compatibility flag | Keep compatibility in `.train()` shim; deprecate then remove |
| `checkpoint_interval` | `ModelCheckpoint(every_n_epochs=...)` | Pass through |
| `num_workers` | `DataModule(num_workers=...)` | Pass through |
| `output_dir` | `Trainer(default_root_dir=...)` | Pass through |

---

## ⚙️ Phase 5: Config System Refactor

### 5.1 Deprecate `populate_args()` and remove it later

`populate_args()` in `main.py` translates Pydantic → argparse Namespace for `engine.py`. Keep it as a deprecated shim during migration, then remove after benchmark/test migration no longer imports it.

| Field currently in `populate_args()` | Action |
|---|---|
| `clip_max_norm` | Promote to `TrainConfig` (users may want to control this) |
| `seed` | Promote to `TrainConfig` |
| `distributed`, `world_size`, `dist_url`, `dist_backend` | **Delete** — PTL handles |
| `start_epoch` | **Delete** — PTL resumes automatically |
| `eval` flag | **Delete** — becomes `rfdetr validate` subcommand |
| `do_benchmark` | Deprecate in `.train()` and route to benchmark callback/subcommand |
| `sync_bn` | Promote to `TrainConfig` (or always-on in DDP) |
| `print_freq` | **Delete** — `Trainer(log_every_n_steps=50)` |
| `fp16_eval` | Promote to `TrainConfig` and map to PTL precision policy |
| `lr_scheduler` | Promote to `TrainConfig` (cosine vs step; currently only in `populate_args`) |
| `lr_min_factor` | Promote to `TrainConfig` (cosine minimum LR factor; only in `populate_args`) |
| `dont_save_weights` | Promote to `TrainConfig` |

### 5.2 `TrainConfig` changes summary

> **CRITICAL: `populate_args()` vs `TrainConfig` default divergence.**
> These fields have conflicting defaults between the two systems. The `TrainConfig` defaults are canonical (they're what users set via `.train(**kwargs)`). `populate_args()` defaults are internal and should NOT be treated as authoritative:
> `ema_decay`: 0.993 (TrainConfig) vs 0.9997 (populate_args) — use 0.993
> `ema_tau`: 100 vs 0 — use 100
> `use_ema`: True vs False — use True
> `warmup_epochs`: 0.0 vs 1 — use 0.0
> `early_stopping`: False vs True — use False

```python
class TrainConfig(BaseModel):
    # ... existing fields ...
    # DROPPED:
    # device: Literal["auto", "cpu", "cuda", "mps"]  ← removed (breaking change)

    # PROMOTED from populate_args:
    clip_max_norm: float = 0.1
    seed: Optional[int] = None
    sync_bn: bool = False
    fp16_eval: bool = False
    lr_scheduler: Literal["step", "cosine"] = "step"  # currently only in populate_args
    lr_min_factor: float = 0.0                          # currently only in populate_args
    dont_save_weights: bool = False                     # currently only in populate_args
```

Note: `ModelConfig.device` is also kept for model placement during inference (used by `Model.__init__` and `predict()`). Only `TrainConfig.device` is dropped.

### 5.3 Long-term direction for `TrainConfig`

`TrainConfig` as a monolithic Pydantic model is a transitional concept. PTL's recommended approach is to distribute configuration naturally across each component's `__init__` signature and let `LightningCLI` + `jsonargparse` auto-generate CLI flags and YAML keys from those signatures — no central config model needed.

Long-term trajectory:

| Today (`TrainConfig` field) | PTL-native home |
|---|---|
| `lr`, `weight_decay`, `batch_size`, `epochs`, `grad_accum_steps`, `clip_max_norm` | `RFDETRModelModule.__init__()` kwargs |
| `num_workers`, `aug_config`, `dataset_dir`, `batch_size` | `RFDETRDataModule.__init__()` kwargs |
| `amp`, `devices`, `strategy`, `sync_bn`, `precision` | `Trainer(...)` kwargs (set in YAML or CLI) |
| `tensorboard`, `wandb`, `mlflow`, `output_dir` | Logger configs in YAML |
| `early_stopping`, `early_stopping_patience`, `checkpoint_interval` | Callback configs in YAML |

This transition is **out of scope for this migration** — `TrainConfig` is kept as-is and threaded through `build_trainer()` / `RFDETRModelModule.__init__()` to preserve the existing user API. The monolithic config can be dissolved in a separate PR once the PTL migration is stable.

---

## 💻 Phase 6: CLI Replacement

### 6.1 LightningCLI

**File:** `src/rfdetr/lit/cli.py`

```python
from lightning.pytorch.cli import LightningCLI

class RFDETRCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_config.resolution", "data.resolution", apply_on="instantiate")
        parser.link_arguments("model.model_config.patch_size", "data.patch_size", apply_on="instantiate")

def main():
    RFDETRCli(RFDETRModelModule, RFDETRDataModule)
```

### 6.2 Entry point

```toml
[project.scripts]
rfdetr = "rfdetr.lit.cli:main"
```

### 6.3 Subcommands

Auto: `fit`, `validate`, `test`, `predict`. Custom (later): `export`, `deploy`, `benchmark`, `convert-checkpoint`.

### 6.4 YAML config (auto-provided by LightningCLI)

```bash
rfdetr fit --config configs/rfdetr_base.yaml
rfdetr fit --trainer.devices 4
rfdetr validate --ckpt_path output/best.ckpt
```

---

## 🪦 Phase 7: Top-Level API Compatibility Layer

### 7.1 `RFDETR` and all subclasses — NOT deprecated (kept as primary API)

The `RFDETR*` class hierarchy (`RFDETRBase`, `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`, `RFDETRLargeNew`, `RFDETRLargeDeprecated`, `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge`, platform models `RFDETR2XLarge`, `RFDETRXLarge`) IS the public API users interact with. These classes are **preserved without deprecation**. Their `.train()` method is refactored to delegate to PTL internally (see Phase 4.2) but the user-facing interface remains stable. The `RFDETRModule`/`RFDETRDataModule` API is offered as an **advanced alternative**, not a replacement.

### 7.2 `Module.train()` naming conflict

`RFDETR` is a plain Python class (not `nn.Module`), so `.train(**kwargs)` has no naming conflict. `RFDETRModule` is a `LightningModule` — users call `Trainer.fit()`, not `.train()` directly. If `RFDETR` ever gains `nn.Module` inheritance, rename `.train()` to `.fit()` first.

### 7.3 `RFDETR.train()` shim — see Phase 4.2

### 7.4 `RFDETR.predict()` — kept as-is

Self-contained; no `engine.py` involvement. `RFDETRModule.predict_step()` is the PTL-native alternative for Trainer-driven inference.

### 7.5 `RFDETR.export()` — kept as-is

Export logic stays on `RFDETR`. The PTL path can offer `RFDETRModule.export()` as an alternative.

### 7.6 `RFDETR.deploy_to_roboflow()` — kept as-is

No change needed.

### 7.7 Updated `__init__.py` exports

```python
# Existing exports (unchanged — these remain the primary API)
from rfdetr.detr import RFDETRBase, RFDETRNano, RFDETRSmall, ...

# New advanced exports (added alongside existing)
from rfdetr.lit import RFDETRModule, RFDETRDataModule, build_trainer
```

---

## 🔄 Phase 8: Checkpoint Converter

**File:** `src/rfdetr/lit/checkpoint.py`

```python
def convert_legacy_checkpoint(old_path: str, new_path: str):
    """Convert old-format .pth checkpoint to PTL .ckpt format."""
    old = torch.load(old_path, map_location="cpu")
    args_obj = old.get("args")
    if isinstance(args_obj, dict):
        hyper_parameters = args_obj
    elif args_obj is None:
        hyper_parameters = {}
    else:
        hyper_parameters = vars(args_obj)

    new = {
        "state_dict": {"model." + k: v for k, v in old["model"].items()},
        "epoch": old.get("epoch", 0),
        "global_step": 0,
        "hyper_parameters": hyper_parameters,
        "legacy_checkpoint_format": True,
    }
    if "ema_model" in old:
        # Keep this as a dedicated key; callback-specific state keys are framework-internal.
        new["legacy_ema_state_dict"] = old["ema_model"]
    torch.save(new, new_path)
```

Auto-detect in `RFDETRModule.on_load_checkpoint()`:

```python
def on_load_checkpoint(self, checkpoint):
    # Legacy .pth loaded directly by Trainer
    if "model" in checkpoint and "state_dict" not in checkpoint:
        checkpoint["state_dict"] = {
            "model." + k: v for k, v in checkpoint["model"].items()
        }
    # If legacy EMA weights are present, hand them to RFDETREMACallback explicitly
    if "legacy_ema_state_dict" in checkpoint:
        self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
```

---

## 🗑️ Phase 9: Cleanup & Deletion

### Files to delete entirely

| File | Reason |
|---|---|
| `src/rfdetr/main.py` | `Model` class + `populate_args()` replaced |
| `src/rfdetr/cli/main.py` | Replaced by LightningCLI |
| `src/rfdetr/datasets/coco_eval.py` | `CocoEvaluator` replaced by torchmetrics; all helpers deleted |
| `src/rfdetr/util/early_stopping.py` | Replaced by PTL `EarlyStopping` callback |

### Functions to delete from remaining files

| File | Function(s) | Reason |
|---|---|---|
| `engine.py` | `train_one_epoch()` | Replaced by `training_step()` + Trainer |
| `engine.py` | legacy body of `evaluate()` | Replaced by PTL validation path; keep thin compatibility wrapper until benchmark migration is complete |
| `engine.py` | `get_autocast_args()`, `_get_cuda_autocast_dtype()` | PTL handles AMP |
| `engine.py` | legacy internals of `coco_extended_metrics()` | Keep public symbol; re-implement using new matching helpers to satisfy `tests/util/test_metrics.py` |
| `util/misc.py` | `MetricLogger`, `SmoothedValue` | PTL's `self.log()` |
| `util/misc.py` | `init_distributed_mode()` | PTL |
| `util/misc.py` | `reduce_dict()` | PTL `sync_dist=True` |
| `util/misc.py` | `get_rank()`, `is_main_process()` | PTL trainer properties (`self.trainer.global_rank`, `self.trainer.is_global_zero`) |
| `util/misc.py` | `get_world_size()`, `is_dist_avail_and_initialized()` | **Do NOT delete.** Still imported by `SetCriterion` in `models/lwdetr.py` for `num_boxes` all_reduce. Keep in `util/misc.py`. |
| `util/misc.py` | `save_on_master()` | PTL checkpointing |
| `util/metrics.py` | `MetricsTensorBoardSink`, `MetricsWandBSink`, `MetricsMLFlowSink`, `MetricsClearMLSink` | PTL built-in loggers |
| `datasets/__init__.py` | `get_coco_api_from_dataset()` | Remove only after benchmark/tests are migrated off `engine.evaluate()` compatibility path |

### Functions to KEEP (reused by PTL code)

| File | What to keep | Used by |
|---|---|---|
| `engine.py` | `sweep_confidence_thresholds()` | `COCOEvalCallback` |
| `engine.py` | new `build_matching_data()` | `COCOEvalCallback` |
| `engine.py` | compatibility `evaluate()` wrapper + `results_json` schema | benchmark tests + migration transition |
| `engine.py` | compatibility `coco_extended_metrics()` symbol | existing tests/imports |
| `models/lwdetr.py` | Entire file | `RFDETRModule` |
| `models/` | All model code | Unchanged |
| `datasets/` | All dataset code + augmentation | `RFDETRDataModule` |
| `datasets/aug_config.py` | All presets | Public API |
| `config.py` | All Pydantic configs (minus `device` field) | Used directly |
| `util/utils.py` | `clean_state_dict` | `BestModelCallback` |
| `util/misc.py` | `strip_checkpoint()` | `BestModelCallback` |
| `util/drop_scheduler.py` | `drop_scheduler()` | `DropPathCallback` |
| `util/get_param_dicts.py` | `get_param_dict()` | `configure_optimizers()` |
| `util/misc.py` | `collate_fn()`, `NestedTensor`, `nested_tensor_from_tensor_list()` | `RFDETRDataModule` |
| `util/misc.py` | `get_world_size()`, `is_dist_avail_and_initialized()` | `SetCriterion.forward()` — `num_boxes` all_reduce for globally-normalized loss |
| `util/metrics.py` | `MetricsPlotSink` logic | `MetricsPlotCallback` |
| `utilities/decorators.py` | `_DeprecatedDict`, `@deprecated` | Deprecation shims |
| `detr.py` | Entire file | Deprecated compat layer |
| `assets/` | Model weights | Unchanged |
| `deploy/` | ONNX export, benchmark | Unchanged |

---

## 📋 Implementation Sequence

> **Commit discipline:** Each checklist item (or tightly related pair) is a **single atomic commit**. This keeps history bisectable and makes every step auditable in a PR review. Commit messages should reference the chapter and phase (e.g., `feat(lit): [Ch1/Phase1] implement RFDETRModule training_step`).
>
> **Chapter gate:** Before opening a PR for the next chapter, the full CPU test suite **must be green** (`uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`). No chapter is considered done until CI passes. GPU tests are validated at the final chapter.

### Prerequisite: Add compatibility harness for migration

Before migrating internals, lock in behavior expected by current tests/benchmarks.

- [ ] Add a migration compatibility checklist in tests:
  - [ ] `engine.evaluate()` output schema keys (`results_json`, `results_json_masks`)
  - [ ] `coco_extended_metrics()` symbol remains importable
  - [ ] `rfdetr.util.*` imports remain valid
- [ ] Add explicit numeric tolerances for metric parity tests (mAP/F1) with fixed seeds and fixed dataset sample sets
- [ ] Add benchmark acceptance gates that mirror existing benchmark tests (`tests/benchmarks/test_coco_inference.py`, `tests/benchmarks/test_synthetic_convergence.py`)

**Milestone:** Baseline behavior is codified before invasive refactors; regressions are caught by tests instead of discovered late.

---

### Chapter 1: Implement `RFDETRModule` and `RFDETRDataModule` _(Phases 0 + 1 + 2)_

**Covers:** Phase 0 (§0.1 dependencies, §0.3 `@deprecated`, §0.4 `lit/` package), Phase 1 (`RFDETRModule`), Phase 2 (`RFDETRDataModule`)

**Pre-conditions:** Prerequisite compatibility harness PR merged and CPU CI green.
**Creates:** `src/rfdetr/lit/__init__.py`, `lit/module.py`, `lit/datamodule.py`, `lit/compat.py`, `lit/callbacks/__init__.py`
**Modifies:** `pyproject.toml` (add `lightning`, `torchmetrics[detection]`, `pyDeprecate`), `src/rfdetr/utilities/decorators.py` (add `@deprecated` via pyDeprecate), `src/rfdetr/engine.py` (add `evaluate()` compat wrapper), `tests/` (new smoke tests)
**Deletes:** nothing

Create the core Lightning module and data module that replicate the current `Model` + `engine` training loop and dataset pipeline. This is the first piece that can be tested in isolation alongside the existing code — no existing files are deleted yet.

- [ ] Add `lightning>=2.6,<3`, `torchmetrics[detection]>=1.2` to `pyproject.toml` (jsonargparse and faster-coco-eval are transitive)
- [ ] Create `src/rfdetr/lit/` package structure (see Phase 0.4)
- [ ] Implement `RFDETRModule` (`lit/module.py`) with `__init__`, `training_step`, `validation_step`, `configure_optimizers`, `on_train_batch_start` (drop path + multi-scale), `transfer_batch_to_device`, `predict_step`
- [ ] Wire `gradient_checkpointing` flag from `ModelConfig` in `RFDETRModule.__init__()`
- [ ] Implement `RFDETRDataModule` (`lit/datamodule.py`) with `prepare_data`, `setup`, `train_dataloader`, `val_dataloader`, `test_dataloader`
- [ ] Add `engine.evaluate()` compatibility wrapper that can consume PTL module outputs and returns the legacy stats schema expected by benchmark tests
- [ ] Verify `Trainer(fast_dev_run=2).fit(module, datamodule)` runs without error for a detection model
- [ ] Verify `Trainer(fast_dev_run=2).fit(module, datamodule)` runs without error for a segmentation model

**Milestone:** PTL training loop runs end-to-end for detection and segmentation; no existing code deleted. Tests: `fast_dev_run` smoke, gradient flow, `NestedTensor` device transfer, `engine.evaluate()` schema compat. **CI gate: full CPU test suite green before Chapter 2.**

---

### Chapter 2: Replace COCO evaluation with torchmetrics and rewrite F1 sweep _(Phase 3 — COCOEvalCallback)_

**Covers:** Phase 3 (`COCOEvalCallback` in `lit/callbacks/coco_eval.py`; `build_matching_data()` and `sweep_confidence_thresholds()` helpers in `engine.py`)

**Pre-conditions:** Chapter 1 PR merged and CPU CI green.
**Creates:** `src/rfdetr/lit/callbacks/coco_eval.py`
**Modifies:** `src/rfdetr/engine.py` (add `build_matching_data()`, `sweep_confidence_thresholds()`), `tests/` (mAP/F1 parity tests, numeric baselines)
**Deletes:** nothing — `CocoEvaluator` intentionally kept for parallel comparison

Replace `CocoEvaluator` (pycocotools-based) with `torchmetrics.detection.MeanAveragePrecision` and rewrite the F1 sweep helper to remove its dependency on `COCOeval.evalImgs` internals. This chapter makes the evaluation stack independent of both `pycocotools.cocoeval` and the planned-but-now-superseded `faster-coco-eval` migration PR.

- [ ] Write `build_matching_data(preds_list, targets_list, iou_threshold=0.5, iou_type="bbox"|"segm")` in `engine.py`:
  - [ ] bbox matching via `torchvision.ops.box_iou`
  - [ ] segmentation matching via boolean mask IoU
  - [ ] greedy highest-score-first matching with crowd handling
- [ ] Implement compact matching accumulator + distributed merge helper for F1 (`all_gather_object` or reduced tensors), so F1 is global across DDP ranks
- [ ] Implement `COCOEvalCallback` (`lit/callbacks/coco_eval.py`) using `MeanAveragePrecision(class_metrics=True)` for mAP and merged matching data + `sweep_confidence_thresholds()` for F1 sweep
- [ ] Support both detection (`iou_type="bbox"`) and segmentation (`iou_type=["bbox","segm"]`, `backend="faster_coco_eval"`)
- [ ] Handle target box conversion from normalized CxCyWH to absolute xyxy inside the callback using device-safe tensor creation
- [ ] Preserve legacy output keys (`results_json`, `results_json_masks`, `coco_eval_bbox`, `coco_eval_masks`) in compatibility wrappers used by benchmarks
- [ ] **Measure** mAP and F1 from the legacy `CocoEvaluator` path on the reference dataset — record as numeric baselines in the test suite
- [ ] **Verify** the new `COCOEvalCallback` produces numbers within explicit tolerances on the same dataset:
  - [ ] detection: `|ΔmAP50| <= 0.005`, `|ΔF1| <= 0.01`
  - [ ] segmentation: `|Δmask mAP50| <= 0.005`, `|Δmask F1| <= 0.01`
- [ ] **Write** tests that assert the new metrics meet the same numeric thresholds — these tests replace the legacy-path measurement once the baselines are met
- [ ] The legacy `CocoEvaluator` code is **not deleted yet** (that happens in Chapter 6); both paths run in parallel during this chapter for comparison

**Milestone:** `COCOEvalCallback` mAP/F1 agree with legacy `CocoEvaluator` within tolerance; F1 global across DDP ranks; legacy code still present. Tests: mAP parity, F1 parity, segm mask F1, `iscrowd` handling, DDP aggregation. **CI gate: full CPU test suite green before Chapter 3.**

---

### Chapter 3: Implement remaining callbacks _(Phase 3 — EMA, DropPath, BestModel, Metrics)_

**Covers:** Phase 3 (`RFDETREMACallback`, `DropPathCallback`, `BestModelCallback`, `MetricsPlotCallback` in `lit/callbacks/`)

**Pre-conditions:** Chapter 2 PR merged and CPU CI green.
**Creates:** `src/rfdetr/lit/callbacks/ema.py`, `lit/callbacks/drop_schedule.py`, `lit/callbacks/best_model.py`, `lit/callbacks/metrics.py`
**Modifies:** `src/rfdetr/lit/__init__.py` (wire callbacks into `build_trainer()`), `tests/` (EMA parity, drop rate, early-stopping trigger, best-model artifact tests)
**Deletes:** nothing

Implement the callbacks that handle EMA, drop-path scheduling, best-model selection, metrics plotting, and early stopping.

- [ ] Implement `RFDETREMACallback` (`lit/callbacks/ema.py`) on PTL 2.6+ `WeightAveraging`/`EMAWeightAveraging` APIs:
  - [ ] strict-parity path: custom `avg_fn` with `decay * (1 - exp(-updates / tau))`
  - [ ] optional simplified path: `EMAWeightAveraging(decay=...)` when tau warmup parity is not required
  - [ ] if callback adds custom state, implement `state_dict()/load_state_dict()` and unique `state_key`
  - [ ] verify EMA weights after N steps match current `ModelEma` output for strict-parity mode
- [ ] Implement `DropPathCallback` (`lit/callbacks/drop_schedule.py`); verify drop rates match `drop_scheduler()` at the same global step values
- [ ] Implement `BestModelCallback` (`lit/callbacks/best_model.py`); tracks regular and EMA metrics, writes `checkpoint_best_regular.pth` and `checkpoint_best_ema.pth`, selects `checkpoint_best_total.pth`, strips optimizer state, and calls `trainer.test()` from `on_fit_end()` when `run_test=True`
- [ ] Implement `MetricsPlotCallback` (`lit/callbacks/metrics.py`) rewriting `MetricsPlotSink` as a PTL callback
- [ ] Implement PTL early-stopping behavior compatible with current `early_stopping_use_ema` semantics (`ema` vs `max(regular, ema)`); verify trigger epoch parity with current callback

**Milestone:** All callbacks wired; full `build_trainer()` run produces EMA weights matching `ModelEma`, correct best-model artifacts, legacy drop path rates. Tests: EMA parity (N=500), drop rate, early stopping trigger, `BestModelCallback` artifacts. **CI gate: full CPU test suite green before Chapter 4.**

---

### Chapter 4: Implement `build_trainer()`, CLI, and config cleanup _(Phases 4 + 5 + 6)_

**Covers:** Phase 4 (`build_trainer()` factory, `Trainer` assembly), Phase 5 (Config System — `TrainConfig` field promotion, `populate_args()` shim), Phase 6 (CLI — `RFDETRCli` / `LightningCLI`)

**Pre-conditions:** Chapter 3 PR merged and CPU CI green.
**Creates:** `src/rfdetr/lit/cli.py`, `configs/` YAML examples (nano/small/medium/base/large detection + segmentation)
**Modifies:** `src/rfdetr/lit/__init__.py` (complete `build_trainer()` factory), `src/rfdetr/config.py` (promote `clip_max_norm`, `seed`, `sync_bn`, `fp16_eval`; remove `device`), `src/rfdetr/main.py` (`populate_args()` → deprecated shim), `pyproject.toml` (entry point → `rfdetr.lit.cli:main`; keep legacy compat adapter), `tests/` (CLI smoke, YAML roundtrip)
**Deletes:** nothing

Assemble the trainer factory, replace the argparse CLI with `LightningCLI`, and clean up config mapping while keeping compatibility shims required by tests/benchmarks.

- [ ] Implement `build_trainer(config, model_config)` in `lit/__init__.py` (see Phase 4.1)
- [ ] Add `clip_max_norm`, `seed`, `sync_bn`, `fp16_eval` as promoted fields to `TrainConfig`; remove `device` field
- [ ] Keep `populate_args()` as a deprecated compatibility shim until benchmark tests no longer import it
- [ ] Implement `RFDETRCli` subclass of `LightningCLI` in `lit/cli.py`
- [ ] Update `pyproject.toml` entry point to `rfdetr.lit.cli:main`
- [ ] Keep a compatibility adapter for legacy `rfdetr.cli.main:trainer` entry point for one deprecation cycle
- [ ] Add example YAML config files for each detection model size (nano, small, medium, base, large)
- [ ] Add example YAML config files for each segmentation model size

**Milestone:** `rfdetr fit/validate/--help` all work; legacy CLI entry points emit deprecation warnings. Tests: CLI smoke, YAML roundtrip, `TrainConfig.device` removal. **CI gate: full CPU test suite green before Chapter 5.**

---

### Chapter 5: Add deprecation shims to the existing public API _(Phases 7 + 8)_

**Covers:** Phase 7 (Top-Level API Compatibility — `RFDETR.train()` delegation, `detr.py` compat layer), Phase 8 (Checkpoint Converter — `convert_legacy_checkpoint()`, `on_load_checkpoint()` auto-detect)

**Pre-conditions:** Chapter 4 PR merged and CPU CI green.
**Creates:** `src/rfdetr/lit/checkpoint.py` (`convert_legacy_checkpoint()`, `on_load_checkpoint()` auto-detect)
**Modifies:** `src/rfdetr/detr.py` (`RFDETR.train()` delegates to `build_trainer().fit()`; absorbs `device=` kwarg silently), `src/rfdetr/__init__.py` (additionally export `RFDETRModule`, `RFDETRDataModule`, `build_trainer`), `tests/` (`.train()` shim, converter, full public API smoke)
**Deletes:** nothing

Implement the `RFDETR.train()` internal delegation to PTL and add the `@deprecated` decorator for internal shims only. The `RFDETR*` class hierarchy is NOT deprecated.

- [ ] Add `@deprecated` function/class decorator to `utilities/decorators.py`
- [ ] `RFDETR` class hierarchy is NOT deprecated — kept as primary API
- [ ] Implement `RFDETR.train()` to delegate to `build_trainer().fit()` internally (see Phase 4.2); absorb the `device=` kwarg silently
- [ ] `RFDETR.export()` and `RFDETR.deploy_to_roboflow()` — kept as-is (no deprecation)
- [ ] Implement checkpoint converter `convert_legacy_checkpoint()` and `on_load_checkpoint()` auto-detect (see Phase 8)
- [ ] Update `__init__.py` to additionally export `RFDETRModule`, `RFDETRDataModule`, `build_trainer`
- [ ] Verify all existing usage patterns (`.train()`, `.predict()`, `.export()`) still work unchanged

**Milestone:** `RFDETR*` API works as before (no deprecation warnings); `.train()` delegates to PTL internally; `convert_legacy_checkpoint()` converts `.pth` → `.ckpt`. Tests: `.train()` shim, converter. **CI gate: full CPU test suite green before Chapter 6.**

---

### Chapter 6: Delete replaced code and update tests _(Phase 9)_

**Covers:** Phase 9 (Cleanup & Deletion — remove legacy `engine.py` internals, `main.py`, `coco_eval.py`, obsolete `util/` symbols, legacy metric sinks)

**Pre-conditions:** Chapter 5 PR merged and CPU CI green; benchmark tests migrated off all legacy APIs.
**Creates:** nothing
**Modifies:** `src/rfdetr/engine.py` (delete `train_one_epoch()` internals, obsolete AMP helpers), `src/rfdetr/util/misc.py` (delete `MetricLogger`, `SmoothedValue`, `init_distributed_mode`, `reduce_dict`, `get_rank`, `is_main_process`, `save_on_master`; **keep** `get_world_size`, `is_dist_avail_and_initialized`), `src/rfdetr/util/metrics.py` (delete legacy `Metrics*Sink` classes), `src/rfdetr/datasets/__init__.py` (delete `get_coco_api_from_dataset()`), `src/rfdetr/lit/cli.py` (`link_arguments` narrowed to field-level paths after Phase 6.3 constructor decomposition), `tests/` (migrate all tests to PTL entry points)
**Deletes:** `src/rfdetr/main.py`, `src/rfdetr/datasets/coco_eval.py`, `src/rfdetr/util/early_stopping.py`

Remove all code that has been superseded by the PTL stack, update tests to exercise the new paths, and confirm the package is in a clean state.

- [x] Promote `device` kwarg to a proper `TrainConfig.accelerator` field (see Decision #11 / risk table row 4):
  - `TrainConfig.accelerator` is now first-class and consumed by `build_trainer()`. The legacy `.train(device=...)` shim remains for compatibility (`"cpu"` maps to `accelerator="cpu"`; other legacy values emit deprecation warnings and defer to PTL/device env configuration).
  - Compatibility target is RF-DETR `1.5` only; intermediate migration-stage compatibility is not a requirement.
- [ ] Migrate benchmark tests to PTL-native/compat APIs while preserving current acceptance thresholds:
  - [ ] `tests/benchmarks/test_coco_inference.py`
  - [ ] `tests/benchmarks/test_synthetic_convergence.py`
- [ ] Keep legacy metric output schema in compatibility wrappers until the benchmark migration above is complete
- [ ] Delete `src/rfdetr/main.py` (`Model` class and `populate_args()`)
- [ ] Delete `src/rfdetr/datasets/coco_eval.py` (`CocoEvaluator` and all helpers)
- [ ] Delete `src/rfdetr/util/early_stopping.py`
- [ ] Delete legacy `train_one_epoch()` internals and obsolete AMP helpers from `engine.py`; keep/remove compatibility wrappers based on benchmark migration completion
- [ ] Delete `MetricLogger`, `SmoothedValue`, `init_distributed_mode()`, `reduce_dict()`, `get_rank()`, `is_main_process()`, `save_on_master()` from `util/misc.py` — **keep** `get_world_size()`/`is_dist_avail_and_initialized()` (used by `SetCriterion`; see Decision #13)
- [ ] Delete `MetricsTensorBoardSink`, `MetricsWandBSink`, `MetricsMLFlowSink`, `MetricsClearMLSink` from `util/metrics.py`
- [ ] Delete `get_coco_api_from_dataset()` from `datasets/__init__.py` only when no remaining imports in `src/`, `tests/`, and docs examples
- [ ] Update all existing tests to use the new PTL-based entry points
- [ ] Confirm full test suite passes
- [ ] Remove legacy metric comparison scaffolding that was used in Chapter 2 (now superseded by the torchmetrics-based performance tests)
- [ ] Decompose `RFDETRDataModule.__init__(model_config, train_config)` into individual parameters (Phase 6.3); then narrow `RFDETRCli.link_arguments` from full-object linking (`model.model_config` → `data.model_config`) to field-level linking per the original spec (`model.model_config.resolution` → `data.resolution`, `model.model_config.patch_size` → `data.patch_size`)

**Milestone:** No deleted-module imports remain; benchmark thresholds hold; `pycocotools.cocoeval` unused in training/eval path; full test suite green; `pip install .` clean; detection/segmentation mAP/F1 match pre-migration baselines. **CI gate: full CPU + GPU test suites green; migration complete.**

---

## ⚠️ Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| mAP numbers differ between torchmetrics MAP and pycocotools COCOeval | High | Compare on reference dataset before removing old evaluator; torchmetrics uses COCO-identical algorithm |
| `build_matching_data()` F1 sweep differs from `coco_extended_metrics()` | High | Test against known-good outputs from current implementation before cutting over; lock numeric tolerances in tests |
| F1 is computed per-rank instead of globally under DDP | High | Merge compact matching data across ranks before sweeping confidence thresholds |
| EMA behavior differs from current `ModelEma` | Medium | Use PTL 2.6+ `WeightAveraging` with custom `avg_fn` (tau warmup) and validate weight parity after N steps |
| EMA callback used with sharded strategy | Medium-High | Detect FSDP/DeepSpeed-style strategies and disable EMA (or provide a dedicated strategy-aware implementation) |
| Callback ordering assumptions create race in best-model selection | Medium | Make best-model selection read `trainer.callback_metrics` in `on_validation_end()`, not order-coupled hooks |
| Gradient accumulation semantics differ | **Critical** | Current code loads `effective_batch_size` into DataLoader and manually splits; PTL loads `batch_size` and delays `optimizer.step()`. DataLoader batch size MUST change from `effective_batch_size` to `batch_size`. Verify gradient magnitudes match on a synthetic run. |
| Multi-scale resize with `random.seed(global_step)` differs from current step counting | Medium | Verify scales match exactly on deterministic run |
| `NestedTensor` + PTL's `on_after_batch_transfer` device handling | Low | PTL respects custom `collate_fn`; verify `NestedTensor.to(device)` called correctly |
| `TrainConfig.device` removal breaks existing user code | Low-Medium | Document in release notes; `@deprecated` shim absorbs `device=` kwarg silently |
| segmentation mask format for torchmetrics (`bool` tensor vs RLE) | Medium | Verify `PostProcess` mask output is compatible; add `bool()` cast if needed |
| Benchmark regressions during migration (`tests/benchmarks/*`) | High | Keep legacy stats schema wrappers until benchmark tests are fully migrated and green |
| `get_world_size`/`is_dist_avail_and_initialized` deleted from `util/misc.py` | High | Imported by `models/lwdetr.py` for `num_boxes` all_reduce — not engine boilerplate. Keep in `util/misc.py`; never delete. |

---

## ✅ Resolved Design Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | EMA callback | Use PTL 2.6+ `WeightAveraging`/`EMAWeightAveraging` APIs; implement strict parity via custom `avg_fn` and avoid private internals |
| 2 | Per-epoch regular vs EMA metrics | Preserve both regular and EMA metrics/checkpoints for compatibility (`checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_total.pth`) |
| 3 | torchmetrics backend | `faster_coco_eval` for segmentation; omitted (torchvision default) for detection-only |
| 4 | `MeanAveragePrecision` creation | `setup()` hook — correct device placement after DDP init |
| 5 | `class_names` propagation | `COCOEvalCallback.on_fit_start()` reads `trainer.datamodule.class_names` |
| 6 | F1 sweep crowd handling | Pass `iscrowd` from targets; defaults to absent for YOLO/Roboflow datasets; compute global F1 across ranks |
| 7 | `NestedTensor` device transfer | Override `RFDETRModule.transfer_batch_to_device()` explicitly |
| 8 | `gradient_checkpointing` | Wired in `RFDETRModule.__init__()` after `build_model()` |
| 9 | Test-set eval after training | `trainer.test()` called from `BestModelCallback.on_fit_end()` when `run_test=True` |
| 10 | `callbacks` dict backward compat | Deprecated shim emits extra warning if non-empty; no bridging |
| 11 | `TrainConfig.device` | Dropped (breaking change); absorbed silently in deprecated shim |
| 12 | `fp16_eval` | Mapped to PTL precision policy (`bf16-mixed` when supported, otherwise `16-mixed`) |
| 13 | `SetCriterion` `num_boxes` all_reduce | **Keep as-is.** `SetCriterion` calls `all_reduce(num_boxes)` + `/ get_world_size()` before every loss to produce a globally-consistent denominator across ranks — essential for DDP stability. PTL uses the same process groups so this works unchanged. Refactoring into `training_step()` still needs `torch.distributed` in two places, changes the criterion API, and gains nothing. `get_world_size`/`is_dist_avail_and_initialized` stay in `util/misc.py`; never deleted. |
| 14 | `RFDETRCli.link_arguments` paths | **Keep full-object linking** (`model.model_config` → `data.model_config`, `model.train_config` → `data.train_config`, `apply_on="parse"`) until after user testing. The spec's field-level paths (`data.resolution`, `data.patch_size`) require decomposing `RFDETRDataModule(model_config, train_config)` into individual params (Phase 6.3). That decomposition is deferred to Chapter 6 to avoid breaking changes before the API is validated with real users. |

## 📝 Notes for Implementers

**`build_matching_data()` implementation notes** (the most non-trivial new piece of code):

- Lives in `engine.py` alongside `sweep_confidence_thresholds()`
- IoU via `torchvision.ops.box_iou` for bbox and boolean-mask IoU for segmentation
- Matching: greedy highest-score-first, each GT matched at most once (identical to COCO algorithm)
- Fixed IoU threshold 0.5 (mirrors current `iou50_idx` extraction)
- `iscrowd` from targets: crowd GT instances excluded from denominator; `dtIgnore` set for detections matched to crowd instances — for YOLO/Roboflow datasets `iscrowd` key is absent and defaults to all-zero
- Return compact per-class arrays that can be merged across DDP ranks before confidence sweeping

**`RFDETREMACallback` tau warmup implementation note:**

The strict-parity path should keep the current `ModelEma` formula:
`effective_decay = decay * (1 - exp(-updates / tau))` when `tau > 0`; otherwise `effective_decay = decay`.
Implement this through PTL's `WeightAveraging(avg_fn=...)` hook and verify parity against current `ModelEma` on a fixed synthetic run.
