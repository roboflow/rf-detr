# TODOs: Replace DINOv2 backbone with DINOv3 in RF-DETR

This document lists the required changes to integrate **DINOv3** as a backbone in place of **DINOv2**.

---

## 1. Dependencies
- [x] Add Hugging Face `transformers>=4.40` or enable `torch.hub.load()` for DINOv3 models.
- [ ] Update deployment scripts to download DINOv3 checkpoints (official links from the [DINOv3 repo](https://github.com/facebookresearch/dinov3).

---

## 2. Configuration
- [x] In `rfdetr/config.py`, extend the `encoder` literal to include `"dinov3_base"`, `"dinov3_small"`, etc.
- [x] Define DINOv3 defaults:
  - `patch_size = 16`
  - `out_feature_indexes = [2,5,8,11]` (for 12-layer ViT-B/16 backbone).
  - `positional_encoding_size = resolution / patch_size`
  - `hidden_dim = 768` (for `dinov3-vitb16`).
  - Update `projector_scale` accordingly.

---

## 3. DINOv3 Backbone Class
- [x] Create `rfdetr/models/backbone/dinov3.py`.
- [x] Implement a `DinoV3` class:
  - Load backbone via:
    ```python
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        output_hidden_states=True
    )
    ```
    or via:
    ```python
    torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16")
    ```
  - Extract intermediate hidden states (`out_feature_indexes`) and reshape to `(B, C, H, W)`.
  - Set `_out_feature_channels = [768] * len(out_feature_indexes)`.
  - Implement `export()` to handle positional embeddings resize.

---

## 4. Backbone Wrapper
- [x] In `rfdetr/models/backbone/backbone.py`:
  - Allow `encoder` names beginning with `"dinov3"`.
  - Dispatch to `DinoV3` when prefix is `"dinov3"`.
- [x] Add a mapping:
  ```python
    size_to_width_dinov3 = {
        "small": 384,
        "base": 768,
        "large": 1024,
    }
    ```
  - [x] Implement get_dinov3_lr_decay_rate() and get_dinov3_weight_decay_rate() similar to the DINOv2 versions, adapted to DINOv3's parameter naming (blocks.0, blocks.1, …).

---

## 5. Training and Data Pipeline
- [ ] Update input transforms to use **ImageNet normalization**:
  ```python
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
- [ ] If using satellite-pretrained weights, use alternate normalization.
- [ ] Update positional encoding logic to match 16×16 patch size.
- [ ] Adjust detection head/projector assumptions about feature dimensions.

## 6. Registration

- [x] Update rfdetr/models/backbone/__init__.py to import and expose DinoV3.
- [x] Update CLI help/docs to mention new "dinov3_*" encoder options.

## 7. Testing

- [ ] Train/evaluate on a small dataset to validate:
  - Backbone returns correct feature map shapes.
  - Projector merges features properly.
  - Learning rate decay works as expected.
