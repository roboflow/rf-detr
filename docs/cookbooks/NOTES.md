# Notebooks

Each `.ipynb` file here is rendered as a page under `/cookbooks/` in the docs site.

Cards on the cookbooks landing page are driven by [`cards.yaml`](cards.yaml). The MkDocs hook
`docs/hooks/cookbooks_cards.py` loads that file and exposes it to `docs/theme/notebooks.html`,
which renders each entry as a card via a Jinja loop.

## Converting a jupytext `.py` to `.ipynb`

Cookbook source files live as jupytext percent-format `.py` scripts (e.g. `fine-tune_keypoints.py` at the project root). Each script has **two output notebooks** that must be regenerated together after every edit:

```bash
# Runnable copy (for users who want to run the notebook directly)
jupytext --to notebook fine-tune_keypoints.py --output notebooks/fine-tune_keypoints.ipynb

# Docs render copy (served by mkdocs-jupyter at /cookbooks/)
jupytext --to notebook fine-tune_keypoints.py --output docs/cookbooks/fine-tune_keypoints.ipynb
```

Both commands must be run to keep the two copies in sync. Omit `--execute` — notebooks are rendered statically by `mkdocs-jupyter` with `execute: false`, so pre-run outputs in the `.ipynb` are displayed as-is.

If jupytext is not installed: `pip install jupytext` (or `uv add jupytext --dev`).

## Adding a notebook

1. Add the `.ipynb` file here, named after its content (e.g. `custom-augmentations.ipynb`, `onnx-export.ipynb`).
2. Add a new entry to `docs/cookbooks/cards.yaml` under the `cards:` list:

<!-- prettier-ignore -->

```yaml
  - href: content-slug/
    name: Short Title
    labels: [LABEL1, LABEL2]
    version: vX.Y.0
    author: GitHubUsername
    description: One sentence describing what the notebook demonstrates.
```

Available labels (reuse these to keep tags standardised): `TRAINING`, `AUGMENTATION`, `EXPORT`, `TFLITE`, `PYTORCH LIGHTNING`, `INFERENCE`, `SEGMENTATION`, `DEPLOY`.
Current tag colours are assigned dynamically by the docs UI, so they may change if cards or labels are added or reordered.

## Removing a notebook

1. Delete the `.ipynb` file.
2. Remove the matching entry (the `- href: content-slug/` block) from `docs/cookbooks/cards.yaml`.

## Current notebooks

| File                         | Card title                                      | Version |
| ---------------------------- | ----------------------------------------------- | ------- |
| `custom-augmentations.ipynb` | Custom Augmentations and Live Training Progress | v1.5.0  |
| `fine-tune_keypoints.ipynb`  | Fine-Tune RF-DETR Keypoint Detection            | v1.8.0  |
| `pytorch-lightning.ipynb`    | Training with PyTorch Lightning                 | v1.6.0  |
