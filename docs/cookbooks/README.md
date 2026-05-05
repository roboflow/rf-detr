# Notebooks

Each `.ipynb` file here is rendered as a page under `/cookbooks/` in the docs site.

## Adding a notebook

1. Add the `.ipynb` file here, named `release-demo_<version>.ipynb` (e.g. `release-demo_1-8.ipynb`).
2. Open `docs/theme/notebooks.html` and add a new card inside `<div class="custom-grid">`:

<!-- prettier-ignore -->

```html
<a href="release-demo_X-Y/">
 <p class="card repo-card" data-author="GitHubUsername" data-description="One sentence describing what the notebook demonstrates." data-labels="LABEL1,LABEL2" data-name="Short Title" data-version="vX.Y.0">
 </p>
</a>
```

Available labels (reuse for consistent tag colouring): `TRAINING`, `AUGMENTATION`, `EXPORT`, `TFLITE`, `PYTORCH LIGHTNING`, `INFERENCE`, `SEGMENTATION`, `DEPLOY`.

## Removing a notebook

1. Delete the `.ipynb` file.
2. Remove the matching `<a href="...">...</a>` block from `docs/theme/notebooks.html`.

## Current notebooks

| File                     | Card title                                      | Version |
| ------------------------ | ----------------------------------------------- | ------- |
| `release-demo_1-5.ipynb` | Custom Augmentations and Live Training Progress | v1.5.0  |
| `release-demo_1-6.ipynb` | PyTorch Lightning Building Blocks               | v1.6.0  |
