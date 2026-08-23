---
description: RF-DETR dataset format guide for COCO JSON and YOLO. Auto-detection, directory structure, annotation schemas, and format conversion.
---

# Dataset Formats

RF-DETR supports training on datasets in two popular formats: **COCO** and **YOLO**. The format is automatically detected based on your dataset's directory structure—simply pass your dataset directory to the `train()` method.

## Automatic Format Detection

When you call `model.train(dataset_dir=<path>)`, RF-DETR checks the following:

1. **COCO format**: Looks for `train/_annotations.coco.json`
2. **YOLO format**: Looks for `data.yaml` (or `data.yml`) and `train/images/` directory

If neither format is detected, an error is raised with instructions on what's expected.

!!! tip "Roboflow Export"

    [Roboflow](https://roboflow.com/annotate) can export datasets in both COCO and YOLO formats. When downloading from Roboflow, select the appropriate format based on your preference.

---

## COCO Format

COCO (Common Objects in Context) format uses JSON files to store annotations in a structured format with images, categories, and annotations.

### Directory Structure

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

### Annotation File Structure

Each `_annotations.coco.json` file contains:

```json
{
  "info": {
    "description": "Dataset description",
    "version": "1.0"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cat",
      "supercategory": "animal"
    },
    {
      "id": 2,
      "name": "dog",
      "supercategory": "animal"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [
        100,
        150,
        200,
        180
      ],
      "area": 36000,
      "iscrowd": 0
    }
  ]
}
```

#### Key Fields

| Field         | Description                                                           |
| ------------- | --------------------------------------------------------------------- |
| `images`      | List of image metadata including `id`, `file_name`, `width`, `height` |
| `categories`  | List of object categories with `id` and `name`                        |
| `annotations` | List of object annotations linking images to categories               |
| `bbox`        | Bounding box in `[x, y, width, height]` format (top-left corner)      |
| `area`        | Area of the bounding box                                              |
| `iscrowd`     | 0 for individual objects, 1 for crowd regions                         |

### Segmentation Annotations

For training segmentation models, your COCO annotations must include a `segmentation` key with polygon coordinates:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [
    100,
    150,
    200,
    180
  ],
  "area": 36000,
  "iscrowd": 0,
  "segmentation": [
    [
      100,
      150,
      150,
      150,
      200,
      200,
      150,
      250,
      100,
      200
    ]
  ]
}
```

The `segmentation` field contains a list of polygons, where each polygon is a flat list of coordinates: `[x1, y1, x2, y2, x3, y3, ...]`.

---

### Keypoint Annotations

For training the keypoint preview model, use COCO JSON keypoint annotations. Roboflow-style COCO exports are supported when the split files are named `train/_annotations.coco.json` and `valid/_annotations.coco.json`.

Each keypoint annotation must include a bounding box plus COCO keypoint fields:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 0,
  "bbox": [
    100,
    150,
    200,
    180
  ],
  "area": 36000,
  "iscrowd": 0,
  "num_keypoints": 17,
  "keypoints": [
    110,
    160,
    2,
    125,
    158,
    2
  ]
}
```

The category should declare the keypoint schema:

```json
{
  "id": 0,
  "name": "person",
  "supercategory": "person",
  "keypoints": [
    "nose",
    "left_eye",
    "right_eye"
  ],
  "skeleton": []
}
```

The `keypoints` array above is shortened for readability. In a valid COCO person-keypoint annotation it contains `17 * 3` values: `x`, `y`, and visibility for each keypoint.

The keypoint preview model is pretrained on COCO person-style keypoints. Its default COCO schema is `[17]`, so keypoint-bearing categories are mapped onto the active keypoint label slot during COCO loading. Legacy checkpoints may still report a background-first `[0, 17]` schema, which RF-DETR accepts for compatibility. Custom keypoint training can also use YOLO pose labels, described below.

---

## YOLO Format

YOLO format uses separate text files for each image's annotations and a `data.yaml` configuration file that defines class names.

### Directory Structure

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── test/
    ├── images/
    │   ├── image1.jpg
    │   └── ...
    └── labels/
        ├── image1.txt
        └── ...
```

### data.yaml Configuration

The `data.yaml` file at the root of your dataset directory defines the class names:

```yaml
names:
  - cat
  - dog
  - bird

nc: 3

train: train/images
val: valid/images
test: test/images
```

| Field                  | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `names`                | List of class names (0-indexed)                    |
| `nc`                   | Number of classes                                  |
| `train`, `val`, `test` | Paths to image directories (relative to data.yaml) |

!!! note "Alternative format"

    Some YOLO datasets use a dictionary format for names:

    ```yaml
    names:
      0: cat
      1: dog
      2: bird
    ```

    Both formats are supported.

### Label File Format

Each image has a corresponding `.txt` file in the `labels/` directory with the same base name. Each line in the label file represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`image1.txt`):

```
0 0.5 0.4 0.3 0.2
1 0.2 0.6 0.15 0.25
```

#### Coordinate Format

| Field      | Range        | Description                                     |
| ---------- | ------------ | ----------------------------------------------- |
| `class_id` | 0, 1, 2, ... | Zero-indexed class ID from `names` in data.yaml |
| `x_center` | 0.0 - 1.0    | Normalized x-coordinate of bounding box center  |
| `y_center` | 0.0 - 1.0    | Normalized y-coordinate of bounding box center  |
| `width`    | 0.0 - 1.0    | Normalized width of bounding box                |
| `height`   | 0.0 - 1.0    | Normalized height of bounding box               |

All coordinates are normalized relative to image dimensions. For example, if an image is 640×480 pixels and the bounding box center is at (320, 240):

- `x_center` = 320 / 640 = 0.5
- `y_center` = 240 / 480 = 0.5

### Segmentation Labels (YOLO-Seg)

For segmentation, YOLO format extends the label format with polygon coordinates:

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

**Example** (`image1.txt` with segmentation):

```
0 0.1 0.2 0.3 0.2 0.4 0.5 0.2 0.6 0.1 0.4
```

The coordinates after the class ID represent the polygon vertices in normalized format.

---

### Pose Labels (YOLO Pose)

For keypoint preview training, RF-DETR supports Ultralytics YOLO pose labels in the same directory layout shown above. The `data.yaml` file must declare `kpt_shape`:

```yaml
names:
  0: person

kpt_shape: [17, 3] # [number_of_keypoints, dimensions]; dimensions must be 2 or 3
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
```

`kpt_names` is optional. When omitted, RF-DETR creates placeholder names such as `keypoint_0`. `flip_idx` is an Ultralytics-style length-`K` permutation used to infer RF-DETR's flat `keypoint_flip_pairs` for horizontal-flip augmentation.

Each pose label row contains a bounding box followed by keypoints:

```text
<class_id> <x_center> <y_center> <width> <height> <px1> <py1> <v1> ... <pxK> <pyK> <vK>
```

For `kpt_shape: [K, 2]`, omit the visibility value:

```text
<class_id> <x_center> <y_center> <width> <height> <px1> <py1> ... <pxK> <pyK>
```

All box and keypoint coordinates are normalized to `[0, 1]`. RF-DETR converts keypoints to COCO-style `(x, y, visibility)` tensors internally. For `[K, 3]`, the visibility values are preserved. For `[K, 2]`, visibility is synthesized: nonzero points are marked visible (`2`) and `(0, 0)` points are marked absent (`0`).

Use the YOLO schema helper when you want to configure a model explicitly:

```python
from pathlib import Path

from rfdetr import RFDETRKeypointPreview
from rfdetr.datasets._keypoint_schema import infer_yolo_keypoint_schema

DATASET_DIR = Path("/path/to/yolo-pose-dataset")
schema = infer_yolo_keypoint_schema(DATASET_DIR / "data.yaml")

model = RFDETRKeypointPreview(
    num_classes=len(schema.class_names),
    num_keypoints_per_class=schema.num_keypoints_per_class,
)

model.train(
    dataset_file="yolo",
    dataset_dir=str(DATASET_DIR),
    class_names=schema.class_names,
    keypoint_oks_sigmas=schema.keypoint_oks_sigmas,
)
```

!!! note "flip_idx and keypoint_flip_pairs"

    `flip_idx` is a permutation, while `keypoint_flip_pairs` is a flat pair list. During `model.train()`, RF-DETR infers the pair list automatically from `flip_idx` when no explicit `keypoint_flip_pairs` is provided.

---

## WebDataset Shards (Sequential I/O)

A COCO or YOLO split stored as loose files costs one `open()` per image per epoch. Raising `num_workers` to keep a GPU fed multiplies that into thousands of concurrent random opens — cheap against a local NVMe disk, expensive against network storage, an object-store mount, or any filesystem whose metadata operations are slow.

`dataset_file="webdataset"` reads the same images from a handful of `.tar` shards instead. Each worker walks its own shards front to back, so an epoch becomes a set of large sequential reads and one `open()` per shard rather than one per image. Augmentation is unchanged: shards decode into the same `(image, target)` pairs the loose-file loader produces, go through the same CPU Albumentations/torchvision pipeline, and reach the GPU through the same `pin_memory` hand-off and Kornia stage.

This is an alternative input path, not a replacement — `coco`, `roboflow` and `yolo` behave exactly as before.

!!! note "It is not a general speed-up"

    Whether packing helps depends on where your input pipeline is actually blocked. Measured on a 32-vCPU machine with an NVMe-attached persistent disk, on COCO 2017, throughput was the same either way — between 0.92x and 1.03x across six configurations. That pipeline was limited by decode and augmentation, not by file access: `iowait` stayed under 1%, and raw reads on the same disk sustained roughly 5.5x the images per second the loader consumed. Packing pays off where per-file access genuinely is the constraint — network filesystems, object-store mounts, directories with millions of small files. What it improves regardless is construction.

### What it does change, everywhere

Building the dataset stops parsing the split's annotation file. `dataset_file="coco"` loads `instances_train2017.json` into a `pycocotools` index before the first batch; the streaming path reads a small JSON listing shards, sample count and categories, and takes each image's annotations from the shard next to its pixels.

On full COCO 2017 that is a median 19.9 s and 2,810 MB of resident memory before the first batch, against 0.012 s and 0.9 MB (three cold runs per arm; the spread is under 0.13 s and RSS is identical to the tenth of a MB). Both are per process, so per rank under DDP.

### Packing a split

Shards are written with the standard library, so packing needs no extra dependency:

```bash
python -m rfdetr.datasets.webdataset_io \
    --image-dir /data/coco/train2017 \
    --annotations /data/coco/annotations/instances_train2017.json \
    --output-dir /data/coco-shards \
    --split train
```

Repeat once per split, into the same `--output-dir`:

```
coco-shards/
├── train-000000.tar        # ~100 MB each, override with --max-shard-mb
├── train-000001.tar
├── ...
├── train-index.json        # shard list, sample count, categories
├── val-000000.tar
└── val-index.json
```

Image bytes are copied verbatim — no re-encode — so a packed split decodes to exactly the same pixels as the directory it came from. A `.json` member next to each image carries that image's `image_id`, `file_name` and annotation list, segmentation polygons included.

`--category-ids` chooses the label space, and the index records the choice so the loader never has to guess:

| Value             | Labels                                                       | Matches                   |
| ----------------- | ------------------------------------------------------------ | ------------------------- |
| `remap` (default) | contiguous `0..N-1`, unannotated grouping categories dropped | `dataset_file="roboflow"` |
| `raw`             | the source `category_id` values                              | `dataset_file="coco"`     |

### Training from shards

```python
from rfdetr import RFDETRSmall

# num_classes is a model argument, not a train() one. 80 matches the `--category-ids remap`
# default the packing command above used; see below for the `raw` formula.
model = RFDETRSmall(num_classes=80)
model.train(
    dataset_dir="/data/coco-shards",
    dataset_file="webdataset",
    epochs=10,
    batch_size=16,
    num_workers=16,
)
```

Install the reader with `pip install "rfdetr[webdataset]"`.

`num_classes` has to be given explicitly: the auto-detection the other formats use looks for `train/_annotations.coco.json` or `data.yaml`, and a shard directory has neither, so the model keeps whatever count it was built with. Read the right number from `train-index.json` — under `raw`, that is the **highest `id` in `categories`, plus one** (COCO's own ids run 1-90 with gaps for its 80 categories, so `len(categories)` undercounts it — the same `max_obj_id + 1` convention `dataset_file="coco"` uses); under `remap`, it is simply the number of categories left after grouping nodes are dropped. Class *names* need no such help: they are read from the same index, indexed by the same labels the loader emits, so checkpoints and `predict()` label output are unaffected.

### How an epoch is sized

A streaming dataset has no index to sample from, so the two loaders size an epoch differently:

- **Training** gives every worker a fixed number of samples, floored to a whole number of *accumulation windows* (`batch_size × grad_accum_steps`), so a partial window at the tail never fires the optimizer early. The epoch length is then exact, which is what the LR schedule needs — it is derived from `trainer.estimated_stepping_batches`. The cost is that a worker holding fewer shards than average repeats some of its own samples inside the epoch.
- **Validation and test** let every worker drain its shards once, so the split is scored exactly once with nothing repeated or dropped. Those loaders report no length, so their progress bar shows no total.

The test stage uses a packed `test` split when the directory has one, and falls back to `val` with a log line when it does not — a shard directory only has a split someone packed deliberately.

Training raises a `ValueError` rather than starting if `world_size × num_workers` exceeds the shard count, because a worker left without a shard would silently shorten every epoch. Pack with a smaller `--max-shard-mb` to get more shards, or lower `num_workers`.

### Pack plenty of shards

Shards are split across workers by count, so an uneven split leaves the worst-served worker short of the samples a fixed epoch asks of it: it repeats some of its own while better-supplied workers leave some unseen. This is measurable. On a 2,000-image, 3-epoch fine-tune at `num_workers=8`, packing into 39 shards — the short worker 18% under the average — scored about 0.011 `mAP@50:95` below the loose-file loader; re-packing the identical split into 290 shards, 0.6% under, matched it. The loader logs a warning when that shortfall passes 5%, measured from each shard's real sample count (shards are cut by byte size, so a count-only estimate can be badly wrong when image sizes vary a lot within a split).

Aim for a shard count that divides `world_size × num_workers`, or simply for many more shards than workers. `--max-shard-mb` is the knob.

### Shuffling

Shard visiting order is shuffled, and samples are shuffled again in a reservoir buffer as they stream. That is a local shuffle, not the global permutation `shuffle=True` gives a map-style loader: two samples in the same shard stay more likely to land in the same epoch region. Both are reseeded every epoch from PyTorch's own per-epoch worker seeding, so `seed_everything` governs them the same way it governs the rest of training.

### Not covered

Keypoint training rejects `dataset_file="webdataset"` with an explicit error. Its label space is inferred from a whole parsed COCO annotation file, which a shard index does not carry — use `coco`, `roboflow` or `yolo` for keypoints. Detection and segmentation splits are supported.

---

## Converting Between Formats

### YOLO to COCO

You can use the [supervision](https://github.com/roboflow/supervision) library to convert datasets:

```python
import supervision as sv

# Load YOLO dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="path/to/images",
    annotations_directory_path="path/to/labels",
    data_yaml_path="path/to/data.yaml",
)

# Save as COCO
dataset.as_coco(images_directory_path="output/images", annotations_path="output/annotations.json")
```

### COCO to YOLO

```python
import supervision as sv

# Load COCO dataset
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="path/to/images", annotations_path="path/to/annotations.json"
)

# Save as YOLO
dataset.as_yolo(
    images_directory_path="output/images", annotations_directory_path="output/labels", data_yaml_path="output/data.yaml"
)
```

### Using Roboflow

[Roboflow](https://roboflow.com) provides a web interface to:

1. Upload datasets in any format
2. Annotate new images or edit existing annotations
3. Export in COCO, YOLO, or other formats

This is often the easiest way to convert between formats while also having the option to augment your data.

---

## Which Format Should I Use?

Both formats work equally well with RF-DETR. Choose based on your workflow:

| Consideration                     | COCO                       | YOLO                    |
| --------------------------------- | -------------------------- | ----------------------- |
| **Annotation storage**            | Single JSON file per split | One text file per image |
| **Human readability**             | JSON structure, verbose    | Simple text, compact    |
| **Other framework compatibility** | DETR family, MMDetection   | Ultralytics YOLO        |
| **Segmentation support**          | Full polygon support       | Full polygon support    |
| **Editing annotations**           | Requires JSON parsing      | Simple text editing     |

!!! tip "Recommendation"

    If you're exporting from Roboflow or already have a dataset in one format, simply use that format. RF-DETR handles both identically.

---

## Troubleshooting

### Format Detection Fails

If you see an error like:

```
Could not detect dataset format in /path/to/dataset
```

Check that:

**For COCO format:**

- `train/_annotations.coco.json` exists
- The JSON file is valid

**For YOLO format:**

- `data.yaml` or `data.yml` exists at the root
- `train/images/` directory exists with images

### Empty Annotations

If images have no objects, handle them as follows:

**COCO format:** Include the image in the `images` array but don't add any annotations for it.

**YOLO format:** Create an empty `.txt` file (0 bytes) for the image, or omit the label file entirely.

### Class ID Mismatch

**COCO format:** Category IDs in annotations must match IDs defined in the `categories` array.

**YOLO format:** Class IDs in label files must be valid indices (0 to `nc-1`) based on the `names` list in `data.yaml`.
