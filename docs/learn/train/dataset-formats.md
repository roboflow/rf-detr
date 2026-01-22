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
      "bbox": [100, 150, 200, 180],
      "area": 36000,
      "iscrowd": 0
    }
  ]
}
```

#### Key Fields

| Field | Description |
|-------|-------------|
| `images` | List of image metadata including `id`, `file_name`, `width`, `height` |
| `categories` | List of object categories with `id` and `name` |
| `annotations` | List of object annotations linking images to categories |
| `bbox` | Bounding box in `[x, y, width, height]` format (top-left corner) |
| `area` | Area of the bounding box |
| `iscrowd` | 0 for individual objects, 1 for crowd regions |

### Segmentation Annotations

For training segmentation models, your COCO annotations must include a `segmentation` key with polygon coordinates:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 150, 200, 180],
  "area": 36000,
  "iscrowd": 0,
  "segmentation": [[100, 150, 150, 150, 200, 200, 150, 250, 100, 200]]
}
```

The `segmentation` field contains a list of polygons, where each polygon is a flat list of coordinates: `[x1, y1, x2, y2, x3, y3, ...]`.

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

| Field | Description |
|-------|-------------|
| `names` | List of class names (0-indexed) |
| `nc` | Number of classes |
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

| Field | Range | Description |
|-------|-------|-------------|
| `class_id` | 0, 1, 2, ... | Zero-indexed class ID from `names` in data.yaml |
| `x_center` | 0.0 - 1.0 | Normalized x-coordinate of bounding box center |
| `y_center` | 0.0 - 1.0 | Normalized y-coordinate of bounding box center |
| `width` | 0.0 - 1.0 | Normalized width of bounding box |
| `height` | 0.0 - 1.0 | Normalized height of bounding box |

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

## Converting Between Formats

### YOLO to COCO

You can use the [supervision](https://github.com/roboflow/supervision) library to convert datasets:

```python
import supervision as sv

# Load YOLO dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="path/to/images",
    annotations_directory_path="path/to/labels",
    data_yaml_path="path/to/data.yaml"
)

# Save as COCO
dataset.as_coco(
    images_directory_path="output/images",
    annotations_path="output/annotations.json"
)
```

### COCO to YOLO

```python
import supervision as sv

# Load COCO dataset
dataset = sv.DetectionDataset.from_coco(
    images_directory_path="path/to/images",
    annotations_path="path/to/annotations.json"
)

# Save as YOLO
dataset.as_yolo(
    images_directory_path="output/images",
    annotations_directory_path="output/labels",
    data_yaml_path="output/data.yaml"
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

| Consideration | COCO | YOLO |
|--------------|------|------|
| **Annotation storage** | Single JSON file per split | One text file per image |
| **Human readability** | JSON structure, verbose | Simple text, compact |
| **Other framework compatibility** | DETR family, MMDetection | Ultralytics YOLO |
| **Segmentation support** | Full polygon support | Full polygon support |
| **Editing annotations** | Requires JSON parsing | Simple text editing |

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
