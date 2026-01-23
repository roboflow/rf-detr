# Benchmarks

This page reports RF-DETR benchmark results for object detection and instance segmentation on Microsoft COCO and RF100-VL. All benchmark numbers and plots match the latest released checkpoints and tables shown below. Latency values are measured on an NVIDIA T4 with TensorRT in FP16 at batch size 1. For full methodology details and architectural context, see the RF-DETR paper.

## Methodology

Accuracy is reported using standard COCO metrics computed with pycocotools. For object detection, we report COCO AP50 and COCO AP50:95, and the same metrics are also reported for RF100-VL. COCO results are evaluated on the validation split, following common practice in detector benchmarking. RF100-VL results are averaged across all 100 datasets to reflect performance under diverse real-world data distributions.

Latency is measured as single-image inference latency rather than sustained throughput. All latency numbers are obtained on an NVIDIA T4 GPU using TensorRT 10.4 and CUDA 12.4 with FP16 inference and batch size 1. To reduce variance caused by GPU power throttling and thermal effects, a 200 ms buffer is inserted between consecutive forward passes. This procedure improves reproducibility of latency measurements but is not intended to measure maximum throughput.

Accuracy and latency are always measured using the same model artifact and the same numerical precision. This avoids reporting FP32 accuracy together with FP16 latency, which can lead to misleading comparisons because naive FP16 conversion can significantly degrade accuracy for some models.

## Detection

<img alt="rf_detr_1-4_latency_accuracy_object_detection" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_object_detection.png" />

| Architecture | COCO AP<sub>50</sub> |  COCO AP<sub>50:95</sub>   |  RF100VL AP<sub>50</sub>   | RF100VL AP<sub>50:95</sub>  |  Latency (ms)   |   Params (M) |   Resolution  |
|:------------:|:--------------------:|:--------------------------:|:--------------------------:|:---------------------------:|:---------------:|:------------:|:-------------:|
|  RF-DETR-N   |         67.6         |            48.4            |            85.0            |            57.7             |       2.3       |         30.5 |       384x384 |
|  RF-DETR-S   |         72.1         |            53.0            |            86.7            |            60.2             |       3.5       |         32.1 |       512x512 |
|  RF-DETR-M   |         73.6         |            54.7            |            87.4            |            61.2             |       4.4       |         33.7 |       576x576 |
|  RF-DETR-L   |         75.1         |            56.5            |            88.2            |            62.2             |       6.8       |         33.9 |       704x704 |
|  RF-DETR-XL  |         77.4         |            58.6            |            88.5            |            62.9             |       11.5      |        126.4 |       700x700 |
|  RF-DETR-2XL |         78.5         |            60.1            |            89.0            |            63.2             |       17.2      |        126.9 |       880x880 |
|   YOLO11-N   |         52.0         |            37.4            |            81.4            |            55.3             |       2.5       |          2.6 |       640x640 |
|   YOLO11-S   |         59.7         |            44.4            |            82.3            |            56.2             |       3.2       |          9.4 |       640x640 |
|   YOLO11-M   |         64.1         |            48.6            |            82.5            |            56.5             |       5.1       |         20.1 |       640x640 |
|   YOLO11-L   |         64.9         |            49.9            |            82.2            |            56.5             |       6.5       |         25.3 |       640x640 |
|   YOLO11-X   |         66.1         |            50.9            |            81.7            |            56.2             |       10.5      |         56.9 |       640x640 |
|   YOLO26-N   |         55.8         |            40.3            |            76.7            |            52.0             |       1.7       |          2.6 |       640x640 |
|   YOLO26-S   |         64.3         |            47.7            |            82.7            |            57.0             |       2.6       |          9.4 |       640x640 |
|   YOLO26-M   |         69.7         |            52.5            |            84.4            |            58.7             |       4.4       |         20.1 |       640x640 |
|   YOLO26-L   |         71.1         |            54.1            |            85.0            |            59.3             |       5.7       |         25.3 |       640x640 |
|   YOLO26-X   |         74.0         |            56.9            |            85.6            |            60.0             |       9.6       |         56.9 |       640x640 |
|  LW-DETR-T   |         60.7         |            42.9            |            84.7            |            57.1             |       1.9       |         12.1 |       640x640 |
|  LW-DETR-S   |         66.8         |            48.0            |            85.0            |            57.4             |       2.6       |         14.6 |       640x640 |
|  LW-DETR-M   |         72.0         |            52.6            |            86.8            |            59.8             |       4.4       |         28.2 |       640x640 |
|  LW-DETR-L   |         74.6         |            56.1            |            87.4            |            61.5             |       6.9       |         46.8 |       640x640 |
|  LW-DETR-X   |         76.9         |            58.3            |            87.9            |            62.1             |       13.0      |        118.0 |       640x640 |
|   D-FINE-N   |         60.2         |            42.7            |            84.4            |            58.2             |       2.1       |          3.8 |       640x640 |
|   D-FINE-S   |         67.6         |            50.6            |            85.3            |            60.3             |       3.5       |         10.2 |       640x640 |
|   D-FINE-M   |         72.6         |            55.0            |            85.5            |            60.6             |       5.4       |         19.2 |       640x640 |
|   D-FINE-L   |         74.9         |            57.2            |            86.4            |            61.6             |       7.5       |         31.0 |       640x640 |
|   D-FINE-X   |         76.8         |            59.3            |            86.9            |            62.2             |       11.5      |         62.0 |       640x640 |

## Segmentation

<img alt="rf_detr_1-4_latency_accuracy_instance_segmentation" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_instance_segmentation.png" />

|   Architecture   | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |
|:----------------:|:--------------------:|:-----------------------:|:------------:|:----------:|:----------:|
|  RF-DETR-Seg-N   |         63.0         |          40.3           |     3.4      |    33.6    |  312x312   |
|  RF-DETR-Seg-S   |         66.2         |          43.1           |     4.4      |    33.7    |  384x384   |
|  RF-DETR-Seg-M   |         68.4         |          45.3           |     5.9      |    35.7    |  432x432   |
|  RF-DETR-Seg-L   |         70.5         |          47.1           |     8.8      |    36.2    |  504x504   |
|  RF-DETR-Seg-XL  |         72.2         |          48.8           |     13.5     |    38.1    |  624x624   |
| RF-DETR-Seg-2XL  |         73.1         |          49.9           |     21.8     |    38.6    |  768x768   |
|   YOLOv8-N-Seg   |         45.6         |          28.3           |     3.5      |    3.4     |  640x640   |
|   YOLOv8-S-Seg   |         53.8         |          34.0           |     4.2      |    11.8    |  640x640   |
|   YOLOv8-M-Seg   |         58.2         |          37.3           |     7.0      |    27.3    |  640x640   |
|   YOLOv8-L-Seg   |         60.5         |          39.0           |     9.7      |    46.0    |  640x640   |
|  YOLOv8-XL-Seg   |         61.3         |          39.5           |     14.0     |    71.8    |  640x640   |
|  YOLOv11-N-Seg   |         47.8         |          30.0           |     3.6      |    2.9     |  640x640   |
|  YOLOv11-S-Seg   |         55.4         |          35.0           |     4.6      |    10.1    |  640x640   |
|  YOLOv11-M-Seg   |         60.0         |          38.5           |     6.9      |    22.4    |  640x640   |
|  YOLOv11-L-Seg   |         61.5         |          39.5           |     8.3      |    27.6    |  640x640   |
|  YOLOv11-XL-Seg  |         62.4         |          40.1           |     13.7     |    62.1    |  640x640   |
|   YOLO26-N-Seg   |         54.3         |          34.7           |     2.31     |    2.7     |  640x640   |
|   YOLO26-S-Seg   |         62.4         |          40.2           |     3.47     |    10.4    |  640x640   |
|   YOLO26-M-Seg   |         67.8         |          44.0           |     6.32     |    23.6    |  640x640   |
|   YOLO26-L-Seg   |         69.8         |          45.5           |     7.58     |    28.0    |  640x640   |
|   YOLO26-X-Seg   |         71.6         |          46.8           |    12.92     |    62.8    |  640x640   |
