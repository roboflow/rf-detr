RF-DETR is the first real-time model to exceed 60 AP on the Microsoft COCO benchmark alongside competitive performance at base sizes. It also achieves state-of-the-art performance on RF100-VL, an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is comparable speed to current real-time objection models.

On this page, we outline our results from our Microsoft COCO benchmarks, benchmarks across all seven categories in the RF100-VL benchmark, and notes from our benchmarking.

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto.png)

## RF-DETR on Microsoft COCO

| Model            | params<br><sup>(M) | mAP<sup>COCO val<br>@0.50:0.95 | mAP<sup>RF100-VL<br>Average @0.50 | mAP<sup>RF100-VL<br>Average @0.50:95 | Total Latency<br><sup>T4 bs=1<br>(ms) |
|------------------|--------------------|--------------------------------|-----------------------------------|---------------------------------------|---------------------------------------|
| D-FINE-M         | 19.3               | <ins>55.1</ins>                | N/A                               | N/A                                   | 6.3                                   |
| LW-DETR-M        | 28.2               | 52.5                           | 84.0                              | 57.5                                  | 6.0                                   |
| YOLO11m          | 20.0               | 51.5                           | 84.9                              | 59.7                                  | <ins>5.7</ins>                        |
| YOLOv8m          | 28.9               | 50.6                           | 85.0                              | 59.8                                  | 6.3                                   |
| RF-DETR-Medium    | 33.7               | 54.8                           | <ins>86.6</ins>                   | <ins>60.6</ins>                       | <ins>4.31</ins>                                   |                               |

## RF-DETR on RF100-VL

<img src="https://github.com/user-attachments/assets/e61a7ba4-5294-40a9-8cd7-4fc924639924" alt="rf100-vl-map50">

### RF100-VL Benchmarking Notes

- The "Total Latency" reported is measured on a T4 GPU using TensorRT10 FP16 (ms/img) and was introduced by LW-DETR. Unlike transformer-based models, YOLO models perform Non-Maximum Suppression (NMS) after generating predictions to refine bounding box candidates. While NMS boosts accuracy, it also slightly reduces speed due to the additional computation required, which varies with the number of objects in an image. Notably, many YOLO benchmarks include NMS in accuracy measurements but exclude it from speed metrics. By contrast, our benchmarking—following LW-DETR’s approach—factors in NMS latency to provide a uniform measure of the total time needed to obtain a final result across all models on the same hardware.

- D-FINE’s fine-tuning capability is currently unavailable, making its domain adaptability performance inaccessible. The authors [caution](https://github.com/Peterande/D-FINE) that “if your categories are very simple, it might lead to overfitting and suboptimal performance.” Furthermore, several open issues ([#108](https://github.com/Peterande/D-FINE/issues/108), [#146](https://github.com/Peterande/D-FINE/issues/146), [#169](https://github.com/Peterande/D-FINE/issues/169), [#214](https://github.com/Peterande/D-FINE/issues/214)) currently prevent successful fine-tuning. We have opened an additional issue in hopes of ultimately benchmarking D-FINE with RF100-VL.
</details>