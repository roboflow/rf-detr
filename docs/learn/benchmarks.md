RF-DETR is the first real-time model to exceed 60 AP on the Microsoft COCO benchmark alongside competitive performance at base sizes. It also achieves state-of-the-art performance on RF100-VL, an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is fastest and most accurate for its size when compared current real-time objection models.

On this page, we outline our results from our Microsoft COCO benchmarks, benchmarks across all seven categories in the RF100-VL benchmark, and notes from our benchmarking.

The table below shows the performance of RF-DETR, compared to other object detection models:

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto.png)

|family|size  |coco_map50|coco_map5095|rf100vl_map50|rv100vl_map5095|latency|
|------|------|----------|------------|-------------|---------------|-------|
|RF-DETR|Nano  |67.6      |48.4        |84.1         |57.1           |2.32   |
|RF-DETR|Small |72.1      |53.0        |85.9         |59.6           |3.52   |
|RF-DETR|Medium|73.6      |54.7        |86.6         |60.6           |4.52   |
|YOLO11|n     |52.0      |37.4        |81.4         |55.3           |2.49   |
|YOLO11|s     |59.7      |44.4        |82.3         |56.2           |3.16   |
|YOLO11|m     |64.1      |48.6        |82.5         |56.5           |5.13   |
|YOLO11|l     |65.3      |50.2        |x            |x              |6.65   |
|YOLO11|x     |66.5      |51.2        |x            |x              |11.92  |
|LW-DETR|Tiny  |60.7      |42.9        |x            |x              |1.91   |
|LW-DETR|Small |66.8      |48.0        |84.5         |58.0           |2.62   |
|LW-DETR|Medium|72.0      |52.6        |85.2         |59.4           |4.49   |
|D-FINE |Nano  |60.2      |42.7        |83.6         |57.7           |2.12   |
|D-FINE |Small |67.6      |50.7        |84.5         |59.9           |3.55   |
|D-FINE |Medium|72.6      |55.1        |84.6         |60.2           |5.68   |

_We are actively working on RF-DETR Large and X-Large models using the same techniques we used to achieve the strong accuracy that RF-DETR Medium attains. This is why RF-DETR Large and X-Large is not yet reported on our pareto charts. Check back in the next few weeks for the launch of new RF-DETR Large and X-Large models._