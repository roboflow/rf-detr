Routes (the letter is how the [fastest-route tables](index.md#fastest-route-across-nvidia-gpus) name a route):

- **C**, `inference()` graph, FP16: `model.inference()` with fp16 and a captured CUDA graph.
- **F**, `.trt` engine, FP16: a static engine from `RFDETR.export(format="tensorrt")`, loaded by `RFDETRInference`.
- **I**, `.trt` engine, INT8: the same export with `quantization="int8"` and calibration images.
- **M**, managed engine: the TensorRT package that `inference-models` selects itself, so its precision is not set here. `no engine` means `inference-models` 0.39.8 selected none (`NoModelPackagesAvailableError`).
- **FP32**: a `.trt` engine with `fp16=False`. It has no letter and is only in the [per-GPU tabs](tensorrt.md#latency-across-nvidia-gpus), because it is never the fastest.
