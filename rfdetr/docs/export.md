After training a custom RF-DETR model it is often desirable to export the model.
RF-DETR supports exporting models to both ONNX and TensorRT formats.
Exporting models to ONNX enables interoperability with various inference frameworks and can improve deployment efficiency.
Exporting to TensorRT typically reduces inference latency and model size.

## ONNX export

> [!IMPORTANT]
> Starting with RF-DETR 1.2.0, you'll have to run `pip install rfdetr[onnxexport]` before exporting model weights to ONNX format.  

To export your model, simply initialize it and call the `.export()` method. There are several optional arguments that you can pass to the `.export()` method. 

*   `output_dir`: The directory where the ONNX model should be saved.
*   `infer_dir`: A directory where a single sample image exists.
*   `simplify`: A boolean indicating whether you want to simplify the ONNX model. This improves inference speed and reduces model complexity and size.
*   `backbone_only`: A boolean indicating whether you want to export the backbone only. Setting this boolean to true renders the model unable to perform object detection.

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export(output_dir="onnx-models", infer_dir=None, simplify=True,  backbone_only=False)
```

## TensorRT conversion

> [!IMPORTANT]
> TensorRT conversion must be done on the same device where you want to run inference. 

The ONNX model can be exported to TensorRT for faster inference and reduced model size.
First download and install TensorRT>=8.6.1 from [TensorRT](https://developer.nvidia.com/tensorrt/download), make sure that the TensorRT is compatible with your OS (`lsb_release -a`) and CUDA (`nvcc --version`) version.

To export your ONNX model to TensorRT, initialize and call the `trtexec()` method with the path to your ONNX model and three arguments:

1. `verbose` [tensorRT_docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.)html#trtexec
2. If you want to use nsight-systems profiling install it from [nsigh-systems_install](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html). Documentation is available at [night-systems_docs](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli). This provides you with detailed information about TensorRT execution on the GPU.
3. Setting `dry_run` to true enables you to print the command that would be executed.

Run the code below to convert your ONNX model to TensorRT. Change `onnx_model_path`.

```python
from rfdetr.deploy.export import trtexec
import argparse

args = argparse.Namespace()
args.verbose = True
args.profile = False
args.dry_run = False
args.wandb = False # This is required for rf-detr 1.0.0 and 1.1.0! (Will be dropped from 1.2.0 onwards)
onnx_model_path = "your_onnx_model.onnx"

trtexec(onnx_model_path, args)
```
This script will create a file named `your_onnx_model.engine`.

## Deploying on Jetson Orin Nano (8 GB developer kit)

The exported .engine model can be used to perform real-time inference.
Replace `your_onnx_model.engine` with your .engine file in the script below.

The script below shows an example for real-time inference using the TensorRT engine file it is partly based on `rfdetr/deploy/benchmark.py`:

```python
import os
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import tensorrt as trt
from PIL import Image

import rfdetr.datasets.transforms as T

transforms = T.Compose([
    # Note: adjust the line below based on the `resolution` RF-DETR was trained.
    T.SquareResize([1120]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_file_path: str) -> trt.ICudaEngine | None:
    """
    Load a serialized engine from file.

    Args:
        engine_file_path (str): Path to the serialized engine file.

    Returns:
        trt.ICudaEngine | None: A TensorRT ICudaEngine object or None if loading failed.

    Examples:
        >>> engine = load_engine("your_onnx_model.engine")
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def load_trt_engine_context() -> tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """
    Load the TensorRT engine and create an execution context.

    Returns:
        tuple[trt.ICudaEngine, trt.IExecutionContext]: A tuple containing the TensorRT engine and execution context.
    """

    # Note: adjust this based on the path where you saved the engine file.
    engine = load_engine("your_onnx_model.engine")
    context = engine.create_execution_context()
    return engine, context


engine, context = load_trt_engine_context()


def run_sync(context: trt.IExecutionContext, engine: trt.ICudaEngine, input_data: np.ndarray) -> dict:
    """
    Run inference on the input data using the TensorRT engine and context.

    Args:
        context (trt.IExecutionContext): The TensorRT execution context.
        engine (trt.ICudaEngine): The TensorRT engine.
        input_data (np.ndarray): Input data for inference.

    Returns:
        dict: A dictionary containing the output tensors.
    """
    bindings = {}
    bindings_addr = {}
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    for name in input_names:
        shape = context.get_tensor_shape(name)
        size = trt.volume(shape) * np.dtype(np.float32).itemsize
        device_input = cuda.mem_alloc(size)
        cuda.memcpy_htod(
            device_input, np.ascontiguousarray(input_data).ravel())
        bindings[name] = device_input
        bindings_addr[name] = int(device_input)

    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    for name in output_names:
        shape = context.get_tensor_shape(name)
        size = trt.volume(shape) * np.dtype(np.float32).itemsize
        device_output = cuda.mem_alloc(size)
        bindings[name] = device_output
        bindings_addr[name] = int(device_output)

    context.execute_v2(list(bindings_addr.values()))

    outputs = {}
    for name in output_names:
        shape = context.get_tensor_shape(name)
        host_output = cuda.pagelocked_empty(
            trt.volume(shape), dtype=np.float16)
        cuda.memcpy_dtoh(host_output, bindings[name])
        outputs[name] = host_output.reshape(shape)

    return outputs


def detect_items(frame: np.ndarray) -> tuple[list, list]:
    """
    Detect items in the given frame using the TensorRT engine.

    Args:
        frame (np.ndarray): The input frame for detection.

    Returns:
        tuple[list, list]: A tuple containing the detected class ids and their confidence scores.
    """
    frame = Image.fromarray(frame)
    img_transformed, _ = transforms(frame, None)
    img_transformed = img_transformed.unsqueeze(0)
    input_np = img_transformed.numpy().astype(np.float32)

    output_tensors = run_sync(context, engine, input_np)
    pred_boxes = output_tensors['dets']
    logits = output_tensors['labels']

    scores = torch.sigmoid(torch.from_numpy(logits))
    max_scores, pred_labels = scores.max(-1)
    confidence_mask = max_scores.squeeze(0) > 0.5
    filtered_scores = max_scores.squeeze(0)[confidence_mask]
    top_indices = filtered_scores.argsort(descending=True)[:5]
    top_5_scores = filtered_scores[top_indices].tolist()
    top_5_labels = pred_labels.squeeze(
        0)[confidence_mask][top_indices].tolist()

    print(f"Top 5 predictions (class ids and scores):")
    for i, (label_id, score) in enumerate(zip(top_5_labels, top_5_scores)):
        print(
            f"Prediction {i+1}: class id: {label_id}, confidence score: {score}")

    return top_5_labels, top_5_scores


# Note: the device index on the line below might be different on your device.
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Failed to open video stream.")

start_time = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detected_items = detect_items(frame)
    frames += 1
    fps = frames / (time.time() - start_time)
    print(f"Detected objects: {detected_items} | FPS: {fps:.2f}")
cap.release()
```
