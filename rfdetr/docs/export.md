After training a custom RF-DETR model it is often desirable to export the model. RF-DETR supports exporting models to the ONNX and TensorRT format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

## ONNX export

> [!IMPORTANT]
> Starting with RF-DETR 1.2.0, you'll have to run `pip install rfdetr[onnxexport]` before exporting model weights to ONNX format.  

To export your model, simply initialize it and call the `.export()` method. There are several optional arguments that you can pass to the `.export()` method. 

The directory where the ONNX model should be saved (`output_dir`). A directory where a single sample image exists (`infer_dir`). `simplify`, a boolean indicating whether you want to simplify the ONNX model, this improves inference speed and reduces model complexity and size. `backbone_only`, a boolean indicating whether you want to export the backbone only. Setting this boolean to true renders the model unable to perform object detection. 

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export(output_dir="onnx-models", infer_dir=None, simplify=True,  backbone_only=False)
```

## TensorRT conversion

> [!IMPORTANT]
> TensorRT conversion must be done on the same device where you want to run inference on. 
> 
(Tested on: WSL2 - CUDA 12.0 - Ubuntu 24.04.1 LTS - Python 3.12.2)
(Tested on: )

The ONNX model can be exported to TensorRT for faster inference.
First download and install TensorRT>=8.6.1 from [TensorRT](https://developer.nvidia.com/tensorrt/download), make sure that the TensorRT is compatible with your OS (`lsb_release -a`) and CUDA (`nvcc --version`) version.   


## Deploying on Jetson Orin Nano (8 GB developer kit)

