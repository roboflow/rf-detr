# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
This tool provides performance benchmarks by using ONNX Runtime and TensorRT
to run inference on a given model with the COCO validation set. It offers
reliable measurements of inference latency using ONNX Runtime or TensorRT
on the device.
"""

import contextlib
import json
import os
import os.path as osp
import time
from collections import OrderedDict, namedtuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
except ImportError:
    cuda = None

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def get_image_list(ann_file):
    with open(ann_file, "r") as fin:
        data = json.load(fin)
    return data["images"]


def load_image(file_path):
    return Image.open(file_path).convert("RGB")


def infer_transforms():
    from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

    from rfdetr.datasets.transforms import Normalize

    return Compose(
        [
            Resize((640, 640)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def post_process(outputs, target_sizes):
    out_logits, out_bbox = outputs["labels"], outputs["dets"]

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)]

    return results


def infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device, repeats=1):
    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = infer_transforms()(image, None)  # target is None

        samples = image_tensor[None].numpy()

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                res = sess.run(None, {"input": samples})
        time_list.append(time_profile.total / repeats)
        outputs = {}
        outputs["labels"] = torch.Tensor(res[1]).to(device)
        outputs["dets"] = torch.Tensor(res[0]).to(device)

        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        results = post_process(outputs, orig_target_sizes)
        res = {img_dict["id"]: results[0]}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    logger.info("Model latency with ONNX Runtime: {}ms".format(1000 * sum(time_list) / len(img_list)))

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


def infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device, repeats=1):
    time_list = []
    for img_dict in tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict["file_name"]))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = infer_transforms()(image, None)  # target is None

        samples = image_tensor[None].to(device)
        _, _, h, w = samples.shape
        # torch.Tensor(np.array([h, w]).reshape((1, 2)).astype(np.float32)).to(device)
        # torch.Tensor(np.array([h / height, w / width]).reshape((1, 2)).astype(np.float32)).to(device)

        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                outputs = model({"input": samples})

        time_list.append(time_profile.total / repeats)
        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        if coco_evaluator is not None:
            results = post_process(outputs, orig_target_sizes)
            res = {img_dict["id"]: results[0]}
            coco_evaluator.update(res)

    logger.info("Model latency with TensorRT: {}ms".format(1000 * sum(time_list) / len(img_list)))

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        logger.info(stats)


class TRTInference(object):
    """TensorRT inference engine"""

    def __init__(
        self, engine_path="dino.engine", device="cuda:0", sync_mode: bool = False, max_batch_size=32, verbose=False
    ):
        if not trt:
            raise ImportError("TensorRT is not installed. Please install TensorRT to use TRTInference.")

        self.engine_path = engine_path
        self.device = device
        self.sync_mode = sync_mode
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.stream = None

        if not self.sync_mode:
            if not cuda:
                raise ImportError(
                    "pycuda is not installed. Please install `pycuda` to use TRTInference with async mode."
                )

            self.stream = cuda.Stream()

        # self.time_profile = TimeProfiler()
        self.time_profile = None

    def get_dummy_input(self, batch_size: int):
        blob = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                logger.info(f"make dummy input {name} with shape {binding.shape}")
                blob[name] = torch.rand(batch_size, *binding.shape[1:]).float().to("cuda:0")
        return blob

    def load_engine(self, path):
        """load engine"""
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(
        self,
    ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(
        self,
    ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        """build binddings"""
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                raise NotImplementedError

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_sync(self, blob):
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def run_async(self, blob):
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        outputs = {n: self.bindings[n].data for n in self.output_names}
        self.stream.synchronize()
        return outputs

    def __call__(self, blob):
        if self.sync_mode:
            return self.run_sync(blob)
        else:
            return self.run_async(blob)

    def synchronize(
        self,
    ):
        if self.sync_mode:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return

        if self.stream is not None:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def speed(self, blob, n):
        self.time_profile.reset()
        with self.time_profile:
            for _ in range(n):
                _ = self(blob)
        return self.time_profile.total / n

    def build_engine(self, onnx_file_path, engine_file_path, max_batch_size=32):
        """Takes an ONNX file and creates a TensorRT engine to run inference with
        http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57
        """
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with (
            trt.Builder(self.logger) as builder,
            builder.create_network(EXPLICIT_BATCH) as network,
            trt.OnnxParser(network, self.logger) as parser,
            builder.create_builder_config() as config,
        ):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1024 MiB
            config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    logger.error("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None

            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)

            return serialized_engine


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(
        self,
    ):
        self.total = 0

    def __enter__(
        self,
    ):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(
        self,
    ):
        self.total = 0

    def time(
        self,
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


def main(
    path: str,
    coco_path: str = "data/coco",
    device: int = 0,
    run_benchmark: bool = False,
    disable_eval: bool = False,
) -> None:
    """Performance benchmark tool for ONNX/TRT models.

    Args:
        path: Engine file path (.onnx or .engine).
        coco_path: COCO dataset path.
        device: CUDA device index.
        run_benchmark: Repeat inference 10x to measure latency.
        disable_eval: Skip COCO evaluation.
    """
    logger.info(
        {
            "path": path,
            "coco_path": coco_path,
            "device": device,
            "run_benchmark": run_benchmark,
            "disable_eval": disable_eval,
        }
    )
    coco_gt = osp.join(coco_path, "annotations/instances_val2017.json")
    img_list = get_image_list(coco_gt)
    prefix = osp.join(coco_path, "val2017")
    if run_benchmark:
        repeats = 10
        logger.info("Inference for each image will be repeated 10 times ...")
    else:
        repeats = 1

    if not disable_eval:
        from rfdetr.evaluation.coco_eval import CocoEvaluator

        coco_evaluator = CocoEvaluator(coco_gt, ("bbox",))
    else:
        coco_evaluator = None
    time_profile = TimeProfiler()

    if path.endswith(".onnx"):
        import onnxruntime as nxrun

        sess = nxrun.InferenceSession(path, providers=["CUDAExecutionProvider"])
        infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    elif path.endswith(".engine"):
        model = TRTInference(path, sync_mode=True, device=f"cuda:{device}")
        infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device=f"cuda:{device}", repeats=repeats)
    else:
        raise NotImplementedError('Only model file names ending with ".onnx" and ".engine" are supported.')


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
