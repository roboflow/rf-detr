# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
This tool provides performance benchmarks by using ONNX Runtime and TensorRT 
to run inference on a given model with the COCO validation set. It offers 
reliable measurements of inference latency using ONNX Runtime or TensorRT 
on the device.
"""
import argparse
import copy
import contextlib
import datetime
import json
import os
import os.path as osp
import random
import time
import ast
from pathlib import Path
from collections import namedtuple, OrderedDict

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
import torchvision.transforms.functional as F
import tqdm

import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as nxrun
import tensorrt as trt


def parser_args():
    parser = argparse.ArgumentParser('performance benchmark tool for onnx/trt model')
    parser.add_argument('--path', type=str, help='engine file path')
    parser.add_argument('--coco_path', type=str, default="data/coco", help='coco dataset path')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--run_benchmark', action='store_true', help='repeat the inference to benchmark the latency')
    parser.add_argument('--disable_eval', action='store_true', help='disable evaluation')
    return parser.parse_args()


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = COCO(coco_gt)
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # Running per image evaluation...
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs

def convert_to_xywh(boxes):
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


def get_image_list(ann_file):
    with open(ann_file, 'r') as fin:
        data = json.load(fin)
    return data['images']


def load_image(file_path):
    return Image.open(file_path).convert("RGB")


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class SquareResize(object):
    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        rescaled_img=F.resize(img, (size, size))
        w, h = rescaled_img.size
        if target is None:
            return rescaled_img, None
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(rescaled_img.size, img.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        target["size"] = torch.tensor([h, w])

        return rescaled_img, target


def infer_transforms():
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return Compose([
        SquareResize([640]),
        normalize,
    ])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)


def post_process(outputs, target_sizes):
    out_logits, out_bbox = outputs['labels'], outputs['dets']

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    return results


def infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device, repeats=1):
    time_list = []
    for img_dict in tqdm.tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict['file_name']))
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
        outputs['labels'] = torch.Tensor(res[1]).to(device)
        outputs['dets'] = torch.Tensor(res[0]).to(device)

        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        results = post_process(outputs, orig_target_sizes)
        res = {img_dict['id']: results[0]}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
    
    print("Model latency with ONNX Runtime: {}ms".format(1000 * sum(time_list) / len(img_list)))

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        print(stats)


def infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device, repeats=1):
    time_list = []
    for img_dict in tqdm.tqdm(img_list):
        image = load_image(os.path.join(prefix, img_dict['file_name']))
        width, height = image.size
        orig_target_sizes = torch.Tensor([height, width])
        image_tensor, _ = infer_transforms()(image, None)  # target is None

        samples = image_tensor[None].to(device)
        _, _, h, w = samples.shape
        im_shape = torch.Tensor(np.array([h, w]).reshape((1, 2)).astype(np.float32)).to(device)
        scale_factor = torch.Tensor(np.array([h / height, w / width]).reshape((1, 2)).astype(np.float32)).to(device)
        
        time_profile.reset()
        with time_profile:
            for _ in range(repeats):
                outputs = model({"input": samples})

        time_list.append(time_profile.total / repeats)
        orig_target_sizes = torch.stack([orig_target_sizes], dim=0).to(device)
        if coco_evaluator is not None:
            results = post_process(outputs, orig_target_sizes)
            res = {img_dict['id']: results[0]}
            coco_evaluator.update(res)
    
    print("Model latency with TensorRT: {}ms".format(1000 * sum(time_list) / len(img_list)))

    # accumulate predictions from all images
    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        print(stats)


class TRTInference(object):
    """TensorRT inference engine
    """
    def __init__(self, engine_path='dino.engine', device='cuda:0', sync_mode:bool=False, max_batch_size=32, verbose=False):
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
        
        if not self.sync_mode:
            self.stream = cuda.Stream()

        # self.time_profile = TimeProfiler()
        self.time_profile = None

    def get_dummy_input(self, batch_size:int):
        blob = {}
        for name, binding in self.bindings.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f"make dummy input {name} with shape {binding.shape}")
                blob[name] = torch.rand(batch_size, *binding.shape[1:]).float().to('cuda:0')
        return blob
    
    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names
    
    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        '''build binddings
        '''
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                raise NotImplementedError
            
            if False:
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

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

    def synchronize(self, ):
        if not self.sync_mode and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.sync_mode:
            self.stream.synchronize()
    
    def speed(self, blob, n):
        self.time_profile.reset()
        with self.time_profile:
            for _ in range(n):
                _ = self(blob)
        return self.time_profile.total / n 


    def build_engine(self, onnx_file_path, engine_file_path, max_batch_size=32):
        '''Takes an ONNX file and creates a TensorRT engine to run inference with
        http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57
        '''
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.logger) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, self.logger) as parser, \
            builder.create_builder_config() as config:
            
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1024 MiB
            config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)

            return serialized_engine


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


def main(args):
    print(args)

    coco_gt = osp.join(args.coco_path, 'annotations/instances_val2017.json')
    img_list = get_image_list(coco_gt)
    prefix = osp.join(args.coco_path, 'val2017')
    if args.run_benchmark:
        repeats = 10
        print('Inference for each image will be repeated 10 times to obtain '
              'a reliable measurement of inference latency.')
    else:
        repeats = 1 

    if args.disable_eval:
        coco_evaluator = None
    else:
        coco_evaluator = CocoEvaluator(coco_gt, ('bbox',))
    
    time_profile = TimeProfiler()

    if args.path.endswith(".onnx"):
        sess = nxrun.InferenceSession(args.path, providers=['CUDAExecutionProvider'])
        infer_onnx(sess, coco_evaluator, time_profile, prefix, img_list, device=f'cuda:{args.device}', repeats=repeats)
    elif args.path.endswith(".engine"):
        model = TRTInference(args.path, sync_mode=True, device=f'cuda:{args.device}')
        infer_engine(model, coco_evaluator, time_profile, prefix, img_list, device=f'cuda:{args.device}', repeats=repeats)
    else:
        raise NotImplementedError('Only model file names ending with ".onnx" and ".engine" are supported.')


if __name__ == '__main__':
    args = parser_args()
    main(args)
