from rfdetr import RFDETRBase

# model = RFDETRBase(pretrain_weights="rf-detr-base.pth")

# model.export(output_dir="onnx-models", infer_dir=None,
#              simplify=True,  backbone_only=False)

from rfdetr.deploy.export import trtexec
import argparse

args = argparse.Namespace()
args.verbose = True
args.profile = True
args.dry_run = False

onnx_model_path = "rfdetr.onnx"

trtexec(onnx_model_path, args)
