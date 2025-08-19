# train_single.py
import argparse
import os
import platform

ENCODER_ALIASES = {
    "dinov2": "dinov2_windowed_small",
    "v2": "dinov2_windowed_small",
    "dinov2_small": "dinov2_windowed_small",
    "dinov2_base": "dinov2_windowed_base",
    "dinov3": "dinov3_base",
    "v3": "dinov3_base",
}

VALID_ENCODERS = {
    "dinov2_windowed_small",
    "dinov2_windowed_base",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
}

def parse_args():
    ap = argparse.ArgumentParser("RF-DETR Medium single-run trainer (v2 or v3)")
    ap.add_argument("--data", required=True, help="Dataset root with train/valid/test")
    ap.add_argument("--out", default="./runs", help="Root output dir for TB & checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8, help="Batch size per iteration") ##TODO #Not actually applying, not worked out why
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default: 0 on Windows, else 2)")
    ap.add_argument("--encoder", default="dinov2",
                    help=("dinov2|v2|dinov2_small|dinov2_base|dinov3|v3|"
                          "dinov3_small|dinov3_base|dinov3_large or exact name"))
    ap.add_argument("--name", default=None, help="Optional run name (subdir under --out)")

    # Optional local DINOv3 assets
    ap.add_argument("--dinov3-repo", default=None, help="Local DINOv3 repo (sets DINOV3_REPO)")
    ap.add_argument("--dinov3-weights", default=None, help="Path to DINOv3 .pth (sets DINOV3_WEIGHTS)")
    return ap.parse_args()

def resolve_encoder(enc_str: str) -> str:
    enc_str = enc_str.strip().lower()
    enc = ENCODER_ALIASES.get(enc_str, enc_str)
    if enc not in VALID_ENCODERS:
        raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
    return enc

def main():
    args = parse_args()

    # Safer default on Windows for DataLoader workers
    if args.workers is None:
        import platform
        args.workers = 0 if platform.system() == "Windows" else 2

    encoder_name = resolve_encoder(args.encoder)

    # Set env *before* importing your package (your Pydantic defaults read env)
    os.environ["RFD_ENCODER"] = encoder_name
    if args.dinov3_repo:
        os.environ["DINOV3_REPO"] = args.dinov3_repo
    if args.dinov3_weights:
        os.environ["DINOV3_WEIGHTS"] = args.dinov3_weights

    # Now import project code
    from rfdetr import RFDETRBase, RFDETRMedium
    from rfdetr.config import RFDETRBaseConfig, RFDETRMediumConfig  # to override config cleanly

    # Two thin wrappers so we can control encoder and pretrain at construction time
    class RFDETRBaseV2(RFDETRMedium):
        def get_model_config(self, **kwargs):
            # keep RF-DETR pretrain (default) for v2
            return RFDETRMediumConfig(encoder="dinov2_windowed_small", **kwargs)

    class RFDETRBaseV3(RFDETRMedium):
        def get_model_config(self, **kwargs):
            # IMPORTANT: disable RF-DETR pretrain for v3 to avoid shape mismatches
            return RFDETRMediumConfig(encoder="dinov3_base", pretrain_weights=None, **kwargs)

    # Output dir (separate subdirs so TB shows two runs side-by-side)
    if args.name:
        run_name = args.name
    else:
        tag = "DINOv2" if encoder_name.startswith("dinov2") else "DINOv3"
        run_name = f"{tag}_Base"
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build and train
    ModelCls = RFDETRBaseV3 if encoder_name.startswith("dinov3") else RFDETRBaseV2
    model = ModelCls()

    print(f"\n=== Training RF-DETR Base with encoder: {encoder_name} ===")
    print(f"Dataset: {args.data}")
    print(f"Output : {out_dir}")

    train_kwargs = dict(
        dataset_dir=args.data,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.bs,  #TODO #Not actually applying, not worked out why
        num_workers=args.workers,
        tensorboard=True,
        run_test=True,
    )
    # NOTE: train() expects kwargs, not a TrainConfig instance
    model.train(**train_kwargs)

    print("\nDone. View in TensorBoard with:")
    print(f"  tensorboard --logdir {args.out}")
    print("Open http://127.0.0.1:6006")

if __name__ == "__main__":
    main()
