from detr import ViTLarge
import torch

if __name__ == "__main__":
    model = ViTLarge()
    model.train(
        dataset_dir="<path>",
        epochs=12,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )