import torch
import pytest
from rfdetr import RFDETRBase, RFDETRLarge


@pytest.mark.parametrize("model_class", [RFDETRBase, RFDETRLarge])
@pytest.mark.parametrize("channels", [1, 4])
def test_multispectral_support(model_class, channels: int) -> None:
    model = model_class(num_channels=channels, device="cpu")
    image = torch.zeros(channels, 224, 224).to("cpu")
    model.predict(image, threshold=0.5)
