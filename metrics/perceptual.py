from PIL import Image
from utils import to_tensor
from .lpips import LPIPS
from .watson import LossProvider


def compute_lpips(image1, image2, mode="vgg"):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    assert mode in ["vgg", "alex"]

    lpips_func = LPIPS(net=mode)
    image1_tensor = to_tensor([image1])
    image2_tensor = to_tensor([image2])
    return lpips_func(image1_tensor, image2_tensor).cpu().item()


def compute_watson(image1, image2, mode="vgg"):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    assert mode in ["vgg", "fft"]

    provider = LossProvider()
    watson_func = provider.get_loss_function(
        "Watson-" + mode, colorspace="RGB", pretrained=True, reduction="sum"
    )

    image1_tensor = to_tensor([image1])
    image2_tensor = to_tensor([image2])
    return watson_func(image1_tensor, image2_tensor).cpu().item()
