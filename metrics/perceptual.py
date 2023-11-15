import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from utils import to_tensor
from .lpips import LPIPS
from .watson import LossProvider


def get_perceptual_model(metric_name, mode):
    assert metric_name in ["lpips", "watson"]
    if metric_name == "lpips":
        assert mode in ["vgg", "alex"]
        perceptual_model = LPIPS(net=mode)
    elif metric_name == "watson":
        assert mode in ["vgg", "fft"]
        perceptual_model = LossProvider().get_loss_function(
            "Watson-" + mode, colorspace="RGB", pretrained=True, reduction="sum"
        )
    else:
        assert False
    return perceptual_model


# Compute metric between two images
def compute_metric(image1, image2, perceptual_model):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    image1_tensor = to_tensor([image1])
    image2_tensor = to_tensor([image2])
    return perceptual_model(image1_tensor, image2_tensor).cpu().item()


# Compute LPIPS distance between two images
def compute_lpips(image1, image2, mode="vgg"):
    perceptual_model = get_perceptual_model("lpips", mode)
    return compute_metric(image1, image2, perceptual_model)


# Compute Watson distance between two images
def compute_watson(image1, image2, mode="vgg"):
    perceptual_model = get_perceptual_model("watson", mode)
    return compute_metric(image1, image2, perceptual_model)


# Compute metrics between pairs of images
def compute_metric_repeated(images1, images2, metric_name, mode="vgg", verbose=False):
    # Accept list of PIL images
    assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
    assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
    assert len(images1) == len(images2)

    perceptual_model = get_perceptual_model(metric_name, mode)
    values = []
    for image1, image2 in (
        tqdm(zip(images1, images2), total=len(images1), desc=f"{metric_name.upper()} ")
        if verbose
        else zip(images1, images2)
    ):
        values.append(compute_metric(image1, image2, perceptual_model))
    return values


# Compute LPIPS distance between pairs of images
def compute_lpips_repeated(images1, images2, mode="vgg", verbose=False):
    return compute_metric_repeated(images1, images2, "lpips", mode, verbose)


# Compute Watson distance between pairs of images
def compute_watson_repeated(images1, images2, mode="vgg", verbose=False):
    return compute_metric_repeated(images1, images2, "watson", mode, verbose)
