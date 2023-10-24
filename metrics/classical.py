import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from utils import to_pil


# Compute the PSNR between two images
def compute_psnr(image1, image2):
    # Convert the images to PIL Images if they are tensors
    if isinstance(image1, Image.Image):
        assert isinstance(image2, Image.Image)
    elif isinstance(image1, torch.Tensor):  # image1 and image2 are tensors
        assert isinstance(image2, torch.Tensor)
        image1, image2 = to_pil(torch.stack([image1, image2]))
    else:
        assert False

    # Convert the PIL Images to numpy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    assert image1_np.shape == image2_np.shape

    # Calculate the PSNR
    psnr_value = peak_signal_noise_ratio(image1_np, image2_np)
    return psnr_value


def compute_psnr_repeated(images1, images2):
    pass
