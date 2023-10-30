import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from utils import to_pil


# Compute PSNR between two images
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


# Compute SSIM between two images
def compute_ssim(image1, image2):
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

    # Calculate the SSIM
    ssim_value = metrics.structural_similarity(image1_np, image2_np, multichannel=True)
    return ssim_value


# Compute PSNR for multiple pairs of images
def compute_psnr_repeated(images1, images2):
    assert len(images1) == len(images2)
    psnr_values = []
    for image1, image2 in zip(images1, images2):
        psnr_values.append(compute_psnr(image1, image2))
    return np.mean(psnr_values), np.std(psnr_values)


# Compute SSIM for multiple pairs of images
def compute_ssim_repeated(images1, images2):
    assert len(images1) == len(images2)
    ssim_values = []
    for image1, image2 in zip(images1, images2):
        ssim_values.append(compute_ssim(image1, image2))
    return np.mean(ssim_values), np.std(ssim_values)
