import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity as structural_similarity_index_measure,
)
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor


# Process images to numpy arrays
def convert_image_pair_to_numpy(image1, image2):
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)

    image1_np = np.array(image1)
    image2_np = np.array(image2)
    assert image1_np.shape == image2_np.shape

    return image1_np, image2_np


# Compute MSE for a single pair of images
def compute_mse(image1, image2):
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return mean_squared_error(image1_np, image2_np)


# Compute PSNR for a single pair of images
def compute_psnr(image1, image2):
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return peak_signal_noise_ratio(image1_np, image2_np)


# Compute SSIM for a single pair of images
def compute_ssim(image1, image2):
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return structural_similarity_index_measure(image1_np, image2_np, channel_axis=2)


# Compute metrics
def compute_metric_repeated(
    images1, images2, metric_func, num_workers=None, verbose=False
):
    assert len(images1) == len(images2)

    if num_workers is not None:
        assert 1 <= num_workers <= os.cpu_count()
    else:
        num_workers = max(torch.cuda.device_count() * 4, 8)

    metric_name = metric_func.__name__.split("_")[1].upper()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = executor.map(metric_func, images1, images2)
        values = (
            list(tasks)
            if not verbose
            else list(
                tqdm(
                    tasks,
                    total=len(images1),
                    desc=f"{metric_name} ",
                )
            )
        )
    return np.mean(values), np.std(values)


# Compute MSE for pairs of images
def compute_mse_repeated(images1, images2, num_workers=None, verbose=False):
    return compute_metric_repeated(images1, images2, compute_mse, num_workers, verbose)


# Compute PSNR for pairs of images
def compute_psnr_repeated(images1, images2, num_workers=None, verbose=False):
    return compute_metric_repeated(images1, images2, compute_psnr, num_workers, verbose)


# Compute SSIM for pairs of images
def compute_ssim_repeated(images1, images2, num_workers=None, verbose=False):
    return compute_metric_repeated(images1, images2, compute_ssim, num_workers, verbose)