import os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from utils import set_random_seed
from utils import to_pil
from .pytorch_fid.fid_score import calculate_fid_given_paths


# Save images to temp dir
def save_images_to_temp(images):
    assert isinstance(images, list) and isinstance(images[0], Image.Image)
    temp_dir = tempfile.mkdtemp()
    for i, image in enumerate(images):
        save_path = os.path.join(temp_dir, f"{i}.png")
        image.save(save_path, "PNG")
    return temp_dir


# Calculate FID
def compute_fid(
    images1,
    images2,
    batch_size=50,
    dims=2048,
    device=None,
    num_workers=None,
    verbose=False,
    return_paths=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is not None:
        assert 1 <= num_workers <= os.cpu_count()
    else:
        num_workers = min(os.cpu_count(), 8)

    # Check images
    if not isinstance(images1, list):
        assert isinstance(images1, str) and os.path.exists(images1)
        assert isinstance(images2, str) and os.path.exists(images2)
        path1 = images1
        path2 = images2
    else:
        if not isinstance(images1[0], Image.Image):
            images1 = to_pil(images1)
            images2 = to_pil(images2)
        # Save images to temp dir if needed
        path1 = save_images_to_temp(images1)
        path2 = save_images_to_temp(images2)

    # Calculate FID
    fid_score = calculate_fid_given_paths(
        paths=[path1, path2],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers,
        verbose=verbose,
    )

    if return_paths:
        return fid_score, (path1, path2)
    else:
        if os.path.exists(path1):
            shutil.rmtree(path1)
        if os.path.exists(path2):
            shutil.rmtree(path2)
        return fid_score


def compute_fid_repeated(
    images1,
    images2,
    num_repeats,
    sample_size,
    pairwise=False,
    batch_size=50,
    dims=2048,
    device=None,
    num_workers=None,
    verbose=False,
    sampling_seed=None,
):
    if sampling_seed is not None:
        set_random_seed(sampling_seed)
    # The minimum number of images is 2048 for FID calculation
    assert num_repeats >= 1 and sample_size >= 2048
    # If pairwise, we assume that images1 and images2 are paired
    if pairwise:
        assert len(images1) == len(images2)
    # Calculate FID scores for each pair of sampled sets
    fid_scores = []
    for _ in trange(num_repeats) if verbose else range(num_repeats):
        selected_indices = np.random.choice(len(images1), sample_size, replace=False)
        images1_sample = [images1[i] for i in selected_indices]
        # If pairwise, we use the same indices for images2
        if not pairwise:
            selected_indices = np.random.choice(
                len(images2), sample_size, replace=False
            )
        images2_sample = [images2[i] for i in selected_indices]
        fid_score = compute_fid(
            images1_sample,
            images2_sample,
            batch_size=batch_size,
            dims=dims,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
        )
        fid_scores.append(fid_score)
    return np.mean(fid_scores), np.std(fid_scores)
