import os
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm, trange
from utils import set_random_seed
from utils import to_pil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from PIL import Image
from .clean_fid import fid


def save_single_image(i, image, temp_dir):
    save_path = os.path.join(temp_dir, f"{i}.png")
    image.save(save_path, "PNG")


def save_images_to_temp(images, num_workers, verbose=False):
    assert isinstance(images, list) and isinstance(images[0], Image.Image)
    temp_dir = tempfile.mkdtemp()

    # Using ProcessPoolExecutor to save images in parallel
    func = partial(save_single_image, temp_dir=temp_dir)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = executor.map(func, range(len(images)), images)
        list(tasks) if not verbose else list(
            tqdm(
                tasks,
                total=len(images),
                desc="Saving images ",
            )
        )
    return temp_dir


# Compute FID between two sets of images
def compute_fid(
    images1,
    images2,
    mode="legacy",
    device=None,
    num_workers=None,
    verbose=False,
    return_paths=False,
):
    # Support four types of FID scores
    assert mode in ["legacy", "clean", "clip", "kid"]
    if mode == "legacy":
        mode = "legacy_pytorch"
        model_name = "inception_v3"
    elif mode == "clean":
        mode = "clean"
        model_name = "inception_v3"
    elif mode == "clip":
        mode = "clean"
        model_name = "clip_vit_b_32"
    elif mode == "kid":
        pass
    else:
        assert False

    # Set up device and num_workers
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if num_workers is not None:
        assert 1 <= num_workers <= os.cpu_count()
    else:
        num_workers = max(torch.cuda.device_count() * 4, 8)

    # Check images, can be paths or lists of PIL images
    if not isinstance(images1, list):
        assert isinstance(images1, str) and os.path.exists(images1)
        assert isinstance(images2, str) and os.path.exists(images2)
        path1 = images1
        path2 = images2
    else:
        assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
        assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
        # Save images to temp dir if needed
        path1 = save_images_to_temp(images1, num_workers=num_workers, verbose=verbose)
        path2 = save_images_to_temp(images2, num_workers=num_workers, verbose=verbose)

    if mode != "kid":
        fid_score = fid.compute_fid(
            path1,
            path2,
            mode=mode,
            model_name=model_name,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
        )
    else:
        fid_score = fid.compute_kid(
            path1,
            path2,
            device=device,
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


# Compute FID for multiple times
def compute_fid_repeated(
    images1,
    images2,
    mode="legacy",
    num_repeats=1,
    sample_size=2048,
    pairwise=False,
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
    for _ in trange(num_repeats, desc="Repeats ") if verbose else range(num_repeats):
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
            mode=mode,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
            return_paths=False,
        )
        fid_scores.append(fid_score)
    return np.mean(fid_scores), np.std(fid_scores)
