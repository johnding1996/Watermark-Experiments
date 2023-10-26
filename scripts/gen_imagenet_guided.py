# Package imports
import os
import sys
import argparse
import itertools
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

# Relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tree_ring import *
from guided_diffusion import *
from utils import *


# Find devices
devices = [torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())]

# Dictionary mapping ImageNet labels to WordNet IDs (wnids) as a global variable
labels_to_winds = get_imagenet_wnids()


def worker_func(args):
    (
        labels,
        diffusion_seed,
        device,
        image_size,
        dataset_dir_path,
    ) = args

    # Load guided diffusion models which are class-conditional diffusion models trained on ImageNet
    model, diffusion = load_guided_diffusion_model(image_size, device)

    # Generate images without watermark
    images_wo = guided_diffusion_without_watermark(
        model,
        diffusion,
        labels,
        image_size=image_size,
        diffusion_seed=diffusion_seed,
        return_image=True,
    )

    # Save images with watermark
    for label, image_wo in zip(labels, images_wo):
        save_path = os.path.join(
            dataset_dir_path,
            f"{labels_to_winds[label]}",
            f"{diffusion_seed}.png",
        )
        image_wo.save(save_path, "PNG")


# Create jobs for workers
def create_jobs(
    batch_size,
    labels,
    diffusion_seeds,
    devices,
    image_size,
    dataset_dir_path,
):
    for diffusion_seed in diffusion_seeds:
        for i in range(0, len(labels), batch_size):
            batched_labels = labels[i : min(i + batch_size, len(labels))]
            device = devices[i % len(devices)]
            yield (
                batched_labels,
                diffusion_seed,
                device,
                image_size,
                dataset_dir_path,
            )


# Generate non-watermarked guided-diffusion images
def generate_guided_diffusion_images(
    image_size,
    dataset_template,
    num_diffusion_seeds,
    batch_size,
):
    # Get labels
    wnids_to_labels = {wnid: label for label, wnid in get_imagenet_wnids().items()}
    if dataset_template == "ImageNet":
        labels = list(range(1000))
    elif dataset_template == "Tiny-ImageNet":
        assert image_size == 64
        with open("./datasets/tiny-imagenet-200/wnids.txt", "r") as f:
            wnids = sorted([line.strip() for line in f.readlines()])
        labels = [wnids_to_labels[wnid] for wnid in wnids]
    elif dataset_template == "Imagenette":
        assert image_size == 256
        wnids = [name for name in os.listdir("./datasets/imagenette2-320/train")]
        labels = [wnids_to_labels[wnid] for wnid in wnids]
    else:
        assert False

    # Define seeds
    diffusion_seeds = list(range(num_diffusion_seeds))

    # Create dataset directory
    dataset_dir_name = f"imagenet_guided_{image_size}_{dataset_template.lower()}"
    assert os.path.exists("./datasets/")
    dataset_dir_path = os.path.join("./datasets/", dataset_dir_name, "train")
    if os.path.exists(dataset_dir_path):
        print("Dataset directory already exists, aborted.")
        return
    os.makedirs(dataset_dir_path)
    for label in labels:
        os.makedirs(os.path.join(dataset_dir_path, f"{labels_to_winds[label]}"))

    # Create jobs for workers
    jobs = create_jobs(
        batch_size,
        labels,
        diffusion_seeds,
        devices,
        image_size,
        dataset_dir_path,
    )
    num_jobs = sum(
        1
        for _ in create_jobs(
            batch_size,
            labels,
            diffusion_seeds,
            devices,
            image_size,
            dataset_dir_path,
        )
    )

    # Generate watermarked images
    with Pool(processes=len(devices)) as pool:
        for _ in tqdm(pool.imap_unordered(worker_func, jobs), total=num_jobs):
            pass

    # Print message
    print(f"Generated guided diffusion images saved to: {dataset_dir_path}")


# Parse arguments
def main():
    parser = argparse.ArgumentParser(
        description="Generate images using guided diffusion model and pack into a dataset"
    )

    # Image size argument
    parser.add_argument(
        "--image_size",
        type=int,
        choices=[64, 256],
        default=64,
        help="Image size",
    )

    # Dataset template argument
    parser.add_argument(
        "--dataset_template",
        type=str,
        choices=["ImageNet", "Tiny-ImageNet", "Imagenette"],
        default="Tiny-ImageNet",
        help="Dataset template",
    )

    # Number of diffusion seeds argument
    parser.add_argument(
        "--num_diffusion_seeds",
        type=int,
        default=100,
        help="Number of diffusion seeds (positive integer)",
    )

    # Batch size argument
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (positive integer)",
    )

    args = parser.parse_args()

    if args.num_diffusion_seeds <= 0 or args.batch_size <= 0:
        assert False

    generate_guided_diffusion_images(
        args.image_size,
        args.dataset_template,
        args.num_diffusion_seeds,
        args.batch_size,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
