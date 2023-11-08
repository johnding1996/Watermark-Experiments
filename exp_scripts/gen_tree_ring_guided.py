# Package imports
import os
import sys
import argparse
import itertools
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tree_ring import *
from guided_diffusion import *
from utils import *


# Find devices
devices = [torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())]

# Default tree-ring parameters
tree_ring_paras = dict(
    w_channel=2,
    w_pattern="ring",
    w_mask_shape="circle",
    w_radius=10,
    w_measurement="l1_complex",
    w_injection="complex",
)

# Dictionary mapping ImageNet labels to WordNet IDs (wnids) as a global variable
labels_to_winds = get_imagenet_wnids()


def worker_func(args):
    (
        labels,
        key_seed,
        message_seed,
        diffusion_seed,
        device,
        image_size,
        dataset_dir_path,
    ) = args

    # Load guided diffusion models which are class-conditional diffusion models trained on ImageNet
    model, diffusion = load_guided_diffusion_model(image_size, device)

    # Generate one watermark key (which is the mask in tree-ring's paper)
    key = generate_tree_ring_key(
        key_seed=key_seed,
        image_size=image_size,
        tree_ring_paras=tree_ring_paras,
        device=device,
    )

    # Generate the watermark message (which is the key in tree-ring's paper)
    message = generate_tree_ring_message(
        message_seed=message_seed,
        image_size=image_size,
        tree_ring_paras=tree_ring_paras,
        device=device,
    )

    # Generate images with watermark
    images_w = guided_diffusion_with_tree_ring(
        model,
        diffusion,
        labels,
        keys=key,
        messages=message,
        tree_ring_paras=tree_ring_paras,
        image_size=image_size,
        diffusion_seed=diffusion_seed,
        return_image=True,
    )

    # Save images with watermark
    for label, image_w in zip(labels, images_w):
        save_path = os.path.join(
            dataset_dir_path,
            f"{labels_to_winds[label]}_{key_seed}_{message_seed}",
            f"{diffusion_seed}.png",
        )
        image_w.save(save_path, "PNG")


# Create jobs for workers
def create_jobs(
    batch_size,
    labels,
    key_seeds,
    message_seeds,
    diffusion_seeds,
    devices,
    image_size,
    dataset_dir_path,
):
    for key_seed, message_seed, diffusion_seed in itertools.product(
        key_seeds, message_seeds, diffusion_seeds
    ):
        for i in range(0, len(labels), batch_size):
            batched_labels = labels[i : min(i + batch_size, len(labels))]
            device = devices[i % len(devices)]
            yield (
                batched_labels,
                key_seed,
                message_seed,
                diffusion_seed,
                device,
                image_size,
                dataset_dir_path,
            )


# Generate watermarked images
def generate_watermarked_images(
    image_size,
    dataset_template,
    num_key_seeds,
    num_message_seeds,
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
    key_seeds = list(range(num_key_seeds))
    message_seeds = list(range(num_message_seeds))
    diffusion_seeds = list(range(num_diffusion_seeds))

    # Create dataset directory
    dataset_dir_name = f"tree_ring_guided_{image_size}_{dataset_template.lower()}_{num_key_seeds}k_{num_message_seeds}m"
    assert os.path.exists("./datasets/")
    dataset_dir_path = os.path.join("./datasets/", dataset_dir_name, "train")
    if os.path.exists(dataset_dir_path):
        print("Dataset directory already exists, aborted.")
        return
    os.makedirs(dataset_dir_path)
    for label, key_seed, message_seed in itertools.product(
        labels, key_seeds, message_seeds
    ):
        os.makedirs(
            os.path.join(
                dataset_dir_path, f"{labels_to_winds[label]}_{key_seed}_{message_seed}"
            )
        )

    # Create jobs for workers
    jobs = create_jobs(
        batch_size,
        labels,
        key_seeds,
        message_seeds,
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
            key_seeds,
            message_seeds,
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

    # Save tree-ring keys and messages
    keys = []
    for key_seed in key_seeds:
        keys.append(
            generate_tree_ring_key(
                key_seed=key_seed,
                image_size=image_size,
                tree_ring_paras=tree_ring_paras,
                device=torch.device("cuda:0"),
            )
        )
    messages = []
    for message_seed in message_seeds:
        messages.append(
            generate_tree_ring_message(
                message_seed=message_seed,
                image_size=image_size,
                tree_ring_paras=tree_ring_paras,
                device=torch.device("cuda:0"),
            )
        )
    dataset_dir_path = os.path.join("./datasets/", dataset_dir_name)
    torch.save(keys, os.path.join(dataset_dir_path, "keys.pt"))
    torch.save(messages, os.path.join(dataset_dir_path, "messages.pt"))

    # Print message
    print(f"Generated watermarked images saved to: {dataset_dir_path}")


# Parse arguments and call generate_watermarked_images
def main():
    parser = argparse.ArgumentParser(
        description="Generate watermarked images using guided diffusion model and original tree-ring watermark and pack into a dataset"
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

    # Number of message seeds argument
    parser.add_argument(
        "--num_key_seeds",
        type=int,
        default=1,
        help="Number of key seeds (positive integer)",
    )

    # Number of message seeds argument
    parser.add_argument(
        "--num_message_seeds",
        type=int,
        default=1,
        help="Number of message seeds (positive integer)",
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

    if (
        args.num_key_seeds <= 0
        or args.num_message_seeds <= 0
        or args.num_diffusion_seeds <= 0
        or args.batch_size <= 0
    ):
        assert False

    generate_watermarked_images(
        args.image_size,
        args.dataset_template,
        args.num_key_seeds,
        args.num_message_seeds,
        args.num_diffusion_seeds,
        args.batch_size,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
