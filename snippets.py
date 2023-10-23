# Visualizing the evolution of tree-ring watermark through diffusion

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random


def paired_transforms(img1, img2, img3, type=None, index=None):
    # Ensure img1 and img2 are PIL Images
    assert type in ["Rotation", "RandomResizedCrop", "RandomErasing", "IndexedErasing"]

    if type == "Rotation":
        # Rotation
        angle = random.uniform(-30, 30)  # Random rotation by up to 30 degrees
        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)
    elif type == "RandomResizedCrop":
        # Random Resized Crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img1, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)
        )
        img1 = F.resized_crop(img1, i, j, h, w, (64, 64))
        img2 = F.resized_crop(img2, i, j, h, w, (64, 64))
    elif type == "RandomErasing":
        # Cutout
        x, y = random.randint(0, img1.width), random.randint(0, img1.height)
        h, w = random.randint(
            int(0.02 * img1.width), int(0.33 * img1.width)
        ), random.randint(int(0.02 * img1.height), int(0.33 * img1.height))
        img1, img2 = transforms.ToTensor()(img1), transforms.ToTensor()(img2)
        img1 = F.erase(img1, x, y, h, w, v=0)
        img2 = F.erase(img2, x, y, h, w, v=0)
        img1, img2 = transforms.ToPILImage()(img1), transforms.ToPILImage()(img2)
    elif type == "IndexedErasing":
        assert index >= 0 and index <= 8 * 8
        # Cutout
        x, y = 8 * (index // 8), 8 * (index % 8)
        h, w = 16, 16
        img1, img2, img3 = (
            transforms.ToTensor()(img1),
            transforms.ToTensor()(img2),
            transforms.ToTensor()(img3),
        )
        img1 = F.erase(img1, x, y, h, w, v=0)
        img2 = F.erase(img2, x, y, h, w, v=0)
        img3 = F.erase(img3, x, y, h, w, v=0)
        img1, img2, img3 = (
            transforms.ToPILImage()(img1),
            transforms.ToPILImage()(img2),
            transforms.ToPILImage()(img3),
        )
    return img1, img2, img3


def generate_and_compare_reverse(
    model,
    diffusion,
    prompt,
    key,
    image_size,
    tree_ring_paras,
    init_latents_w,
    watermarking_mask,
    diffusion_seed,
):
    set_random_seed(diffusion_seed)
    # For this class-conditioned diffusion model, prompts are just class ids
    assert isinstance(prompt, int) and 0 <= prompt < 1000
    # For simplicity, fix batch size to one
    shape = (1, 3, image_size, image_size)
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
    # Unnormalize for
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    unnormalize = transforms.Normalize(
        (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
        (1 / std[0], 1 / std[1], 1 / std[2]),
    )
    # First pic
    # Diffusion w/o watermark
    no_wm_iter = diffusion.ddim_sample_loop_progressive(
        model=model,
        shape=shape,
        noise=init_latents_w,
        clip_denoised=True,
        model_kwargs=dict(y=torch.tensor([prompt], device=device)),
        device=device,
    )
    # Diffusion w watermark
    wm_iter = diffusion.ddim_sample_loop_progressive(
        model=model,
        shape=shape,
        noise=inject_watermark(init_latents_w, watermarking_mask, key, tree_ring_args),
        clip_denoised=True,
        model_kwargs=dict(y=torch.tensor([prompt], device=device)),
        device=device,
    )
    # Main loop
    image_list = []
    for no_wm_sample, wm_sample in zip(no_wm_iter, wm_iter):
        diff_init = (
            torch.abs(
                unnormalize(no_wm_sample["sample"][0])
                - unnormalize(wm_sample["sample"][0])
            )
            * 10
        )
        break

    # Diffusion
    no_wm_output = diffusion.ddim_sample_loop(
        model=model,
        shape=shape,
        noise=init_latents_w,
        clip_denoised=True,
        model_kwargs=dict(y=torch.tensor([prompt], device=device)),
        device=device,
        return_image=True,
    )
    wm_output = diffusion.ddim_sample_loop(
        model=model,
        shape=shape,
        noise=inject_watermark(init_latents_w, watermarking_mask, key, tree_ring_args),
        clip_denoised=True,
        model_kwargs=dict(y=torch.tensor([prompt], device=device)),
        device=device,
        return_image=True,
    )
    no_wm_image, wm_image = no_wm_output[0], wm_output[0]
    # no_wm_image = unnormalize(no_wm_image).permute(1, 2, 0).cpu()*255
    # wm_image = unnormalize(wm_image).permute(1, 2, 0).cpu()*255

    image_list = []
    for index in trange(8 * 8):
        # Augmentation
        no_wm_image_aug, wm_image_aug, diff_init_aug = paired_transforms(
            no_wm_image,
            wm_image,
            transforms.ToPILImage()(diff_init),
            type="IndexedErasing",
            index=index,
        )

        # Reverse Diffusion w/o watermark
        no_wm_iter_reverse = diffusion.ddim_reverse_sample_loop_progressive(
            model=model,
            shape=shape,
            image=no_wm_image_aug,
            clip_denoised=True,
            model_kwargs=dict(y=torch.tensor([prompt], device=device)),
            device=device,
        )
        # Reverse Diffusion w watermark
        wm_iter_reverse = diffusion.ddim_reverse_sample_loop_progressive(
            model=model,
            shape=shape,
            image=wm_image_aug,
            clip_denoised=True,
            model_kwargs=dict(y=torch.tensor([prompt], device=device)),
            device=device,
        )
        # Main loop
        for no_wm_sample_reverse, wm_sample_reverse in zip(
            no_wm_iter_reverse, wm_iter_reverse
        ):
            no_wm_image_reverse, wm_image_reverse = (
                no_wm_sample_reverse["sample"][0],
                wm_sample_reverse["sample"][0],
            )
        # Plot
        fft_diff = torch.abs(
            torch.fft.fftshift(
                torch.fft.fft2(
                    unnormalize(no_wm_image_reverse) - unnormalize(wm_image_reverse)
                )  # , dim=(-1, -2) check if this is making difference
            )
        )
        fft_diff = fft_diff / fft_diff.max()
        fig = visualize_images(
            [
                [
                    transforms.ToTensor()(no_wm_image_aug),
                    transforms.ToTensor()(wm_image_aug),
                    transforms.ToTensor()(diff_init_aug),
                    torch.abs(
                        transforms.ToTensor()(no_wm_image_aug)
                        - transforms.ToTensor()(wm_image_aug)
                    )
                    * 10,
                    unnormalize(no_wm_image_reverse),
                    unnormalize(wm_image_reverse),
                    torch.abs(
                        unnormalize(no_wm_image_reverse) - unnormalize(wm_image_reverse)
                    )
                    * 10,
                    fft_diff,
                ]
            ],
            [
                "w/o watermark",
                "w/ watermark",
                "delta*10 (inited)",
                "delta*10",
                "w/o watermark (reversed)",
                "w/ watermark (reversed)",
                "delta*10 (reversed)",
                "fft-delta (reversed, normd)",
            ],
            [""],
            fontsize=10,
        )

        buf = BytesIO()  # in-memory binary stream
        fig.savefig(
            buf, format="png", dpi=200, bbox_inches="tight"
        )  # save figure to the stream in PNG format
        buf.seek(0)
        image_list.append(imageio.imread(buf))  # read image from the stream
        plt.show(fig)
        plt.close(fig)
    imageio.mimsave("vis_64x64_0_3.gif", image_list, loop=1, duration=0.5)
    return None


set_random_seed(0)
key = create_key(key_seed=0, image_size=image_size, tree_ring_paras=tree_ring_paras)
shape = (1, 3, image_size, image_size)
tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
init_latents_w = torch.randn(*shape, device=device)
watermarking_mask = get_watermarking_mask(init_latents_w, tree_ring_args, device=device)
generate_and_compare_reverse(
    model,
    diffusion,
    0,
    key,
    image_size,
    tree_ring_paras,
    init_latents_w,
    watermarking_mask,
    3,
)


# Package imports
import torch

# Relative imports
from tree_ring import *
from utils import *

# device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
num_gpus = torch.cuda.device_count()

# Experiment setups
dataset_name = "Tiny-ImageNet"
# All dataset sizes are actually doubled cause there are watermarked counterparts
train_size = 1500
test_size = 500
image_size = 64
exp_rand_seed = 0

# Load dataset and split into train and test sets
dataset, class_names = load_imagenet_subset(dataset_name)
train_images, train_labels, test_images, test_labels = sample_images_and_labels(
    train_size, test_size, dataset, exp_rand_seed
)

# Visualize
visualize_imagenet_subset(dataset, class_names, n_classes=3, n_samples_per_class=3)
