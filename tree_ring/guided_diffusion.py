import numpy as np
import torch
from sklearn import metrics
from PIL import Image
from utils import set_random_seed
from collections import namedtuple
from guided_diffusion.script_util import NUM_CLASSES
from .optim_utils import (
    get_watermarking_pattern,
    get_watermarking_mask,
    inject_watermark,
    eval_watermark,
)
from utils.data_utils import to_tensor_and_normalize, unnormalize_and_to_pil


# Guided diffusion without watermark
def guided_diffusion_without_watermark(
    model,
    diffusion,
    labels,
    image_size,
    diffusion_seed,
    progressive=False,
    return_image=False,
):
    # Diffusion seed is the random seed for diffusion sampling
    set_random_seed(diffusion_seed)
    # For guided diffusion, prompts are class ids
    assert isinstance(labels, list) and all(
        isinstance(label, int) and 0 <= label < NUM_CLASSES for label in labels
    )
    # Device and shape
    device = next(model.parameters()).device
    shape = (len(labels), 3, image_size, image_size)
    # The random initial latent is determined by the diffusion seed, so no need to keep it
    init_latents_wo = torch.randn(*shape, device=device)
    # Diffusion
    if not progressive:
        output = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latents_wo,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
            return_image=return_image,
            clip_denoised=False,
        )
        return output
    else:
        output = []
        for sample in diffusion.ddim_sample_loop_progressive(
            model=model,
            shape=shape,
            noise=init_latents_wo,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
        ):
            if not return_image:
                output.append(sample["sample"])
            else:
                output.append(
                    unnormalize_and_to_pil(sample["sample"], norm_type="Naive")
                )
        return output


# Generate a message (which is the key in tree-ring's paper) with a specific message seed
def generate_message(message_seed, image_size, tree_ring_paras, device):
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", list(tree_ring_paras.keys()) + ["w_seed"])(
        **tree_ring_paras, w_seed=message_seed
    )
    shape = (1, 3, image_size, image_size)
    # Generate the message, which is the key in tree-ring's paper
    message = get_watermarking_pattern(None, tree_ring_args, device, shape)
    # Message's shape is (1, 3, image_size, image_size)
    return message


# Generate a key (which is the mask in tree-ring's paper) with a specific key seed
def generate_key(key_seed, image_size, tree_ring_paras, device):
    # For tree-ring, the key (watermarking mask) is not randomized and fully determined by the w_radius and image size
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
    shape = (1, 3, image_size, image_size)
    # In get_watermarking_mask, only the shape of init_latents_w matters, not its values
    # So we can just use random values
    init_latents_w = torch.randn(*shape, device=device)
    # Generate the key, which is the mask in tree-ring's paper
    key = get_watermarking_mask(init_latents_w, tree_ring_args, device=device)
    # Key's shape is (1, 3, image_size, image_size)
    return key


# Guided diffusion with watermark
def guided_diffusion_with_watermark(
    model,
    diffusion,
    labels,
    keys,
    messages,
    tree_ring_paras,
    image_size,
    diffusion_seed,
    progressive=False,
    return_image=False,
):
    # Diffusion seed is the random seed for diffusion sampling
    set_random_seed(diffusion_seed)
    # For guided diffusion, prompts are class ids
    assert isinstance(labels, list) and all(
        isinstance(label, int) and 0 <= label < NUM_CLASSES for label in labels
    )
    # Assert key and message are on the same device as the model
    assert keys.device == messages.device == next(model.parameters()).device
    # Can either use the same key or message for all images, or use different keys or messages for different images
    if keys.size()[0] == 1:
        keys = keys.repeat(len(labels), 1, 1, 1)
    if messages.size()[0] == 1:
        messages = messages.repeat(len(labels), 1, 1, 1)
    assert keys.size() == messages.size() == (len(labels), 3, image_size, image_size)

    # Device and shape
    device = next(model.parameters()).device
    shape = (len(labels), 3, image_size, image_size)
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
    # The random initial latent is determined by the diffusion seed, so no need to keep it
    init_latents_wo = torch.randn(*shape, device=device)
    # Inject watermark
    init_latents_w = inject_watermark(init_latents_wo, keys, messages, tree_ring_args)
    # Diffusion
    if not progressive:
        output = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latents_w,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
            return_image=return_image,
        )
        return output
    else:
        output = []
        for sample in diffusion.ddim_sample_loop_progressive(
            model=model,
            shape=shape,
            noise=init_latents_w,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
        ):
            if not return_image:
                output.append(sample["sample"])
            else:
                output.append(
                    unnormalize_and_to_pil(sample["sample"], norm_type="Naive")
                )
        return output


# Reverse guided diffusion
def reverse_guided_diffusion(
    model,
    diffusion,
    images,
    image_size,
    default_labels=0,
    progressive=False,
    return_image=False,
):
    # Reverse diffusion of DDIM smapling is deterministic, so this line has no effect
    set_random_seed(0)
    # Device and shape
    device = next(model.parameters()).device
    shape = (len(images), 3, image_size, image_size)
    # If default labels is a single int, repeat it for all images
    if isinstance(default_labels, int):
        default_labels = [default_labels] * len(images)
    # Check whether the inputs are PIL images
    input_image = isinstance(images[0], Image.Image)
    # Reversed diffusion
    if not progressive:
        output = diffusion.ddim_reverse_sample_loop(
            model=model,
            shape=shape,
            image=images,
            # Reverse diffusion does not depends on the labels, thus pass in dummy labels
            model_kwargs=dict(y=torch.tensor(default_labels, device=device)),
            device=device,
            input_image=input_image,
            clip_denoised=False,
        )
        if not return_image:
            return output
        else:
            return unnormalize_and_to_pil(output, norm_type="Naive")
    else:
        output = []
        for sample in diffusion.ddim_reverse_sample_loop_progressive(
            model=model,
            shape=shape,
            image=images,
            # Reverse diffusion does not depends on the labels, thus pass in dummy labels
            model_kwargs=dict(y=torch.tensor(default_labels, device=device)),
            device=device,
            input_image=input_image,
        ):
            if not return_image:
                output.append(sample["sample"])
            else:
                output.append(
                    unnormalize_and_to_pil(sample["sample"], norm_type="Naive")
                )
        return output


# Detect and evaluate watermark
def detect_evaluate_watermark(
    reversed_latents_wo, reversed_latents_w, keys, messages, tree_ring_paras, image_size
):
    # Assert key and message are on the same device
    assert keys.device == messages.device
    # Check whether the inputs are PIL images
    if isinstance(reversed_latents_wo[0], Image.Image):
        reversed_latents_wo = to_tensor_and_normalize(
            reversed_latents_wo, norm_type="Naive"
        ).to(keys.device)
        reversed_latents_w = to_tensor_and_normalize(
            reversed_latents_w, norm_type="Naive"
        ).to(keys.device)
    else:
        reversed_latents_wo = reversed_latents_wo.to(keys.device)
        reversed_latents_w = reversed_latents_w.to(keys.device)
    # To evaluate, currently we require images with and without watermark appear in pairs
    assert len(reversed_latents_wo) == len(reversed_latents_w)
    # Can either use the same key or message for all images, or use different keys or messages for different images
    if keys.size()[0] == 1:
        keys = keys.repeat(len(reversed_latents_wo), 1, 1, 1)
    if messages.size()[0] == 1:
        messages = messages.repeat(len(reversed_latents_wo), 1, 1, 1)
    assert (
        keys.size()
        == messages.size()
        == (len(reversed_latents_wo), 3, image_size, image_size)
    )
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)

    # Evaluation by measuring the L1 distance to the true message under key
    distances_wo, distances_w = eval_watermark(
        reversed_latents_wo, reversed_latents_w, keys, messages, tree_ring_args
    )
    # Calculate the AUROC scores and related
    y_true = [1] * len(distances_wo) + [0] * len(distances_w)
    y_score = distances_wo + distances_w
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.01)[0][-1]]
    return auc, acc, low
