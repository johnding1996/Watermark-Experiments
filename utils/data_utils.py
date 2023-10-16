import torch
from torchvision import datasets, transforms
from PIL import Image
import json


# Normalize image tensors
def normalize_tensor(images, norm_type):
    assert norm_type in ["ImageNet", "Naive"]
    # Two possible normalization conventions
    if norm_type == "ImageNet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
    elif norm_type == "Naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
    else:
        assert False
    return torch.stack([normalize(image) for image in images])


# Unnormalize image tensors
def unnormalize_tensor(images, norm_type):
    assert norm_type in ["ImageNet", "Naive"]
    # Two possible normalization conventions
    if norm_type == "ImageNet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    elif norm_type == "Naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    else:
        assert False
    return torch.stack([unnormalize(image) for image in images])


# Convert PIL images to tensors and normalize
def to_tensor_and_normalize(images, norm_type=None):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images


# Unnormalize tensors and convert to PIL images
def unnormalize_and_to_pil(images, norm_type=None):
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [transforms.ToPILImage()(image) for image in images.cpu()]


# Get ImageNet class names
def get_imagenet_class_names(labels=None):
    with open("./datasets/imagenet_class_index.json", "r") as file:
        cid_to_wnid_and_words = json.load(file)
    if labels is None:
        return {int(cid): words for cid, (wnid, words) in cid_to_wnid_and_words.items()}
    else:
        return [cid_to_wnid_and_words[str(cid)][1] for cid in labels]


# Load dataset and class names
def load_imagenet_subset(dataset_name):
    assert dataset_name in ["Tiny-ImageNet", "Imagenette"]
    # Load WordNet IDs and class names
    with open("./datasets/tiny-imagenet-200/wnids.txt", "r") as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_words = {}
    with open("./datasets/tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            wnid, words = line.strip().split("\t")
            wnid_to_words[wnid] = words
    # Tiny-ImageNet
    if dataset_name == "Tiny-ImageNet":
        # Tiny-ImageNet dataset
        data_dir = "./datasets/tiny-imagenet-200"
        dataset = datasets.ImageFolder(
            f"{data_dir}/train",
            lambda x: to_tensor_and_normalize([x], norm_type="ImageNet"),
        )
        assert len(dataset) == 100000
        # Tiny-ImageNet class names
        class_names = [wnid_to_words[wnid] for wnid in sorted(wnids)]
        assert len(class_names) == 200
    # Imagenette
    elif dataset_name == "Imagenette":
        # Imagenette dataset
        data_dir = "./datasets/imagenette2-320"
        dataset = datasets.ImageFolder(
            f"{data_dir}/train",
            transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(256),
                    lambda x: to_tensor_and_normalize([x], norm_type="ImageNet"),
                ]
            ),
        )
        assert len(dataset) == 9469
        # Imagenette class names
        class_names = [wnid_to_words[wnid] for wnid in dataset.classes]
        assert len(class_names) == 10
    return dataset, class_names


# Sample images and labels from imagenet subsets
def sample_images_and_labels(train_size, test_size, dataset, exp_rand_seed):
    assert (train_size + test_size) <= len(dataset)
    # Reset random seed
    torch.manual_seed(exp_rand_seed)
    # Random sample without replacement
    indices = torch.randperm(len(dataset))[: train_size + test_size]
    images = [dataset[idx][0] for idx in indices]
    labels = [dataset[idx][1] for idx in indices]
    return (
        images[:train_size],
        labels[:train_size],
        images[-test_size:],
        labels[-test_size:],
    )
