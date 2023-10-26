# Package imports
import os
import sys
import argparse
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


# Relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adversarial import *
from tree_ring import *
from utils import *

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Merged dataset class
class MergedDataset(Dataset):
    def __init__(self, dataset_wo, dataset_w):
        self.dataset_wo = dataset_wo
        self.dataset_w = dataset_w
        self.length_wo = len(dataset_wo)
        self.length_w = len(dataset_w)

    def __len__(self):
        return self.length_wo + self.length_w

    def __getitem__(self, index):
        if index < self.length_wo:
            image = self.dataset_wo[index][0].squeeze()
            label = 0
        else:
            image = self.dataset_w[index - self.length_wo][0].squeeze()
            label = 1
        return image, label


def train_surrogaate_classifier(
    image_size,
    dataset_template,
    learning_rate,
    num_epochs,
    split_ratio,
    batch_size,
    exp_random_seed,
    verbose,
):
    # Set random seed
    set_random_seed(exp_random_seed)

    # Load generated images without watermarks
    dataset_wo, class_names_wo = load_imagenet_guided(
        image_size, dataset_template, convert_to_tensor=True, norm_type="imagenet"
    )

    # Load generated images with watermarks
    dataset_w, class_names_w, keys, messages = load_tree_ring_guided(
        image_size,
        dataset_template,
        num_key_seeds=1,
        num_message_seeds=1,
        convert_to_tensor=True,
        norm_type="imagenet",
    )

    # Create the merged dataset
    merged_dataset = MergedDataset(dataset_wo, dataset_w)

    # Define the split sizes
    train_size = int(split_ratio * len(merged_dataset))
    test_size = len(merged_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the surrogate classifier
    model = train_classifier(
        train_loader, test_loader, device, learning_rate, num_epochs, verbose=verbose
    )

    # Save and print to console
    save_path = f"./models/surrogate_classifier_tree_ring_guided_{image_size}_{dataset_template}_1k_1m.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Trained surrogate classifier saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate clasifier on generate watermarked watermarked and not watermarked images."
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

    # Training hyper-parameters
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Split raio.",
    )
    parser.add_argument(
        "--batch_size",
        type=float,
        default=256,
        help="Batch size.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose.",
    )

    # Experiment random seed
    parser.add_argument(
        "--exp_random_seed",
        type=int,
        default=0,
        help="Experiment random seed.",
    )

    args = parser.parse_args()

    train_surrogaate_classifier(
        args.image_size,
        args.dataset_template,
        args.learning_rate,
        args.num_epochs,
        args.split_ratio,
        args.batch_size,
        args.exp_random_seed,
        args.verbose,
    )


if __name__ == "__main__":
    main()
