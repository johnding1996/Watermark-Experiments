# Package imports
import os
import sys
import argparse
import torch
from tqdm import trange
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from dotenv import load_dotenv

from guided_diffusion.generate import reverse_guided_diffusion

load_dotenv()

# Relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adversarial import *
from guided_diffusion import *
from tree_ring import *
from utils import *

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Tree-ring parameters
tree_ring_paras = dict(
    w_channel=2,
    w_pattern="ring",
    w_mask_shape="circle",
    w_radius=10,
    w_measurement="l1_complex",
    w_injection="complex",
)


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


def attack_surrogate_classifier(
    image_size,
    dataset_template,
    eps,
    alpha,
    steps,
    num_warm_up_iters,
    split_ratio,
    batch_size,
    exp_random_seed,
    verbose,
    eval_before_attack,
):
    # Set random seed for reproducibility
    set_random_seed(exp_random_seed)

    # Load datasets (as in the provided code)
    dataset_wo, class_names_wo = load_imagenet_guided(
        image_size, dataset_template, convert_to_tensor=True, norm_type="imagenet"
    )
    dataset_w, class_names_w, keys, messages = load_tree_ring_guided(
        image_size,
        dataset_template,
        num_key_seeds=1,
        num_message_seeds=1,
        convert_to_tensor=True,
        norm_type="imagenet",
    )
    merged_dataset = MergedDataset(dataset_wo, dataset_w)

    train_size = int(split_ratio * len(merged_dataset))
    # The test set is the same as the one used in the surrogate classifier training
    # Controlled by the experiment random seed
    train_size = int(split_ratio * len(merged_dataset))
    test_size = len(merged_dataset) - train_size
    train_dataset, test_dataset = random_split(merged_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet-18 model

    # Save and print to console
    save_path = f"./models/surrogate_classifier_tree_ring_guided_{image_size}_{dataset_template}_1k_1m.pt"
    classifier_model = load_classifier(save_path)
    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    # Load guided diffusion models which are class-conditional diffusion models trained on ImageNet
    model, diffusion = load_guided_diffusion_model(image_size, device)

    # Warmup to get the average delta for the attack
    average_delta_list = []
    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)

        # Attack images with watermark (we know watermark labels because it's on the training set)
        images_w = images[labels == 1]
        images_w_adv = pgd_attack_classifier(
            classifier_model,
            eps,
            alpha,
            steps,
            images_w,
            torch.ones(images_w.size(0)),
            init_delta=None,
        )
        average_delta_list.append((images_w_adv - images_w).mean(dim=0))

        if verbose and i > 0 and i % 5 == 0:
            average_delta_before = torch.cat(average_delta_list[:-1], dim=0).mean(dim=0)
            average_delta_after = torch.cat(average_delta_list, dim=0).mean(dim=0)
            abs_diff = torch.abs(average_delta_before - average_delta_after)
            max_abs_diff = torch.max(abs_diff)
            relative_diff = abs_diff / torch.clamp_min(
                torch.max(average_delta_before, average_delta_after), 1e-10
            )
            max_relative_diff = torch.max(relative_diff)

            print(
                f"Warmup Iteration [{i:04d}] Max Absolute Diff: {max_abs_diff.item():.3e} Max Relative Diff: {max_relative_diff.item():.3e}"
            )

        if i >= num_warm_up_iters:
            break

    average_delta = torch.cat(average_delta_list, dim=0).mean(dim=0)

    # Attack the classifier on the test set
    reversed_latnents_wo_list, reversed_latnents_w_list = [], []
    reversed_latnents_adv_wo_list, reversed_latnents_adv_w_list = [], []
    for i, (images, labels) in enumerate(test_loader):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)

        if eval_before_attack:
            # Reverse diffusion on images with and without watermark
            reversed_latents = reverse_guided_diffusion(
                model,
                diffusion,
                images=renormalize_tensor(
                    images, in_norm_type="imagenet", out_norm_type="naive"
                ),
                image_size=image_size,
                return_image=False,
            )
            reversed_latnents_wo_list.append(reversed_latents[labels == 0].cpu())
            reversed_latnents_w_list.append(reversed_latents[labels == 1].cpu())

            # Detect and evaluate watermark
            auc, acc, low = detect_tree_ring(
                torch.cat(reversed_latnents_wo_list, dim=0),
                torch.cat(reversed_latnents_w_list, dim=0),
                keys=keys[0],
                messages=messages[0],
                tree_ring_paras=tree_ring_paras,
                image_size=image_size,
            )
            print(
                f"Iteration [{(i+1):04d}] Before Attack So Far; ACC={100*acc:.2f}% AUC={auc:.4f} TPR@1%FPR={low:.4f}"
            )

        # Attack images with and without watermark  (always attack to non-watermark class)
        images_adv = pgd_attack_classifier(
            classifier_model,
            eps,
            alpha,
            steps,
            images,
            labels,
            init_delta=average_delta,
        )

        # Reverse diffusion on images with and without watermark
        reversed_latents_adv = reverse_guided_diffusion(
            model,
            diffusion,
            images=renormalize_tensor(
                images_adv, in_norm_type="imagenet", out_norm_type="naive"
            ),
            image_size=image_size,
            return_image=False,
        )
        reversed_latnents_adv_wo_list.append(reversed_latents_adv[labels == 0].cpu())
        reversed_latnents_adv_w_list.append(reversed_latents_adv[labels == 1].cpu())

        # Detect and evaluate watermark
        auc, acc, low = detect_tree_ring(
            torch.cat(reversed_latnents_adv_wo_list, dim=0),
            torch.cat(reversed_latnents_adv_w_list, dim=0),
            keys=keys[0],
            messages=messages[0],
            tree_ring_paras=tree_ring_paras,
            image_size=image_size,
        )
        if verbose:
            print(
                f"Iteration [{(i+1):04d}] After Attack So Far;  ACC={100*acc:.2f}% AUC={auc:.4f} TPR@1%FPR={low:.4f}"
            )

    print(f"After Attack; ACC={acc:.4f} AUC={auc:.4f} TPR@1%FPR={low:.4f}")

    assert False


def main():
    parser = argparse.ArgumentParser(
        description="PGD Attack using the finetuned surrogate clasifier on test split."
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

    # Attack hyper-parameters
    parser.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="Epsilon.",
    )
    parser.add_argument(
        "--alpha_to_eps_ratio",
        type=float,
        default=0.05,
        help="Epsilon.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Step size.",
    )
    parser.add_argument(
        "--num_warm_up_iters",
        type=int,
        default=20,
        help="Number of warm up iterations.",
    )

    # Other hyper-parameters
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Split raio.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--eval_before_attack",
        action="store_true",
        help="Evalute before attack peformance for snaity check.",
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

    attack_surrogate_classifier(
        args.image_size,
        args.dataset_template,
        args.eps,
        args.eps * args.alpha_to_eps_ratio,
        args.steps,
        args.num_warm_up_iters,
        args.split_ratio,
        args.batch_size,
        args.exp_random_seed,
        args.verbose,
        args.eval_before_attack,
    )


if __name__ == "__main__":
    main()
