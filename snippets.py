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

# Tree-ring watermark parameters
tree_ring_paras = dict(
    w_channel=2,
    w_pattern="ring",
    w_mask_shape="circle",
    w_radius=10,
    w_measurement="l1_complex",
    w_injection="complex",
    w_pattern_const=0,
)

# Load dataset and split into train and test sets
dataset, class_names = load_imagenet_subset(dataset_name)
train_images, train_labels, test_images, test_labels = sample_images_and_labels(
    train_size, test_size, dataset, exp_rand_seed
)

# Load guided diffusion models which are class-conditional diffusion models trained on ImageNet
model, diffusion = load_guided_diffusion_model(image_size, device)

# Visualize
visualize_imagenet_subset(dataset, class_names, n_classes=3, n_samples_per_class=3)
