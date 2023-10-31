import argparse
from datasets import Dataset, load_from_disk
from torchvision.datasets import ImageNet, CocoCaptions
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import trange
from metrics import load_open_clip_tokenizer, compute_open_clip_num_tokens
from dotenv import load_dotenv

load_dotenv()


def prepare_standard_dataset(image_size, dataset_source, prompt_type, dataset_size):
    if dataset_source == "MSCOCO":
        assert image_size in [64, 256, 512]
        assert dataset_size in ["50", "200", "1k", "5k"]
        assert prompt_type == "string"
        # Need to download the dataset from https://cocodataset.org/#download first
        dataset = CocoCaptions(
            root="./datasets/source/mscoco/val2017/",
            annFile="./datasets/source/mscoco/annotations/captions_val2017.json",
        )
        images, prompts = [], []
        for i in trange(len(dataset)):
            image = dataset[i][0]
            transform = Compose(
                [CenterCrop(min(image.size)), Resize((image_size, image_size))]
            )
            images.append(transform(image))
            prompts.append(dataset[i][1][0])
        dataset = Dataset.from_dict({"image": images, "prompt": prompts}, num_proc=8)
        # Filter out prompts with 0 or >=74 CLIP tokens
        # For MSCOCO, this does not filter out any data
        clip_tokenizer = load_open_clip_tokenizer()
        dataset = dataset.filter(
            lambda data: 0
            < compute_open_clip_num_tokens(
                data["prompt"], clip_tokenizer=clip_tokenizer
            )
            < 74
        )
        dataset.save_to_disk("./datasets/mscoco_5k/")

    elif dataset_source == "DiffusionDB":
        # Only support <=10k for now
        assert dataset_size in ["50", "200", "1k", "5k", "10k"]
        assert prompt_type == "string"
        dataset = load_from_disk("./datasets/source/diffusiondb/2m_random_50k")
        dataset = dataset.filter(
            # filter width and height
            lambda x: (x["width"] == 512 and x["height"] == 512)
            # filter diffusion hyperparameters
            and (x["step"] == 50 and x["cfg"] == 7 and x["sampler"] == "k_lms")
            # filter nsfw
            and (x["image_nsfw"] < 0.2 and x["prompt_nsfw"] < 0.1)
        )
        dataset = dataset.remove_columns(
            [
                "seed",
                "step",
                "cfg",
                "sampler",
                "width",
                "height",
                "user_name",
                "timestamp",
                "image_nsfw",
                "prompt_nsfw",
            ]
        )
        # Filter out prompts with 0 or >=74 CLIP tokens
        # For DiffusionDB, this is necessary as many prompts are too long
        clip_tokenizer = load_open_clip_tokenizer()
        dataset = dataset.filter(
            lambda data: 0
            < compute_open_clip_num_tokens(
                data["prompt"], clip_tokenizer=clip_tokenizer
            )
            < 74
        )
        dataset.save_to_disk("./datasets/diffusiondb_5k/")

    elif dataset_source == "DALLE3":
        # Only support <=5k for now
        assert dataset_size in ["50", "200", "1k", "5k"]

    elif dataset_source == "ImageNet":
        # Only support >=1k and <=50k for now
        assert dataset_size in ["1k", "5k", "10k", "50k"]
        assert prompt_type in ["string", "label"]
        dataset = ImageNet(root="./datasets/source/imagenet/imagenet-val", split="val")

    elif dataset_source == "TinyImageNet":
        # Only support >=200 and <=10k for now
        assert dataset_size in ["200", "1k", "5k", "10k"]
        assert prompt_type in ["string", "label"]
        dataset = ImageNet(root="./datasets/source/tiny-imagenet-200/val", split="val")

    else:
        assert False


# Parse arguments
def main():
    parser = argparse.ArgumentParser(
        description="Prepare standard datasets for watermark and attack evaluation."
    )

    # Image size argument
    parser.add_argument(
        "--image_size",
        type=int,
        choices=[64, 256, 512, 1024],
        default=512,
        help="Image size",
    )

    # Dataset source argument
    parser.add_argument(
        "--dataset_source",
        type=str,
        choices=["MSCOCO", "DiffusionDB", "DALLE3", "ImageNet", "TinyImageNet"],
        default="DiffusionDB",
        help="Dataset source",
    )

    # Prompt type argument
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["string", "label"],
        default="string",
        help="Prompt type",
    )

    # Dataset source argument
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["50", "200", "1k", "5k", "10k", "50k"],
        default="5k",
        help="Dataset size",
    )

    args = parser.parse_args()

    prepare_standard_dataset(
        args.image_size,
        args.dataset_source,
        args.prompt_type,
        args.dataset_size,
    )


if __name__ == "__main__":
    main()
