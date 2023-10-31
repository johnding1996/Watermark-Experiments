import argparse
import torch
from datasets import Dataset, load_from_disk
from torchvision.datasets import ImageNet, CocoCaptions
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import open_clip
from dotenv import load_dotenv

load_dotenv()

clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")


def calculate_num_clip_tokens(prompt):
    assert isinstance(prompt, str)
    return (clip_tokenizer(prompt) == 0).sum().item()


def calculate_perplexity(prompt, stride=512):
    assert isinstance(prompt, str)
    encodings = ppl_tokenizer(prompt, return_tensors="pt")
    max_length = ppl_model.config.n_positions
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


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
        dataset = dataset.filter(
            lambda data: 0 < calculate_num_clip_tokens(data["prompt"]) < 74
        )
        dataset = dataset.map(
            lambda data: {
                "image": data["image"],
                "prompt": data["prompt"],
                "perplexity": calculate_perplexity(data["prompt"]),
            }
        )
        dataset = dataset.sort("perplexity")
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
        dataset = dataset.map(
            lambda data: {
                "image": data["image"],
                "prompt": data["prompt"],
                "perplexity": calculate_perplexity(data["prompt"]),
            }
        )
        dataset = dataset.sort("perplexity")
        # Skip the first 50 images with very low perplexity as there are repeatitions in prompts
        dataset = dataset.select(range(50, 50 + int(dataset_size[:-1]) * 1000))
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
