import os
import torch
import PIL
import warnings
import diffusers
from diffusers import DPMSolverMultistepScheduler
from tree_ring import InversableStableDiffusionPipeline
from utils import to_tensor
import torch.multiprocessing as mp
from multiprocessing import Manager
from tqdm.auto import tqdm

diffusers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def validate_image_directory_name(path):
    attack_dirname = str(path).split("/")[-1]
    if not len(attack_dirname.split("-")) == 3:
        raise ValueError(
            f"Attack directory name {attack_dirname} is not in the format of 'attack_name-strength-source_name'"
        )
    attack_name, attack_strength, source_name = attack_dirname.split("-")
    try:
        attack_strength = float(attack_strength)
        if attack_strength <= 0:
            raise ValueError("Strength must be positive")
    except ValueError:
        raise ValueError("Strength must be a number")
    if not source_name in [
        "real",
        "stable_sig",
        "stegastamp",
        "tree_ring",
        "real_stable_sig",
        "real_stegastamp",
        "real_tree_ring",
    ]:
        raise ValueError(
            "Source name must be one of ['real', 'stable_sig', 'stegastamp', 'tree_ring']"
        )
    return attack_name, attack_strength, source_name


def validate_image_directory_files(path, limit=5000):
    found_filenames = list(os.listdir(path))
    expected_filenames = [f"{i}.png" for i in range(limit)]
    for filename in expected_filenames:
        if filename not in found_filenames:
            raise FileNotFoundError(f"Image file {filename} not found in {path}")


def check_existing_reversed_latents(path, limit=5000):
    found_filenames = list(os.listdir(path))
    expected_filenames = [f"{i}_reversed.pkl" for i in range(limit)]
    found_indices = []
    for filename in expected_filenames:
        if filename in found_filenames:
            found_indices.append(int(filename.split("_")[0]))
    return found_indices


# Function to initialize the model
def init_model(device):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# Reverse image function adapted for multiprocessing
def reverse_image(pipe, path, idx, device):
    num_inference_steps = 50
    tester_prompt = ""  # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    image = PIL.Image.open(os.path.join(path, f"{idx}.png"))
    image_transformed = to_tensor([image]).to(text_embeddings.dtype).to(device)
    image_latents = pipe.get_image_latents(image_transformed, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        verbose=False,
    )
    torch.save(reversed_latents, os.path.join(path, f"{idx}_reversed.pkl"))


# Worker function for multiprocessing
def worker(gpu, path, indices, counter, lock):
    device = f"cuda:{gpu}"
    pipe = init_model(device)
    for idx in indices:
        reverse_image(pipe, path, idx, device)
        with lock:
            counter.value += 1


def main(path, limit=5000):
    attack_name, attack_strength, source_name = validate_image_directory_name(path)
    if not source_name in ["real", "tree_ring", "real_tree_ring"]:
        raise ValueError(
            f"Reverse diffusion is only required for real and tree_ring watermarked images, not {source_name}"
        )
    print(" -- Attack name:", attack_name)
    print(" -- Attack strength:", attack_strength)
    print(" -- Source name:", source_name)
    print()
    validate_image_directory_files(path, limit)
    found_indices = check_existing_reversed_latents(path, limit)
    print(f"Found {limit} images, and {len(found_indices)} reversed latents")
    if len(found_indices) == limit:
        print("All reversed latents are found, exiting")
        return
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    print(f"Using {num_gpus} GPUs for processing")
    remained_indices = list(set(range(limit)) - set(found_indices))
    chunk_size = len(remained_indices) // num_gpus

    with Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        processes = []

        for gpu in range(num_gpus):
            start_idx = gpu * chunk_size
            end_idx = None if gpu == num_gpus - 1 else (gpu + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(gpu, path, remained_indices[start_idx:end_idx], counter, lock),
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(remained_indices)) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(remained_indices):
                        break

        for p in processes:
            p.join()


if __name__ == "__main__":
    current_path = os.getcwd()
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    main(current_path)
