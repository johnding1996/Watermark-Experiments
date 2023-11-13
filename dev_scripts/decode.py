import os
import click
import torch
import numpy as np
import onnxruntime as ort
import torch.multiprocessing as mp
import orjson
import base64
from PIL import Image, ImageOps
import warnings
from tqdm.auto import tqdm
import dotenv
from space import (
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
    encode_array_to_string,
)
from utils import to_tensor

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")


def get_indices(mode, path, quiet, subset, limit, subset_limit):
    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    reversed_latents_existences = check_file_existence(
        path, name_pattern="{}_reversed.pkl", limit=limit
    )
    if not quiet:
        print(
            f"Found {sum(image_existences)} images, and {sum(reversed_latents_existences)} reversed latents"
        )
    if mode == "tree_ring":
        existences = reversed_latents_existences
    elif mode in ["stable_sig", "stegastamp"]:
        existences = image_existences
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-decode.json",
    )
    if not os.path.exists(json_path):
        indices = existence_to_indices(
            existences, limit=limit if not subset else subset_limit
        )
    else:
        data = load_json(json_path)
        decoded_existences = [data[str(i)][mode] is not None for i in range(limit)]
        indices = existence_to_indices(
            existence_operation(existences, decoded_existences, op="difference"),
            limit=limit if not subset else subset_limit,
        )
    return indices


def init_decoder_model(mode, gpu):
    if mode == "tree_ring":
        size = 64
        radius = 10
        channel = 3
        mask = torch.zeros((1, 4, size, size), dtype=torch.bool)
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        mask[:, channel] = torch.tensor(((x - x0) ** 2 + (y - y0) ** 2) <= radius**2)
        return mask
    elif mode == "stable_sig":
        model = torch.jit.load(
            os.path.join(os.environ.get("MODEL_DIR"), "stable_signature.pt")
        ).to(f"cuda:{gpu}")
        return model
    elif mode == "stegastamp":
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        return ort.InferenceSession(
            os.path.join(os.environ.get("MODEL_DIR"), "stega_stamp.onnx"),
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )


def decode_image_or_reversed_latents(mode, model, gpu, path, idx):
    if mode == "tree_ring":
        reversed_latents = torch.load(os.path.join(path, f"{idx}_reversed.pkl"))
        reversed_latents_fft = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents), dim=(-1, -2)
        )[model].flatten()
        return (
            torch.concatenate([reversed_latents_fft.real, reversed_latents_fft.imag])
            .cpu()
            .numpy()
        )
    elif mode == "stable_sig":
        image = Image.open(os.path.join(path, f"{idx}.png"))
        return (
            (model(to_tensor([image]).to(f"cuda:{gpu}")) > 0)
            .squeeze()
            .cpu()
            .numpy()
            .astype(bool)
        )
    elif mode == "stegastamp":
        image = Image.open(os.path.join(path, f"{idx}.png"))
        image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        secret = np.zeros((1, 100), dtype=np.float32)
        inputs = {
            "image": image,
            "secret": secret,
        }
        outputs = model.run(None, inputs)
        return outputs[2].flatten().astype(bool)


def worker(mode, gpu, path, indices, lock, counter, results):
    model = init_decoder_model(mode, gpu)
    for idx in indices:
        message = decode_image_or_reversed_latents(mode, model, gpu, path, idx)
        with lock:
            counter.value += 1
            results[idx] = encode_array_to_string(message)


def process(mode, indices, path, quiet):
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    chunk_size = len(indices) // num_gpus

    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()

        processes = []
        for rank in range(num_gpus * 1):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_gpus * 1 - 1 else (rank + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    mode,
                    rank % num_gpus,
                    path,
                    indices[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(indices), desc="Decoding images or reversed latents", unit="file"
        ) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break

        for p in processes:
            p.join()

        return dict(results)
        # return {idx: orjson.loads(message) for idx, message in dict(results).items()}


def report(mode, path, results, quiet, limit):
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-decode.json",
    )
    if not os.path.exists(json_path):
        data = {}
        for idx in range(limit):
            data[str(idx)] = {
                "tree_ring": results.get(idx) if mode == "tree_ring" else None,
                "stable_sig": results.get(idx) if mode == "stable_sig" else None,
                "stegastamp": results.get(idx) if mode == "stegastamp" else None,
            }
    else:
        data = load_json(json_path)
        for idx, message in results.items():
            data[str(idx)][mode] = message
    save_json(data, json_path)
    if not quiet:
        print(f"Decoded messages saved to {json_path}")


def single_mode(mode, path, quiet, subset, limit, subset_limit):
    if not quiet:
        print(f"Decoding {mode} messages")
    indices = get_indices(mode, path, quiet, subset, limit, subset_limit)
    if len(indices) == 0:
        if not quiet:
            print("All messages requested already decoded")
        return
    results = process(mode, indices, path, quiet)
    report(mode, path, results, quiet, limit)


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(path, dry, subset, quiet, limit=5000, subset_limit=1000):
    _, _, _, source_name = parse_image_dir_path(path, quiet=quiet)
    # Reverse is only required for real and tree_ring watermarked images
    if source_name == "real":
        for mode in ["tree_ring", "stable_sig", "stegastamp"]:
            single_mode(mode, path, quiet, subset, limit, subset_limit)
    elif source_name.endswith("tree_ring"):
        single_mode("tree_ring", path, quiet, subset, limit, subset_limit)
    elif source_name.endswith("stable_sig"):
        single_mode("stable_sig", path, quiet, subset, limit, subset_limit)
    elif source_name.endswith("stegastamp"):
        single_mode("stegastamp", path, quiet, subset, limit, subset_limit)
    else:
        raise ValueError(f"Invalid source name {source_name} encountered")


if __name__ == "__main__":
    main()
