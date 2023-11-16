import os
import click
import torch
import numpy as np
import torch.multiprocessing as mp
from PIL import Image
import warnings
from tqdm.auto import tqdm
import dotenv
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from metrics import (
    compute_fid,
    compute_psnr_repeated,
    compute_ssim_repeated,
    compute_nmi_repeated,
    load_aesthetics_and_artifacts_models,
    compute_aesthetics_and_artifacts_scores,
    load_open_clip_model_preprocess_and_tokenizer,
    compute_clip_score,
)
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    QUALITY_METRICS,
    get_all_image_dir_paths,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
)
from utils import to_tensor

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")

DELTA_METRICS = ["lpips", "aesthetics", "artifacts", "clip_score"]
SINGLE_METRICS = ["legacy_fid", "clip_fid", "psnr", "ssim"]


def get_indices(
    mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit
):
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-metric.json",
    )
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        if mode != "aesthetics_and_artifacts":
            measured_existences = [data[str(i)][mode] is not None for i in range(limit)]
        else:
            measured_existences = [
                (data[str(i)]["aesthetics"] is not None)
                and (data[str(i)]["artifacts"] is not None)
                for i in range(limit)
            ]
        if (not subset and sum(measured_existences) == limit) or (
            subset and sum(measured_existences[:subset_limit]) == subset_limit
        ):
            return []
    clean_image_existences = check_file_existence(
        clean_path, name_pattern="{}.png", limit=limit
    )
    attacked_image_existences = (
        check_file_existence(attacked_path, name_pattern="{}.png", limit=limit)
        if attacked_path is not None
        else [True] * limit
    )
    if mode.endswith("_fid") and sum(clean_image_existences) != limit:
        raise ValueError(f"Cannot compute FID if not all {limit} clean images exist")
    if not quiet:
        if attacked_path is None:
            print(f"Found {sum(clean_image_existences)} not-attacked images")
        else:
            print(
                f"Found {sum(attacked_image_existences)} attacked images and {sum(clean_image_existences)} corresponding not-attacked images"
            )
    existences = existence_operation(
        clean_image_existences, attacked_image_existences, op="union"
    )
    if os.path.exists(json_path):
        existences = existence_operation(
            existences, measured_existences, op="difference"
        )
    indices = existence_to_indices(
        existences,
        limit=limit if not subset else subset_limit,
    )
    return indices


def process_single(mode, indices, clean_path, attacked_path, quiet, limit):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    if mode.endswith("_fid"):
        metric = compute_fid(
            clean_path,
            attacked_path,
            mode=mode.split("_")[0],
            device=torch.device("cuda"),
            batch_size=128,
            num_workers=8,
            verbose=not quiet,
        )
        results = {
            idx: metric
            for idx in existence_to_indices(
                check_file_existence(attacked_path, name_pattern="{}.png", limit=limit)
            )
        }
        return results
    elif mode in ["psnr", "ssim", "nmi"]:
        clean_images = [
            Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices
        ]
        attacked_images = [
            Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices
        ]
        metrics = globals()[f"compute_{mode}_repeated"](
            clean_images, attacked_images, num_workers=8, verbose=not quiet
        )
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        return results


def init_model(mode, gpu):
    if mode == "lpips":
        return LearnedPerceptualImagePatchSimilarity(
            net_type="alex", reduction="mean"
        ).to(f"cuda:{gpu}")
    elif mode == "aesthetics_and_artifacts":
        return load_aesthetics_and_artifacts_models(device=torch.device(f"cuda:{gpu}"))
    elif mode == "clip_score":
        return load_open_clip_model_preprocess_and_tokenizer(
            device=torch.device(f"cuda:{gpu}")
        )


def load_files(mode, path, clean_path, attacked_path, indices):
    if mode == "lpips":
        clean_images = [
            Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices
        ]
        attacked_images = [
            Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices
        ]
        return clean_images, attacked_images
    elif mode == "aesthetics_and_artifacts":
        return [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]
    elif mode == "clip_score":
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]
        prompts = list(
            load_json(
                os.path.join(
                    os.environ.get("RESULT_DIR"),
                    str(path).split("/")[-2],
                    "prompts.json",
                )
            ).values()
        )
        return images, prompts


def measure(mode, model, gpu, inputs):
    if mode == "lpips":
        metric = (
            model(
                to_tensor(inputs[0]).to(f"cuda:{gpu}"),
                to_tensor(inputs[1]).to(f"cuda:{gpu}"),
            )
            .cpu()
            .item()
        )
        return [metric] * len(inputs[0])
    elif mode == "aesthetics_and_artifacts":
        return [
            compute_aesthetics_and_artifacts_scores(image, model) for image in inputs
        ]
    elif mode == "clip_score":
        return [
            compute_clip_score(image, prompt, model)
            for image, prompt in zip(inputs[0], inputs[1])
        ]


def worker(mode, gpu, path, clean_path, attacked_path, indices, lock, counter, results):
    model = init_model(mode, gpu)
    batch_size = {
        "lpips": 1,
        "aesthetics_and_artifacts": 1,
        "clip_score": 1,
    }[mode]
    for it in range(0, len(indices), batch_size):
        inputs = load_files(
            mode,
            path,
            clean_path,
            attacked_path,
            indices[it : min(it + batch_size, len(indices))],
        )
        metrics = measure(mode, model, gpu, inputs)
        with lock:
            counter.value += inputs.shape[0]
            for idx, metric in zip(
                indices[it : min(it + batch_size, len(indices))], metrics
            ):
                results[idx] = metric


def process_parallel(mode, indices, path, clean_path, attacked_path, quiet):
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    num_workers = {
        "lpips": num_gpus,
        "aesthetics": num_gpus,
        "artifacts": num_gpus,
        "clip_score": num_gpus,
    }[mode]
    chunk_size = len(indices) // num_workers
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()

        processes = []
        for rank in range(num_workers):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_workers - 1 else (rank + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    mode,
                    rank % num_gpus,
                    path,
                    clean_path,
                    attacked_path,
                    indices[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(indices), desc="Computing metrics on images", unit="image"
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


def report(mode, path, results, quiet, limit):
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-metric.json",
    )
    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {
                _mode: results.get(idx) if mode == _mode else None
                for _mode in QUALITY_METRICS.keys()
            }
    else:
        for idx, metric in results.items():
            data[str(idx)][mode] = metric
    save_json(data, json_path)
    if not quiet:
        print(f"Computed {mode} metrics saved to {json_path}")


def single_mode(
    mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit
):
    if not quiet:
        print(f"Computing {mode} metrics")
    indices = get_indices(
        mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit
    )
    if len(indices) == 0:
        if not quiet:
            print(f"All {mode} metrics requested already computed")
        return

    if mode in SINGLE_METRICS:
        results = process_single(mode, indices, clean_path, attacked_path, quiet, limit)
    else:
        results = process_parallel(
            mode, indices, path, clean_path, attacked_path, quiet
        )
    if mode == "aesthetics_and_artifacts":
        report(
            "aesthetics",
            path,
            {idx: result[0] for idx, result in results.items()},
            quiet,
            limit,
        )
        report(
            "artifacts",
            path,
            {idx: result[1] for idx, result in results.items()},
            quiet,
            limit,
        )
    else:
        report(mode, path, results, quiet, limit)


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(path, dry, subset, quiet, limit=LIMIT, subset_limit=SUBSET_LIMIT):
    (
        dataset_name,
        attack_name,
        attack_strength,
        source_name,
    ) = parse_image_dir_path(path, quiet=quiet)

    if attack_name is None or attack_strength is None:
        if not quiet:
            print(
                f"Only detla metrices, {DELTA_METRICS}, are required for clean images"
            )
        modes = DELTA_METRICS
        clean_path = path
        attacked_path = None
    else:
        if not quiet:
            print(f"Computing all metrics, {QUALITY_METRICS.keys()}")
        clean_path_dict = get_all_image_dir_paths(
            lambda _dataset_name, _attack_name, _attack_strength, _source_name: (
                _dataset_name == dataset_name
                and attack_name is None
                and attack_strength is None
                and _source_name == source_name
            )
        )
        if not clean_path_dict == 1:
            raise ValueError(
                f"Found {len(clean_path_dict)} corresponding not-attacked image directories, expected exactly 1"
            )
        if not quiet:
            print(f"Found correcponding not-attacked image directory: {clean_path}")

        modes = QUALITY_METRICS.keys()
        clean_path = list(clean_path_dict.values())[0]
        attacked_path = path

    if "aesthetics" in modes or "artifacts" in modes:
        modes = [mode for mode in modes if mode not in ["aesthetics", "artifacts"]] + [
            "aesthetics_and_artifacts"
        ]

    for mode in modes:
        single_mode(
            mode, path, clean_path, attacked_path, quiet, subset, limit, subset_limit
        )


if __name__ == "__main__":
    main()
