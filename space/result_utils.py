import os
import warnings


def parse_json_path(path):
    if not os.path.commonpath(
        [os.environ.get("RESULT_DIR"), str(path)]
    ) == os.path.commonpath([os.environ.get("RESULT_DIR")]):
        raise ValueError(
            f"JSON files should be under the result directory {os.environ.get('RESULT_DIR')}"
        )
    if not str(path).endswith(".json"):
        raise ValueError("Invalid JSON file path, must end with .json")

    dataset_name, filename = str(path).split("/")[-2:]
    if not dataset_name in ["diffusiondb", "mscoco", "dalle3"]:
        raise ValueError(
            f"Dataset name must be one of ['diffusiondb', 'mscoco', 'dalle3'], found {dataset_name}"
        )

    if filename.count("-") == 0:
        attack_name, attack_strength, source_name, result_type = (
            None,
            None,
            None,
            str(filename[:-5]),
        )
    elif filename.count("-") == 1:
        attack_name, attack_strength, source_name, result_type = (
            None,
            None,
            *str(filename[:-5]).split("-"),
        )
    elif filename.count("-") == 3:
        attack_name, attack_strength, source_name, result_type = str(
            filename[:-5]
        ).split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a number")
    else:
        raise ValueError(
            f"Invalid JSON file name {filename}, must be in the format of 'source_name-result_type.json' or 'attack_name-attack_strength-source_name-result_type.json'"
        )
    if not result_type in ["status", "reverse", "decode", "metric", "prompts"]:
        raise ValueError(
            "Invalid result type, must be one of ['status', 'reverse', 'decode', 'metric']"
        )
    if source_name is not None and not source_name in [
        "real",
        "stable_sig",
        "stegastamp",
        "tree_ring",
        "real_stable_sig",
        "real_stegastamp",
        "real_tree_ring",
    ]:
        raise ValueError(
            "Source name must be one of ['real', 'stable_sig', 'stegastamp', 'tree_ring'] or start with 'real_'"
        )

    return dataset_name, attack_name, attack_strength, source_name, result_type


def get_all_json_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be a callable function")
    json_paths = []
    for dataset_name in ["diffusiondb", "mscoco", "dalle3"]:
        for filename in os.listdir(
            os.path.join(os.environ.get("RESULT_DIR"), dataset_name)
        ):
            path = os.path.join(os.environ.get("RESULT_DIR"), dataset_name, filename)
            if os.path.isfile(path):
                json_paths.append(path)
    json_dict = {}
    for path in json_paths:
        try:
            key = parse_json_path(path)
            if criteria is None or criteria(*key):
                json_dict[key] = path
        except ValueError as e:
            warnings.warn(f"Found invalid JSON file {path}, {e}, skipping")
    return json_dict
