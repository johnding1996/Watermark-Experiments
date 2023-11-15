import os
import warnings


def check_file_existence(path, name_pattern, limit):
    found_filenames = set(os.listdir(path))
    return [name_pattern.format(i) in found_filenames for i in range(limit)]


def existence_operation(existences1, existences2, op):
    if op == "difference":
        return [a and not b for a, b in zip(existences1, existences2)]
    elif op == "union":
        return [a and b for a, b in zip(existences1, existences2)]
    else:
        raise ValueError(
            f"Invalid operation {op}, can either be 'difference' or 'union'"
        )


def existence_to_indices(existences, limit):
    indices = []
    for i in range(min(len(existences), limit)):
        if existences[i]:
            indices.append(i)
    return indices


def parse_image_dir_path(path, quiet=True):
    if not os.path.commonpath(
        [os.environ.get("DATA_DIR"), str(path)]
    ) == os.path.commonpath([os.environ.get("DATA_DIR")]):
        raise ValueError(
            f"Image directory should be under the dataset directory {os.environ.get('DATA_DIR')}"
        )
    try:
        mode, dataset_name, dirname = str(path).split("/")[-3:]
    except ValueError:
        raise ValueError("Invalid image directory path, unable to parse")

    if not dataset_name in ["diffusiondb", "mscoco", "dalle3"]:
        raise ValueError(
            f"Dataset name must be one of ['diffusiondb', 'mscoco', 'dalle3'], found {dataset_name}"
        )

    if mode == "attacked":
        if not len(dirname.split("-")) == 3:
            raise ValueError(
                f"Attack directory name {dirname} is not in the format of 'attack_name-attack_strength-source_name'"
            )
        attack_name, attack_strength, source_name = dirname.split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a number")
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
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", attack_name)
            print(" -- Attack strength:", attack_strength)
            print(" -- Source name:", source_name)
        return dataset_name, attack_name, attack_strength, source_name
    elif mode == "main":
        if not dirname in ["real", "stable_sig", "stegastamp", "tree_ring"]:
            raise ValueError(
                "Source name must be one of ['real', 'stable_sig', 'stegastamp', 'tree_ring']"
            )
        source_name = dirname
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", None)
            print(" -- Attack strength:", None)
            print(" -- Source name:", source_name)
        return dataset_name, None, None, source_name
    else:
        raise ValueError("Invalid image directory path, unable to parse")


def get_all_image_dir_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be a callable function")
    dir_paths = []
    for mode in ["main", "attacked"]:
        for dataset_name in ["diffusiondb", "mscoco", "dalle3"]:
            for dirname in os.listdir(
                os.path.join(os.environ.get("DATA_DIR"), mode, dataset_name)
            ):
                path = os.path.join(
                    os.environ.get("DATA_DIR"), mode, dataset_name, dirname
                )
                if os.path.isdir(path):
                    dir_paths.append(path)
    image_dir_dict = {}
    for path in dir_paths:
        try:
            key = parse_image_dir_path(path)
            if criteria is None or criteria(*key):
                image_dir_dict[key] = path
        except ValueError:
            warnings.warn(f"Found invalid image directory {path}, skipping")
    return image_dir_dict
