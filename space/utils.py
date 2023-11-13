import os
import stat
import warnings
import orjson
import base64
from io import BytesIO
from PIL import Image


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


def chmod_group_write(path):
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")
    current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
    os.chmod(path, current_permissions | stat.S_IWGRP)


def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True


def save_json(data, filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as json_file:
            existing_data = orjson.loads(json_file.read())
        if compare_dicts(data, existing_data):
            return
    with open(filepath, "wb") as json_file:
        json_file.write(orjson.dumps(data))
    chmod_group_write(filepath)


def load_json(filepath):
    with open(filepath, "rb") as json_file:
        return orjson.loads(json_file.read())


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


def get_all_image_dir_paths():
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
            image_dir_dict[parse_image_dir_path(path)] = path
        except ValueError:
            warnings.warn(f"Found invalid image directory {path}, skipping")
    return image_dir_dict


def encode_image_to_string(image, size=256, quality=75):
    # Resize the image
    image = image.resize((size, size))
    # Save the image to a byte buffer in JPEG format
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    # Encode the buffer to a base64 string
    return base64.b64encode(buffered.getvalue()).decode()


def decode_image_from_string(base64_str):
    # Decode the base64 string to bytes
    img_data = base64.b64decode(base64_str)
    # Read the image from bytes
    image = Image.open(BytesIO(img_data))
    return image
