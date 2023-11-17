from .constants import LIMIT, WATERMARK_METHODS, GROUND_TRUTH_MESSAGES
from .find import parse_json_path, get_all_json_paths
from .io import load_json, decode_array_from_string, decode_image_from_string
from .eval import message_distance, detection_perforamance


def get_progress_from_json(path):
    (
        _,
        _,
        _,
        source_name,
        result_type,
    ) = parse_json_path(path)
    data = load_json(path)
    if result_type == "status":
        return sum([data[str(i)]["exist"] for i in range(LIMIT)])
    elif result_type == "reverse":
        return sum([data[str(i)] for i in range(LIMIT)])
    elif result_type == "decode":
        for mode in WATERMARK_METHODS.keys():
            if source_name.endswith(mode):
                return sum([data[str(i)][mode] is not None for i in range(LIMIT)])
        return sum(
            [
                (all([data[str(i)][mode] is not None for mode in WATERMARK_METHODS.keys()]))
                for i in range(LIMIT)
            ]
        )
    elif result_type == "metric":
        return 0


def get_example_from_json(path):
    data = load_json(path)
    return [
        decode_image_from_string(data[str(i)]["thumbnail"]) for i in [0, 1, 10, 100]
    ]


def get_distances_from_json(path, mode):
    try:
        data = load_json(path)
        messages = [decode_array_from_string(data[str(i)][mode]) for i in range(LIMIT)]
        return [
            message_distance(message, GROUND_TRUTH_MESSAGES[mode])
            for message in messages
        ]
    except TypeError:
        return None


def get_metrics_from_json(path, mode):
    data = load_json(path)
    return [data[str(i)][mode] for i in range(LIMIT)]
