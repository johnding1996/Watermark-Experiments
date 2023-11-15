from .constants import EVALUATION_SETUPS, PERFORMANCE_METRICS, QUALITY_METRICS
from .find import get_all_json_paths
from .parse import get_distances_from_json, get_metrics_from_json
from .eval import detection_perforamance


def get_performance_from_jsons(watermarked_path, original_path, mode):
    watermarked_distances = get_distances_from_json(watermarked_path, mode)
    original_distances = get_distances_from_json(original_path, mode)
    if watermarked_distances is None or original_distances is None:
        return [None] * len(PERFORMANCE_METRICS)
    return detection_perforamance(watermarked_distances, original_distances)


def get_performance(dataset_name, source_name, attack_name, attack_strength, mode):
    if source_name.startswith("real") or attack_name is None or attack_strength is None:
        raise ValueError(
            f"Cannot compute performance for {dataset_name}, {source_name}, {attack_name}, {attack_strength}"
        )
    if mode not in EVALUATION_SETUPS:
        raise ValueError(f"Unknown evaluation setup {mode}")

    try:
        if mode == "removal":
            watermarked_attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and _source_name == source_name
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            original_clean_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name is None
                        and _attack_strength is None
                        and _source_name == "real"
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            return get_performance_from_jsons(
                watermarked_attacked_path, original_clean_path, source_name
            )

        elif mode == "spoofing":
            watermarked_clean_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name is None
                        and _attack_strength is None
                        and _source_name == source_name
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            original_attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and _source_name.startswith("real")
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            return get_performance_from_jsons(
                watermarked_clean_path, original_attacked_path, source_name
            )

        else:
            watermarked_attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and _source_name == source_name
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            original_attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and _source_name.startswith("real")
                        and _result_type == "decode"
                    )
                ).values()
            )[0]
            print(watermarked_attacked_path, original_attacked_path)
            return get_performance_from_jsons(
                watermarked_attacked_path, original_attacked_path, source_name
            )
    except StopIteration:
        return [None] * len(PERFORMANCE_METRICS)


def get_quality_metrics(dataset_name, source_name, attack_name, attack_strength, mode):
    pass
