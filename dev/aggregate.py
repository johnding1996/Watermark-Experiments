from .constants import EVALUATION_SETUPS, PERFORMANCE_METRICS, QUALITY_METRICS
from .find import get_all_json_paths
from .parse import get_distances_from_json, get_metrics_from_json
from .eval import detection_perforamance


def get_performance_from_jsons(original_path, watermarked_path, mode):
    original_distances = get_distances_from_json(original_path, mode)
    watermarked_distances = get_distances_from_json(watermarked_path, mode)
    if original_distances is None or watermarked_distances is None:
        return [None] * len(PERFORMANCE_METRICS)
    return detection_perforamance(original_distances, watermarked_distances)


def get_performance(dataset_name, source_name, attack_name, attack_strength, mode):
    if source_name.startswith("real") or attack_name is None or attack_strength is None:
        raise ValueError(
            f"Cannot compute performance for {dataset_name}, {source_name}, {attack_name}, {attack_strength}"
        )
    if mode not in EVALUATION_SETUPS:
        raise ValueError(f"Unknown evaluation setup {mode}")

    try:
        if mode == "removal":
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
            return get_performance_from_jsons(
                original_clean_path, watermarked_attacked_path, source_name
            )

        elif mode == "spoofing":
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
            return get_performance_from_jsons(
                original_attacked_path, watermarked_clean_path, source_name
            )

        else:
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
            return get_performance_from_jsons(
                original_attacked_path, watermarked_attacked_path, source_name
            )
    except IndexError:
        return [None] * len(PERFORMANCE_METRICS)


def get_quality_metrics(dataset_name, source_name, attack_name, attack_strength, mode):
    pass
