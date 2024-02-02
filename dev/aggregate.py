import os
from joblib import Memory
from .constants import EVALUATION_SETUPS, PERFORMANCE_METRICS, QUALITY_METRICS
from .find import get_all_json_paths
from .parse import get_distances_from_json, get_metrics_from_json
from .eval import detection_perforamance, mean_and_std, combine_means_and_stds


memory = Memory(os.environ.get("CACHE_DIR"), verbose=0)


def get_performance_from_jsons(original_path, watermarked_path, mode):
    original_distances = get_distances_from_json(original_path, mode)
    watermarked_distances = get_distances_from_json(watermarked_path, mode)
    if original_distances is None or watermarked_distances is None:
        return {key: None for key in PERFORMANCE_METRICS.keys()}
    return detection_perforamance(original_distances, watermarked_distances)


@memory.cache
def get_performance(dataset_name, source_name, attack_name, attack_strength, mode):
    if source_name.startswith("real") or attack_name is None or attack_strength is None:
        raise ValueError(
            f"Cannot compute performance for {dataset_name}, {source_name}, {attack_name}, {attack_strength}"
        )
    if mode not in EVALUATION_SETUPS.keys():
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
                        and (
                            (_source_name == "real")
                            or (
                                _source_name.startswith("real")
                                and _source_name.endswith(source_name)
                            )
                        )
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
                        and (
                            (_source_name == "real")
                            or (
                                _source_name.startswith("real")
                                and _source_name.endswith(source_name)
                            )
                        )
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
        return {key: None for key in PERFORMANCE_METRICS.keys()}


def get_single_quality_from_jsons(clean_path, attacked_path, mode):
    if mode not in ["aesthetics", "artifacts", "clip_score"]:
        attacked_metrics = get_metrics_from_json(attacked_path, mode)
        if attacked_metrics is None:
            return None
        return attacked_metrics
    else:
        attacked_metrics = get_metrics_from_json(attacked_path, mode)
        clean_metrics = get_metrics_from_json(clean_path, mode)
        if attacked_metrics is None or clean_metrics is None:
            return None
        return [
            (clean_metric - attacked_metric)
            for attacked_metric, clean_metric in zip(attacked_metrics, clean_metrics)
        ]


def get_quality_from_jsons(clean_path, attacked_path):
    return {
        mode: mean_and_std(
            get_single_quality_from_jsons(clean_path, attacked_path, mode)
        )
        for mode in QUALITY_METRICS.keys()
    }


@memory.cache
def get_quality(dataset_name, source_name, attack_name, attack_strength, mode):
    if attack_name is None or attack_strength is None:
        raise ValueError(
            f"Cannot compute quality for {dataset_name}, {source_name}, {attack_name}, {attack_strength}"
        )
    if mode is None:
        try:
            clean_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name is None
                        and _attack_strength is None
                        and (
                            (_source_name == source_name)
                            if not source_name.startswith("real")
                            else _source_name.startswith("real")
                        )
                        and _result_type == "metric"
                    )
                ).values()
            )[0]
            attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and (
                            (_source_name == source_name)
                            if not source_name.startswith("real")
                            else _source_name.startswith("real")
                        )
                        and _result_type == "metric"
                    )
                ).values()
            )[0]
            return get_quality_from_jsons(clean_path, attacked_path)
        except IndexError:
            return [None] * len(QUALITY_METRICS)

    if source_name.startswith("real"):
        raise ValueError(
            f"Cannot compute quality for {dataset_name}, {source_name}, {attack_name}, {attack_strength}"
        )
    if mode not in EVALUATION_SETUPS.keys():
        raise ValueError(f"Unknown evaluation setup {mode}")

    try:
        if mode in ["removal", "combined"]:
            watermarked_clean_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name is None
                        and _attack_strength is None
                        and _source_name == source_name
                        and _result_type == "metric"
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
                        and _result_type == "metric"
                    )
                ).values()
            )[0]
            watermarked_quality = get_quality_from_jsons(
                watermarked_clean_path, watermarked_attacked_path
            )

        if mode in ["spoofing", "combined"]:
            original_clean_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name is None
                        and _attack_strength is None
                        and _source_name == "real"
                        and _result_type == "metric"
                    )
                ).values()
            )[0]
            original_attacked_path = list(
                get_all_json_paths(
                    lambda _dataset_name, _attack_name, _attack_strength, _source_name, _result_type: (
                        _dataset_name == dataset_name
                        and _attack_name == attack_name
                        and abs(_attack_strength - attack_strength) < 1e-5
                        and (
                            (_source_name == "real")
                            or (
                                _source_name.startswith("real")
                                and _source_name.endswith(source_name)
                            )
                        )
                        and _result_type == "metric"
                    )
                ).values()
            )[0]
            original_quality = get_quality_from_jsons(
                original_clean_path, original_attacked_path
            )

        if mode == "removal":
            return watermarked_quality
        elif mode == "spoofing":
            return original_quality
        else:
            return {
                mode: combine_means_and_stds(
                    watermarked_quality[mode], original_quality[mode]
                )
                for mode in QUALITY_METRICS.keys()
            }

    except IndexError:
        return {key: None for key in QUALITY_METRICS.keys()}


def clear_aggregated_cache():
    memory.clear(warn=False)
