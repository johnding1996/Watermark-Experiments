import numpy as np
from sklearn import metrics
from scipy.stats import binom, ncx2
from .constants import (
    GROUND_TRUTH_MESSAGES,
    QUALITY_NORMALIZATION_THRESHOLDS,
    QUALITY_NORMALIZATION_INTERVALS,
    QUALITY_NORMALIZATION_WEIGHTS,
)


def bit_error_rate(pred, target):
    if not pred.dtype == target.dtype == bool:
        raise ValueError(f"Cannot compute BER for {pred.dtype} and {target.dtype}")
    return np.mean(pred != target)


def complex_l1(pred, target):
    if not pred.dtype == target.dtype == np.float16:
        raise ValueError(
            f"Cannot compute Complex L1 for {pred.dtype} and {target.dtype}"
        )
    # Cast to float32 to avoid large numerical errors
    pred = pred.astype(np.float32).reshape(2, -1)
    target = target.astype(np.float32).reshape(2, -1)
    return np.sqrt(((pred - target) ** 2).sum(0)).mean()


def complex_l2(pred, target):
    if not pred.dtype == target.dtype == np.float16:
        raise ValueError(
            f"Cannot compute Complex L2 for {pred.dtype} and {target.dtype}"
        )
    # Cast to float32 to avoid large numerical errors
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    return np.sqrt(((pred - target) ** 2).sum())


def message_distance(pred, target, mode):
    if mode == "detection":
        if target.dtype == bool:
            return bit_error_rate(pred, target)
        elif target.dtype == np.float16:
            return complex_l1(pred, target)
        else:
            raise TypeError
    elif mode == "identification":
        if target.dtype == bool:
            return bit_error_rate(pred, target)
        elif target.dtype == np.float16:
            return complex_l2(pred, target)
        else:
            raise TypeError
    else:
        raise ValueError


def detection_perforamance(original_distances, watermarked_distances):
    if not len(original_distances) == len(watermarked_distances):
        raise ValueError(f"Length of distances must be equal")
    y_true = [0] * len(original_distances) + [1] * len(watermarked_distances)
    y_score = (-np.array(original_distances + watermarked_distances)).tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    acc_1 = np.max(1 - (fpr + (1 - tpr)) / 2)
    auc_1 = metrics.auc(fpr, tpr)
    low100_1 = tpr[np.where(fpr < 0.01)[0][-1]]
    low1000_1 = tpr[np.where(fpr < 0.001)[0][-1]]
    return {
        "acc_1": acc_1,
        "auc_1": auc_1,
        "low100_1": low100_1,
        "low1000_1": low1000_1,
    }


def generate_random_message(mode):
    if mode == "tree_ring":
        x, y = np.ogrid[-32:32, -32:32]
        mask = x**2 + y**2 <= 10**2
        fft_latent = np.fft.fftshift(
            np.fft.fft2(np.random.randn(64, 64).astype(np.float16))
        )
        message = fft_latent[mask]
        combined = np.empty(message.size * 2, dtype=np.float16)
        combined[0::2], combined[1::2] = message.real, message.imag
        return combined
    elif mode in ["stable_sig", "stegastamp"]:
        return np.random.choice([True, False], size=len(GROUND_TRUTH_MESSAGES[mode]))
    else:
        raise ValueError


def simulated_identification_performance(watermarked_messages, num_users, mode):
    acc_list = []
    # Simulate 5 times
    for _ in range(5):
        simulated_ground_truth_messages = [
            generate_random_message(mode) for _ in range(1, num_users)
        ]
        fail_count = 0
        for message in watermarked_messages:
            target_distance = message_distance(
                message, GROUND_TRUTH_MESSAGES[mode], "identification"
            )
            for simulated_ground_truth_message in simulated_ground_truth_messages:
                simulated_distance = message_distance(
                    message, simulated_ground_truth_message, "identification"
                )
                if simulated_distance <= target_distance:
                    fail_count += 1
                    break
        acc_list.append(1.0 - fail_count / len(watermarked_messages))
    return (np.mean(acc_list), np.std(acc_list))


def theoretical_distance_cdf(messages, distances, mode):
    length = len(GROUND_TRUTH_MESSAGES[mode])
    if mode == "tree_ring":
        # Cast to float32 to avoid large numerical errors
        return ncx2.cdf(
            distances**2,
            df=length,
            nc=np.array(
                [
                    np.sum(np.square(message.astype(np.float32))) / (np.pi * length)
                    for message in messages
                ]
            ),
            scale=(np.pi * length),
        )
    elif mode in ["stable_sig", "stegastamp"]:
        return binom.cdf(np.floor(length * distances), length, 0.5)
    else:
        raise ValueError


def theoretical_identification_performance(watermarked_messages, num_users, mode):
    distances = np.array(
        [
            message_distance(message, GROUND_TRUTH_MESSAGES[mode], "identification")
            for message in watermarked_messages
        ]
    )
    return np.mean(
        np.exp(
            (num_users - 1)
            * np.log(
                1.0 - theoretical_distance_cdf(watermarked_messages, distances, mode),
            )
        )
    )


def identification_performance(watermarked_messages, mode):
    return {
        "acc_100": theoretical_identification_performance(
            watermarked_messages, 100, mode
        ),
        "acc_1000": theoretical_identification_performance(
            watermarked_messages, 1000, mode
        ),
        "acc_1000000": theoretical_identification_performance(
            watermarked_messages, 1000000, mode
        ),
    }


def mean_and_std(values):
    if values is None:
        return None
    return np.mean(values), np.std(values)


def combine_means_and_stds(mean_and_std1, mean_and_std2):
    if mean_and_std1 is None or mean_and_std2 is None:
        return None
    mean1, std1 = mean_and_std1
    mean2, std2 = mean_and_std2
    mean = (mean1 + mean2) / 2
    std = np.sqrt((std1**2 + std2**2) / 2)
    return mean, std


def normalized_quality(qualities):
    normalized_mean = sum(
        [
            QUALITY_NORMALIZATION_WEIGHTS[mode]
            * (
                (qualities[mode][0] - QUALITY_NORMALIZATION_INTERVALS[mode][0])
                / (
                    QUALITY_NORMALIZATION_INTERVALS[mode][1]
                    - QUALITY_NORMALIZATION_INTERVALS[mode][0]
                )
                * (
                    QUALITY_NORMALIZATION_THRESHOLDS[1]
                    - QUALITY_NORMALIZATION_THRESHOLDS[0]
                )
                + QUALITY_NORMALIZATION_THRESHOLDS[0]
            )
            for mode in QUALITY_NORMALIZATION_INTERVALS.keys()
        ]
    )
    normalized_std = sum(
        [
            QUALITY_NORMALIZATION_WEIGHTS[mode]
            * abs(
                qualities[mode][1]
                / (
                    QUALITY_NORMALIZATION_INTERVALS[mode][1]
                    - QUALITY_NORMALIZATION_INTERVALS[mode][0]
                )
                * (
                    QUALITY_NORMALIZATION_THRESHOLDS[1]
                    - QUALITY_NORMALIZATION_THRESHOLDS[0]
                )
            )
            for mode in QUALITY_NORMALIZATION_INTERVALS.keys()
        ]
    )
    return normalized_mean, normalized_std
