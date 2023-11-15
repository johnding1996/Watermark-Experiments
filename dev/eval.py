import numpy as np
from sklearn import metrics


def bit_error_rate(pred, target):
    if not pred.dtype == target.dtype == bool:
        raise ValueError(f"Cannot compute BER for {pred.dtype} and {target.dtype}")
    return np.mean(pred != target)


def normalized_mse(pred, target):
    if not pred.dtype == target.dtype == np.float16:
        raise ValueError(
            f"Cannot compute Normalized MSE for {pred.dtype} and {target.dtype}"
        )
    # Cast to float32 to avoid numerical error and overflow
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    return (((pred - target) / np.std(pred)) ** 2).mean()


def message_distance(pred, target):
    if target.dtype == bool:
        return bit_error_rate(pred, target)
    elif target.dtype == np.float16:
        return normalized_mse(pred, target)
    else:
        raise ValueError(f"Unsupported dtype {target.dtype}")


def detection_perforamance(watermarked_distances, original_distances):
    if not len(watermarked_distances) == len(original_distances):
        raise ValueError(
            f"Length of watermarked_distances ({len(watermarked_distances)}) and original_distances ({len(original_distances)}) must be equal"
        )
    y_true = [0] * len(watermarked_distances) + [1] * len(original_distances)
    y_score = (-np.array(watermarked_distances + original_distances)).tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    auc = metrics.auc(fpr, tpr)
    low = tpr[np.where(fpr < 0.001)[0][-1]]
    return acc, auc, low
