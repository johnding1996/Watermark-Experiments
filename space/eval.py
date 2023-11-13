import numpy as np


def bit_error_rate(pred, target):
    assert pred.dtype == target.dtype == bool
    return np.mean(pred != target)


def normalized_mse(pred, target):
    assert pred.dtype == target.dtype == np.float16
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    return (((pred - target) / np.std(pred)) ** 2).mean()
