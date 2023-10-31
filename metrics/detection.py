import numpy as np
from sklearn import metrics


# Evaluate the detection performance
def evaluate_detection(distances_wo, distances_w):
    # Calculate the AUROC scores and related
    y_true = [0] * len(distances_wo) + [1] * len(distances_w)
    y_score = (-np.array(distances_wo + distances_w)).tolist()

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    auc = metrics.auc(fpr, tpr)
    low = tpr[np.where(fpr < 0.01)[0][-1]]
    return acc, auc, low
