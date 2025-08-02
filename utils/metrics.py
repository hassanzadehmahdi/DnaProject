import torch
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, average_precision_score

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred_bin),
        "mcc": matthews_corrcoef(y_true, y_pred_bin),
        "ap": average_precision_score(y_true, y_pred)
    }

def group_metrics_by_sequence(results):
    # results = list of (y_true, y_pred, sequence_id)
    grouped = {}
    for y_t, y_p, seq_id in results:
        m = compute_metrics(np.array(y_t), np.array(y_p))
        grouped[seq_id] = m
    return grouped