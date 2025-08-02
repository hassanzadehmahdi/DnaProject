import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y_true, y_scores, label=None, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=label or "PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_sequence_prediction(sequence, y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    vis = []
    for aa, t, p in zip(sequence, y_true, y_pred_bin):
        if t == 1 and p == 1:
            vis.append(f"[{aa}]")  # TP
        elif t == 1:
            vis.append(f"({aa})")  # FN
        elif p == 1:
            vis.append(f"*{aa}*")  # FP
        else:
            vis.append(aa)         # TN
    return "".join(vis)
