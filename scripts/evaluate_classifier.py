import os
import csv
import torch
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score, matthews_corrcoef
)

from utils.dataloader import load_sequence_label_fasta
from utils.embeddings import load_embedding
from models.classifier import ResidueClassifier


@torch.no_grad()
def evaluate(model, entries, emb_dir, score_dir, device="cuda", save_csv=None):
    model.eval()
    model = model.to(device)

    y_true_all = []
    y_pred_all = []
    y_pred_bin_all = []

    csv_rows = []

    for entry in tqdm(entries, desc="Evaluating"):
        seq_id = entry["id"]
        labels = torch.tensor(entry["labels"], dtype=torch.float32)

        emb_path = os.path.join(emb_dir, f"{seq_id}.pt")
        score_path = os.path.join(score_dir, f"{seq_id}.pt")

        if not os.path.exists(emb_path) or not os.path.exists(score_path):
            continue

        emb = load_embedding(emb_path)             # [L, 1024]
        scores = torch.tensor(load_embedding(score_path))  # [L]

        if len(labels) != emb.shape[0] or len(labels) != len(scores):
            continue  # skip corrupted entries

        combined = torch.cat([emb, scores.unsqueeze(1)], dim=1).unsqueeze(0).to(device)  # [1, L, 1025]

        pred = model(combined).squeeze(0).cpu()  # [L]
        pred_bin = (pred >= 0.5).int()

        for i in range(len(labels)):
            y_true_all.append(labels[i].item())
            y_pred_all.append(pred[i].item())
            y_pred_bin_all.append(pred_bin[i].item())

            if save_csv:
                csv_rows.append({
                    "sequence_id": seq_id,
                    "residue_index": i,
                    "true_label": int(labels[i].item()),
                    "pred_score": float(pred[i].item()),
                    "pred_binary": int(pred_bin[i].item()),
                })

    # Compute metrics
    f1 = f1_score(y_true_all, y_pred_bin_all, zero_division=0)
    precision = precision_score(y_true_all, y_pred_bin_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_bin_all, zero_division=0)
    pr_auc = average_precision_score(y_true_all, y_pred_all)
    mcc = matthews_corrcoef(y_true_all, y_pred_bin_all)

    print("\n📊 Evaluation Results:")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"PR-AUC         : {pr_auc:.4f}")
    print(f"MCC            : {mcc:.4f}")

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n✅ Raw predictions saved to: {save_csv}")


if __name__ == "__main__":
    MODEL_PATH = "outputs/models/classifier.pt"
    FASTA_PATH = "../data/PDNA-543.fasta"
    EMB_DIR = "outputs/embeddings"
    SCORE_DIR = "outputs/anomaly_scores"
    CSV_PATH = "outputs/evaluation/predictions.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading model and data...")
    model = ResidueClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    entries = load_sequence_label_fasta(FASTA_PATH)

    evaluate(
        model,
        entries,
        emb_dir=EMB_DIR,
        score_dir=SCORE_DIR,
        device=DEVICE,
        save_csv=CSV_PATH,
    )
