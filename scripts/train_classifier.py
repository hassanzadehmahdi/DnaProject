import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.dataloader import load_sequence_label_fasta
from utils.embeddings import load_embedding
from models.classifier import ResidueClassifier


# ✅ Focal Loss for imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


# ✅ Dataset class
class ResidueClassifierDataset(Dataset):
    def __init__(self, fasta_entries, emb_dir, score_dir, max_len=512):
        self.data = []

        for entry in fasta_entries:
            seq_id = entry["id"]
            label = entry["labels"]
            emb_path = os.path.join(emb_dir, f"{seq_id}.pt")
            score_path = os.path.join(score_dir, f"{seq_id}.pt")

            if not os.path.exists(emb_path) or not os.path.exists(score_path):
                continue

            emb = load_embedding(emb_path)  # [L, 1024]
            scores = torch.tensor(load_embedding(score_path))  # [L]
            label = torch.tensor(label, dtype=torch.float32)   # [L]

            if len(label) != emb.shape[0] or len(label) != len(scores):
                continue

            emb = emb[:max_len]
            scores = scores[:max_len]
            label = label[:max_len]

            pad_len = max_len - emb.shape[0]
            if pad_len > 0:
                emb = torch.cat([emb, torch.zeros(pad_len, emb.shape[1])], dim=0)
                scores = torch.cat([scores, torch.zeros(pad_len)], dim=0)
                label = torch.cat([label, torch.zeros(pad_len)], dim=0)

            combined = torch.cat([emb, scores.unsqueeze(1)], dim=1)  # [max_len, 1025]
            self.data.append((combined, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ✅ Training function
def train_classifier(dataloader, device="cuda", save_path="outputs/models/classifier.pt"):
    model = ResidueClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = FocalLoss(alpha=0.25, gamma=2)

    model.train()
    for epoch in range(10):
        total_loss = 0
        all_preds = []
        all_labels = []

        for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            X = X.to(device)         # [B, L, 1025]
            y = y.to(device)         # [B, L]

            out = model(X)           # [B, L]
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_preds.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

        # Evaluation after epoch
        y_true = torch.cat(all_labels).flatten().numpy()
        y_pred = torch.cat(all_preds).flatten().numpy()
        y_pred_bin = (y_pred >= 0.5).astype(int)

        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Classifier saved to {save_path}")


# ✅ Main entrypoint
if __name__ == "__main__":
    FASTA_PATH = "../data/PDNA-543.fasta"
    EMB_DIR = "outputs/embeddings"
    SCORE_DIR = "outputs/anomaly_scores"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading FASTA entries...")
    entries = load_sequence_label_fasta(FASTA_PATH)

    print("🔹 Building dataset...")
    dataset = ResidueClassifierDataset(entries, emb_dir=EMB_DIR, score_dir=SCORE_DIR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("🔹 Training classifier...")
    train_classifier(dataloader, device=DEVICE)
