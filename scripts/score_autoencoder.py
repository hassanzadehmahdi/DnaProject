import os
import torch
from tqdm import tqdm
from utils.embeddings import load_embedding
from utils.dataloader import load_sequence_label_fasta, DNABindingDataset
from models.autoencoder import ResidueAutoencoder


@torch.no_grad()
def compute_anomaly_scores(model, embedding, device):
    """
    Compute reconstruction error for each residue vector in embedding.
    """
    model.eval()
    embedding = embedding.to(device)
    recon = model(embedding)
    error = torch.mean((recon - embedding) ** 2, dim=1)  # [seq_len]
    return error.cpu().tolist()


def score_all_sequences(fasta_path, emb_dir, model_path, out_dir, device="cuda"):
    entries = load_sequence_label_fasta(fasta_path)
    dataset = DNABindingDataset(entries)

    # Load trained model
    model = ResidueAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    os.makedirs(out_dir, exist_ok=True)

    for tokenized_seq, labels, seq_id in tqdm(dataset, desc="Scoring sequences"):
        emb_path = os.path.join(emb_dir, f"{seq_id}.pt")
        if not os.path.exists(emb_path):
            print(f"⚠️ Missing embedding for {seq_id}")
            continue

        embedding = load_embedding(emb_path)  # [seq_len, 1024]
        scores = compute_anomaly_scores(model, embedding, device)

        # Save scores
        torch.save(scores, os.path.join(out_dir, f"{seq_id}.pt"))


if __name__ == "__main__":
    FASTA_PATH = "../data/PDNA-543.fasta"
    EMB_DIR = "outputs/embeddings"
    MODEL_PATH = "outputs/models/autoencoder.pt"
    SCORE_DIR = "outputs/anomaly_scores"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    score_all_sequences(
        fasta_path=FASTA_PATH,
        emb_dir=EMB_DIR,
        model_path=MODEL_PATH,
        out_dir=SCORE_DIR,
        device=DEVICE
    )
