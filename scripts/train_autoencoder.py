import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.dataloader import load_sequence_label_fasta
from utils.embeddings import load_embedding
from models.autoencoder import ResidueAutoencoder


def collect_label0_embeddings(dataset, embedding_dir):
    """
    Collect all residue embeddings with label = 0 from the dataset.
    """
    vectors = []
    for sample in tqdm(dataset, desc="Loading embeddings"):
        tokenized_seq, labels, seq_id = sample
        emb_path = os.path.join(embedding_dir, f"{seq_id}.pt")

        if not os.path.exists(emb_path):
            print(f"⚠️ Missing embedding for {seq_id}")
            continue

        embedding = load_embedding(emb_path)  # shape: [seq_len, 1024]
        if embedding.shape[0] != len(labels):
            print(f"❌ Shape mismatch for {seq_id}")
            continue

        # Extract only label=0 vectors
        for i in range(len(labels)):
            if labels[i].item() == 0:
                vectors.append(embedding[i].unsqueeze(0))  # [1, 1024]

    all_vectors = torch.cat(vectors, dim=0)  # [N, 1024]
    return all_vectors


def train_autoencoder(vectors, device="cuda", save_path="outputs/models/autoencoder.pt"):
    """
    Train the autoencoder on label=0 embeddings.
    """
    model = ResidueAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(vectors)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Autoencoder saved to {save_path}")


if __name__ == "__main__":
    FASTA_PATH = "../data/PDNA-543.fasta"
    EMB_DIR = "outputs/embeddings"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading dataset...")
    entries = load_sequence_label_fasta(FASTA_PATH)

    from utils.dataloader import DNABindingDataset
    dataset = DNABindingDataset(entries, max_len=512)

    print("🔹 Collecting normal residue vectors...")
    vectors = collect_label0_embeddings(dataset, embedding_dir=EMB_DIR)

    print(f"🔹 Training on {len(vectors)} vectors...")
    train_autoencoder(vectors, device=device)
