import torch
from transformers import T5EncoderModel, T5Tokenizer
from typing import List
import os

# Load ProtT5 model and tokenizer once and reuse
_model = None
_tokenizer = None


def load_prott5_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_name = "Rostlab/prot_t5_xl_uniref50"
        _tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        _model = T5EncoderModel.from_pretrained(model_name)
        _model = _model.eval()  # inference only
    return _tokenizer, _model


@torch.no_grad()
def extract_embeddings(sequence: str, device="cpu") -> torch.Tensor:
    """
    Convert a tokenized amino acid sequence ('A C D E ...') into per-residue embeddings.

    Returns:
        torch.Tensor of shape [seq_len, hidden_dim]
    """
    tokenizer, model = load_prott5_model()
    model = model.to(device)

    # Prepare model input
    input_ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)

    # Forward pass
    output = model(input_ids=input_ids)

    # Remove special tokens ([CLS], [SEP]) from embedding
    embeddings = output.last_hidden_state.squeeze(0)[1:-1]  # shape: [seq_len, hidden_dim]

    return embeddings.cpu()  # return to CPU to avoid GPU memory spike


def save_embedding(embedding: torch.Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embedding, save_path)


def load_embedding(load_path: str) -> torch.Tensor:
    return torch.load(load_path)


from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_and_cache_embeddings(dataset, save_dir="outputs/embeddings", device="cpu"):
    """
    Iterate over dataset and cache embeddings for each sequence to disk.
    Each file is saved as: outputs/embeddings/{sequence_id}.pt
    """
    os.makedirs(save_dir, exist_ok=True)

    for tokenized_seq, _, seq_id in tqdm(dataset, desc="Embedding sequences"):
        save_path = os.path.join(save_dir, f"{seq_id}.pt")

        if os.path.exists(save_path):
            continue  # Skip if already cached

        embedding = extract_embeddings(tokenized_seq, device=device)
        save_embedding(embedding, save_path)



if __name__ == "__main__":
    # from utils.embeddings import extract_embeddings

    # Assume tokenized sequence: 'A C D E F G'
    embedding = extract_embeddings("A C D E F G", device="cuda")

    print(embedding.shape)  # torch.Size([6, 1024])


    ################### for both dataloader and embeddings file #####################

    from utils.dataloader import load_sequence_label_fasta, DNABindingDataset
    # from utils.embeddings import compute_and_cache_embeddings

    # Load sequences
    entries = load_sequence_label_fasta("data/dna_binding.fasta")
    dataset = DNABindingDataset(entries, max_len=512)

    # Embed and cache them
    compute_and_cache_embeddings(dataset, save_dir="outputs/embeddings", device="cuda")

