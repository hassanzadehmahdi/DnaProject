import torch
from torch.utils.data import Dataset
from typing import List, Tuple


def load_sequence_label_fasta(filepath: str) -> List[dict]:
    """
    Load sequences and per-residue labels from a FASTA-like file.
    Format:
        >ID
        SEQUENCE
        LABELS
    """
    data = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            entry = {}
            entry['id'] = lines[i][1:]
            entry['sequence'] = lines[i + 1]
            entry['labels'] = [int(c) for c in lines[i + 2]]

            assert len(entry['sequence']) == len(entry['labels']), \
                f"Mismatch in sequence/label length for {entry['id']}"

            data.append(entry)
            i += 3
        else:
            i += 1
    return data


def tokenize_sequence(seq: str) -> str:
    """
    Convert 'ACDE' to 'A C D E' for ProtT5/ESM compatibility.
    """
    return ' '.join(list(seq))


def pad_sequence(seq: str, labels: List[int], max_len: int) -> Tuple[str, List[int]]:
    """
    Pad or truncate sequence and labels to max_len.
    """
    seq = seq[:max_len]
    labels = labels[:max_len]
    pad_len = max_len - len(seq)

    # Pad with 'X' and label 0
    seq += 'X' * pad_len
    labels += [0] * pad_len

    return seq, labels


class DNABindingDataset(Dataset):
    """
    PyTorch Dataset for DNA-binding residue prediction.
    Returns:
        - tokenized sequence (e.g., 'A C D E F')
        - label tensor (0 or 1 per residue)
        - sequence ID
    """
    def __init__(self, data: List[dict], max_len: int = 512, apply_padding: bool = True):
        self.max_len = max_len
        self.apply_padding = apply_padding
        self.samples = []

        for entry in data:
            seq = entry["sequence"]
            labels = entry["labels"]

            if self.apply_padding:
                seq, labels = pad_sequence(seq, labels, max_len)

            tokenized_seq = tokenize_sequence(seq)

            self.samples.append({
                "id": entry["id"],
                "input": tokenized_seq,
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["input"], item["labels"], item["id"]


if __name__ == '__main__':
    # from utils.dataloader import load_sequence_label_fasta, DNABindingDataset

    # Load your FASTA-like dataset
    entries = load_sequence_label_fasta("../data/PDNA-543.fasta")

    # Wrap in PyTorch Dataset
    dataset = DNABindingDataset(entries, max_len=512)

    # Example batch
    seq, label, seq_id = dataset[0]
    print(seq)  # 'A V K K I S Q ...'
    print(label)  # tensor([0, 0, 1, ...])
    print(seq_id)  # '4I27_A'
