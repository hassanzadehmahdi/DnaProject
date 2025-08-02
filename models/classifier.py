import torch
import torch.nn as nn


class ResidueClassifier(nn.Module):
    """
    Simple per-residue classifier using embedding + anomaly score.
    Optionally includes BiLSTM context.
    """
    def __init__(self, input_dim=1025, hidden_dim=256, use_bilstm=True):
        super().__init__()
        self.use_bilstm = use_bilstm

        if self.use_bilstm:
            self.bilstm = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns: [batch_size, seq_len]
        """
        if self.use_bilstm:
            x, _ = self.bilstm(x)
        logits = self.classifier(x).squeeze(-1)
        probs = self.sigmoid(logits)
        return probs
