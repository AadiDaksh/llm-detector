"""Bidirectional LSTM text classifier for identifying the LLM behind a response."""

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=3,
        dropout=0.3,
        padding_idx=0,
        bidirectional=True,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        directions = 2 if bidirectional else 1
        # Mean- and max-pool concatenated → 2 * (directions * hidden_dim).
        self.classifier = nn.Linear(
            directions * hidden_dim * 2,
            num_classes,
        )

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch_size, sequence_length)"
            )

        # (B, L, 1) mask over non-padding tokens.
        mask = (input_ids != self.padding_idx).unsqueeze(-1).float()

        embedded = self.embedding(input_ids)  # (B, L, E)

        outputs, _ = self.lstm(embedded)  # (B, L, D*H)

        # Mean pool over non-padding positions.
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = summed / counts

        # Max pool, masking padding positions with -inf.
        masked_outputs = outputs.masked_fill(mask == 0, float("-inf"))
        max_pool, _ = masked_outputs.max(dim=1)
        # Safety: if a row is entirely padding, fall back to zeros.
        max_pool = torch.where(
            torch.isinf(max_pool),
            torch.zeros_like(max_pool),
            max_pool,
        )

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)


if __name__ == "__main__":
    model = LSTMClassifier(vocab_size=19337)

    sample_ids = torch.randint(0, 19337, (32, 256))
    logits = model(sample_ids)

    print("Input shape:", tuple(sample_ids.shape))
    print("Output shape:", tuple(logits.shape))

    assert logits.shape == (32, 3)
    print("Works")
