import torch
from torch import nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_filters=100,
        filter_sizes=(3, 4, 5),
        num_classes=3,
        dropout=0.3,
        padding_idx=0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if not filter_sizes or any(size <= 0 for size in filter_sizes):
            raise ValueError("filter_sizes must contain positive integers")

        self.filter_sizes = tuple(filter_sizes)

        # Convert token IDs into learnable word vectors.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # Detect patterns across groups of 3, 4, and 5 tokens.
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=size,
                )
                for size in self.filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Predict GPT, Claude, or Gemini.
        self.classifier = nn.Linear(
            num_filters * len(self.filter_sizes),
            num_classes
        )

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch_size, sequence_length)"
            )

        if input_ids.size(1) < max(self.filter_sizes):
            raise ValueError(
                f"Sequence length must be at least {max(self.filter_sizes)}"
            )

        # Convert token IDs into embeddings.
        embedded = self.embedding(input_ids)

        # Change dimensions for the convolution layers.
        embedded = embedded.transpose(1, 2)

        pooled_features = []

        # Apply convolution, activation, and max pooling.
        for convolution in self.convolutions:
            features = F.relu(convolution(embedded))
            pooled = torch.max(features, dim=2).values
            pooled_features.append(pooled)

        # Combine features from all convolution filters.
        combined = torch.cat(pooled_features, dim=1)
        combined = self.dropout(combined)

        # Return three classification scores.
        return self.classifier(combined)


if __name__ == "__main__":
    model = CNNClassifier(vocab_size=19337)

    sample_ids = torch.randint(0, 19337, (32, 256))
    logits = model(sample_ids)

    print("Input shape:", tuple(sample_ids.shape))
    print("Output shape:", tuple(logits.shape))

    assert logits.shape == (32, 3)
    print("Works")


