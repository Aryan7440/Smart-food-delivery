"""
Transformer-based next-item recommender architecture.

This module contains only the PyTorch model definition and its helper
utilities.  It is intentionally separate from the service-wrapper classes
in ml_models.py so the architecture can be imported, inspected, or
extended without pulling in the rest of the inference stack.
"""
from app.constants import MAX_SEQ_LEN

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    TORCH_AVAILABLE = False


def pad_sequence(seq: list, max_len: int, pad_value: int = 0) -> list:
    """Left-pad *seq* to *max_len*; truncate from the left if already longer."""
    seq = seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq


if TORCH_AVAILABLE:
    class FoodRecommender(nn.Module):
        """
        Transformer-based next-item recommender.

        The architecture must match the checkpoint saved by the training
        notebook (notebook/food-recommandation.ipynb):
          - item / hour / day-of-week embeddings summed with positional embeddings
          - TransformerEncoder with batch_first=True
          - Linear projection to num_items logits
        """

        def __init__(
            self,
            num_items: int,
            hidden_dim: int = 64,
            num_heads: int = 2,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
            self.hour_embedding  = nn.Embedding(25, hidden_dim, padding_idx=0)
            self.dow_embedding   = nn.Embedding(8,  hidden_dim, padding_idx=0)
            self.pos_embedding   = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output  = nn.Linear(hidden_dim, num_items)
            self.dropout = nn.Dropout(dropout)

        def forward(self, item_seq, hour_seq, dow_seq):
            batch_size, seq_len = item_seq.shape
            positions = (
                torch.arange(seq_len, device=item_seq.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            x = (
                self.item_embedding(item_seq)
                + self.hour_embedding(hour_seq)
                + self.dow_embedding(dow_seq)
                + self.pos_embedding(positions)
            )
            x = self.dropout(x)
            padding_mask = item_seq == 0
            x = self.transformer(x, src_key_padding_mask=padding_mask)
            return self.output(x[:, -1, :])

else:
    FoodRecommender = None  # type: ignore[assignment,misc]
