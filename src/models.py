"""
PyTorch Deep Learning Model Architecture for Human Activity Recognition (HAPT 12-Class)
Model: ActivityLSTM (Stacked Bidirectional Long Short-Term Memory Network)
"""

import torch
import torch.nn as nn


class ActivityLSTM(nn.Module):
    """
    Two-layer Stacked Bidirectional LSTM for Temporal Sequence Modeling.
    
    Architecture:
      Input: (Batch_Size, 128 timesteps, 6 sensor channels)
      -> Bi-LSTM Layer 1 (hidden_size=128, bidirectional=True) -> Output (Batch, 128, 256)
      -> Dropout(p=0.3)
      -> Bi-LSTM Layer 2 (hidden_size=64, bidirectional=True)  -> Output (Batch, 128)
      -> BatchNorm1d(128)
      -> Linear(128, 64) + ReLU
      -> Dropout(p=0.4)
      -> Linear(64, num_classes=12) -> Logits (Batch, 12)
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        fc_dim: int = 64,
        num_classes: int = 12,
        dropout_rate: float = 0.3
    ):
        super(ActivityLSTM, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Layer 1: Bi-LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: Bi-LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim_1 * 2,
            hidden_size=hidden_dim_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Batch Normalization across the 128-dimensional context feature
        self.bn = nn.BatchNorm1d(hidden_dim_2 * 2)

        # Fully Connected Representation Layer
        self.fc = nn.Linear(hidden_dim_2 * 2, fc_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        # Final Classification Head (12 Classes)
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len=128, channels=6)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, (h_n, c_n) = self.lstm2(out)

        # Concatenate forward and backward hidden states from final timestep
        forward_hidden = h_n[0, :, :]
        backward_hidden = h_n[1, :, :]
        context = torch.cat([forward_hidden, backward_hidden], dim=1)

        context = self.bn(context)
        rep = self.relu(self.fc(context))
        rep = self.dropout2(rep)
        logits = self.classifier(rep)

        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_input = torch.randn(16, 128, 6)
    lstm_model = ActivityLSTM()
    output = lstm_model(dummy_input)
    print(f"[TEST] Output shape: {output.shape} (Expected: [16, 12])")
    print(f"[TEST] Total Trainable Params: {count_parameters(lstm_model):,}")
