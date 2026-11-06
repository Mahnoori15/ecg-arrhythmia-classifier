"""1D CNN for heartbeat morphology, combined with RR-interval rhythm features."""
import torch
from torch import nn


class ECGNet(nn.Module):
    def __init__(self, n_classes: int = 5, n_rr: int = 4):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.cnn = nn.Sequential(
            block(1, 16, 7), block(16, 32, 5), block(32, 64, 5), block(64, 64, 3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        # Rhythm branch: normalise RR features and lift them to the same size as the
        # shape features, so timing (key for S beats) isn't drowned out by morphology.
        self.rr = nn.Sequential(nn.BatchNorm1d(n_rr), nn.Linear(n_rr, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes)
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        # beat: (batch, 256), rr: (batch, 4)
        z = self.cnn(beat.unsqueeze(1))
        return self.head(torch.cat([z, self.rr(rr)], dim=1))
