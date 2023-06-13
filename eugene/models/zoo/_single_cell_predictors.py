import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import _layers as layers
from ..base import _regularizers as regularizers
from ..base import _blocks as blocks
from ..base import _towers as towers


class scBasset(nn.Module):
    def __init__(self, num_cells, num_batches=None, l1=0.0, l2=0.0):
        super(scBasset, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv_tower = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=288, kernel_size=5, padding=2),
                nn.BatchNorm1d(288),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=288, out_channels=323, kernel_size=5, padding=2),
                nn.BatchNorm1d(323),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=323, out_channels=363, kernel_size=5, padding=2),
                nn.BatchNorm1d(363),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=363, out_channels=407, kernel_size=5, padding=2),
                nn.BatchNorm1d(407),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=407, out_channels=456, kernel_size=5, padding=2),
                nn.BatchNorm1d(456),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv1d(in_channels=456, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.flatten = nn.Flatten()

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=1792, out_features=32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.GELU(),
        )

        self.fc1 = regularizers.L2(
            nn.Linear(in_features=32, out_features=num_cells), weight_decay=l1
        )
        self.sigmoid = nn.Sigmoid()

        if num_batches is not None:
            self.fc2 = regularizers.L2(
                nn.Linear(in_features=32, out_features=num_batches), weight_decay=l2
            )

        self.l1 = l1
        self.l2 = l2

    def forward(self, x, batch=None):
        x = self.conv1(x)
        x = self.conv_tower(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.bottleneck(x)
        if batch is not None:
            x_batch = self.fc2(x)
            x_batch = torch.matmul(x_batch, batch)
            x = self.fc1(x) + x_batch
        else:
            x = self.fc1(x)

        x = self.sigmoid(x)

        return x
