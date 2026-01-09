import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleHARNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(9, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)
