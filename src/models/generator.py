import torch
import torch.nn as nn


class ClipModule(nn.Module):
    def __init__(self, min_threshold, max_threshold):
        super(ClipModule, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    
    def forward(self, x):
        return torch.clamp(x, min=self.min_threshold, max=self.max_threshold)


class FCGenerator(nn.Module):
    def __init__(self, fc1_dim, fc2_dim):
        super(FCGenerator, self).__init__()

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        fully_connected = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.fc1_dim),
            nn.Linear(self.fc1_dim, self.fc2_dim, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(self.fc2_dim, self.num_classes, bias=False),
        )
        self.generator = nn.Sequential(
            fully_connected,
            # nn.Sigmoid()
            ClipModule(min_threshold=0, max_threshold=1),
        )

    def forward(self, x):
        return self.generator(x)