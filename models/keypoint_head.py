import torch.nn as nn


class CourtKeypointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_keypoints, 1),
        )

    def forward(self, x):
        return self.head(x)

