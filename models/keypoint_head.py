import torch.nn as nn


class CourtKeypointHead(nn.Module):
    def __init__(self, in_channels=640, num_keypoints=14):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_keypoints, 1)  # heatmaps
        )

    def forward(self, x):
        return self.head(x)

