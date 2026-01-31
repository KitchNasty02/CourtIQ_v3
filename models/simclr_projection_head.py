import torch.nn as nn

"""
This projection head is added on top of the mobilevit backbone when fine tuning
using contrastive loss. The head absorbs contrastive-specific distortions and 
leaves the backbone with general learning

"""
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=640, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),      # one hidden layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    
