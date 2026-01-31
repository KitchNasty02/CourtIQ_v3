import torch
import torch.nn.functional as F

"""
NTXent Loss (Normalized tempurature-scaled cross entropy loss) 
is used in SSL tasks and specifically contrastive learning


"""
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature


    def forward(self, z1, z2):
        pass


