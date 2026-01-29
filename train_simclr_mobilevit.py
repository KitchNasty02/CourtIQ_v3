from data.simclr_dataset import SimCLRDataset
from data.tennis_augment import TennisAugment

import torch
import timm
import torch.nn as nn


# create MobileViT v2 S backbone
# settings with no classifier and with global embedding for training
backbone = timm.create_model('mobilevit_s', pretrained=True, num_classes=0, global_pool="avg")

backbone.eval()

# Move to gpu if available -- might need to install torch for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)


x = torch.randn(2, 3, 256, 256).to(device)
y = backbone(x)
print(y.shape)


