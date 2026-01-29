import torch
import timm
import torch.nn as nn


# create MobileViT v2 S backbone
# features only does not give final classifier when true
backbone = timm.create_model('mobilevit_s', pretrained=True, features_only=True)

backbone.eval()

# Move to gpu if available -- might need to install torch for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.to(device)







