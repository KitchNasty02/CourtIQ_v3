from models.keypoint_head import CourtKeypointHead
import torch
import timm


# Move to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# ------- Backbone ------ #

backbone = timm.create_model(
    'mobilevit_s', 
    pretrained=False,
    features_only=True
)

weights_path = 'weights\mobilevit_simclr_final.pth'
weights = torch.load(weights_path, map_location=device)


missing, unexpected = backbone.load_state_dict(weights, strict=False)
# these are ok because they were weights in the projection head or layers
# that dont exist in the feature only model
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

backbone.to(device)
backbone.train() # set to training mode

# find feature map shape
with torch.no_grad():
    dummy = torch.zeros(1, 3, 256, 256).to(device)
    features = backbone(dummy)
    last_feat = features[-1]

print(last_feat.shape)  

projection = CourtKeypointHead(
    in_channels=last_feat.shape[1],
    num_keypoints=14
)
projection.to(device)

with torch.no_grad():
    heatmaps = projection(last_feat)

print(heatmaps.shape)






