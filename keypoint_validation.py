from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset import CourtKeypointDataset
from utils.generate_heatmaps import generate_heatmaps_batch
from utils.heatmap_to_keypoints import heatmaps_to_keypoints

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import timm


def validate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")

    # backbone
    backbone = timm.create_model(
        'mobilevit_s',
        pretrained=False,
        features_only=True
    )
    backbone.to(device)

    # compute stride (should be 32)
    with torch.no_grad():
        x = torch.zeros(1, 3, 720, 1280).to(device)
        f = backbone(x)[-1]
        stride = x.shape[-1] // f.shape[-1]

    # keypoint head
    head = CourtKeypointHead(
        in_channels=f.shape[1],
        num_keypoints=14
    )
    head.to(device)

    # load weights
    checkpoint = torch.load(
        "weights/finetune_epoch_20.pth",
        map_location=device
    )

    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])

    # set to evaluation mode
    backbone.eval()
    head.eval()


    # validation dataset
    val_dataset = CourtKeypointDataset(
        img_dir="keypoint_data/images",
        label_file="keypoint_data/data_val.json"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,   # could try different batch size
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    with torch.no_grad():

        for i, (images, keypoints) in enumerate(val_loader):

            images = images.to(device)

            features = backbone(images)
            last_feat = features[-1]
            pred_heatmaps = head(last_feat)

            img_size = images[0].shape[1:]
            pred_keypoints = heatmaps_to_keypoints(pred_heatmaps, img_size, stride)

            gt_keypoints = keypoints[0].cpu().numpy()

            # move image to cpu for plotting
            img = images[0].cpu().permute(1, 2, 0).numpy()

            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            # plt.imshow(pred_heatmaps[0,0].cpu(), alpha=0.5) # shows heatmap at the first keypoint

            plt.title("Predicted Keypoints")

            # red predicted, green truth
            for i, (x, y) in enumerate(pred_keypoints[0]):
                plt.scatter(x, y, c='red', s=40)
                # plt.text(x + 0.1, y, str(i), fontsize=12) # write keypoint number next to point

            for (x, y) in gt_keypoints:
                plt.scatter(x, y, c='green', s=40)

            plt.show()






if __name__ == "__main__":
    validate()


