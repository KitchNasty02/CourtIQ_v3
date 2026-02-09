from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset import CourtKeypointDataset
from utils.generate_heatmaps import generate_heatmaps_batch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import timm
import json


def main():
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
    backbone.load_state_dict(weights, strict=False)


    backbone.to(device)
    backbone.train() # set to training mode


    with torch.no_grad():
        x = torch.zeros(1, 3, 720, 1280).to(device)
        f = backbone(x)[-1]
        stride = x.shape[-1] // f.shape[-1]

    print('Stride: ', stride)


    head = CourtKeypointHead(
        in_channels=f.shape[1],
        num_keypoints=14
    )
    head.to(device)
    head.train()



    dataset = CourtKeypointDataset(
        img_dir="keypoint_data/images",
        label_file="keypoint_data/data_train.json"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,          # keep batch size small
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ADAM optimizer
    optimizer = torch.optim.AdamW(
        list(head.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    # freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False


    epochs = 50
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for images, keypoints in loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            with torch.no_grad():
                features = backbone(images)
            last_feat = features[-1]

            pred_heatmaps = head(last_feat)

            Hm, Wm = pred_heatmaps.shape[-2:]

            gt_heatmaps = generate_heatmaps_batch(
                keypoints, 
                Hm,
                Wm,
                stride=stride
            )

            loss = F.mse_loss(pred_heatmaps, gt_heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict()
                },
                f"checkpoints/keypoints_epoch_{epoch+1}.pth"
            )

        # after 25 epoch, switch to fine tuning
        if (epoch + 1) == 25:
            # unfreeze backbone
            for param in backbone.parameters():
                param.requires_grad = True
            backbone.train()

            # update optimizer to include backbone
            optimizer = torch.optim.AdamW(
                list(backbone.parameters()) + list(head.parameters()),
                lr=1e-4,  # smaller LR for fine-tuning
                weight_decay=1e-4
            )

    with open('losses.json', 'w') as file:
        # save losses to file so I can make graph later
        json.dump(losses, file, indent=4)



if __name__ == '__main__':
    main()

