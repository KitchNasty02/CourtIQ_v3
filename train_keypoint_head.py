from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset import CourtKeypointDataset
from utils.generate_heatmaps import generate_heatmaps_batch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.amp
import torch
import timm
import json


def main():
    # Move to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # ------- Backbone ------ #

    backbone = timm.create_model(
        'mobilevit_s', 
        pretrained=False,
        features_only=True
    )

    weights_path = 'weights/mobilevit_simclr_final.pth'
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
        batch_size=64,          # keep batch size small
        shuffle=True,
        num_workers=2,
        # pin_memory=True
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
  
    scaler = torch.amp.GradScaler(amp_device)

    epochs = 50
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i, (images, keypoints) in enumerate(loader):
            optimizer.zero_grad()
            images = images.to(device)
            keypoints = keypoints.to(device)

            with torch.amp.autocast(amp_device):
                features = backbone(images)
                last_feat = features[-1]

                pred_heatmaps = head(last_feat)

                gt_heatmaps = generate_heatmaps_batch(keypoints, *pred_heatmaps.shape[-2:], stride=stride)
                loss = F.mse_loss(pred_heatmaps, gt_heatmaps)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if i % 50 == 0:
              print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

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

    # Save final model
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head": head.state_dict()
        },
        "checkpoints/keypoints_final.pth"
    )


    with open('keypoint_head_losses.json', 'w') as file:
        # save losses to file so I can make graph later
        json.dump(losses, file, indent=4)



if __name__ == '__main__':
    main()
