from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset import CourtKeypointDataset
from utils.generate_heatmaps import generate_heatmaps_batch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.amp
import timm
import json
import os


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    torch.cuda.empty_cache()

    # ---------------- BACKBONE ---------------- #

    backbone = timm.create_model(
        'mobilevit_s',
        pretrained=False,
        features_only=True
    )

    # Enable gradient checkpointing (VERY IMPORTANT for ViTs)
    backbone.set_grad_checkpointing(True)

    backbone.to(device)
    backbone.train()

    # ---------------- STRIDE ---------------- #

    with torch.no_grad():
        x = torch.zeros(1, 3, 720, 1280).to(device)
        f = backbone(x)[-1]
        stride = x.shape[-1] // f.shape[-1]

    print("Stride:", stride)

    # ---------------- HEAD ---------------- #
 
    head = CourtKeypointHead(
        in_channels=f.shape[1],
        num_keypoints=14
    )
    head.to(device)
    head.train()

    # ---------------- LOAD CHECKPOINT (EPOCH 20) ---------------- #

    checkpoint = torch.load(
        "checkpoints/keypoints_epoch_20.pth",
        map_location=device
    )

    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])

    print("Loaded epoch 20 weights")

    # ---------------- DATA ---------------- #

    dataset = CourtKeypointDataset(
        img_dir="/content/keypoint_data/images",
        label_file="/content/keypoint_data/data_train.json"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ---------------- UNFREEZE ---------------- #

    for param in backbone.parameters():
        param.requires_grad = True

    # ---------------- OPTIMIZER ---------------- #

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler(amp_device)

    epochs = 20
    losses = []

    # ---------------- TRAIN ---------------- #

    for epoch in range(epochs):

        total_loss = 0

        for i, (images, keypoints) in enumerate(loader):

            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device):
                features = backbone(images)
                last_feat = features[-1]
                pred_heatmaps = head(last_feat)

                gt_heatmaps = generate_heatmaps_batch(
                    keypoints,
                    *pred_heatmaps.shape[-2:],
                    stride=stride
                )

                loss = F.mse_loss(pred_heatmaps, gt_heatmaps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        torch.save(
            {
                "backbone": backbone.state_dict(),
                "head": head.state_dict()
            },
            f"checkpoints/finetune_epoch_{epoch+1}.pth"
        )

    print("Fine-tuning complete")

    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head": head.state_dict()
        },
        "checkpoints/keypoints_finetuned_final.pth"
    )

    with open('finetune_losses.json', 'w') as file:
        json.dump(losses, file, indent=4)


if __name__ == '__main__':
    main()


