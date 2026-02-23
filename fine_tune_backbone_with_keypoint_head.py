from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset_augmented import KeypointDatasetWithAugmentation
from utils.generate_heatmaps import generate_heatmaps_batch
from plotting.plot_training_metrics import plot_training_metrics
from utils.validate import validate

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.amp
import timm


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    torch.cuda.empty_cache()

    # ---------------- backbone ---------------- #

    backbone = timm.create_model(
        'mobilevit_s',
        pretrained=False,
        features_only=True
    )

    # enable gradient checkpointing (important for ViTs)
    backbone.set_grad_checkpointing(True)

    backbone.to(device)
    backbone.train()

    # ---------------- stride ---------------- #

    with torch.no_grad():
        x = torch.zeros(1, 3, 720, 1280).to(device)
        f = backbone(x)[-1]
        stride = x.shape[-1] // f.shape[-1]

    print("Stride:", stride)

    # ---------------- head ---------------- #
 
    head = CourtKeypointHead(
        in_channels=f.shape[1],
        num_keypoints=14
    )
    head.to(device)
    head.train()

    # ---------------- load weights ---------------- #

    weights = torch.load(
        "weights/keypoints_final.pth",
        map_location=device
    )

    backbone.load_state_dict(weights["backbone"])
    head.load_state_dict(weights["head"])

    print("Loaded keypoint head weights")

    # ---------------- data ---------------- #

    train_dataset = KeypointDatasetWithAugmentation(
        img_dir="/content/data/data/images",
        label_file="/content/data/data/data_train.json",
        augment=True,
        img_size=(720, 1280)
    )

    val_dataset = KeypointDatasetWithAugmentation(
        img_dir="/content/data/data/images",
        label_file="/content/data/data/data_val.json",
        augment=False,  # no augmentation
        img_size=(720, 1280)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,    
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,    
        shuffle=False,
        num_workers=2,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ---------------- unfreeze ---------------- #

    for param in backbone.parameters():
        param.requires_grad = True

    # ---------------- optimizer ---------------- #

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )

    # ---------------- scheduler ---------------- #

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    scaler = torch.amp.GradScaler(amp_device)

    epochs = 50

    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_pck': [],
        'val_distance': []
    }

    best_pck = 0

    # ---------------- train ---------------- #

    for epoch in range(epochs):
        backbone.train()
        head.train()
        total_train_loss = 0

        for i, (images, keypoints) in enumerate(train_loader):

            optimizer.zero_grad()
            images = images.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device):
                features = backbone(images)
                last_feat = features[-1]

                pred_heatmaps = head(last_feat)

                gt_heatmaps = generate_heatmaps_batch(keypoints, *pred_heatmaps.shape[-2:], stride=stride)
                loss = F.mse_loss(pred_heatmaps, gt_heatmaps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            if i % 50 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        torch.cuda.empty_cache()

        # validation
        val_metrics = validate(backbone, head, val_loader, device, stride, img_size=(720, 1280))

        # update LR scheduler
        scheduler.step(val_metrics['loss'])

        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_pck'].append(val_metrics['pck'])
        metrics['val_distance'].append(val_metrics['avg_distance'])

        # print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(f"  Train Loss:      {avg_train_loss:.4f}")
        print(f"  Val Loss:        {val_metrics['loss']:.4f}")
        print(f"  Val PCK@0.05:    {val_metrics['pck']:.4f} ({val_metrics['pck']*100:.2f}%)")
        print(f"  Val Avg Dist:    {val_metrics['avg_distance']:.2f} pixels\n")

        # save best model
        if val_metrics['pck'] > best_pck:
            best_pck = val_metrics['pck']
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch + 1,
                    "pck": best_pck
                },
                "/content/drive/MyDrive/Projects/CourtIQ/checkpoints/keypoints_best.pth"
            )
            print(f"New best model PCK: {best_pck:.4f}")

        # save checkpoint every 10 epoch
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": metrics
                },
                f"/content/drive/MyDrive/Projects/CourtIQ/checkpoints/keypoints_epoch_{epoch+1}.pth"
            )

            plot_training_metrics(metrics, save_path=f"/content/drive/MyDrive/Projects/CourtIQ/images/metrics_epoch_{epoch+1}.png")



    print("Fine-tuning complete")

    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "metrics": metrics
        },
        "/content/drive/MyDrive/Projects/CourtIQ/checkpoints/keypoints_final.pth"
    )

    print(f"Training Complete:")
    print(f"Best Validation PCK: {best_pck:.4f} ({best_pck*100:.2f}%)")
    print(f"Final Validation PCK: {metrics['val_pck'][-1]:.4f}")
    print(f"Final Avg Distance: {metrics['val_distance'][-1]:.2f} pixels")

    plot_training_metrics(metrics, save_path='/content/drive/MyDrive/Projects/CourtIQ/images/keypoint_metrics_final.png')



if __name__ == '__main__':
    main()


