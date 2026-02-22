from models.keypoint_head import CourtKeypointHead
from data.keypoint_dataset_augmented import KeypointDatasetWithAugmentation
from utils.generate_heatmaps import generate_heatmaps_batch
from utils.heatmap_to_keypoints import heatmaps_to_keypoints
from plotting.plot_training_metrics import plot_training_metrics
from utils.calculate_pck import calculate_pck

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.amp
import torch
import timm
import numpy as np



def validate(backbone, head, val_loader, device, stride, img_size=(720, 1280)):
    """
    Validate model on validation set
    """
    backbone.eval()
    head.eval()
    
    total_loss = 0
    total_pck = 0
    total_distance = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, keypoints in val_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            # Forward pass
            features = backbone(images)
            pred_heatmaps = head(features[-1])
            
            # Calculate loss
            gt_heatmaps = generate_heatmaps_batch(keypoints, *pred_heatmaps.shape[-2:], stride=stride)
            loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
            
            # Convert heatmaps to keypoints for accuracy metrics
            pred_kps_batch = []
            for b in range(pred_heatmaps.shape[0]):
                pred_kps = heatmaps_to_keypoints(
                    pred_heatmaps[b:b+1], 
                    img_size, 
                    stride
                )
                pred_kps_batch.append(pred_kps[0])
            
            pred_kps_tensor = torch.tensor(pred_kps_batch, device=device)
            
            # Calculate PCK
            pck, avg_distance = calculate_pck(pred_kps_tensor, keypoints, threshold=0.05, img_size=img_size)
            
            total_loss += loss.item()
            total_pck += pck
            total_distance += avg_distance
            num_batches += 1
    
    backbone.train()
    head.train()
    
    return {
        'loss': total_loss / num_batches,
        'pck': total_pck / num_batches,
        'avg_distance': total_distance / num_batches
    }




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

    
    train_dataset = KeypointDatasetWithAugmentation(
        img_dir="keypoint_data/images",
        label_file="keypoint_data/data_train.json",
        augment=True,
        img_size=(720, 1280)
    )

    val_dataset = KeypointDatasetWithAugmentation(
        img_dir="keypoint_data/images",
        label_file="keypoint_data/data_val.json",
        augment=False,  # no augmentation
        img_size=(720, 1280)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,    
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,    
        shuffle=False,
        num_workers=2,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ADAM optimizer
    optimizer = torch.optim.AdamW(
        list(head.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
  
    scaler = torch.amp.GradScaler(amp_device)

    epochs = 75
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_pck': [],
        'val_distance': []
    }

    best_pck = 0

    for epoch in range(epochs):
        backbone.train()
        head.train()
        total_train_loss = 0

        for i, (images, keypoints) in enumerate(train_loader):
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

            total_train_loss += loss.item()

            if i % 50 == 0:
              print(f"Epoch [{epoch+1}] Batch {i}/{len(train_loader)} | Loss: {loss.item():.8f}")

        avg_train_loss = total_train_loss / len(train_loader)


        # ----- Validation ----- #
        
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

        # Save best model
        if val_metrics['pck'] > best_pck:
            best_pck = val_metrics['pck']
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch + 1,
                    "pck": best_pck
                },
                "checkpoints/keypoints_best.pth"
            )
            print(f"New best model PCK: {best_pck:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": metrics
                },
                f"checkpoints/keypoints_epoch_{epoch+1}.pth"
            )

            plot_training_metrics(metrics, save_path=f"images/metrics_epoch_{epoch+1}.png")

        # after 50 epoch, switch to fine tuning
        if (epoch + 1) == 50:
            print('\nSwitching to Fine Tuning \n')
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

            # create scheduler for new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )

    # Save final model
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "metrics": metrics
        },
        "checkpoints/keypoints_final.pth"
    )

    print(f"Training Complete:")
    print(f"Best Validation PCK: {best_pck:.4f} ({best_pck*100:.2f}%)")
    print(f"Final Validation PCK: {metrics['val_pck'][-1]:.4f}")
    print(f"Final Avg Distance: {metrics['val_distance'][-1]:.2f} pixels")

    plot_training_metrics(metrics, save_path='images/keypoint_metrics_final.png')





if __name__ == '__main__':
    main()
