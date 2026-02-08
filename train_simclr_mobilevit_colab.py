from data.simclr_dataset import SimCLRDataset
from data.tennis_augment import TennisAugment
from models.simclr_projection_head import ProjectionHead
from torch.utils.data import DataLoader
import torch.amp

from pytorch_metric_learning.losses import NTXentLoss

import torch
import timm
import torch.nn as nn


def main():

    # Move to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # ------- Backbone ------ #

    # create MobileViT v2 S backbone
    # settings with no classifier and with global embedding for training
    backbone = timm.create_model(
        'mobilevit_s', 
        pretrained=True, 
        num_classes=0,      # remove classifier
        global_pool="avg"
    )
    backbone.train() # set to training mode
    backbone.to(device)

    # get input dim (should be 640)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 256, 256).to(device)
        in_dim = backbone(dummy).shape[1]

    projection = ProjectionHead(in_dim)
    projection.to(device)

    # -------- Dataset -------- #

    transform = TennisAugment(image_size=256)
    dataset = SimCLRDataset('/content/drive/MyDrive/Projects/CourtIQ/frames', transform)

    loader = DataLoader(
        dataset,
        batch_size=128,      # larger batches are better for contrastive learning (128, 256)
        shuffle=True,
        num_workers=2,  # 4 is best, recommended 2 in current setup
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )


    loss_func = NTXentLoss(temperature=0.1)     # typically use small values like 0.1 or 0.2

    scaler = torch.amp.GradScaler(amp_device)  # scaler for automatic mixed precision

    # can try different LR and WD
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projection.parameters()),
        lr=3e-4,            # learning rate = 0.00003
        weight_decay=1e-4  
    )

    best_loss = 100  # large number that loss will be less than
    losses = [] # Loss for each epoch

    # train
    epochs = 200  # train like 200 epochs when good

    for epoch in range(epochs):
        backbone.train()
        projection.train()

        total_loss = 0

        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device):

                # ---- forward backbone ----
                f1 = backbone(x1)
                f2 = backbone(x2)

                # ---- projection ----
                z1 = projection(f1)
                z2 = projection(f2)

                # ---- prepare SimCLR loss ----
                embeddings = torch.cat([z1, z2], dim=0) # put embeddings of both images together

                B = z1.size(0)
                labels = torch.arange(B, device=device) # create labels
                labels = torch.cat([labels, labels], dim=0) # dupicate labels

                loss = loss_func(embeddings, labels)


            # backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs} | Loss: {avg_loss:.4f}]')

        # save backbone every 25 epochs, don't need to save head
        if (epoch+1) % 25 == 0:
            print(f'Epoch {epoch+1} weights saved')
            torch.save(
                backbone.state_dict(),
                f"/content/drive/MyDrive/Projects/CourtIQ/checkpoints/mobilevit_simclr_epoch_{epoch+1}.pth"
            )

        # save as best weights if loss is lowest
        if avg_loss < best_loss:
            print(f'Epoch {epoch+1} weights saved as best weights')
            torch.save(
                backbone.state_dict(),
                f"/content/drive/MyDrive/Projects/CourtIQ/checkpoints/mobilevit_simclr_best.pth"
            )


    torch.save(
        backbone.state_dict(),
        f"/content/drive/MyDrive/Projects/CourtIQ/checkpoints/mobilevit_simclr_final.pth"
    )



if __name__ == '__main__':

    main()



