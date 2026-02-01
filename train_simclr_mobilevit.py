from data.simclr_dataset import SimCLRDataset
from data.tennis_augment import TennisAugment
from models.simclr_projection_head import ProjectionHead
from torch.utils.data import DataLoader

from pytorch_metric_learning.losses import NTXentLoss

import torch
import timm
import torch.nn as nn


def main():

    # Move to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dataset = SimCLRDataset('C:/Users/conno/OneDrive/Desktop/Projects/CourtIQ_v2/frames/clay/gkIsvlZDG-Y', transform)

    loader = DataLoader(
        dataset,
        batch_size=32,      # larger batches are better for contrastive learning (128, 256)
        shuffle=True,
        num_workers=4,  # set to 4
        drop_last=True,
        pin_memory=True
    )


    loss_func = NTXentLoss(temperature=0.1)     # typically use small values like 0.1 or 0.2

    # can try different LR and WD
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projection.parameters()),
        lr=3e-4,            # learning rate = 0.00003
        weight_decay=1e-4  
    )


    # train
    epochs = 5  # train like 200 epochs when good

    for epoch in range(epochs):
        backbone.train()
        projection.train()

        total_loss = 0

        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

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

            # backware pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        
        avg_loss = total_loss / len(loader)
        print(f'Epoch [{epoch+1}/{epochs} | Loss: {avg_loss:.4f}]')

        # save backbone every 25 epochs, don't need to save head
        if (epoch+1) % 25 == 0:
            torch.save(
                backbone.state_dict(),
                f"checkpoints/mobilevit_simclr_epoch_{epoch+1}.pth"
            )





if __name__ == '__main__':

    main()



