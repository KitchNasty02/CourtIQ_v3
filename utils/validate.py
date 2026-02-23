from utils.generate_heatmaps import generate_heatmaps_batch
from utils.heatmap_to_keypoints import heatmaps_to_keypoints
from utils.calculate_pck import calculate_pck
import torch.nn.functional as F
import torch

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
            
            # forward pass
            features = backbone(images)
            pred_heatmaps = head(features[-1])
            
            # calculate loss
            gt_heatmaps = generate_heatmaps_batch(keypoints, *pred_heatmaps.shape[-2:], stride=stride)
            loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
            
            # convert heatmaps to keypoints for accuracy metrics
            pred_kps_batch = []
            for b in range(pred_heatmaps.shape[0]):
                pred_kps = heatmaps_to_keypoints(
                    pred_heatmaps[b:b+1], 
                    img_size, 
                    stride
                )
                pred_kps_batch.append(pred_kps[0])
            
            pred_kps_tensor = torch.tensor(pred_kps_batch, device=device)
            
            # calculate PCK
            pck, avg_distance = calculate_pck(pred_kps_tensor, keypoints, threshold=0.05, img_size=img_size)
            
            total_loss += loss.item()
            total_pck += pck
            total_distance += avg_distance
            num_batches += 1

            # delete to free memory
            del images, keypoints, features, pred_heatmaps, gt_heatmaps, pred_kps_tensor
            torch.cuda.empty_cache()
    
    backbone.train()
    head.train()
    
    return {
        'loss': total_loss / num_batches,
        'pck': total_pck / num_batches,
        'avg_distance': total_distance / num_batches
    }

