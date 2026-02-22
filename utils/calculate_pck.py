import numpy as np
import torch


def calculate_pck(pred_keypoints, gt_keypoints, threshold=0.05, img_size=(720, 1280)):
    """
    Calculate Percentage of Correct Keypoints (PCK)
    
    Args:
        pred_keypoints: (B, K, 2) predicted keypoints
        gt_keypoints: (B, K, 2) ground truth keypoints
        threshold: Distance threshold as fraction of image diagonal
        img_size: (height, width)
    
    Returns:
        PCK score (0-1)
    """
    # Calculate image diagonal
    h, w = img_size
    diagonal = np.sqrt(h**2 + w**2)
    threshold_pixels = threshold * diagonal
    
    # Calculate distances
    distances = torch.sqrt(((pred_keypoints - gt_keypoints) ** 2).sum(dim=-1))  # (B, K)
    
    # Count correct predictions (within threshold)
    correct = (distances < threshold_pixels).float()
    pck = correct.mean().item()
    
    return pck, distances.mean().item()


