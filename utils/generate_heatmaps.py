import torch

def generate_heatmaps_batch(keypoints, Hm, Wm, stride, sigma=1.5):
    """
    keypoints: (B, K, 2) in pixel coords
    returns: (B, K, Hm, Wm)
    """
    B, K, _ = keypoints.shape
    device = keypoints.device

    yy, xx = torch.meshgrid(
        torch.arange(Hm, device=device),
        torch.arange(Wm, device=device),
        indexing="ij"
    )

    heatmaps = torch.zeros(B, K, Hm, Wm, device=device)

    for b in range(B):
        for k in range(K):
            cx = keypoints[b, k, 0] / stride
            cy = keypoints[b, k, 1] / stride
            heatmaps[b, k] = torch.exp(
                -((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)
            )

    return heatmaps

