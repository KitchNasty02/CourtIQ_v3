import torch.nn.functional as F

def heatmaps_to_keypoints(heatmaps, img_size, stride):
    B, K, H_hm, W_hm = heatmaps.shape
    H_img, W_img = img_size
    keypoints = []

    for b in range(B):
        kp_list = []
        for k in range(K):
            heatmap = heatmaps[b, k]
            
            # get index of max activation
            flat_idx = heatmap.argmax()
            y = flat_idx // W_hm
            x = flat_idx % W_hm
            
            x = x.item()
            y = y.item()
            
            
            """
            Subpixel refinement using Taylor expansion

                Imagine you have a heatmap peak at position `x=12`. The values around it are:
                    
                    heatmap[y, 11] = 0.3
                    heatmap[y, 12] = 0.9  <-- peak
                    heatmap[y, 13] = 0.5

                The true peak is probably not exactly at x=12, but somewhere between 11 and 13.

                Model the heatmap locally as a quadratic function
                
                    h(x) ≈ h(x₀) + h'(x₀)·Δx + ½h''(x₀)·Δx²
            """
            if 1 <= x < W_hm-1 and 1 <= y < H_hm-1:
                # get neighboring values
                # curvature (second derivative)
                dxx = (heatmap[y, x+1] - 2*heatmap[y, x] + heatmap[y, x-1]).item()
                dyy = (heatmap[y+1, x] - 2*heatmap[y, x] + heatmap[y-1, x]).item()

                # gradient (first derivative)
                dx = (heatmap[y, x+1] - heatmap[y, x-1]).item() * 0.5
                dy = (heatmap[y+1, x] - heatmap[y-1, x]).item() * 0.5
                
                # quadratic peak estimation
                if dxx < 0:  # ensure it's a peak (concave down), not valley
                    x = x - dx / dxx    # taylor expansion formula
                if dyy < 0:
                    y = y - dy / dyy
            
            # convert to image coordinates
            x = x * stride
            y = y * stride

            # clamp to image bounds
            x = min(max(x, 0), W_img - 1)
            y = min(max(y, 0), H_img - 1)

            kp_list.append([x, y])

        keypoints.append(kp_list)

    return keypoints


