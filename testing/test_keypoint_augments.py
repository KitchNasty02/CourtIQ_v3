import matplotlib.pyplot as plt
import numpy as np
from data.keypoint_dataset_augmented import KeypointDatasetWithAugmentation

def visualize_augmentations(dataset, num_samples=6, samples_per_image=3):
    """
    Visualize original and augmented versions of images with keypoints
    
    Args:
        dataset: CourtKeypointDatasetWithAugmentation instance
        num_samples: Number of original images to show
        samples_per_image: How many augmented versions per image
    """
    fig = plt.figure(figsize=(20, 4 * num_samples))
    
    for img_idx in range(num_samples):
        # Get original (no augmentation)
        dataset.augment = False
        img_orig, kp_orig = dataset[img_idx]
        
        # Show original
        ax = plt.subplot(num_samples, samples_per_image + 1, 
                        img_idx * (samples_per_image + 1) + 1)
        plot_image_with_keypoints(img_orig, kp_orig, ax, "Original")
        
        # Show augmented versions
        dataset.augment = True
        for aug_idx in range(samples_per_image):
            img_aug, kp_aug = dataset[img_idx]
            
            ax = plt.subplot(num_samples, samples_per_image + 1,
                           img_idx * (samples_per_image + 1) + aug_idx + 2)
            plot_image_with_keypoints(img_aug, kp_aug, ax, 
                                     f"Augmented {aug_idx + 1}")
    
    plt.tight_layout()
    plt.savefig('augmentation_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'augmentation_validation.png'")


def plot_image_with_keypoints(img_tensor, keypoints, ax, title):
    """Plot image with keypoints - clamp only for display"""
    img_np = img_tensor.permute(1, 2, 0).numpy()
    h, w = img_np.shape[:2]
    
    ax.imshow(img_np)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')
    
    kp_np = keypoints.numpy()
    colors = plt.cm.tab20(np.linspace(0, 1, len(kp_np)))
    
    for i, (x, y) in enumerate(kp_np):
        # Check if in bounds for display
        if 0 <= x < w and 0 <= y < h:
            # Draw keypoint
            ax.plot(x, y, 'o', color=colors[i], markersize=8, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.text(x + 6, y - 6, str(i), color='white', fontsize=8,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                            edgecolor='white', linewidth=1.5, alpha=0.9))
        else:
            # Draw indicator at edge showing direction
            # Clamp for display only
            x_clamped = np.clip(x, 0, w - 1)
            y_clamped = np.clip(y, 0, h - 1)
            
            # Draw with different style (hollow circle) to show it's off-screen
            ax.plot(x_clamped, y_clamped, 'o', color=colors[i], markersize=8,
                   markerfacecolor='none', markeredgecolor=colors[i], 
                   markeredgewidth=3, alpha=0.5)
            
            # Add arrow pointing in direction of true location
            dx = np.clip(x - x_clamped, -20, 20)
            dy = np.clip(y - y_clamped, -20, 20)
            if abs(dx) > 1 or abs(dy) > 1:
                ax.arrow(x_clamped, y_clamped, dx, dy, 
                        head_width=8, head_length=8, 
                        fc=colors[i], ec=colors[i], alpha=0.5)
            
            ax.text(x_clamped + 6, y_clamped - 6, f"{i}*", 
                   color='yellow', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=colors[i], alpha=0.7))


def test_augmentation_single_image(dataset, idx=0, num_augmentations=9):
    """
    Show one image with multiple augmentation variations
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    dataset.augment = True
    
    for i in range(num_augmentations):
        img, kp = dataset[idx]
        plot_image_with_keypoints(img, kp, axes[i], f"Augmentation {i+1}")
    
    plt.tight_layout()
    plt.savefig(f'single_image_augmentations_{idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to 'single_image_augmentations_{idx}.png'")




if __name__ == "__main__":
    # Create augmented dataset
    dataset = KeypointDatasetWithAugmentation(
        img_dir="keypoint_data/images",
        label_file="keypoint_data/data_train.json",
        augment=True
    )
    
    print("Dataset created with augmentation")
    print(f"Total samples: {len(dataset)}")
    
    print("\nGenerating augmentation comparison grid...")
    visualize_augmentations(dataset, num_samples=4, samples_per_image=3)
    
    print("\nGenerating single image variations...")
    test_augmentation_single_image(dataset, idx=0, num_augmentations=9)
    


