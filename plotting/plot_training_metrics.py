import matplotlib.pyplot as plt


def plot_training_metrics(metrics, save_path='images/keypoint_training_metrics.png'):
    """
    Plot training curves.
    """
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCK curve
    axes[0, 1].plot(epochs, metrics['val_pck'], color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('PCK@0.05', fontsize=12)
    axes[0, 1].set_title('Validation PCK (Percentage of Correct Keypoints)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% target')
    axes[0, 1].legend()
    
    # Distance curve
    axes[1, 0].plot(epochs, metrics['val_distance'], color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Average Distance (pixels)', fontsize=12)
    axes[1, 0].set_title('Validation Average Keypoint Distance', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PCK percentage
    axes[1, 1].plot(epochs, [pck * 100 for pck in metrics['val_pck']], color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Validation Accuracy (PCK %)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('images/keypoint_training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Metrics plot saved to '{save_path}'")