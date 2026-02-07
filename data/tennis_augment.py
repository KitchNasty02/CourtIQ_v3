import torchvision.transforms as T
import random
from PIL import Image
import numpy as np

"""
Augmentations:
  Color Jitter - lighting changes
  Horizontal flip - players switching sides
  Random crop and resize - zooming or camera pan
  Gaussian blur - motion blur
  Random occlusion - objects blocking view
  Court mask noise - and synthetic shadows or partial line erase
  Rotation (<10 deg) - camera tilt or bounce distortion
  Elastic distortion - mimics player movement stretch frames

EXPERIMENT WITH DIFFERENT PARAMTERS FOR THE AUGMENTS

"""

# custom distortion to simulate tennis distortions
class TennisAugment:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.augment = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            # might not be good for tennis court -- p is the probability of it being flipped (0.5 default)
            T.RandomHorizontalFlip(p=0.25),
            T.RandomRotation(degrees=5),   # stay with small rotation
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Lambda(self.random_occlusion),  # Simulate net/player occlusion
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    # ran when class is called
    def __call__(self, img):
        return self.augment(img)

    # adds a balck patch to simulate a blocking object
    def random_occlusion(self, tensor):
        if random.random() < 0.2:
            _, H, W = tensor.shape
            h = random.randint(H // 12, H // 4)     # can experiment with different values
            w = random.randint(W // 12, W // 4)
            x = random.randint(0, H - h)
            y = random.randint(0, W - w)
            tensor[:, x:x+h, y:y+w] = 0.0  # black patch
        return tensor
    


