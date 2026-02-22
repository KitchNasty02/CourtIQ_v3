import json
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class KeypointDatasetWithAugmentation(Dataset):
    def __init__(self, img_dir, label_file, augment=True, img_size=(720, 1280)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        with open(label_file) as f:
            self.labels = json.load(f)

        if augment:
            self.transform = self.get_augmentation_pipeline()
        else:
            self.transform = A.Compose([
                A.NoOp()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    def get_augmentation_pipeline(self):
        """
        Create an augmentation pipeline with perspective transforms.
        Simulates different camera angles/heights.
        """
        return A.Compose([
            # main augmentation of different camera angles
            A.OneOf([
                # really low angles
                A.Perspective(
                    scale=(0.3, 0.45),  # (0.15, 0.3)
                    keep_size=True,
                    p=1.0
                ),

                # medium angles
                A.Perspective(
                    scale=(0.1, 0.3),   # (0.05, 0.15)
                    keep_size=True,
                    p=1.0
                ),

                # side angle
                A.Affine(
                    translate_percent={'x': (-0.2, 0.2), 'y': (-0.1, 0.1)},
                    scale=(0.8, 1.1),
                    rotate=(-15, 15),
                    shear={'x': (-10, 10), 'y': (-5, 5)},
                    p=1.0
                )
            ], p=0.8),     # 80% chance to add these augments

            # rotation/zoom
            A.OneOf([
                # small rotation
                A.Rotate(
                    limit=(-10, 10),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                # zoom
                A.RandomScale(
                    scale_limit=(-0.1, 0.1),     # (-0.2, 0.2)
                    p=1.0
                ),
            ], p=0.4),

            # color/lighting
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=1.0
                ),
            ], p=0.6),

            # simulate different lighting
            A.OneOf([
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.RandomToneCurve(scale=0.3, p=1.0),
            ], p=0.3),

            # simulate camera blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.25),

            # different image qualities
            A.OneOf([
                A.GaussNoise(std_range=(0.2, 0.5), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),

            # simulate shadows
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2
            ),

        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)     # keep keypoints even if off-screen
        
        )


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        item = self.labels[idx]

        img_path = os.path.join(self.img_dir, item["id"] + ".png")
        img = cv2.imread(img_path)

        # flip vertically for augments (makes sure low/high angles are correct)
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        keypoints_np = np.array(item["kps"])  # (14, 2)

        # flip keypoints
        keypoints_np[:, 1] = h - keypoints_np[:, 1]
        keypoints_list = keypoints_np.tolist()

        # apply augmentation
        if self.augment:
            transformed = self.transform(image=img, keypoints=keypoints_list)
            img = transformed['image']
            keypoints_list = transformed['keypoints']

        # flip back after augments
        img = cv2.flip(img, 0)
        keypoints_array = np.array(keypoints_list)
        keypoints_array[:, 1] = img.shape[0] - keypoints_array[:, 1]
        
        # back to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        keypoints_tensor = torch.from_numpy(keypoints_array).float()

        return img_tensor, keypoints_tensor
    



