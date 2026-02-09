import json
import os
import cv2
import torch
from torch.utils.data import Dataset

class CourtKeypointDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir
        with open(label_file) as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]

        img_path = os.path.join(self.img_dir, item["id"] + ".png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        keypoints = torch.tensor(item["kps"]).float()  # (14, 2)

        return img, keypoints
