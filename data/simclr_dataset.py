import os
from PIL import Image
from torch.utils.data import Dataset


# loads frames and returns augmented pairs
# transform is the tennis augment being applied twice
class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = []

        # walking makes it so the images can be separated in folders
        for root, _, files in os.walk(root_dir):
            for f in files:
              if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                  self.image_paths.append(os.path.join(root, f))

        # raise error if no frames exist in folder
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2  # for contrastive loss