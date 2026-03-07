import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as T



class MayoCTDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.data_dir = os.path.join(
            root, "mayo_data_arranged_patientwise", split, "Phantom"
        )
        self.files = sorted(os.listdir(self.data_dir))
        if split == "train":
            self.transform = TRANSFORM
        else:
            self.transform = None
        self.dataset = [{"fn": fn} for fn in self.files]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.dataset[idx]["fn"])
        image = np.load(img_name)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {"image": image}


LPNDataset = MayoCTDataset
