import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def synthesize_low_light(img_np, gamma=2.5, noise_std=15):
    img = img_np.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    noise = np.random.normal(0, noise_std / 255.0, img.shape)
    img = np.clip(img + noise, 0, 1)
    return (img * 255).astype(np.uint8)


class NightDriveDataset(Dataset):
    def __init__(self, img_paths, img_size=(256, 256)):
        self.paths = img_paths
        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.resize(img)
        clean = self.to_tensor(img)
        dark_np = synthesize_low_light(np.array(img))
        dark = self.to_tensor(Image.fromarray(dark_np))
        return dark, clean


def get_data_loaders(data_dir="./data", batch_size=16, img_size=(128, 128)):
    img_paths = (
        glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True)
    )
    if len(img_paths) == 0:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Run 01_download_data.py first."
        )
    print(f"Found {len(img_paths)} images.")

    split = int(0.85 * len(img_paths))
    train_ds = NightDriveDataset(img_paths[:split], img_size)
    val_ds   = NightDriveDataset(img_paths[split:], img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader, img_paths