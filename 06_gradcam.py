import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

dataset_mod = importlib.import_module("02_dataset")
model_mod   = importlib.import_module("03_model")
get_data_loaders    = dataset_mod.get_data_loaders
ImageEnhancementNet = model_mod.ImageEnhancementNet


class GradCAMWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.model = base

    def forward(self, x):
        return self.model(x).mean(dim=[2, 3])


def run_gradcam():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, _ = get_data_loaders()

    base_model = ImageEnhancementNet().to(device)
    base_model.load_state_dict(torch.load("models/enhancement_model.pth",
                                           map_location=device))
    base_model.eval()

    wrapped      = GradCAMWrapper(base_model).to(device)
    target_layer = [base_model.bottleneck.block[-2]]
    cam          = GradCAM(model=wrapped, target_layers=target_layer)

    dark_batch, _ = next(iter(val_loader))
    n = min(4, len(dark_batch))

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        inp          = dark_batch[i:i+1].to(device)
        gray_cam     = cam(input_tensor=inp)[0]
        rgb_np       = np.clip(dark_batch[i].permute(1, 2, 0).numpy(),
                               0, 1).astype(np.float32)
        heatmap      = show_cam_on_image(rgb_np, gray_cam, use_rgb=True)

        axes[0, i].imshow(rgb_np)
        axes[0, i].set_title("Input Image")
        axes[0, i].axis("off")
        axes[1, i].imshow(heatmap)
        axes[1, i].set_title("Grad-CAM Heatmap")
        axes[1, i].axis("off")

    plt.suptitle("Grad-CAM: Model Attention on Enhancement Regions",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/gradcam_results.png", dpi=150)
    plt.show()
    print("Saved to outputs/gradcam_results.png")


if __name__ == "__main__":
    run_gradcam()