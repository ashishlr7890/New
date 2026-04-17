import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

dataset_mod = importlib.import_module("02_dataset")
model_mod   = importlib.import_module("03_model")
get_data_loaders    = dataset_mod.get_data_loaders
ImageEnhancementNet = model_mod.ImageEnhancementNet


def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, _ = get_data_loaders()

    model = ImageEnhancementNet().to(device)
    model.load_state_dict(torch.load("models/enhancement_model.pth",
                                     map_location=device))
    model.eval()

    psnr_list, ssim_list, samples = [], [], []

    with torch.no_grad():
        for dark, clean in val_loader:
            pred = model(dark.to(device)).cpu()
            for p, d, c in zip(pred, dark, clean):
                p_np = p.permute(1, 2, 0).numpy()
                c_np = c.permute(1, 2, 0).numpy()
                psnr_list.append(psnr(c_np, p_np, data_range=1.0))
                ssim_list.append(ssim(c_np, p_np, data_range=1.0,
                                      channel_axis=2))
            if len(samples) < 3:
                samples.append((dark[0], pred[0], clean[0]))

    print(f"\nAverage PSNR : {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM : {np.mean(ssim_list):.4f}")

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for col, title in enumerate(["Input (Dark)", "Enhanced", "Ground Truth"]):
        axes[0, col].set_title(title, fontweight="bold")

    for row, (d, e, c) in enumerate(samples):
        for col, img in enumerate([d, e, c]):
            axes[row, col].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
            axes[row, col].axis("off")

    plt.suptitle("Enhancement Results", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/visual_results.png", dpi=150)
    plt.show()
    print("Saved to outputs/visual_results.png")


if __name__ == "__main__":
    evaluate()