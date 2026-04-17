import importlib
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

dataset_mod = importlib.import_module("02_dataset")
model_mod   = importlib.import_module("03_model")
get_data_loaders    = dataset_mod.get_data_loaders
ImageEnhancementNet = model_mod.ImageEnhancementNet


def detect(yolo_model, img_tensor):
    img_np  = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    results = yolo_model(img_np, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    return annotated, len(results.boxes)


def run_detection():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, _ = get_data_loaders()

    enhance_model = ImageEnhancementNet().to(device)
    enhance_model.load_state_dict(torch.load("models/enhancement_model.pth",
                                              map_location=device))
    enhance_model.eval()

    yolo = YOLO("yolov8n.pt")

    dark_batch, _ = next(iter(val_loader))
    with torch.no_grad():
        enhanced = enhance_model(dark_batch.to(device)).cpu()

    n = min(4, len(dark_batch))
    fig, axes    = plt.subplots(2, n, figsize=(4 * n, 8))
    total_before = 0
    total_after  = 0

    for i in range(n):
        before_img, nb = detect(yolo, dark_batch[i])
        after_img,  na = detect(yolo, enhanced[i])
        total_before  += nb
        total_after   += na

        axes[0, i].imshow(before_img)
        axes[0, i].set_title(f"Before: {nb} objects", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(after_img)
        axes[1, i].set_title(f"After: {na} objects", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle(
        f"YOLOv8 Detection  |  Before: {total_before}  →  After: {total_after}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/yolo_detection.png", dpi=150)
    plt.show()
    print("Saved to outputs/yolo_detection.png")


if __name__ == "__main__":
    run_detection()