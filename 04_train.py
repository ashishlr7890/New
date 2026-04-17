import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import importlib

dataset_mod = importlib.import_module("02_dataset")
model_mod   = importlib.import_module("03_model")
get_data_loaders    = dataset_mod.get_data_loaders
ImageEnhancementNet = model_mod.ImageEnhancementNet


def edge_loss(pred, target):
    device = pred.device
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                            dtype=torch.float32).view(1,1,3,3).to(device)
    sobel_y = sobel_x.transpose(2, 3)

    def edges(img):
        gray = img.mean(dim=1, keepdim=True)
        ex = F.conv2d(gray, sobel_x, padding=1)
        ey = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(ex**2 + ey**2 + 1e-6)

    return F.l1_loss(edges(pred), edges(target))


def total_loss(pred, target, alpha=0.8, beta=0.2):
    return alpha * F.mse_loss(pred, target) + beta * edge_loss(pred, target)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_data_loaders()
    model = ImageEnhancementNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    EPOCHS = 5
    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for dark, clean in train_loader:
            dark, clean = dark.to(device), clean.to(device)
            pred = model(dark)
            loss = total_loss(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for dark, clean in val_loader:
                dark, clean = dark.to(device), clean.to(device)
                v_loss += total_loss(model(dark), clean).item()

        history["train"].append(t_loss / len(train_loader))
        history["val"].append(v_loss / len(val_loader))
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train: {history['train'][-1]:.4f} | "
              f"Val: {history['val'][-1]:.4f}")

    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/enhancement_model.pth")
    print("Model saved to models/enhancement_model.pth")

    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    plt.show()
    print("Training curve saved to outputs/training_curves.png")


if __name__ == "__main__":
    train()