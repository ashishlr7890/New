import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import importlib
import os

# ==============================
# Load Model
# ==============================
model_mod = importlib.import_module("03_model")
ImageEnhancementNet = model_mod.ImageEnhancementNet

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = ImageEnhancementNet().to(device)

# Check if model exists
model_path = "models/enhancement_model.pth"
if not os.path.exists(model_path):
    print("❌ Model not found! Train first using 04_train.py")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# ==============================
# Input Image
# ==============================
img_path = input("\nEnter image path: ").strip().strip('"').strip("'")

print("📂 Using path:", img_path)

if not os.path.exists(img_path):
    print("❌ File not found. Please check the path.")
    exit()

# ==============================
# Load & Preprocess Image
# ==============================
try:
    img = Image.open(img_path).convert("RGB")
except Exception as e:
    print("❌ Error loading image:", e)
    exit()

img = img.resize((256, 256))

img_np = np.array(img) / 255.0
img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

# ==============================
# Run Model
# ==============================
with torch.no_grad():
    enhanced = model(img_tensor).cpu()[0]

enhanced_np = enhanced.permute(1, 2, 0).numpy()

# ==============================
# Show Results
# ==============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.clip(enhanced_np, 0, 1))
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==============================
# Save Output
# ==============================
os.makedirs("outputs", exist_ok=True)

output_path = "outputs/enhanced_result.png"
cv2.imwrite(output_path, (enhanced_np * 255).astype(np.uint8))

print(f"\n✅ Saved enhanced image → {output_path}")