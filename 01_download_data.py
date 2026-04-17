import os
import subprocess

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Downloading Nighttime Driving Dataset from Kaggle...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "ipythonx/nighttime-driving-dataset",
    "-p", "./data", "--unzip"
], check=True)
print("Download complete! Files saved to ./data/")