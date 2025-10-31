import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- USER INPUT ---
image_path = input("Enter the path to your image file: ").strip()

if not os.path.exists(image_path):
    print("⚠️ Image file not found!")
    exit()

# --- AUTO-DETECT MODALITY FROM IMAGE PATH ---
image_path_lower = image_path.lower()
if "xray" in image_path_lower or "x_ray" in image_path_lower:
    modality = "xray"
elif "ct" in image_path_lower:
    modality = "ct"
elif "mri" in image_path_lower:
    modality = "mri"
else:
    print("⚠️ Could not detect modality from image name or folder. Please include 'xray', 'ct', or 'mri' in the path.")
    exit()

# --- LOAD MODEL BASED ON DETECTED MODALITY ---
model_path = f"./checkpoints/{modality}/SRCNN_{modality}.pth"
if not os.path.exists(model_path):
    print(f"⚠️ Model not found for {modality.upper()} at: {model_path}")
    exit()

print(f"\n🟢 Loading {modality.upper()} model...")

model = SRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- LOAD AND PROCESS IMAGE ---
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("⚠️ Invalid image format or corrupted file.")
    exit()

# Resize to 256x256 for consistency
img = cv2.resize(img, (256, 256))
# Simulate low-res input
lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
lr_up = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

# Normalize and convert to tensor
input_tensor = torch.tensor(lr_up / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# --- INFERENCE ---
with torch.no_grad():
    output = model(input_tensor)

output_img = (output.cpu().numpy().squeeze() * 255.0).clip(0, 255).astype(np.uint8)

# --- DISPLAY RESULTS (Original + Enhanced only) ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_img, cmap='gray')
plt.title("Enhanced Image")
plt.axis('off')

plt.suptitle(f"{modality.upper()} Image Enhancement Result", fontsize=14)
plt.show()

# --- SAVE RESULT AUTOMATICALLY ---
os.makedirs("./results/single_uploads", exist_ok=True)
save_path = os.path.join("./results/single_uploads", f"enhanced_{os.path.basename(image_path)}")
cv2.imwrite(save_path, output_img)

print(f"\n✅ {modality.upper()} Image Enhanced Successfully!")
print(f"💾 Enhanced image saved at: {save_path}")

