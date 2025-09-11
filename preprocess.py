import os
import cv2
import numpy as np

# Main dataset folder (raw images)
input_folder = "Dataset"   #  normal/ , pneumonia/
output_folder = "Preprocessed_dataset"

# Target image size
target_size = (224, 224)

# Create processed folder structure
for category in ["Normal", "Pneumonia"]:
    os.makedirs(os.path.join(output_folder, category), exist_ok=True)

def preprocess_image(image_path, save_path):
    try:
        # Read image (grayscale → 1 channel)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping corrupted file: {image_path}")
            return

        # Resize
        img = cv2.resize(img, target_size)

        # Convert grayscale (224,224,1) → RGB-like (224,224,3)
        img_rgb = np.stack((img,)*3, axis=-1)

        # Save processed image (no normalization here)
        cv2.imwrite(save_path, img_rgb)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Loop through both folders
for category in ["Normal", "Pneumonia"]:
    input_path = os.path.join(input_folder, category)
    output_path = os.path.join(output_folder, category)

    for file_name in os.listdir(input_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            in_file = os.path.join(input_path, file_name)
            out_file = os.path.join(output_path, file_name)
            preprocess_image(in_file, out_file)

print(" Preprocessing completed. Resized images saved in:", output_folder)