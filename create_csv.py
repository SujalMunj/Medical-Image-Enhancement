import os
import csv
import random

# Main dataset folder
main_folder = r"C:\Users\prajy\OneDrive\Desktop\SY\Mini Project\Dataset"  # Update your path

# Output CSV
output_csv = "Raw_dataset.csv"

# Generate patient ID
def generate_patient_id(index):
    return f"P{index:04d}"  # P0001 → P3000

# Collect all images (from both folders)
all_images = []
for label_folder in ["normal", "pneumonia"]:  # folders
    label_path = os.path.join(main_folder, label_folder)
    if os.path.isdir(label_path):
        for image_name in os.listdir(label_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(label_path, image_name)
                all_images.append((label_folder, image_path))

# Shuffle for mixing
random.shuffle(all_images)

# Limit to 3000 images (exactly as per your dataset size)
if len(all_images) > 3000:
    all_images = all_images[:3000]

# Create CSV data
data = []
for index, (label, path) in enumerate(all_images, start=1):
    patient_id = generate_patient_id(index)
    data.append([patient_id, label, path, "Raw"])

# Write to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Patient_ID", "Label", "Image_Path", "Status"])
    writer.writerows(data)

print(f"CSV created: {output_csv} ({len(data)} images)")
