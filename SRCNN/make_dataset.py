import os
import cv2
import h5py
import numpy as np

dataset_root = "./Dataset"
modalities = ["Xray", "CT", "MRI"]
patch_size = 64

# Augmentation function
def augment_image(img, modality):
    imgs = [img]
    # Flip / Rotate / Noise augmentation
    if modality == "Xray":
        imgs.append(cv2.flip(img, 0))
        imgs.append(cv2.flip(img, 1))
        imgs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    elif modality == "MRI":
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        imgs.append(cv2.add(img, noise))
        imgs.append(cv2.GaussianBlur(img, (3,3), 0))
        imgs.append(cv2.convertScaleAbs(img, alpha=1.2, beta=0))
    elif modality == "CT":
        imgs.append(cv2.resize(img, (img.shape[1]-5, img.shape[0]-5)))
        imgs.append(cv2.rotate(img, cv2.ROTATE_180))
    return imgs

for modality in modalities:
    hr_patches = []
    lr_patches = []
    folder = os.path.join(dataset_root, modality)
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        img_list = augment_image(img, modality)
        
        for im in img_list:
            # Create LR image
            lr = cv2.resize(im, (128, 128))
            lr = cv2.resize(lr, (256, 256))
            
            # Patch extraction with shape check
            for i in range(0, 256 - patch_size + 1, patch_size):
                for j in range(0, 256 - patch_size + 1, patch_size):
                    hr_patch = im[i:i+patch_size, j:j+patch_size]
                    lr_patch = lr[i:i+patch_size, j:j+patch_size]

                    if hr_patch.shape == (patch_size, patch_size) and lr_patch.shape == (patch_size, patch_size):
                        hr_patches.append(hr_patch)
                        lr_patches.append(lr_patch)
    
    hr_patches = np.array(hr_patches, dtype=np.float32)/255.0
    lr_patches = np.array(lr_patches, dtype=np.float32)/255.0
    
    save_path = f"train_{modality.lower()}.h5"
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('hr', data=hr_patches)
        hf.create_dataset('lr', data=lr_patches)
    
    print(f"✅ {modality} dataset created — {save_path}")
