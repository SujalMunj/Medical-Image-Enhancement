# Medical-Image-Enhancement
# 🩻 Enhancing Medical Image Quality using Deep Learning




## 📌 Project Overview

Medical imaging plays a crucial role in diagnosing diseases such as Tuberculosis, Pneumonia, and Lung Cancer.


However, X-ray images often suffer from low resolution, noise, and poor clarity, making diagnosis difficult and sometimes inaccurate.


* This project focuses on:

   Enhancing medical X-ray images using deep learning models (SRCNN & VDSR).


   Improving diagnostic accuracy by generating super-resolved images.


   Extending the system to perform disease detection in later phases.




## 🎯 Objectives

Develop a pipeline to improve the quality of low-resolution X-rays.


Train and compare SRCNN and VDSR models.


Measure performance using PSNR and SSIM.


Provide insights into how image enhancement can improve disease diagnosis.







## 🧠 Methodology

1. Data Collection
Dataset used: Kaggle Chest X-ray Pneumonia Dataset (~3000 images)



2. Preprocessing
Resizing, normalization, and augmentation (rotation, zoom, flip).
Dataset is split into:
80% training
10% validation
10% testing



3. Model Training

     SRCNN (Super-Resolution CNN)

     VDSR (Very Deep Super-Resolution)



5. Evaluation Metrics
Performance of the models was measured using two key metrics:



Peak Signal-to-Noise Ratio (PSNR) – assesses image clarity.
Structural Similarity Index (SSIM) – measures structural similarity between enhanced and original images.

➡️ Error maps were also generated to visualize differences between predicted and ground-truth images.





## 📊 Results 

* SRCNN: Achieved baseline improvement in resolution and reduced noise with low computation cost.


* VDSR: Outperformed SRCNN by producing sharper, high-quality images, though requiring more computational resources.



VDSR achieved higher PSNR and SSIM scores, proving more effective for medical image enhancement.






## 🔮 Future Scope

Extend the project to disease detection (Pneumonia, TB, Lung Cancer) using CNN/Transfer Learning models.


Explore GAN-based architectures for even better super-resolution results.


Deploy the system as a web application to assist doctors with real-time X-ray enhancement and diagnosis.






## 🛠️ Tech Stack

* Programming Language: Python


* Frameworks: TensorFlow, PyTorch


* Libraries: NumPy, OpenCV, Matplotlib, Scikit-learn


* Evaluation Tools: PSNR, SSIM

  


## 📌 Dataset



Chest X-ray Pneumonia Dataset (Kaggle) → Download here (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Total ~3000 images (Normal + Pneumonia).



Both original dataset and preprocessed dataset are stored locally (not uploaded to GitHub).
