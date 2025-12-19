# 🍊 Citrus Leaf Disease Classification & Detection System

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-red?logo=pytorch)
![GUI](https://img.shields.io/badge/GUI-PyQt6-green?logo=qt)

A high-performance Deep Learning application designed to classify and detect diseases in citrus leaves. This project utilizes the **EfficientNetV2-S** architecture with Transfer Learning techniques, achieving an impressive accuracy. It also features a user-friendly Desktop GUI for real-time inference on local images.

## ✨ Key Features

* **State-of-the-Art Model:** Built on `EfficientNetV2-S` architecture, offering an optimal balance between inference speed and classification accuracy.
* **Two-Stage Training:** Implements a robust training pipeline starting with Feature Extraction followed by Fine-Tuning strategies to maximize model performance.
* **Hardware Optimized:** Leverages Mixed Precision (`BFloat16`/`Float16`), Memory Format optimization (`Channels Last`), and RAM Caching to ensure fast and efficient training on NVIDIA GPUs.
* **Smart Preprocessing:** Includes HSV-based Leaf Segmentation algorithms to filter out background noise before prediction, improving real-world reliability.
* **User-Friendly GUI:** A clean and intuitive desktop interface built with **PyQt6**, allowing easy image loading and instant disease analysis.

## 📂 Project Structure

```text
├── 📂 Dataset/                # Root directory for the image dataset
│   ├── Citrus_Canker/
│   ├── Black_Spot/
│   └── ...
├── 📂 Model/                  # Directory to store trained model weights
│   ├── checkpoint_fe.pth
│   └── model_final.pth
├── CitrusDetectorApp.py       # Main Python script for the Desktop GUI Application
├── train.ipynb                # Jupyter Notebook containing the full Training Pipeline
├── requirements.txt
└── README.md
```

## 🧠 Model & Classes

The project employs the **EfficientNetV2-S** (Small) architecture, trained using *Transfer Learning*. This specific model variant was selected for its superior trade-off between computational efficiency and accuracy, making it suitable for desktop deployment even on limited hardware.

The model is trained to classify citrus leaves into **5 specific categories**:

1.  **Citrus Canker**
2.  **Nutrient Deficiency**
3.  **Multiple Diseases**
4.  **Healthy Leaf**
5.  **Young Healthy Leaf**

> **Model Performance**: The model achieved an accuracy of **~98.70%** on the validation dataset after the Fine-Tuning stage.

---

### 🖥️ Development Environment
The model was trained and tested on the following system specifications:

#### ⚙️ Hardware
| Component | Specification |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7500F |
| **GPU** | NVIDIA GeForce RTX 5060 |
| **RAM** | 2 x 16GB 6000MHz CL30 |

#### 💻 Software & Environment
| Component | Version / Detail |
| :--- | :--- |
| **OS** | Windows 11 Pro 25H2 (26200.7462) |
| **AMD** | 7.11.26.2142 (Chipset Driver) |
| **NVIDIA** | 591.44 (Studio Driver) |
| **CUDA** | 13.1 (NVCC), 9.17 (CUDNN) |
| **Python** | 3.13.11 (pip 25.3) |
| **PyTorch** | 2.9.1 (Stable CUDA 13.0) |