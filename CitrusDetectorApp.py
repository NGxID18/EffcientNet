import sys
import os
import torch
import numpy as np
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights
from PIL import Image
from torch.utils.checkpoint import checkpoint

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

# --- 1. KONFIGURASI GLOBAL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MODEL_PATH = 'citrus_efficientnetv2l_final.pth' # Pastikan file model ada di folder yang sama

CLASS_NAMES = [
    'Citrus_Canker_Diseases_Leaf_Orange',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange',
    'Healthy_Leaf_Orange',
    'Multiple_Diseases_Leaf_Orange',
    'Young_Healthy_Leaf_Orange'
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_MAPPING = {
    'Citrus_Canker_Diseases_Leaf_Orange': 'Penyakit Kanker Jeruk (Canker)',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange': 'Kekurangan Nutrisi (Menguning)',
    'Healthy_Leaf_Orange': 'Daun Sehat',
    'Multiple_Diseases_Leaf_Orange': 'Berbagai Penyakit (Multiple)',
    'Young_Healthy_Leaf_Orange': 'Daun Muda Sehat',
}

MIN_CONFIDENCE_THRESHOLD = 40.0 
MIN_LEAF_AREA_RATIO = 0.05 

# --- 2. KELAS BACKEND MODEL ---

class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return checkpoint(self.module, x, use_reentrant=False)

class CitrusPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transform()
        print(f"Model siap di {self.model.device.type.upper()}.")

    def _get_transform(self):
        weights = EfficientNet_V2_L_Weights.DEFAULT
        return weights.transforms(antialias=True) 

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH}.")

        model = models.efficientnet_v2_l(weights=None)
        
        # Bungkus ulang layer dengan CheckpointWrapper agar struktur sama dengan saat training
        for i, block_sequence in enumerate(model.features):
            if isinstance(block_sequence, nn.Sequential):
                model.features[i] = CheckpointWrapper(block_sequence)

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        try:
            # Load ke CPU dulu agar aman
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise e

        model.to(DEVICE)
        model.eval()
        model.device = DEVICE 
        return model

    def segment_leaf(self, image_path):
        """
        Segmentasi menggunakan Pillow & NumPy (Kompatibel Python 3.14).
        """
        try:
            pil_img = Image.open(image_path).convert("RGB")
            pil_img_small = pil_img.resize((224, 224))
            
            # Convert ke HSV (Pillow Range: 0-255 untuk H, S, V)
            hsv_img = pil_img_small.convert("HSV")
            h, s, v = hsv_img.split()
            
            np_h = np.array(h)
            np_s = np.array(s)
            np_v = np.array(v)

            # Threshold Warna Daun (Hijau-Kuning)
            # Disesuaikan untuk range Pillow (0-255)
            lower_h, upper_h = 20, 130
            lower_s, upper_s = 40, 255
            lower_v, upper_v = 40, 255

            mask = (np_h >= lower_h) & (np_h <= upper_h) & \
                   (np_s >= lower_s) & (np_s <= upper_s) & \
                   (np_v >= lower_v) & (np_v <= upper_v)

            leaf_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            ratio = leaf_pixels / total_pixels

            # Buat visualisasi hasil segmentasi
            img_array = np.array(pil_img_small)
            segmented_array = np.zeros_like(img_array)
            segmented_array[mask] = img_array[mask]

            is_leaf_detected = ratio >= MIN_LEAF_AREA_RATIO
            
            return segmented_array, is_leaf_detected, ratio

        except Exception as e:
            print(f"Segmentasi Error: {e}")
            return None, False, 0.0

    def predict(self, image_path: str):
        # 1. Segmentasi
        segmented_img_np, is_valid_leaf, leaf_ratio = self.segment_leaf(image_path)
        
        if not is_valid_leaf:
            return "Non-Citrus", 0.0, {}, segmented_img_np

        # 2. Prediksi AI
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return "File Error", 0.0, None, None

        input_tensor = self.transform(image).unsqueeze(0).to(self.model.device)

        # Cek support tipe data untuk Autocast
        dtype_infer = torch.float32
        if self.model.device.type == 'cuda':
            if torch.cuda.is_bf16_supported():
                dtype_infer = torch.bfloat16
            else:
                dtype_infer = torch.float16

        with torch.no_grad():
            with torch.autocast(device_type=self.model.device.type, dtype=dtype_infer, enabled=(self.model.device.type == 'cuda')):
                output = self.model(input_tensor)
        
        # --- FIX UTAMA: .float() ditambahkan di sini ---
        # Mengubah BFloat16/FP16 ke Float32 agar NumPy tidak crash
        probabilities = torch.softmax(output, dim=1).float().cpu().numpy()[0]
        
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index] * 100
        
        predicted_class = CLASS_NAMES[predicted_index]
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
             predicted_class = "Unknown/Uncertain"

        class_details = {name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probabilities)}
        
        return predicted_class, confidence, class_details, segmented_img_np

# --- 3. GUI UTAMA ---

class CitrusClassifierApp(QWidget):
    def __init__(self, predictor_instance):
        super().__init__()
        self.predictor = predictor_instance
        self.setWindowTitle("🍊 Citrus Disease Classifier (Python 3.14 Compatible)")
        self.setGeometry(100, 100, 1000, 600)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Panel Kiri (Kontrol)
        left_panel = QVBoxLayout()
        self.select_button = QPushButton("1. Pilih Gambar")
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setStyleSheet("padding: 10px; font-weight: bold;")
        left_panel.addWidget(self.select_button)
        
        self.path_input = QLineEdit(readOnly=True)
        left_panel.addWidget(QLabel("Path:"))
        left_panel.addWidget(self.path_input)

        self.predict_button = QPushButton("2. Analisis")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)
        self.predict_button.setStyleSheet("padding: 10px; background-color: #2ecc71; color: white; font-weight: bold;")
        left_panel.addWidget(self.predict_button)
        
        self.status_label = QLabel(f"GPU: {'ON' if torch.cuda.is_available() else 'OFF'}")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        left_panel.addWidget(self.status_label)
        
        left_panel.addSpacing(20)
        self.result_label = QLabel("Diagnosis: -")
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.result_label.setStyleSheet("border: 1px solid #ddd; padding: 10px; background: #f9f9f9;")
        self.result_label.setWordWrap(True)
        left_panel.addWidget(self.result_label)

        self.confidence_label = QLabel("Keyakinan: -")
        left_panel.addWidget(self.confidence_label)
        
        left_panel.addSpacing(10)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        left_panel.addWidget(self.details_label)
        
        left_panel.addStretch(1)
        main_layout.addLayout(left_panel, stretch=1)

        # Panel Kanan (Visualisasi)
        right_panel = QVBoxLayout()
        
        self.lbl_orig = QLabel("Gambar Asli")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_orig = QLabel()
        self.view_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_orig.setFixedSize(300, 300)
        self.view_orig.setStyleSheet("border: 2px dashed #aaa;")
        
        self.lbl_seg = QLabel("Area Daun Terdeteksi")
        self.lbl_seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_seg = QLabel()
        self.view_seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_seg.setFixedSize(300, 300)
        self.view_seg.setStyleSheet("border: 2px solid #333; background: black;")
        
        right_panel.addWidget(self.lbl_orig)
        right_panel.addWidget(self.view_orig)
        right_panel.addWidget(self.lbl_seg)
        right_panel.addWidget(self.view_seg)
        
        main_layout.addLayout(right_panel, stretch=2)
        self.setLayout(main_layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.path_input.setText(file_name)
            self.display_image(file_name, self.view_orig)
            self.view_seg.clear()
            self.view_seg.setText("...")
            self.result_label.setText("Diagnosis: Siap")
            self.result_label.setStyleSheet("")
            self.predict_button.setEnabled(True)

    def display_image(self, source, label_widget):
        if isinstance(source, str):
            pixmap = QPixmap(source)
        elif isinstance(source, np.ndarray):
            h, w, ch = source.shape
            bytes_per_line = 3 * w
            q_img = QImage(source.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        else:
            return

        if not pixmap.isNull():
            scaled = pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label_widget.setPixmap(scaled)

    def run_prediction(self):
        path = self.path_input.text()
        if not path: return

        self.result_label.setText("Menganalisis...")
        QApplication.processEvents()

        res, conf, details, seg_img = self.predictor.predict(path)

        if seg_img is not None:
            self.display_image(seg_img, self.view_seg)

        if res == "Non-Citrus":
            self.result_label.setText("BUKAN DAUN JERUK")
            self.result_label.setStyleSheet("background-color: #c0392b; color: white; padding: 5px;")
            self.confidence_label.setText("Area daun hijau terlalu sedikit.")
            self.details_label.setText("")
        elif res == "Unknown/Uncertain":
            self.result_label.setText("TIDAK YAKIN")
            self.result_label.setStyleSheet("background-color: #f1c40f; padding: 5px;")
            self.confidence_label.setText(f"Skor: {conf:.1f}%")
        else:
            display_name = CLASS_MAPPING.get(res, res)
            self.result_label.setText(display_name)
            self.result_label.setStyleSheet("background-color: #27ae60; color: white; padding: 5px;")
            self.confidence_label.setText(f"Skor: {conf:.1f}%")
            
            if details:
                txt = "<b>Probabilitas:</b><br>" + "<br>".join([f"{k}: {v}" for k,v in sorted(details.items(), key=lambda x: float(x[1].strip('%')), reverse=True)])
                self.details_label.setText(txt)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        predictor = CitrusPredictor()
        window = CitrusClassifierApp(predictor)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error Init: {e}")