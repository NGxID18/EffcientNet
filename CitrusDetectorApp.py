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
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

# --- 1. KONFIGURASI GLOBAL ---
# Menggunakan CUDA jika tersedia (karena ini adalah konfigurasi yang Anda konfirmasi berfungsi)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MODEL_PATH = 'EfficientNetV2-L_Citrus.pth' 

CLASS_NAMES = [
    'Citrus_Canker_Diseases_Leaf_Orange',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange',
    'Healthy_Leaf_Orange',
    'Multiple_Diseases_Leaf_Orange',
    'Young_Healthy_Leaf_Orange'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- 2. KELAS UTAMA MODEL/BACKEND ---

class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        # Memerlukan impor 'from torch.utils.checkpoint import checkpoint'
        from torch.utils.checkpoint import checkpoint 
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
        
        # Terapkan CheckpointWrapper
        for i, block_sequence in enumerate(model.features):
            if isinstance(block_sequence, nn.Sequential):
                model.features[i] = CheckpointWrapper(block_sequence)

        # Ubah layer klasifikasi
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        # Muat bobot dan pindahkan ke DEVICE (CUDA/CPU)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        # Tambahkan atribut 'device' ke model untuk pengecekan status
        model.device = DEVICE 
        return model

    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return "File Error", 0.0, None

        input_tensor = self.transform(image).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            # Menggunakan torch.autocast/AMP untuk performa GPU
            with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=(self.model.device.type == 'cuda')):
                output = self.model(input_tensor)
        
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index] * 100

        class_details = {name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probabilities)}
        
        return predicted_class, confidence, class_details

# --- 3. KELAS GUI UTAMA/FRONTEND ---

class CitrusClassifierApp(QWidget):
    def __init__(self, predictor_instance):
        super().__init__()
        self.predictor = predictor_instance # Menerima instance model yang sudah dimuat
        self.setWindowTitle("🍊 Citrus Disease Classifier (EfficientNetV2-L)")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        self.predict_button.setEnabled(True) # Langsung aktif karena model sudah dimuat

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Panel Kiri (Kontrol dan Hasil)
        left_panel = QVBoxLayout()
        self.select_button = QPushButton("1. Pilih Gambar")
        self.select_button.clicked.connect(self.select_image)
        left_panel.addWidget(self.select_button)
        
        self.path_input = QLineEdit(readOnly=True)
        left_panel.addWidget(QLabel("Path Gambar:"))
        left_panel.addWidget(self.path_input)

        self.predict_button = QPushButton("2. Prediksi")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False) # Diaktifkan setelah model berhasil dimuat
        left_panel.addWidget(self.predict_button)
        
        self.status_label = QLabel(f"Model Aktif: {self.predictor.model.device.type.upper()}")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_panel.addWidget(self.status_label)
        
        left_panel.addWidget(QLabel("--- Hasil ---"))
        self.result_label = QLabel("Hasil:")
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.result_label.setStyleSheet("color: darkgreen;")
        left_panel.addWidget(self.result_label)

        self.confidence_label = QLabel("Keyakinan:")
        left_panel.addWidget(self.confidence_label)
        
        self.details_label = QLabel("Detail Probabilitas:")
        self.details_label.setWordWrap(True)
        left_panel.addWidget(self.details_label)
        
        left_panel.addStretch(1)
        main_layout.addLayout(left_panel)

        # Panel Kanan (Tampilan Gambar)
        self.image_label = QLabel("Tampilan Gambar")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.image_label)
        right_panel.addStretch(1)
        
        main_layout.addLayout(right_panel)
        self.setLayout(main_layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 
            "Pilih Gambar Daun Jeruk", 
            "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.path_input.setText(file_name)
            self.display_image(file_name)
            self.result_label.setText("Hasil: Menunggu prediksi...")
            self.confidence_label.setText("Keyakinan:")
            self.details_label.setText("Detail Probabilitas:")
            self.status_label.setText(f"Gambar dimuat. Tekan 'Prediksi'. Model aktif: {self.predictor.model.device.type.upper()}")

    def display_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText("Gagal memuat gambar.")
            return
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def run_prediction(self):
        image_path = self.path_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Perhatian", "Pilih file gambar yang valid.")
            return

        self.status_label.setText("Memprediksi...")
        QApplication.processEvents()

        predicted_class, confidence, details = self.predictor.predict(image_path)

        if predicted_class == "File Error":
            self.result_label.setText(f"Hasil: ERROR FILE")
            self.confidence_label.setText(f"Pesan: Gagal memproses gambar.")
            self.status_label.setText("Prediksi gagal.")
            return
            
        self.result_label.setText(f"Hasil: **{predicted_class}**")
        self.confidence_label.setText(f"Keyakinan: {confidence:.2f}%")
        
        detail_text = "Detail Probabilitas:\n"
        for k in CLASS_NAMES:
            detail_text += f"- {k}: {details.get(k, 'N/A')}\n"
        self.details_label.setText(detail_text)
        
        self.status_label.setText("Prediksi selesai.")

# --- 4. EKSEKUSI UTAMA ---

if __name__ == '__main__':
    # 1. Inisialisasi QApplication (Wajib pertama untuk GUI)
    app = QApplication(sys.argv)
    
    # 2. Muat model terlebih dahulu (Backend)
    try:
        predictor = CitrusPredictor()
    except FileNotFoundError as e:
        QMessageBox.critical(None, "File Model Error", f"Model tidak ditemukan: {e}")
        sys.exit(1)
    except Exception as e:
        # Menangkap error DLL / CUDA/PyTorch
        QMessageBox.critical(None, "PyTorch Runtime Error", f"Gagal menginisialisasi PyTorch: {e}")
        sys.exit(1)

    # 3. Buat dan jalankan GUI (Frontend)
    ex = CitrusClassifierApp(predictor)
    ex.show()
    sys.exit(app.exec())