import sys
import os
import torch
import numpy as np
import cv2
import platform
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'model_final.pth')

CLASS_NAMES = [
    'Citrus_Canker_Diseases_Leaf_Orange',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange',
    'Healthy_Leaf_Orange',
    'Multiple_Diseases_Leaf_Orange',
    'Young_Healthy_Leaf_Orange'
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_MAPPING = {
    'Citrus_Canker_Diseases_Leaf_Orange': 'Citrus Canker',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange': 'Nutrient Deficiency (Yellowing)',
    'Healthy_Leaf_Orange': 'Healthy Leaf',
    'Multiple_Diseases_Leaf_Orange': 'Multiple Diseases',
    'Young_Healthy_Leaf_Orange': 'Young Healthy Leaf',
}

MIN_CONFIDENCE_THRESHOLD = 40.0

class CitrusPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _get_transform(self):
        weights = EfficientNet_V2_S_Weights.DEFAULT
        return weights.transforms(antialias=True)

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        model = models.efficientnet_v2_s(weights=None)
        
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise e

        model.to(DEVICE)
        model.eval()
        return model

    def segment_leaf_grabcut(self, image_path):
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None: return None

            height, width = img_bgr.shape[:2]
            scale = 400 / width
            new_w, new_h = 400, int(height * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))

            mask = np.zeros(img_resized.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            margin_w = int(new_w * 0.1)
            margin_h = int(new_h * 0.1)
            rect = (margin_w, margin_h, new_w - 2*margin_w, new_h - 2*margin_h)

            cv2.grabCut(img_resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask_clean = np.zeros_like(mask2)
                cv2.drawContours(mask_clean, [largest_contour], -1, 1, thickness=cv2.FILLED)
                mask2 = mask_clean
            
            img_resized = img_resized * mask2[:, :, np.newaxis]

            y_indices, x_indices = np.where(mask2 > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                img_final = img_resized[y_min:y_max, x_min:x_max]
            else:
                img_final = img_resized

            result_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            return result_rgb

        except Exception:
            return None

    def predict(self, image_path: str):
        segmented_img_np = self.segment_leaf_grabcut(image_path)
        
        if segmented_img_np is not None:
            final_input_image = Image.fromarray(segmented_img_np)
        else:
            try:
                final_input_image = Image.open(image_path).convert("RGB")
                segmented_img_np = np.array(final_input_image)
            except:
                 return "File Error", 0.0, None, None

        input_tensor = self.transform(final_input_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                output = self.model(input_tensor)
        
        probabilities = torch.softmax(output, dim=1).float().cpu().numpy()[0]
        
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index] * 100
        
        raw_class = CLASS_NAMES[predicted_index]
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
             predicted_class = "Unknown/Uncertain"
        else:
             predicted_class = raw_class

        class_details = {
            CLASS_MAPPING.get(name, name): f"{prob*100:.2f}%" 
            for name, prob in zip(CLASS_NAMES, probabilities)
        }
        
        return predicted_class, confidence, class_details, segmented_img_np

class CitrusClassifierApp(QWidget):
    def __init__(self, predictor_instance):
        super().__init__()
        self.predictor = predictor_instance
        self.setWindowTitle("🍊 Citrus Disease Detection (Segmentation Input Mode)")
        self.setGeometry(100, 100, 1100, 650)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Segoe UI, Arial;
            }
            QPushButton {
                border-radius: 5px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #3b3b3b;
                border: 1px solid #555;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_panel = QVBoxLayout()
        
        title = QLabel("Citrus Disease Detection")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        left_panel.addWidget(title)

        self.select_button = QPushButton("📂 Select Image")
        self.select_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        left_panel.addWidget(self.select_button)
        
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("No file selected...")
        self.path_input.setReadOnly(True)
        left_panel.addWidget(self.path_input)

        self.predict_button = QPushButton("🔍 Analyze Segmented Image")
        self.predict_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; margin-top: 10px;
            }
            QPushButton:hover { background-color: #219150; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        left_panel.addWidget(self.predict_button)
        
        self.status_label = QLabel(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-top: 5px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        left_panel.addWidget(self.status_label)
        
        left_panel.addSpacing(20)

        self.result_box = QWidget()
        self.result_box.setObjectName("resultBox")
        self.result_box.setStyleSheet("#resultBox { background-color: #3b3b3b; border-radius: 8px; border: 1px solid #555; }")
        res_layout = QVBoxLayout(self.result_box)
        
        self.result_label = QLabel("Waiting for Input...")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.result_label.setStyleSheet("color: #888; border: none;")
        self.result_label.setWordWrap(True)
        
        self.confidence_label = QLabel("-")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 12px; border: none;")

        res_layout.addWidget(self.result_label)
        res_layout.addWidget(self.confidence_label)
        left_panel.addWidget(self.result_box)
        
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("font-size: 12px; color: #dddddd; margin-top: 10px;")
        left_panel.addWidget(self.details_label)
        
        left_panel.addStretch(1)
        main_layout.addLayout(left_panel, stretch=35)

        right_panel = QVBoxLayout()
        
        self.lbl_orig = QLabel("📸 Original Image")
        self.lbl_orig.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.view_orig = QLabel()
        self.view_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_orig.setFixedSize(350, 350)
        self.view_orig.setStyleSheet("background: #1e1e1e; border: 2px dashed #555; border-radius: 8px;")
        
        self.lbl_seg = QLabel("🧠 Actual Input to AI (Segmented)")
        self.lbl_seg.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.view_seg = QLabel()
        self.view_seg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_seg.setFixedSize(350, 350)
        self.view_seg.setStyleSheet("background: #000; border: 2px solid #27ae60; border-radius: 8px;")
        
        right_panel.addWidget(self.lbl_orig)
        right_panel.addWidget(self.view_orig)
        right_panel.addSpacing(10)
        right_panel.addWidget(self.lbl_seg)
        right_panel.addWidget(self.view_seg)
        
        main_layout.addLayout(right_panel, stretch=65)
        self.setLayout(main_layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.path_input.setText(file_name)
            self.display_image(file_name, self.view_orig)
            
            self.view_seg.clear()
            self.view_seg.setText("...")
            self.result_label.setText("Ready to Analyze")
            self.result_label.setStyleSheet("color: #3498db; border: none; font-size: 16px; font-weight: bold;")
            self.confidence_label.setText("-")
            self.details_label.setText("")
            
            self.predict_button.setEnabled(True)

    def display_image(self, source, label_widget):
        pixmap = QPixmap()
        
        if isinstance(source, str):
            pixmap.load(source)
        elif isinstance(source, np.ndarray):
            source = np.ascontiguousarray(source).astype(np.uint8)
            h, w, ch = source.shape
            bytes_per_line = ch * w
            q_img = QImage(source.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                label_widget.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            label_widget.setPixmap(scaled)

    def run_prediction(self):
        path = self.path_input.text()
        if not path: return

        self.result_label.setText("Processing...")
        self.result_label.setStyleSheet("color: #f39c12; border: none; font-size: 16px; font-weight: bold;")
        QApplication.processEvents()

        raw_class, conf, details, seg_img = self.predictor.predict(path)

        if seg_img is not None:
            self.display_image(seg_img, self.view_seg)

        if raw_class == "Non-Citrus":
            self.result_label.setText("NOT A CITRUS LEAF")
            self.result_label.setStyleSheet("color: #e74c3c; border: none; font-weight: bold; font-size: 16px;")
            self.confidence_label.setText("Leaf object not detected")
            self.details_label.setText("")
            
        elif raw_class == "Unknown/Uncertain":
            self.result_label.setText("UNCERTAIN RESULT")
            self.result_label.setStyleSheet("color: #f39c12; border: none; font-size: 16px; font-weight: bold;")
            self.confidence_label.setText(f"Low Confidence: {conf:.1f}%")
            
        else:
            display_name = CLASS_MAPPING.get(raw_class, raw_class)
            self.result_label.setText(display_name)
            self.result_label.setStyleSheet("color: #2ecc71; border: none; font-weight: bold; font-size: 16px;")
            self.confidence_label.setText(f"Confidence Score: {conf:.2f}%")
            
            if details:
                sorted_details = sorted(details.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
                detail_text = "<b>Full Probability Breakdown:</b><br>"
                for k, v in sorted_details:
                    if k == display_name:
                        detail_text += f"<span style='color: #2ecc71;'>• <b>{k}: {v}</b></span><br>"
                    else:
                        detail_text += f"• {k}: {v}<br>"
                self.details_label.setText(detail_text)

if __name__ == '__main__':
    print(f"{'='*40}")
    print(f"APP RUNNING ON:")
    print(f"OS      : {platform.system()} {platform.release()}")
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"{'='*40}\n")

    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    try:
        predictor = CitrusPredictor()
        window = CitrusClassifierApp(predictor)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Critical Error: {e}")