import sys
import os
import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QMessageBox, QStackedWidget, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QPainter, QPainterPath


class CameraWidget(QWidget):
    image_captured = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Camera display - mobile optimized
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(250, 210)  # Mobile friendly size
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 3px solid #E8E8F0;
                border-radius: 20px;
                background-color: #F8F8FC;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("📱 Camera View")

        # Control buttons - larger for touch
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        self.start_camera_btn = QPushButton("🎥 Start Camera")
        self.start_camera_btn.setFixedSize(260, 50)
        self.start_camera_btn.setStyleSheet(self.get_mobile_button_style("#667EEA"))
        self.start_camera_btn.clicked.connect(self.start_camera)

        self.capture_btn = QPushButton("📸 Capture Photo")
        self.capture_btn.setFixedSize(260, 50)
        self.capture_btn.setStyleSheet(self.get_mobile_button_style("#4CAF50"))
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)

        self.stop_camera_btn = QPushButton("⏹️ Stop Camera")
        self.stop_camera_btn.setFixedSize(260, 50)
        self.stop_camera_btn.setStyleSheet(self.get_mobile_button_style("#FF5722"))
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)

        button_layout.addWidget(self.start_camera_btn, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.capture_btn, alignment=Qt.AlignCenter)
        button_layout.addWidget(self.stop_camera_btn, alignment=Qt.AlignCenter)

        layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_mobile_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {color}DD;
            }}
            QPushButton:pressed {{
                background-color: {color}AA;
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
                color: #888888;
            }}
        """

    def start_camera(self):
        try:
            camera_indices = [0, 1, 2]
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

            self.cap = None

            for backend in backends:
                for index in camera_indices:
                    try:
                        test_cap = cv2.VideoCapture(index, backend)
                        if test_cap is not None and test_cap.isOpened():
                            # Test if we can actually read a frame
                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                self.cap = test_cap
                                break
                        if test_cap is not None:
                            test_cap.release()
                    except Exception as e:
                        print(f"Failed to open camera {index} with backend {backend}: {e}")
                        continue
                if self.cap is not None:
                    break

            if self.cap is None or not self.cap.isOpened():
                QMessageBox.warning(self, "Camera Error",
                                    "Could not access camera. Please check:\n"
                                    "• Camera permissions\n"
                                    "• Camera is not being used by another app\n"
                                    "• Camera drivers are installed")
                return

            # Set camera properties for better compatibility
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Test frame capture before starting timer
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.cap.release()
                QMessageBox.warning(self, "Camera Error", "Camera opened but cannot capture frames")
                return

            # Start the timer and update UI
            self.timer.start(50)  # Reduced frequency to 20 FPS for stability
            self.start_camera_btn.setEnabled(False)
            self.capture_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape

                    # Create QImage properly
                    bytes_per_line = ch * w
                    from PyQt5.QtGui import QImage
                    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                    # Convert to QPixmap and scale
                    qt_image = QPixmap.fromImage(q_image).scaled(
                        280, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    self.camera_label.setPixmap(qt_image)
                else:
                    # If frame reading fails, stop the camera
                    print("Failed to read frame from camera")
                    self.stop_camera()
                    QMessageBox.warning(self, "Camera Error", "Lost connection to camera")
            except Exception as e:
                print(f"Error updating frame: {e}")
                self.stop_camera()
                QMessageBox.warning(self, "Camera Error", f"Camera error: {str(e)}")

    def capture_image(self):
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.image_captured.emit(frame)
                    self.stop_camera()
                else:
                    QMessageBox.warning(self, "Capture Error", "Failed to capture image")
            except Exception as e:
                QMessageBox.critical(self, "Capture Error", f"Error capturing image: {str(e)}")
                self.stop_camera()

    def stop_camera(self):
        try:
            if self.timer.isActive():
                self.timer.stop()

            if self.cap is not None:
                self.cap.release()
                self.cap = None

        except Exception as e:
            print(f"Error stopping camera: {e}")
        finally:
            # Always reset UI state
            self.start_camera_btn.setEnabled(True)
            self.capture_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(False)
            self.camera_label.clear()
            self.camera_label.setText("📱 Camera View")


class ParkinsonDetector:
    def __init__(self):
        self.spiral_model = None
        self.wave_model = None
        self.le = LabelEncoder()
        self.load_models()

    def quantify_image(self, image):
        features = feature.hog(image, orientations=9,
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
        return features

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return image

    def load_models(self):
        self.spiral_model = RandomForestClassifier(random_state=1)
        self.wave_model = RandomForestClassifier(random_state=1)

        # Demo data
        dummy_data = np.random.rand(10, 7056)
        dummy_labels = [0, 1] * 5

        self.spiral_model.fit(dummy_data, dummy_labels)
        self.wave_model.fit(dummy_data, dummy_labels)

    def predict(self, image, test_type="spiral"):
        processed_image = self.preprocess_image(image)
        features = self.quantify_image(processed_image)

        model = self.spiral_model if test_type == "spiral" else self.wave_model
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]

        return prediction, max(probability)


class MobileCard(QFrame):
    def __init__(self, title, icon, color, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 120)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {color}F0, stop:1 {color}D0);
                border-radius: 25px;
                border: none;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(20, 15, 20, 15)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        icon_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2D2D2D;
                background: transparent;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = ParkinsonDetector()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Parkinson Detection")
        self.setFixedSize(360, 640)  # Mobile screen size (similar to iPhone/Android)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667EEA, stop:0.5 #764BA2, stop:1 #F093FB);
            }
        """)

        # Central widget with stacked layout
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        # Create pages
        self.welcome_page = self.create_welcome_page()
        self.detection_page = self.create_detection_page()
        self.result_page = self.create_result_page()

        # Add pages to stack
        self.central_widget.addWidget(self.welcome_page)
        self.central_widget.addWidget(self.detection_page)
        self.central_widget.addWidget(self.result_page)

    def create_welcome_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 40, 30, 40)

        # Status bar space
        layout.addStretch(1)

        # App Icon/Logo
        icon_label = QLabel("🧠")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 64px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 40px;
                padding: 20px;
                margin: 20px;
            }
        """)

        # Title
        title = QLabel("PARKINSON\nDETECTION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: white;
                margin: 10px 0px;
                background: transparent;
                line-height: 1.2;
            }
        """)

        # Subtitle
        subtitle = QLabel("AI-Powered Early Detection\nQuick • Accurate • Reliable")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: rgba(255, 255, 255, 0.9);
                margin-bottom: 40px;
                background: transparent;
                line-height: 1.4;
            }
        """)

        # Start button - mobile style
        start_btn = QPushButton("START DETECTION")
        start_btn.setFixedSize(280, 55)
        start_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.95);
                color: #4A4A6A;
                border: none;
                border-radius: 27px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 1.0);
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background: rgba(240, 240, 240, 0.95);
            }
        """)
        start_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))

        # Info text
        info_text = QLabel("Draw spirals or waves to test\nfor Parkinson's symptoms")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: rgba(255, 255, 255, 0.7);
                background: transparent;
                margin-top: 20px;
            }
        """)

        # Layout
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        layout.addWidget(info_text)
        layout.addStretch(2)

        page.setLayout(layout)
        return page

    def create_detection_page(self):
        page = QWidget()
        page.setStyleSheet("background: white;")

        # Create scroll area for mobile
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(25)
        layout.setContentsMargins(20, 30, 20, 30)

        # Header with back button
        header_layout = QHBoxLayout()

        back_btn = QPushButton("←")
        back_btn.setFixedSize(40, 40)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                color: #666666;
                border: none;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)
        back_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))

        header_title = QLabel("Choose Method")
        header_title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2D2D2D;
                background: transparent;
            }
        """)

        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel(""))  # Spacer

        # Instruction text
        instruction = QLabel("Select how you want to capture the drawing:")
        instruction.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                background: transparent;
                margin: 10px 0px;
            }
        """)
        instruction.setAlignment(Qt.AlignCenter)

        # Camera option card
        camera_card = MobileCard("Take Photo", "📷", "#4FACFE")
        camera_card.mousePressEvent = lambda e: self.open_camera()
        camera_card.setStyleSheet(camera_card.styleSheet() + """
            QFrame:hover {
                transform: scale(1.02);
            }
        """)

        # Gallery option card
        gallery_card = MobileCard("From Gallery", "🖼️", "#A8E6CF")
        gallery_card.mousePressEvent = lambda e: self.import_from_gallery()
        gallery_card.setStyleSheet(gallery_card.styleSheet() + """
            QFrame:hover {
                transform: scale(1.02);
            }
        """)

        # Tips section
        tips_frame = QFrame()
        tips_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0px;
            }
        """)

        tips_layout = QVBoxLayout()

        tips_title = QLabel("💡 Tips for best results:")
        tips_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #495057;
                background: transparent;
                margin-bottom: 8px;
            }
        """)

        tips_text = QLabel(
            "• Use a dark pen on white paper\n• Draw clearly and steadily\n• Ensure good lighting\n• Keep the image sharp")
        tips_text.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #6C757D;
                background: transparent;
                line-height: 1.4;
            }
        """)

        tips_layout.addWidget(tips_title)
        tips_layout.addWidget(tips_text)
        tips_frame.setLayout(tips_layout)

        # Layout
        layout.addLayout(header_layout)
        layout.addWidget(instruction)
        layout.addWidget(camera_card, alignment=Qt.AlignCenter)
        layout.addWidget(gallery_card, alignment=Qt.AlignCenter)
        layout.addWidget(tips_frame)
        layout.addStretch()

        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)

        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        page.setLayout(page_layout)

        return page

    def create_result_page(self):
        page = QWidget()
        page.setStyleSheet("background: white;")

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 30, 20, 30)

        # Header
        header_layout = QHBoxLayout()

        back_btn = QPushButton("←")
        back_btn.setFixedSize(40, 40)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0;
                color: #666666;
                border: none;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        back_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))

        header_title = QLabel("Results")
        header_title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2D2D2D;
                background: transparent;
            }
        """)

        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel(""))

        # Result card
        result_card = QFrame()
        result_card.setFixedHeight(180)
        result_card.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 20px;
                border: 2px solid #E8E8F0;
            }
        """)

        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(20, 20, 20, 20)

        # Result icon and text
        self.result_icon = QLabel("🔍")
        self.result_icon.setAlignment(Qt.AlignCenter)
        self.result_icon.setStyleSheet("font-size: 48px; background: transparent;")

        self.result_label = QLabel("Analyzing...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                background: transparent;
                margin: 10px 0px;
            }
        """)

        self.confidence_label = QLabel("Please wait...")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                background: transparent;
            }
        """)

        result_layout.addWidget(self.result_icon)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.confidence_label)
        result_card.setLayout(result_layout)

        # Image display
        self.result_image = QLabel("Processed image will appear here")
        self.result_image.setFixedSize(280, 200)
        self.result_image.setStyleSheet("""
            QLabel {
                border: 2px solid #E8E8F0;
                border-radius: 15px;
                background-color: #F8F9FA;
                color: #6C757D;
                font-size: 14px;
            }
        """)
        self.result_image.setAlignment(Qt.AlignCenter)

        # Action buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(12)

        new_test_btn = QPushButton("🔄 New Test")
        new_test_btn.setFixedSize(280, 50)
        new_test_btn.setStyleSheet("""
            QPushButton {
                background-color: #667EEA;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A67D8;
            }
        """)
        new_test_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))

        home_btn = QPushButton("🏠 Home")
        home_btn.setFixedSize(280, 50)
        home_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        home_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))

        button_layout.addWidget(new_test_btn, alignment=Qt.AlignCenter)
        button_layout.addWidget(home_btn, alignment=Qt.AlignCenter)

        # Layout
        layout.addLayout(header_layout)
        layout.addWidget(result_card)
        layout.addWidget(self.result_image, alignment=Qt.AlignCenter)
        layout.addLayout(button_layout)
        layout.addStretch()

        page.setLayout(layout)
        return page

    def open_camera(self):
        from PyQt5.QtWidgets import QDialog

        camera_dialog = QDialog()
        camera_dialog.setWindowTitle("Camera")
        camera_dialog.setFixedSize(320, 480)  # Mobile dialog size
        camera_dialog.setStyleSheet("background: white;")

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("📷 Take Photo")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2D2D2D;
                margin: 10px 0px 20px 0px;
                background: transparent;
            }
        """)

        camera_widget = CameraWidget()
        camera_widget.image_captured.connect(
            lambda img: self.process_captured_image(img, camera_dialog)
        )

        layout.addWidget(header)
        layout.addWidget(camera_widget)
        camera_dialog.setLayout(layout)
        camera_dialog.exec_()

    def import_from_gallery(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.process_image(image)
            else:
                QMessageBox.warning(self, "Error", "Could not load the selected image")

    def process_captured_image(self, image, dialog):
        dialog.close()
        self.process_image(image)

    def process_image(self, image):
        # Switch to result page first
        self.central_widget.setCurrentIndex(2)

        try:
            # Predict using the model
            prediction, confidence = self.detector.predict(image, "spiral")

            # Display result
            if prediction == 0:
                self.result_icon.setText("✅")
                self.result_label.setText("HEALTHY")
                self.result_label.setStyleSheet("""
                    QLabel {
                        font-size: 20px;
                        font-weight: bold;
                        color: #2E7D32;
                        background: transparent;
                        margin: 10px 0px;
                    }
                """)
                self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")
                result_card = self.result_page.findChild(QFrame)
                if result_card:
                    result_card.setStyleSheet("""
                        QFrame {
                            background: #E8F5E8;
                            border-radius: 20px;
                            border: 2px solid #4CAF50;
                        }
                    """)
            else:
                self.result_icon.setText("⚠️")
                self.result_label.setText("RISK DETECTED")
                self.result_label.setStyleSheet("""
                    QLabel {
                        font-size: 20px;
                        font-weight: bold;
                        color: #C62828;
                        background: transparent;
                        margin: 10px 0px;
                    }
                """)
                self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%\nConsult a healthcare professional")
                result_card = self.result_page.findChild(QFrame)
                if result_card:
                    result_card.setStyleSheet("""
                        QFrame {
                            background: #FFEBEE;
                            border-radius: 20px;
                            border: 2px solid #F44336;
                        }
                    """)

            # Display the processed image
            processed_img = self.detector.preprocess_image(image.copy())
            height, width = processed_img.shape
            bytes_per_line = width
            qt_image = QPixmap.fromImage(
                processed_img.data, width, height, bytes_per_line, QPixmap.Format_Grayscale8
            ).scaled(280, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.result_image.setPixmap(qt_image)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")


from PyQt5.QtWidgets import QDialog


def main():
    app = QApplication(sys.argv)

    # Set application style for mobile-like appearance
    app.setStyle('Fusion')

    # Set high DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()