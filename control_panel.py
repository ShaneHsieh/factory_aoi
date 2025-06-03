import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QHBoxLayout, QSpinBox
)

class FolderComboBox(QComboBox):
    def showPopup(self):
        if hasattr(self, 'update_callback') and callable(self.update_callback):
            self.update_callback()
        super().showPopup()

class ControlPanel(QWidget):
    def __init__(self, camera_app):
        super().__init__()
        self.setWindowTitle("控制面板")
        self.camera_app = camera_app
        # 在這裡建立所有控制元件
        self.folder_combo = FolderComboBox()
        self.folder_combo.update_callback = self.update_folder_list
        self.folder_combo.addItem("")
        self.set_sample_btn = QPushButton("設定檢測樣本")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(75)
        self.threshold_spin.setPrefix("threshold: ")
        self.erode_spin = QSpinBox()
        self.erode_spin.setRange(0, 20)
        self.erode_spin.setValue(2)
        self.erode_spin.setPrefix("n_erode: ")
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 20)
        self.dilate_spin.setValue(2)
        self.dilate_spin.setPrefix("n_dilate: ")
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 100)
        self.min_samples_spin.setValue(25)
        self.min_samples_spin.setPrefix("min_samples: ")
        self.capture_btn = QPushButton("📸 拍照")

        # 註冊事件
        self.capture_btn.clicked.connect(self.camera_app.snapshot_image)
        self.set_sample_btn.clicked.connect(self.camera_app.set_sample)

        layout = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("選擇資料夾："))
        hbox.addWidget(self.folder_combo)
        hbox.addWidget(self.set_sample_btn)
        layout.addLayout(hbox)
        param_hbox = QHBoxLayout()
        param_hbox.addWidget(self.threshold_spin)
        param_hbox.addWidget(self.erode_spin)
        param_hbox.addWidget(self.dilate_spin)
        param_hbox.addWidget(self.min_samples_spin)
        layout.addLayout(param_hbox)
        layout.addWidget(self.capture_btn)
        self.setLayout(layout)
        self.update_folder_list()

    def update_folder_list(self):
        current = self.folder_combo.currentText()
        base_path = os.path.dirname(os.path.abspath(__file__))
        folders = [
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
            and name not in ['.ipynb_checkpoints', 'templates','.git','__pycache__']
        ]
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        self.folder_combo.addItem("")  # 保留一個空白
        for folder in folders:
            self.folder_combo.addItem(folder)
        # 恢復上次選擇
        idx = self.folder_combo.findText(current)
        if idx >= 0:
            self.folder_combo.setCurrentIndex(idx)
        else:
            self.folder_combo.setCurrentIndex(0)
        self.folder_combo.blockSignals(False)
