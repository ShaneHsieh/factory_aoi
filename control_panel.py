import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QHBoxLayout, QSpinBox, QFileDialog
)
from PyQt5.QtGui import QFont

class FolderComboBox(QComboBox):
    def showPopup(self):
        if hasattr(self, 'update_callback') and callable(self.update_callback):
            self.update_callback()
        super().showPopup()

class ControlPanel(QWidget):
    def __init__(self, camera_app):
        super().__init__()
        self.resize(1200, 400)  # 設定視窗大小更大
        self.setWindowTitle("控制面板")
        self.camera_app = camera_app
        # 設定字體大小
        ui_font = QFont()
        ui_font.setPointSize(20)
        main_title_font = QFont()
        main_title_font.setPointSize(28)
        
        self.setFont(ui_font)
        # 在這裡建立所有控制元件
        label_select_folder = QLabel("選擇資料夾：", font=ui_font)
        label_select_folder.setFixedWidth(240)
        self.folder_combo = FolderComboBox()
        self.folder_combo.setFont(ui_font)
        self.folder_combo.update_callback = self.update_folder_list
        self.folder_combo.currentTextChanged.connect(self.camera_app.on_folder_changed) # 當選項改變時，通知 camera_app
        self.folder_combo.addItem("")
        self.set_sample_btn = QPushButton("設定檢測樣本")
        self.set_sample_btn.setFont(ui_font)
        self.set_sample_btn.setFixedWidth(260)
        #self.set_sample_btn.clicked.connect(self.camera_app.set_sample)
        hbox = QHBoxLayout()
        hbox.addWidget(label_select_folder)
        hbox.addWidget(self.folder_combo)
        hbox.addWidget(self.set_sample_btn)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setFont(ui_font)
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(75)
        self.threshold_spin.setPrefix("threshold: ")
        self.erode_spin = QSpinBox()
        self.erode_spin.setFont(ui_font)
        self.erode_spin.setRange(0, 20)
        self.erode_spin.setValue(2)
        self.erode_spin.setPrefix("n_erode: ")
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setFont(ui_font)
        self.dilate_spin.setRange(0, 20)
        self.dilate_spin.setValue(2)
        self.dilate_spin.setPrefix("n_dilate: ")
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setFont(ui_font)
        self.min_samples_spin.setRange(1, 100)
        self.min_samples_spin.setValue(50)
        self.min_samples_spin.setPrefix("min_samples: ")
        param_hbox = QHBoxLayout()
        param_hbox.addWidget(self.threshold_spin)
        param_hbox.addWidget(self.erode_spin)
        param_hbox.addWidget(self.dilate_spin)
        param_hbox.addWidget(self.min_samples_spin)

        control_hbox_wrapper = QVBoxLayout()
        control_title = QLabel("檢測")
        control_title.setFont(main_title_font)
        control_hbox_wrapper.addWidget(control_title)    
        control_hbox_wrapper.addLayout(hbox)
        control_hbox_wrapper.addLayout(param_hbox)
        
        self.open_folder_btn = QPushButton("選擇存檔資料夾")
        self.open_folder_btn.setFont(ui_font)
        self.open_folder_btn.clicked.connect(self.choose_folder)
        self.open_folder_btn.setFixedWidth(280)
        self.folder_save_path = os.path.abspath(os.getcwd()+"/test")
        self.folder_label = QLabel(self.folder_save_path)
        self.folder_label.setFont(ui_font)
        self.capture_btn = QPushButton("📸")
        self.capture_btn.setFont(ui_font)
        self.capture_btn.clicked.connect(self.camera_app.snapshot_positive_image)
        self.capture_btn.setFixedWidth(120)
        folder_hbox = QHBoxLayout()
        folder_hbox.addWidget(self.open_folder_btn)
        folder_hbox.addWidget(self.folder_label)
        folder_hbox.addWidget(self.capture_btn)

        photo_vbox = QVBoxLayout()
        photo_title = QLabel("拍攝正樣本")
        photo_title.setFont(main_title_font)
        photo_vbox.addWidget(photo_title)
        photo_vbox.addLayout(folder_hbox)

        layout = QVBoxLayout()
        layout.addLayout(control_hbox_wrapper)
        layout.addSpacing(30)  # 兩區塊間隔 30px
        layout.addLayout(photo_vbox)
        self.setLayout(layout)

        self.update_folder_list()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇存檔資料夾", self.folder_save_path)
        if folder:
            self.folder_save_path = folder
            self.folder_label.setText(self.folder_save_path)

    def update_folder_list(self):
        current = self.folder_combo.currentText()
        base_path = os.path.dirname(os.path.abspath(__file__))
        folders = [
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
            and name not in ['.ipynb_checkpoints', 'templates','.git','__pycache__','Shane' ,'nagetive_samples']
            and 'AOI_202' not in name 
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
