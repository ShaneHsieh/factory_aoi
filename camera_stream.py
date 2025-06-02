import sys
import os
import cv2
import time
import numpy as np

import time

from match_template import cv_aoi
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QFileDialog, QHBoxLayout, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal, QPoint

class FolderComboBox(QComboBox):
    def showPopup(self):
        if hasattr(self, 'update_callback') and callable(self.update_callback):
            self.update_callback()
        super().showPopup()

class AOILabel(QLabel):
    aoi_point_signal = pyqtSignal(QPoint)
    aoi_clear_signal = pyqtSignal()  # 新增右鍵清除訊號
    def mousePressEvent(self, event):
        if event.button() == 1:  # 左鍵
            self.aoi_point_signal.emit(event.pos())
        elif event.button() == 2:  # 右鍵
            self.aoi_clear_signal.emit()
        super().mousePressEvent(event)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Stream (Local UI)")
        self.image_label = AOILabel()
        self.capture_btn = QPushButton("📸 拍照")
        self.folder_combo = FolderComboBox()
        self.folder_combo.update_callback = self.update_folder_list
        self.folder_combo.addItem("")  # 預設空白
        self.set_sample_btn = QPushButton("設定檢測樣本")  # 新增按鈕
        self.update_folder_list()

        # 新增四個 SpinBox
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

        self.capture_btn.clicked.connect(self.snapshot_image)
        self.set_sample_btn.clicked.connect(self.set_sample)  # 綁定事件
        self.image_label.aoi_point_signal.connect(self.handle_aoi_point)
        self.image_label.aoi_clear_signal.connect(self.clear_aoi_rect)  # 綁定右鍵清除
        self.aoi_points = []  # 用來記錄 AOI 兩點
        self.aoi_rect = None  # AOI 區域

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("選擇資料夾："))
        hbox.addWidget(self.folder_combo)
        hbox.addWidget(self.set_sample_btn)  # 加入新按鈕
        layout.addLayout(hbox)
        # 新增參數設定區
        param_hbox = QHBoxLayout()
        param_hbox.addWidget(self.threshold_spin)
        param_hbox.addWidget(self.erode_spin)
        param_hbox.addWidget(self.dilate_spin)
        param_hbox.addWidget(self.min_samples_spin)
        layout.addLayout(param_hbox)
        layout.addWidget(self.capture_btn)
        self.setLayout(layout)

        self.video_width = 3840
        self.video_height = 2160
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, -6)
        self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.aoi_model = cv_aoi()
        self.goldens = []

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

    def handle_aoi_point(self, pos):
        if self.aoi_rect is not None:
            return
        self.aoi_points.append((pos.x(), pos.y()))
        if len(self.aoi_points) == 2:
            x1, y1 =(np.array(self.aoi_points[0]).astype(float) / np.array([1280.0 / self.video_width, 720.0 / self.video_height])).astype(int)
            x2, y2 =(np.array(self.aoi_points[1]).astype(float) / np.array([1280.0 / self.video_width, 720.0 / self.video_height])).astype(int)
            # 計算 AOI 區域的矩形
            self.aoi_rect = [min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)]  # [top, bottom, left, right]
            self.aoi_points = []
            print(f"AOI 設定為: {self.aoi_rect}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            display_frame = frame.copy()
            if self.aoi_rect:
                # 在 AOI 區域畫紅框
                t, b, l, r = self.aoi_rect
                cv2.rectangle(display_frame, (l, t), (r, b), (0, 0, 255), 4)
            #frame = cv2.imread('C:/Users/yuan/Desktop/factory/snapshot_20250529-161051.bmp')[150:2000,800:2800]

            if self.goldens:
                #frame= frame[150:2000,800:2800]
                if self.aoi_rect:
                    aoi = self.aoi_rect
                    frame = frame[aoi[0]:aoi[1], aoi[2]:aoi[3]]
                else:
                    aoi = None
                mask, mask_mean, mask_min, a = self.aoi_model.match_template(frame, self.goldens , aoi = aoi )
                
                # 讀取 UI 參數
                threshold = self.threshold_spin.value()
                n_erode = self.erode_spin.value()
                n_dilate = self.dilate_spin.value()
                min_samples = self.min_samples_spin.value()
                
                m = self.aoi_model.post_proc(mask_min, threshold, n_erode, n_dilate)
                res = (np.stack([np.maximum(m,a[:,:,0]), a[:,:,1]*(m==0), a[:,:,2]*(m==0)], axis=-1))

                circle_image , n = self.aoi_model.draw_circle(m, res, min_samples=min_samples)
                    
                print("draw circle", n , "clusters")
                
                #print("successfully matched template, circle_image shape:", circle_image.shape)
                #print("circle_image.shape " , circle_image.shape)
                #cv2.imwrite("test.bmp" , circle_image)
                

                #rgb_image = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                #rgb_image = cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB)
                rgb_image = circle_image.copy()
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage. Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                   1280, 720, aspectRatioMode=1
                ))
            else:
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage. Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    1280, 720, aspectRatioMode=1
                ))
            
    def snapshot_image(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder = self.folder_combo.currentText()
            filename = f"snapshot_{timestamp}.bmp"
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            if folder:
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, filename)
            cv2.imwrite(save_path, self.current_frame)

    def set_sample(self):
        self.goldens = []
        folder = self.folder_combo.currentText()
        if not folder:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "設定檢測樣本", "請先選擇資料夾！")
            return
        base_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "設定檢測樣本", "資料夾不存在！")
            return
        # 只保留 bmp 檔案
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.bmp')]
        if not files:
            msg = "該資料夾內沒有 bmp 檔案。"
        else:
            msg = "\n".join(files)
        
        for img_path in files:
            golden_img = cv2.imread(img_path)
            kp, des = self.aoi_model.get_keypoint(golden_img)
            self.goldens.append([golden_img, kp, des])

    def clear_aoi_rect(self):
        self.aoi_rect = None
        print("AOI 已清除")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    win.show()
    sys.exit(app.exec_())