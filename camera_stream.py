import sys
import os
import cv2
import time
import numpy as np

from match_template import cv_aoi
from control_panel import ControlPanel

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal, QPoint, Qt

class AOILabel(QLabel):
    aoi_rect_changed = pyqtSignal(object)  # 新增 AOI 區域變更訊號
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignCenter) 
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 讓 label 填滿空間
        self.label_width = self.width()   # 新增：儲存 label 寬度
        self.label_height = self.height() # 新增：儲存 label 高度
        self.aoi_points = []  # 移進 AOILabel
        self.aoi_rect = None  # 移進 AOILabel
        self.video_width = 3840  # 預設值，CameraApp 會設置
        self.video_height = 2160

    def resizeEvent(self, event):
        self.label_width = self.width()   # 更新 label 寬度
        self.label_height = self.height() # 更新 label 高度
        super().resizeEvent(event)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtWidgets import QStyle
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._pixmap:
            widget_w, widget_h = self.width(), self.height()
            pixmap_w, pixmap_h = self._pixmap.width(), self._pixmap.height()
            scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
            new_w, new_h = int(pixmap_w * scale), int(pixmap_h * scale)
            x = (widget_w - new_w) // 2
            y = (widget_h - new_h) // 2
            scaled_pixmap = self._pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(x, y, scaled_pixmap)
        # 不要呼叫 super().paintEvent(event)，避免 QLabel 預設繪圖覆蓋

    def mousePressEvent(self, event):
        if event.button() == 1:  # 左鍵
            self.handle_aoi_point(event.pos())
        elif event.button() == 2:  # 右鍵
            self.clear_aoi_rect()
        super().mousePressEvent(event)

    def handle_aoi_point(self, pos):
        if self.aoi_rect is not None:
            return
        self.aoi_points.append((pos.x(), pos.y()))
        if len(self.aoi_points) == 2:
            x1, y1 = (np.array(self.aoi_points[0]).astype(float) / np.array([self.label_width / self.video_width, self.label_height / self.video_height])).astype(int)
            x2, y2 = (np.array(self.aoi_points[1]).astype(float) / np.array([self.label_width / self.video_width, self.label_height / self.video_height])).astype(int)
            self.aoi_rect = [min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)]  # [top, bottom, left, right]
            self.aoi_points = []
            print(f"AOI 設定為: {self.aoi_rect}")
            self.aoi_rect_changed.emit(self.aoi_rect)

    def clear_aoi_rect(self):
        self.aoi_rect = None
        print("AOI 已清除")
        self.aoi_rect_changed.emit(self.aoi_rect)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Stream (Local UI)")

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

        self.image_label = AOILabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 讓 image_label 填滿
        self.image_label.video_width = self.video_width  # 傳遞給 AOILabel
        self.image_label.video_height = self.video_height
        self.image_label.aoi_rect_changed.connect(self.on_aoi_rect_changed)
        self.control_panel = ControlPanel(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 讓 layout 沒有邊框
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.showFullScreen()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.aoi_model = cv_aoi()
        self.goldens = []
        self.aoi_rect = None  # 用於同步 AOI 狀態

    def on_aoi_rect_changed(self, rect):
        self.aoi_rect = rect

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            display_frame = frame.copy()
            if self.aoi_rect:
                t, b, l, r = self.aoi_rect
                cv2.rectangle(display_frame, (l, t), (r, b), (0, 0, 255), 4)
            if self.goldens:
                if self.aoi_rect:
                    aoi = self.aoi_rect
                    frame = frame[aoi[0]:aoi[1], aoi[2]:aoi[3]]
                else:
                    aoi = None
                mask, mask_mean, mask_min, a = self.aoi_model.match_template(frame, self.goldens , aoi = aoi )
                
                threshold = self.control_panel.threshold_spin.value()
                n_erode = self.control_panel.erode_spin.value()
                n_dilate = self.control_panel.dilate_spin.value()
                min_samples = self.control_panel.min_samples_spin.value()
                
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
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            else:
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def snapshot_image(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            folder = self.control_panel.folder_combo.currentText()
            filename = f"snapshot_{timestamp}.bmp"
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            if folder:
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, filename)
            cv2.imwrite(save_path, self.current_frame)

    def set_sample(self):
        self.goldens = []
        folder = self.control_panel.folder_combo.currentText()
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
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.bmp')]
        if not files:
            msg = "該資料夾內沒有 bmp 檔案。"
        else:
            msg = "\n".join(files)
        
        for img_path in files:
            golden_img = cv2.imread(img_path)
            kp, des = self.aoi_model.get_keypoint(golden_img)
            self.goldens.append([golden_img, kp, des])

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == 16777216:  # Qt.Key_Escape
            self.control_panel.close()  # 新增：關閉 ControlPanel
            self.close()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    panel = win.control_panel
    win.showFullScreen()
    panel.show()
    sys.exit(app.exec_())