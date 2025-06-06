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
from PyQt5.QtCore import QTimer, pyqtSignal, QPoint, Qt, QObject, QThread

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
        self.video_width = 2560  # 預設值，CameraApp 會設置
        self.video_height = 1440

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
            if len(self.aoi_points) > 0:
                self.aoi_points = []
            elif self.aoi_rect is not None:
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

class AOIMatchWorker(QObject):
    finished = pyqtSignal(object, object, object, object, float)
    def __init__(self, aoi_model, frame, goldens, aoi):
        super().__init__()
        self.aoi_model = aoi_model
        self.frame = frame
        self.goldens = goldens
        self.aoi = aoi
    def run(self):
        import time
        start_time = time.time()
        mask, mask_mean, mask_min, a , index= self.aoi_model.match_template(self.frame, self.goldens, aoi=self.aoi)
        end_time = time.time()        
        print(f"frame = {self.frame.shape} time = {end_time - start_time}  , index = {index}")
        self.finished.emit(mask, mask_mean, mask_min, a, end_time - start_time)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Stream (Local UI)")

        self.video_width = 3840
        self.video_height = 2160
        self.cap = cv2.VideoCapture(0)
        
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
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
        self.n_label = QLabel("")  # 新增：顯示圈數
        self.n_label.setStyleSheet("color: yellow; font-size: 32px; font-weight: bold; background: rgba(0,0,0,0.5);")
        self.n_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 讓 layout 沒有邊框
        layout.addWidget(self.image_label)
        layout.addWidget(self.n_label)  # 新增：加到 layout
        self.setLayout(layout)
        self.showFullScreen()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.aoi_model = cv_aoi()
        self.goldens = []
        self.aoi_rect = None  # 用於同步 AOI 狀態
        self.matching = False  # 新增：避免重複啟動 worker
        
        self.match_thread = None

    def on_aoi_rect_changed(self, rect):
        self.aoi_rect = rect

    def handle_match_result(self, mask, mask_mean, mask_min, a, elapsed):
        try:
            threshold = self.control_panel.threshold_spin.value()
            n_erode = self.control_panel.erode_spin.value()
            n_dilate = self.control_panel.dilate_spin.value()
            min_samples = self.control_panel.min_samples_spin.value()
            m = self.aoi_model.post_proc(mask_min, threshold, n_erode, n_dilate)
            res = (np.stack([np.maximum(m,a[:,:,0]), a[:,:,1]*(m==0), a[:,:,2]*(m==0)], axis=-1))
            circle_image , n = self.aoi_model.draw_circle(m, res, min_samples=min_samples)
            #print("draw circle", n , "clusters")
            rgb_image = circle_image.copy()
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
        
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.n_label.setText(f"圈數: {n}")  # 新增：顯示圈數
            self.matching = False
        except:
            print("handle_match_result crash")
            self.n_label.setText("")  # 若失敗則清空
            self.matching = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 上下左右顛倒
            frame = cv2.flip(frame, -1)
            self.current_frame = frame
            display_frame = frame.copy()
            if self.aoi_rect:
                t, b, l, r = self.aoi_rect
                cv2.rectangle(display_frame, (l, t), (r, b), (0, 0, 255), 4)
            if self.goldens:
                if self.aoi_rect:
                    aoi = self.aoi_rect
                    frame_aoi = frame[aoi[0]:aoi[1], aoi[2]:aoi[3]]
                else:
                    aoi = None
                    frame_aoi = frame
                try:
                    if not self.matching and (self.match_thread is None or not self.match_thread.isRunning()):
                        self.matching = True
                        self.match_thread = QThread()
                        self.worker = AOIMatchWorker(self.aoi_model, frame_aoi, self.goldens, aoi)
                        self.worker.moveToThread(self.match_thread)
                        self.match_thread.started.connect(self.worker.run)
                        self.worker.finished.connect(self.handle_match_result)
                        self.worker.finished.connect(self.worker.deleteLater)
                        self.worker.finished.connect(self.match_thread.quit)
                        self.match_thread.finished.connect(self.match_thread.deleteLater)
                        self.match_thread.start()
                except:
                    self.match_thread = None
                    print("self.matching crash")
                    print("self.matching = " , self.matching)
                    print("self.match_thread = " , self.match_thread)
                    #print("self.match_thread.isRunning() = " , self.match_thread.isRunning())
            else:
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_image))
                self.n_label.setText("")  # 新增：沒比對時清空圈數
        # ...existing code...
    def snapshot_image(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 優先用 control_panel.folder_path
            folder = self.control_panel.folder_save_path if hasattr(self.control_panel, 'folder_save_path') else None
            filename = f"snapshot_{timestamp}.bmp"
            if folder:
                save_path = os.path.join(folder, filename)
            else:
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
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
        # 修正：確保 thread 已結束
        if self.match_thread is not None and self.match_thread.isRunning():
            self.match_thread.quit()
            self.match_thread.wait()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.control_panel.close()  # 新增：關閉 ControlPanel
            self.close()
        elif event.key() == Qt.Key_P:  # 新增：按下 P 鍵叫出 panel 並顯示在最上層
            self.control_panel.show()
            self.control_panel.raise_()
            self.control_panel.activateWindow()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    panel = win.control_panel
    win.showFullScreen()
    panel.show()
    sys.exit(app.exec_())