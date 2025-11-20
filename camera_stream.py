import sys
import os
import cv2
import time
import numpy as np
import shutil

from match_template import cv_aoi
from control_panel import ControlPanel
from LT_300H_control import LT300HControl 

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal, QPoint, Qt, QObject, QThread, QWaitCondition, QMutex
from PyQt5.QtWidgets import QMessageBox
from pygrabber.dshow_graph import FilterGraph

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
    match_done = pyqtSignal(object, int)  # 改用 match_done signal
    def __init__(self, aoi_model):
        super().__init__()
        self.aoi_model = aoi_model
        self.frame = None
        self.goldens = None
        self.aoi = None
        self.threshold = None
        self.n_erode = None
        self.n_dilate = None
        self.min_samples = None
        # 新增：QWaitCondition 與 QMutex
        self._wait_cond = QWaitCondition()
        self._mutex = QMutex()
        self._should_run = False
        self._running = True  # 控制 while 迴圈

    def set_params(self, frame, goldens, aoi, threshold, n_erode, n_dilate, min_samples):
        self.frame = frame
        self.goldens = goldens
        self.aoi = aoi
        self.threshold = threshold
        self.n_erode = n_erode
        self.n_dilate = n_dilate
        self.min_samples = min_samples

    def run(self):
        while self._running:
            self._mutex.lock()
            if not self._should_run:
                self._wait_cond.wait(self._mutex)
            self._should_run = False
            self._mutex.unlock()
            if not self._running:
                break
            # 執行比對
            if self.frame is None or self.goldens is None:
                self.match_done.emit(None, -1)
                continue
            import time
            start_time = time.time()
            mask, mask_mean, mask_min, a , index= self.aoi_model.match_template(self.frame, self.goldens, aoi=self.aoi)
            circle_image , n  = self.match_result(mask_min, a)
            end_time = time.time()        
            print(f"frame = {self.frame.shape} time = {end_time - start_time}  , index = {index}")
            self.match_done.emit(circle_image, n)

    def wake(self):
        self._mutex.lock()
        self._should_run = True
        self._wait_cond.wakeOne()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._should_run = True
        self._wait_cond.wakeOne()
        self._mutex.unlock()

    def match_result(self, mask_min, a):
        m = self.aoi_model.post_proc(mask_min, self.threshold, self.n_erode, self.n_dilate)
        #res = (np.stack([np.maximum(m,a[:,:,0]) , a[:,:,1]*(m==0), a[:,:,2]*(m==0)], axis=-1))
        res = (np.stack([a[:,:,0]*(m==0), a[:,:,1]*(m==0)  ,np.maximum(m,a[:,:,2]) ], axis=-1))
        return self.aoi_model.draw_circle(m, res, min_samples=self.min_samples)

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Stream (Local UI)")
        graph = FilterGraph()
        temp_cap = None
        for i, name in enumerate(graph.get_input_devices()):
            if "SC0710 PCI, Video 01 Capture" in name:
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                self.video_width = 3840
                self.video_height = 2160
                if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width):
                    print("Can not set frame width to", self.video_width)
                if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height):
                    print("Can not set frame height to", self.video_height)
            elif "NeuroEye" in name:
                temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not temp_cap.set(cv2.CAP_PROP_EXPOSURE, -5):
                    print("Can not set exposure to -8")
                if not temp_cap.set(cv2.CAP_PROP_AUTO_WB, 0):
                    print("Can not set auto white balance to 0")
                if not temp_cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000):
                    print("Can not set white balance blue U to 4000")
                if not temp_cap.set(cv2.CAP_PROP_SETTINGS, 1):
                    print("Can not open camera settings")
                if not temp_cap.set(cv2.CAP_PROP_BRIGHTNESS, 100):
                    print("Can not set brightness to 100")
                if not temp_cap.set(cv2.CAP_PROP_CONTRAST, 100):
                    print("Can not set contrast to 100")
                if not temp_cap.set(cv2.CAP_PROP_SHARPNESS, 0):
                    print("Can not set sharpness to 0")
                #temp_cap.release()

        if not hasattr(self, 'cap'):
            QMessageBox.warning(self, "採集卡失效", "SC0710 找不到")
            sys.exit(1)

        if temp_cap is None:
            QMessageBox.warning(self, "Type C 沒有接上", "Type C 沒有接上")
            sys.exit(1)
        else:
            temp_cap.release()


        self.display_video = True 
        self.current_frame = None

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
        self.timer.start(16)

        self.aoi_model = cv_aoi()
        self.goldens = []
        self.aoi_rect = None  # 用於同步 AOI 狀態
        self.matching = False  # 新增：避免重複啟動 worker
        # 新增：常駐 thread/worker
        self.match_thread = QThread()
        self.worker = AOIMatchWorker(self.aoi_model)
        self.worker.moveToThread(self.match_thread)
        self.worker.match_done.connect(self.show_image)
        self.match_thread.started.connect(self.worker.run)
        self.match_thread.start()  # 直接啟動 thread，讓 worker 進入等待

        self.LT_300H_dev = LT300HControl(port="COM7", baudrate=115200, timeout=1.0)
        self.LT_300H_dev.open()
        self.LT_300H_dev.start_reading()
        self.LT_300H_dev.set_move_speed(100)
        self.LT_300H_dev.move_to(0, 0, 0)
        time.sleep(0.5)  # 等待回應
        self.LT_300H_dev.cur_x, self.LT_300H_dev.cur_y, self.LT_300H_dev.cur_z = self.LT_300H_dev.last_line.split(",")
        # width 104 mm heights 60 mm

    def on_aoi_rect_changed(self, rect):
        self.aoi_rect = rect

    def show_image(self, display_image , circle_count):
        if display_image is None:
            print("display_image is None")
            return
        try:
            rgb_image = display_image.copy() if display_image is not None else np.zeros((100,100,3), dtype=np.uint8)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.n_label.setText(f"圈數: {circle_count}")  # 新增：顯示圈數
            self.matching = False
        except:
            print("show_image crash")
            self.n_label.setText("")  # 若失敗則清空
            self.matching = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 上下左右顛倒
            frame = cv2.flip(frame, -1)
            self.current_frame = frame.copy()
            if self.display_video == True:
                display_frame = frame.copy()
                if self.aoi_rect:
                    t, b, l, r = self.aoi_rect
                    cv2.rectangle(display_frame, (l, t), (r, b), (0, 0, 255), 4)
                self.show_image(display_frame, -1) 

    def snapshot_path(self, frame, path , message=False):
        if not os.path.isdir(path):
            os.makedirs(path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"snapshot_{timestamp}.bmp"
        save_path = os.path.join(path, filename)
        cv2.imwrite(save_path, frame)

        if message == False:
            return save_path

        self.show_message_box("拍照完成", f"已儲存於：\n{save_path}")

        return save_path

    def handle_manual_match(self):
        if self.current_frame is not None and self.goldens and not self.matching :
            frame = self.current_frame.copy()
            today = "AOI_" + time.strftime("%Y%m%d")
            base_path = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_path, today)
            self.last_frame = self.snapshot_path(self.current_frame , save_dir)

            self.display_video = False
            if self.aoi_rect:
                t, b, l, r = self.aoi_rect
                frame_aoi = frame[t:b, l:r]
            else:
                frame_aoi = frame
            threshold = self.control_panel.threshold_spin.value()
            n_erode = self.control_panel.erode_spin.value()
            n_dilate = self.control_panel.dilate_spin.value()
            min_samples = self.control_panel.min_samples_spin.value()
            self.worker.set_params(frame_aoi, self.goldens, self.aoi_rect, threshold, n_erode, n_dilate, min_samples)
            self.matching = True
            self.worker.wake()  # 用 wake 觸發 worker 執行

    def snapshot_positive_image(self, retry_count=5):
        if retry_count == False :
            retry_count = 5
        if self.current_frame is not None:
            frame_copy = self.current_frame.copy()
            if frame_copy.max() == 0:
                if retry_count > 0:
                    print(f"frame is all black, retrying... ({6 - retry_count}/ {retry_count})")
                    QTimer.singleShot(100, lambda: self.snapshot_positive_image(retry_count=retry_count-1))
                    return
                else:
                    print("frame is all black after 3 retries, abort snapshot.")
                    QMessageBox.warning(self, "拍照失敗", "連續 3 次取得的影像皆為全黑，請檢查攝影機狀態。")
                    return
                
            folder = self.control_panel.folder_save_path if hasattr(self.control_panel, 'folder_save_path') else os.path.dirname(os.path.abspath(__file__))
            self.snapshot_path(frame_copy , folder , message=True)

    def set_sample(self):
        folder = self.control_panel.folder_combo.currentText()
        if not folder:
            if self.goldens is not None:
                self.goldens = []
                self.display_video = True 
                return
            QMessageBox.information(self, "設定檢測樣本", "請先選擇資料夾！")
            return
        self.goldens = []
        base_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            QMessageBox.information(self, "設定檢測樣本", "資料夾不存在！")
            return
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.bmp')]
        if not files:
            QMessageBox.information(self, "設定檢測樣本", "資料夾內無BMP檔！")
            return
        
        for img_path in files:
            golden_img = cv2.imread(img_path)
            kp, des = self.aoi_model.get_keypoint(golden_img)
            self.goldens.append([golden_img, kp, des])

    def closeEvent(self, event):
        self.cap.release()
        # 修正：確保 thread 已結束
        if self.match_thread is not None and self.match_thread.isRunning():
            self.worker.stop()  # 通知 worker 結束 while 迴圈
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
        elif event.key() == Qt.Key_Space:  # 新增：空白鍵觸發手動圈數計算
            self.handle_manual_match()
        elif event.key() == Qt.Key_F1:
            folder_path = self.control_panel.folder_combo.currentText()
            base_path = os.path.dirname(os.path.abspath(__file__))
            folder = os.path.join(base_path, folder_path)
            os.makedirs(folder, exist_ok=True)
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return
            filename = os.path.basename(self.last_frame)
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame, new_path)
            message_text = f"已新增正樣本圖片到 {new_path}"
            self.show_message_box("新增正樣本", message_text, 3000)
            self.set_sample()

        elif event.key() == Qt.Key_F2:
            # F2: 你可以在這裡加上你要的功能
            folder_path = "nagetive_samples"
            base_path = os.path.dirname(os.path.abspath(__file__))
            folder = os.path.join(base_path, folder_path)
            os.makedirs(folder, exist_ok=True)
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return
            filename = os.path.basename(self.last_frame)
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame, new_path)

            folder = f"已保存未檢測到瑕疵圖片 {new_path}"
            self.show_message_box("未檢測到", folder, 3000)
        else:
            super().keyPressEvent(event)

    def show_message_box(self, title, text, close_time=5000):
        if hasattr(self, 'msg') and self.msg is not None:
            self.msg.close()
            self.msg = None
        self.msg = QMessageBox()
        self.msg.setWindowTitle(title)
        self.msg.setText(text)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.msg.show()

        def close_msg():
            if self.msg is not None:
                self.msg.close()

        QTimer.singleShot(close_time, close_msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    if win is not None:
        panel = win.control_panel
        win.showFullScreen()
        panel.show()
    sys.exit(app.exec_())