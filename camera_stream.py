import sys
import os
import cv2
import time
import numpy as np
import shutil
import threading
import configparser

from match_template import cv_aoi
from control_panel import ControlPanel
from LT_300H_control import LT300HControl 
from find_contour import FindContour

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot, QPoint, Qt, QObject, QThread, QWaitCondition, QMutex
from PyQt5.QtWidgets import QMessageBox
from pygrabber.dshow_graph import FilterGraph

def get_file(folder_path, extension='.bmp'):
    if os.path.isdir(folder_path):
        # 支援單個副檔名（字串）或多個副檔名（列表/元組）
        extensions = extension if isinstance(extension, (list, tuple)) else [extension]
        
        # 確保所有副檔名都以 . 開頭
        extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
        extensions = [ext.lower() for ext in extensions]
        
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and any(f.lower().endswith(ext) for ext in extensions)]
    else:
        return []

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
        # 新增 zoom 狀態
        self.zoom_scale = 1.0
        self.zoom_center = None  # (x, y) in image coordinates
        self.left = 0
        self.top = 0

    def resizeEvent(self, event):
        self.label_width = self.width()   # 更新 label 寬度
        self.label_height = self.height() # 更新 label 高度
        super().resizeEvent(event)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def wheelEvent(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x, y = int(pos.x()), int(pos.y())
        delta = event.angleDelta().y()
        # 呼叫 zoom_at
        self.zoom_at(x, y, delta)
        self.update()
        # 不呼叫 super().wheelEvent(event) 以避免父類預設行為

    def zoom_at(self, x, y, delta):
        # delta > 0 放大, < 0 縮小
        new_scale = self.zoom_scale + 0.25 if delta > 0 else self.zoom_scale - 0.25
        new_scale = max(1.0, min(new_scale, 20.0))  # 限制縮放範圍
        if self._pixmap is None:
            return
        widget_w, widget_h = self.width(), self.height()
        pixmap_w, pixmap_h = self._pixmap.width(), self._pixmap.height()
        # 目前顯示區域的左上角在原圖的座標
        if self.zoom_scale > 1.01 and self.zoom_center is not None:
            #None
            #zx, zy = self.zoom_center
            crop_w = int(pixmap_w / self.zoom_scale)
            crop_h = int(pixmap_h / self.zoom_scale)
            #self.left = max(0, zx - crop_w // 2)
            #self.top = max(0, zy - crop_h // 2)
        else:
            self.left, self.top = 0, 0
            crop_w, crop_h = pixmap_w, pixmap_h
        # 滑鼠在 widget 的比例座標
        rel_x = x / widget_w
        rel_y = y / widget_h
        # 新的 zoom_center 以目前顯示區域為基準
        img_x = int(self.left + rel_x * crop_w)
        img_y = int(self.top + rel_y * crop_h)
        self.zoom_scale = new_scale
        self.zoom_center = (img_x, img_y)

    def paintEvent(self, event):
        from PyQt5.QtWidgets import QStyle
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._pixmap:
            widget_w, widget_h = self.width(), self.height()
            pixmap_w, pixmap_h = self._pixmap.width(), self._pixmap.height()
            # zoom 狀態
            if self.zoom_scale > 1.01 and self.zoom_center is not None:
                zx, zy = self.zoom_center

                # 維持原圖比例裁切
                crop_w = int(pixmap_w / self.zoom_scale)
                crop_h = int(pixmap_h / self.zoom_scale)
                self.left = max(0, zx - crop_w // 2)
                right = min(pixmap_w, self.left + crop_w)
                self.top = max(0, zy - crop_h // 2)
                bottom = min(pixmap_h, self.top + crop_h)
                cropped = self._pixmap.copy(self.left, self.top, right - self.left, bottom - self.top)
                scaled_pixmap = cropped.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
                new_w, new_h = int(pixmap_w * scale), int(pixmap_h * scale)
                scaled_pixmap = self._pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (widget_w - scaled_pixmap.width()) // 2
            y = (widget_h - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
        # 不要呼叫 super().paintEvent(event)，避免 QLabel 預設繪圖覆蓋

    def mousePressEvent(self, event):
        # if event.button() == 1:  # 左鍵
        #     self.handle_aoi_point(event.pos())
        # elif event.button() == 2:  # 右鍵
        #     if len(self.aoi_points) > 0:
        #         self.aoi_points = []
        #     elif self.aoi_rect is not None:
        #         self.clear_aoi_rect()
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
    match_done = pyqtSignal(int)
    message_box = pyqtSignal(str, str, int)
    def __init__(self, aoi_model):
        super().__init__()
        self.aoi_model = aoi_model

        self.aoi_rect = None
        self.threshold = None
        self.n_erode = None
        self.n_dilate = None
        self.min_samples = None
        self.circle_image = None
        # 新增：QWaitCondition 與 QMutex
        self._wait_cond = QWaitCondition()
        self._mutex = QMutex()
        self._should_run = False
        self._running = True  # 控制 while 迴圈

        #self.frame = None
        self.detect_frame = []
        self.goldens = []
        self.detect_result = []
        self.detect_index = 0 
        self.detect_max_index = 4
        
    def run(self):
        while self._running:
            self._mutex.lock()
            #if not self._should_run:
            self._wait_cond.wait(self._mutex)
            #self._should_run = False
            self._mutex.unlock()
            if not self._running:
                break
            while self.detect_index != self.detect_max_index:
                if len(self.detect_frame) == 0: # neet to fix shane
                    time.sleep(0.1)
                    continue
                self.match(-1)
                self.match_done.emit()
                

    def wake(self):
        #self._mutex.lock()
        #self._should_run = True
        self._wait_cond.wakeOne()
        #self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._should_run = True
        self._wait_cond.wakeOne()
        self._mutex.unlock()

    def set_params(self, threshold, n_erode, n_dilate, min_samples):
        self.threshold = threshold
        self.n_erode = n_erode
        self.n_dilate = n_dilate
        self.min_samples = min_samples
        self.circle_image = None
        return True
    
    def set_frame_and_goldens(self, frame ,folder_path):
        self.detect_frame.append(frame.copy())
        self.goldens , self.aoi_rect = self.set_goldens_sample(folder_path)
        if self.goldens is None or self.aoi_rect is None:
            return False
        return True

    def set_frame(self, frame):
        self.detect_frame.append(frame.copy())

    def set_goldens_sample(self , folder_path):
        """
        遍歷 folder_path 下的所有子資料夾，為每個子資料夾分別載入樣本和 aoi_rect。
        返回二維的 goldens 列表和一維的 aoi_rects 列表。
        """
        all_goldens = []
        all_aoi_rects = []

        if not os.path.isdir(folder_path):
            self.message_box.emit("設定檢測樣本", f"資料夾不存在！\n{folder_path}", 3000)
            return None, None

        subdirectories = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        if not subdirectories:
            self.message_box.emit("設定檢測樣本", "找不到任何子資料夾！", 3000)
            return None, None

        for sub_dir in subdirectories:
            goldens_in_subdir = []
            bmp_files = get_file(sub_dir, extension='.bmp')
            if not bmp_files:
                continue  # 如果子資料夾沒有 bmp 檔，就跳過

            for img_path in bmp_files:
                golden_img = cv2.imread(img_path)
                kp, des = self.aoi_model.get_keypoint_grid(golden_img)
                goldens_in_subdir.append([golden_img, kp, des])
            
            # 讀取該子資料夾的 ini 檔
            aoi_rect = None
            inifiles = get_file(sub_dir, extension='.ini')
            if inifiles:
                try:
                    config = configparser.ConfigParser()
                    config.read(inifiles[0])
                    if config.has_section('Rect'):
                        x = int(config.get('Rect', 'x'))
                        y = int(config.get('Rect', 'y'))
                        width = int(config.get('Rect', 'width'))
                        height = int(config.get('Rect', 'height'))
                        aoi_rect = [y, y + height, x, x + width]
                except Exception as e:
                    self.message_box.emit("讀取 .ini 失敗", f"讀取 {inifiles[0]} 時發生錯誤: {e}", 3000)
            
            all_goldens.append(goldens_in_subdir)
            all_aoi_rects.append(aoi_rect)

        return all_goldens, all_aoi_rects

    def match(self , index = -1):
        if self.detect_index > len(self.detect_frame):
            return None , 0
        frame = self.detect_frame[self.detect_index]
                
        if frame is None or self.goldens[self.detect_index] is None:
            return None , 0
        
        start_time = time.time()
        mask, mask_mean, mask_min, a , index= self.aoi_model.match_template(frame, self.goldens[self.detect_index], aoi=self.aoi_rect[self.detect_index])
        self.circle_image , n  = self.match_result(mask_min, a)
        end_time = time.time()
        print(f"frame = {frame.shape} time = {end_time - start_time}  , index = {index} , n = {n}")

        if index == -1:
            self.detect_index += 1
            self.detect_result.append((self.circle_image , n))
        else:
            self.detect_result[index] = (self.circle_image , n)

    def match_result(self, mask_min, a):
        m = self.aoi_model.post_proc(mask_min, self.threshold, self.n_erode, self.n_dilate)
        #res = (np.stack([np.maximum(m,a[:,:,0]) , a[:,:,1]*(m==0), a[:,:,2]*(m==0)], axis=-1))
        res = (np.stack([a[:,:,0]*(m==0), a[:,:,1]*(m==0)  ,np.maximum(m,a[:,:,2]) ], axis=-1))
        return self.aoi_model.draw_circle(m, res, min_samples=self.min_samples)

class CameraMoveWorker(QObject):
    camera_move_done = pyqtSignal()  # 拍照完成訊號
    def __init__(self, camera_app):
        super().__init__()

        self.LT_300H_dev = LT300HControl(port="COM9", baudrate=115200, timeout=1.0)
        self.LT_300H_dev.set_start_position(35, 48, 21)
        self.LT_300H_dev.set_max_limit_position(150, 110, 21)

        # x 方向：1 表示向右 (0->200)，-1 表示向左 (200->0)
        self._x_dir = 1
        self.step_x = 90
        self.step_y = 53
        self.position = self.calculate_all_position()
        self.current_position_index = 0
        print(f"position = {self.position}")

        self.camera_app = camera_app
        self._running = True
        # self._stop_event = threading.Event()
        self._wait_cond = QWaitCondition()
        self._mutex = QMutex()
        self.start_move = False
        self.move_check_wait_cond = QWaitCondition()
        self.move_check_mutex = QMutex()
        self.move_camera_get_frame_done_callback = None
        self.positive_folder = None
        self.breakpoint = False

    def wake(self):
        self._mutex.lock()
        self.start_move = True
        self._wait_cond.wakeOne()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._wait_cond.wakeOne()
        self.start_move = False
        self._mutex.unlock()
        self.move_check_wait_cond.wakeOne()
    
    def run(self):
        """在背景 thread 中執行拍照迴圈"""
        while self._running:
            self._mutex.lock()
            self._wait_cond.wait(self._mutex)
            self._mutex.unlock()

            while self.start_move:
                frame_copy = None
                if self.camera_app.current_frame is not None:
                    try:
                        print("move camera get frame")
                        frame_copy = self.camera_app.current_frame.copy()
                    except Exception:
                        frame_copy = None

                if frame_copy is None or getattr(frame_copy, 'size', 0) == 0:
                    time.sleep(0.05)
                    continue
                
                print("CameraMoveWorker worker_callback call")
                # 如果有外部指定 move_camera_get_frame_done_callback ，直接在 worker 執行
                if callable(self.move_camera_get_frame_done_callback):
                    try:
                        self.move_camera_get_frame_done_callback(frame_copy)
                    except TypeError:
                        print(f"move camera callback error: {e}")

                print("CameraMoveWorker worker_callback done")

                if self.breakpoint == True:
                    break
                # 移動到下一個位置（非阻塞），等待裝置到達
                try:
                    print("move camera to next position ")
                    self.move_to_next_position()
                    print("move camera to next position done")
                except Exception as e:
                    print(f"move_to_next_position error: {e}")

                self.move_check_mutex.lock()
                self.move_check_wait_cond.wait(self.move_check_mutex)
                self.move_check_mutex.unlock()

                # 檢查是否回到起點
                if (self.LT_300H_dev.target_x == self.LT_300H_dev.start_x and 
                    self.LT_300H_dev.target_y == self.LT_300H_dev.start_y):
                    self.start_move = False
                    #self.breakpoint = False
                    break

                time.sleep(0.1)  # 避免 CPU 佔用過高

            self.camera_move_done.emit()

    def calculate_all_position(self):
        """計算從 start_position 到 max_limit_position 的所有掃描位置。"""
        positions = []
        if not self.LT_300H_dev:
            return positions

        start_x = self.LT_300H_dev.start_x
        start_y = self.LT_300H_dev.start_y
        limit_x_min, limit_x_max = self.LT_300H_dev.limit_x
        limit_y_min, limit_y_max = self.LT_300H_dev.limit_y

        cur_x = start_x
        cur_y = start_y
        x_dir = 1  # 1 for right, -1 for left

        while cur_y <= limit_y_max:
            positions.append((cur_x, cur_y))
            next_x = cur_x + (x_dir * self.step_x)

            if limit_x_min <= next_x <= limit_x_max:
                cur_x = next_x
            else:
                cur_y += self.step_y
                x_dir *= -1 # Reverse direction
                if cur_y > limit_y_max:
                    break
        
        return positions

    def move_to_next_position(self):
        if not self.position:
            print("⚠️ 位置列表是空的，無法移動。")
            return

        # 取得下一個位置
        if self.current_position_index < len(self.position):
            new_x, new_y = self.position[self.current_position_index]
            self.current_position_index += 1
        else:
            # 掃描完成，回到起點
            self.current_position_index = 0
            new_x = self.LT_300H_dev.start_x
            new_y = self.LT_300H_dev.start_y

        # 回調函數
        def on_position_arrived(context):
            if context == "move_to_next_position":
                # 使用 self 來存取 CameraMoveWorker 實例
                self.move_check_wait_cond.wakeOne()

        try:
            # 呼叫裝置移動，並傳遞回調函數
            self.LT_300H_dev.move_to(new_x, new_y, self.LT_300H_dev.start_z, 
                                     callback=lambda ctx: on_position_arrived(ctx),
                                     context="move_to_next_position")
        except Exception as e:
            print(f"move_to_next_position failed: {e}")

class CameraApp(QWidget):
    message_box = pyqtSignal(str, str, int)
    def __init__(self):
        super().__init__()

        # self.LT_300H_dev = LT300HControl(port="COM9", baudrate=115200, timeout=1.0)
        # self.LT_300H_dev.set_start_position(35, 48, 21)
        # self.LT_300H_dev.set_max_limit_position(150, 110, 300)
        # x 方向：1 表示向右 (0->200)，-1 表示向左 (200->0)
        # self._x_dir = 1

        self.camera_setting()

        self.setWindowTitle("Camera Stream (Local UI)")
        
        #test
        self.find_contour = FindContour()
        self.find_contour_result = None

        self.display_video = True 
        self.current_frame = None

        self.image_label = AOILabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 讓 image_label 填滿
        self.image_label.video_width = self.video_width  # 傳遞給 AOILabel
        self.image_label.video_height = self.video_height
        self.image_label.aoi_rect_changed.connect(self.on_aoi_rect_changed)
        self.control_panel = ControlPanel(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 讓 layout 沒有邊框
        layout.addWidget(self.image_label)
        # layout.addWidget(self.n_label)  # 新增：加到 layout
        self.setLayout(layout)
        self.showFullScreen()

        # connect message signal to slot to allow cross-thread messaging
        self.message_box.connect(self.show_message_box)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

        self.aoi_model = cv_aoi()
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.fount_back = None
        self.current_folder = ""
        #self.goldens = []
        #self.aoi_rect = [21 , 2160 - 21 , 38 , 3840 -38 ]#None  # 用於同步 AOI 狀態

        # 新增：常駐 thread/worker
        self.match_thread = QThread()
        self.AOI_worker = AOIMatchWorker(self.aoi_model)
        self.AOI_worker.moveToThread(self.match_thread)
        self.AOI_worker.match_done.connect(self.match_around_done)
        self.AOI_worker.message_box.connect(self.show_message_box)
        self.match_thread.started.connect(self.AOI_worker.run)
        self.match_thread.start()  # 直接啟動 thread，讓 worker 進入等待
        
        # 新增：snapshot positive worker
        self.camera_move_thread = QThread()
        self.camera_move_worker = CameraMoveWorker(self)
        self.camera_move_worker.moveToThread(self.camera_move_thread)
        self.camera_move_worker.camera_move_done.connect(self.on_move_done)
        self.camera_move_thread.started.connect(self.camera_move_worker.run)
        self.camera_move_thread.start()

    def on_folder_changed(self, folder_name: str):
        """當控制面板的資料夾下拉選單變更時的回呼函式"""
        if not folder_name:
            print("資料夾選擇已清除。")
            self.current_folder = ""
            self.AOI_worker.goldens = [] # 清空樣本
            return
        self.current_folder = os.path.join(self.base_path, self.control_panel.folder_combo.currentText())
        self.AOI_worker.set_goldens_sample(self.current_folder)
        print(f"選擇的樣本資料夾已變更為: {self.current_folder}")

    def on_aoi_rect_changed(self, rect):
        self.aoi_rect = rect

    def show_image(self, display_image):
        if display_image is None:
            print("display_image is None")
            return
        try: 
            rgb_image = display_image.copy() if display_image is not None else np.zeros((100,100,3), dtype=np.uint8)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        except:
            print("show_image crash")

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
                self.show_image(display_frame) 

    def snapshot_path(self, frame, path , message=False):
        if not os.path.isdir(path):
            os.makedirs(path)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}.bmp"
        save_path = os.path.join(path, filename)
        cv2.imwrite(save_path, frame)
        if message == False:
            return save_path
        self.show_message_box("拍照完成", f"已儲存於：\n{save_path}")
        return save_path

    # def set_sample(self):
    #     folder_path = self.control_panel.folder_combo.currentText()
    #     if not folder_path:
    #         QMessageBox.information(self, "設定檢測樣本", "請先選擇資料夾！")
    #         return
    #     folder_path = self.get_move_folder(os.path.join(self.base_path, folder_path) , self.fount_back)
    #     self.goldens = []
    #     if not os.path.isdir(folder_path):            
    #         print(f"資料夾不存在！ folder_path = {folder_path}" )
    #         self.show_message_box("設定檢測樣本", "資料夾不存在！", 3000)      
    #         return
    #     files = get_file(folder_path , extension='.bmp')
    #     if not files:
    #         print(f"資料夾內無BMP檔！ {folder_path}")
    #         self.show_message_box("設定檢測樣本", "資料夾內無BMP檔！", 3000)               
    #         return
        
    #     for img_path in files:
    #         golden_img = cv2.imread(img_path)
    #         kp, des = self.aoi_model.get_keypoint_grid(golden_img)
    #         self.goldens.append([golden_img, kp, des])

    #     # 嘗試讀取 ini 檔案並設定 aoi_rect
    #     inifiles = get_file(folder_path , extension='.ini')
    #     if inifiles:
    #         try:
    #             config = configparser.ConfigParser()
    #             config.read(inifiles[0])
                
    #             if config.has_section('Rect'):
    #                 x = int(config.get('Rect', 'x'))
    #                 y = int(config.get('Rect', 'y'))
    #                 width = int(config.get('Rect', 'width'))
    #                 height = int(config.get('Rect', 'height'))
                    
    #                 # 轉換為 aoi_rect 格式 [top, bottom, left, right]
    #                 self.aoi_rect = [y, y + height, x, x + width]
    #         except Exception as e:
    #             print(f"讀取 INI 檔案失敗: {e}")

    def camera_move_done_callback(self , frame):
        # be fix
        today = "AOI_" + time.strftime("%Y%m%d")
        save_dir = self.get_move_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), today) , self.fount_back)
        self.last_frame = self.snapshot_path(self.current_frame , save_dir)
        ##
        self.AOI_worker.detect_frame.append(frame.copy())
        self.AOI_worker.wake()

        #if detect_frame :
        #    today = "AOI_" + time.strftime("%Y%m%d")
        #    save_dir = self.get_move_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), today) , self.fount_back)
        #    self.last_frame = self.snapshot_path(self.current_frame , save_dir)
#
        #    self.display_video = False
#
        #    circle_image , n = self.AOI_worker.match()
        #    self.AOI_worker.match_done.emit(circle_image)
        #    if n > 0 or n == -1:
        #        self.camera_move_worker.breakpoint = True
        #        self.message_box.emit("檢測瑕疵", f"有瑕疵: {n} 個", -1)

    def snapshot_positive_image(self):
        """啟動正樣本拍照 (在背景 thread 中執行)
        """
        start_time = time.time()
        positive_folder = (self.control_panel.folder_save_path 
                          if hasattr(self.control_panel, 'folder_save_path') 
                          else os.path.dirname(os.path.abspath(__file__)))
        
        if positive_folder is None:
            self.show_message_box("正樣資料夾錯誤", f'找不到資料夾{positive_folder}', 3000)
            return

        front_samples = get_file(self.get_move_folder(positive_folder , 0), extension='.bmp')
        back_samples = get_file(self.get_move_folder(positive_folder , 1), extension='.bmp')

        if len(front_samples) == 0 and len(back_samples) == 0:
            self.fount_back = 0 # not sample
        else:
            fount_back = self.aoi_model.get_fount_back_sample(self.current_frame, front_samples , back_samples)

            if fount_back < 0 :
                #self.fount_back = 0
                if len(front_samples) == 0 :
                    self.fount_back = 0
                elif len(back_samples) == 0:
                    self.fount_back = 1
                else:
                    self.show_message_box("判斷失敗", f'無法判斷正反面', 3000)
                    return
            else:
                self.fount_back = fount_back
        
        end_time = time.time()
        
        print(f"len(front_samples) = {len(front_samples)} len(back_samples) = {len(back_samples)} time = {end_time - start_time} ")

        def worker_callback(frame_copy):
            print("CameraMoveWorker worker_callback called")
            folder = self.get_move_folder(positive_folder , fount_back)
            os.makedirs(folder, exist_ok=True)
            try:
                self.snapshot_path(frame_copy, folder, message=False)
                self.detect_and_save_aoi_rect(frame_copy, folder)
            except Exception as e:
                print(f"snapshot save error: {e}")

        self.camera_move_worker.move_camera_get_frame_done_callback = worker_callback
        self.camera_move_worker.wake()
        
    def on_move_done(self):
        """拍照完成時的回調"""
        #if self.LT_300H_dev.check_current_position(self.LT_300H_dev.start_x , self.LT_300H_dev.start_y ,self.LT_300H_dev.start_z) != True:
        #    return  # 只在回到起點時顯示訊息
        if self.camera_move_worker.move_camera_get_frame_done_callback == None:
            self.show_message_box("拍照完成", "正樣本拍照完成")
        elif self.camera_move_worker.move_camera_get_frame_done_callback == self.camera_move_done_callback:
            self.show_message_box("檢測完成", "PASS" , -1)

    def detect_and_save_aoi_rect(self, frame, folder_path):
        """
        檢測並保存 AOI 矩形
        
        Args:
            frame: 要檢測的影像
            folder_path: 儲存 config.ini 的資料夾路徑
        """
        try:
            self.find_contour_result = self.find_contour.detect_by_color(
                frame, lower_bound =(40, 60, 60) , upper_bound =(80, 255, 255) , draw_result=True
            )
            
            if self.find_contour_result['success']:
                self.aoi_rect = self.find_contour_result['rect2']
                #self.aoi_rect can not bigger than [21 , 2160 - 21 , 38 , 3840 -38 ]
                self.aoi_rect[0] = max(21, self.aoi_rect[0] - 5)
                self.aoi_rect[1] = min(2160 - 21, self.aoi_rect[1] + 5)
                self.aoi_rect[2] = max(38, self.aoi_rect[2] - 5)
                self.aoi_rect[3] = min(3840 - 38, self.aoi_rect[3] + 5)

                # 保存成 config.ini 
                config = configparser.ConfigParser()
                
                # 轉換 aoi_rect [top, bottom, left, right] 為 [x, y, width, height]
                top, bottom, left, right = self.aoi_rect
                x = left
                y = top
                width = right - left
                height = bottom - top
                
                config.add_section('Rect')
                config.set('Rect', 'x', str(x))
                config.set('Rect', 'y', str(y))
                config.set('Rect', 'width', str(width))
                config.set('Rect', 'height', str(height))
                
                # 保存到 config.ini
                os.makedirs(folder_path, exist_ok=True)
                config_path = os.path.join(folder_path, 'config.ini')
                with open(config_path, 'w') as f:
                    config.write(f)
            else:
                print("no contour found")
                self.aoi_rect = [21, 2160 - 21, 38, 3840 - 38]
        except Exception as e:
            print(f"detect_and_save_aoi_rect error: {e}")
            self.aoi_rect = [21, 2160 - 21, 38, 3840 - 38]

    def closeEvent(self, event):
        self.LT_300H_dev.close()
        self.cap.release()
        
        # 清理 snapshot worker
        if hasattr(self, 'snapshot_worker'):
            self.snapshot_worker.stop()
        if hasattr(self, 'snapshot_thread') and self.snapshot_thread.isRunning():
            self.snapshot_thread.quit()
            self.snapshot_thread.wait()
        # 清理 match worker
        if self.match_thread is not None and self.match_thread.isRunning():
            self.match_thread.quit()
            self.match_thread.wait()
        event.accept()

    # def move_to_next_position(self):
    #     try:
    #         cur_x = int(float(self.LT_300H_dev.cur_x))
    #     except Exception:
    #         cur_x = 0
    #     try:
    #         cur_y = int(float(self.LT_300H_dev.cur_y))
    #     except Exception:
    #         cur_y = 0
            
    #     # 掃描邏輯（蛇形）：沿 x 方向前進，抵達邊界則在相同 x 做 y 步進，並反向 x
    #     step_x = 90
    #     step_y = 53

    #     # 若尚未設定方向，預設向右
    #     if not hasattr(self, '_x_dir'):
    #         self._x_dir = 1

    #     next_x = cur_x + (self._x_dir * step_x)
    #     new_x = cur_x
    #     new_y = cur_y

    #     # 如果 next_x 在合法範圍內，則沿 x 前進
    #     if self.LT_300H_dev.limit_x[0] <= next_x <= self.LT_300H_dev.limit_x[1]:
    #         new_x = next_x
    #     else:
    #         # 已到達邊界，先在相同 x 做 y 步進，再反向 x
    #         new_x = cur_x
    #         new_y = cur_y + step_y
    #         if new_y > self.LT_300H_dev.limit_y[1]: #到最後一次的 y
    #             new_y = self.LT_300H_dev.limit_y[0]
    #             new_x = self.LT_300H_dev.limit_x[0]
    #             self._x_dir = 1
    #         else:
    #             # 反轉 x 方向，下一次會沿相反方向走
    #             self._x_dir *= -1
    #     try:
    #         # 呼叫裝置移動，並傳遞回調函數
    #         self.LT_300H_dev.move_to(new_x, new_y, self.LT_300H_dev.start_z, 
    #                                  callback=self.on_position_arrived,
    #                                  context="move_to_next_position")

    #     except Exception as e:
    #         print(f"move_to_next_position failed: {e}")

    def check_test_fount_back(self,folder_name: str) -> int:
        #sample_path = os.path.join(self.base_path, self.control_panel.folder_combo.currentText())
        fount_sample = get_file(self.get_move_folder(folder_name , 0) , extension='.bmp')
        back_sample = get_file(self.get_move_folder(folder_name , 1) , extension='.bmp')
        fount_back = self.aoi_model.get_fount_back_sample(self.current_frame, fount_sample , back_sample)
        return fount_back

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.control_panel.close()  # 新增：關閉 ControlPanel
            self.close()
        elif event.key() == Qt.Key_P:  # 新增：按下 P 鍵叫出 panel 並顯示在最上層
            self.control_panel.show()
            self.control_panel.raise_()
            self.control_panel.activateWindow()

        elif event.key() == Qt.Key_Space:  # 新增：空白鍵 -> 移動並觸發手動圈數計算
            self.fount_back = self.check_test_fount_back(self.current_folder)

            if self.fount_back < 0 :
                self.show_message_box("沒有樣本", f"{self.current_folder} 沒有樣本", 3000)
                return

            self.AOI_worker.detect_frame = []
            self.AOI_worker.detect_result = []
            self.AOI_worker.detect_index = 0 
            self.AOI_worker.detect_max_index = 4

            threshold = self.control_panel.threshold_spin.value()
            n_erode = self.control_panel.erode_spin.value()
            n_dilate = self.control_panel.dilate_spin.value()
            min_samples = self.control_panel.min_samples_spin.value()
            self.AOI_worker.set_params(threshold, n_erode, n_dilate, min_samples)

            self.camera_move_worker.move_camera_get_frame_done_callback = self.camera_move_done_callback
            self.camera_move_worker.wake()

        elif event.key() == Qt.Key_D:  # 移動並觸發手動圈數計算
            self.fount_back = self.check_test_fount_back(self.current_folder)

            if self.fount_back < 0 :
                self.show_message_box("沒有樣本", f"{self.current_folder} 沒有樣本", 3000)
                return
            
            if len(self.AOI_worker.detect_frame) <= self.camera_move_worker.current_position_index:
                self.show_message_box("沒有測試過", f"neet to fix by Shane", 3000)
                return

            self.AOI_worker.detect_frame[self.camera_move_worker.current_position_index] = self.current_frame.copy()
            self.AOI_worker.detect_index = self.camera_move_worker.current_position_index
            #self.AOI_worker.detect_max_index = self.camera_move_worker.current_position_index + 1
        
            threshold = self.control_panel.threshold_spin.value()
            n_erode = self.control_panel.erode_spin.value()
            n_dilate = self.control_panel.dilate_spin.value()
            min_samples = self.control_panel.min_samples_spin.value()
            self.AOI_worker.set_params(threshold, n_erode, n_dilate, min_samples)
        
            #self.camera_move_done_callback()
            self.AOI_worker.match(self.camera_move_worker.current_position_index)
            self.match_done_to_show_image(self.camera_move_worker.current_position_index)
            #self.show_message_box("", f"neet to fix by Shane", 3000)
            n = self.AOI_worker.detect_result[self.camera_move_worker.current_position_index][1]
            if n > 0:
                self.show_message_box.emit("檢測瑕疵", f"有瑕疵: {n} 個", -1)
            else:
                self.show_message_box.emit("檢測瑕疵", f"PASS", -1)

        elif event.key() == Qt.Key_F1: # F1 新增正樣本
            folder_path = self.control_panel.folder_combo.currentText()
            folder = os.path.join(self.base_path, folder_path)
            folder = self.get_move_folder(folder , self.fount_back)
            os.makedirs(folder, exist_ok=True)
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return
            filename = os.path.basename(self.last_frame)
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame, new_path)
            message_text = f"已新增正樣本圖片到 {new_path}"
            self.show_message_box("新增正樣本", message_text, 3000)
            self.set_sample()

        elif event.key() == Qt.Key_F2: # F2 負樣本
            folder_path = "nagetive_samples"
            folder = os.path.join(self.base_path, folder_path)
            os.makedirs(folder, exist_ok=True)
            if not hasattr(self, 'last_frame') or self.last_frame is None:
                return
            filename = os.path.basename(self.last_frame)
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame, new_path)

            folder = f"已保存未檢測到瑕疵圖片 {new_path}"
            self.show_message_box("未檢測到", folder, 3000)
        
        elif event.key() == Qt.Key_V:  # V for show camera video or AOI result
            if self.display_video == False:
                self.display_video = True
            else:
                self.display_video = False
                time.sleep(0.1)
                #self.show_image(self.worker.circle_image)
                self.match_done_to_show_image(self.camera_move_worker.current_position_index)
        else:
            super().keyPressEvent(event)

    def match_around_done(self):
        self.match_done_to_show_image(0)
        n = self.AOI_worker.detect_result[self.camera_move_worker.current_position_index][1]
            if n > 0:
                self.show_message_box.emit("檢測瑕疵", f"有瑕疵: {n} 個", -1)
            else:
                self.show_message_box.emit("檢測瑕疵", f"PASS", -1)


    def match_done_to_show_image(self,index):
        self.show_image(self.AOI_worker.detect_result[index][0])


    @pyqtSlot(str, str, int)
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
        if close_time > 0:
            QTimer.singleShot(close_time, close_msg)

    def get_move_folder(self , folder_name: str , fount: int) -> str:
        return os.path.join(folder_name, str(fount) + "_" + str(self.LT_300H_dev.cur_x) + "_" + str(self.LT_300H_dev.cur_y) + "_" + str(self.LT_300H_dev.cur_z) + "\\")

    def camera_setting(self):
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
                if not temp_cap.set(cv2.CAP_PROP_EXPOSURE, -6):
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    try:
        if win is not None:
            panel = win.control_panel
            win.showFullScreen()
            panel.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        None
        #win.closeEvent()