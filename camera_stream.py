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
from aoi_label import AOILabel

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
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

class AOIMatchWorker(QObject):
    match_done = pyqtSignal()
    message_box = pyqtSignal(str, str, int)
    def __init__(self, aoi_model):
        super().__init__()
        self.aoi_model = aoi_model

        self.aoi_rect = None
        self.threshold = None
        self.n_erode = None
        self.n_dilate = None
        self.min_samples = None
        # 新增：QWaitCondition 與 QMutex
        self._wait_cond = QWaitCondition()
        self._mutex = QMutex()
        self._should_run = False
        self._running = True  # 控制 while 迴圈

        #self.frame = None
        self.detect_frame = []
        self.goldens = []
        self.detect_result = []
        self.detect_index = -1
        self.detect_max_index = 4
        self.fount_back = 0
        
    def run(self):
        while self._running:
            self._mutex.lock()
            #if not self._should_run:
            self._wait_cond.wait(self._mutex)
            #self._should_run = False
            self._mutex.unlock()
            if not self._running:
                break
            if self.detect_index == -1 :
                self.detect_index = 0
                while self.detect_index != self.detect_max_index:
                    if len(self.detect_frame) <= self.detect_index:
                        time.sleep(0.1)
                        continue
                    self.match(-1)
            else:
                self.match(self.detect_index)
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
        return True
    
    def set_frame_and_goldens(self, frame ,folder_path):
        self.detect_frame.append(frame.copy())
        self.set_goldens_sample(folder_path)
        if self.goldens is None or self.aoi_rect is None:
            return False
        return True

    def set_frame(self, frame):
        self.detect_frame.append(frame.copy())

    def set_goldens_sample(self, folder_path, camera_move_worker):
        """根據 camera_move_worker 的路徑順序，載入所有樣本和 aoi_rect。"""
        all_goldens = []
        all_aoi_rects = []

        if not os.path.isdir(folder_path):
            self.message_box.emit("設定檢測樣本", f"資料夾不存在！\n{folder_path}", 3000)
            return
        if not camera_move_worker or not hasattr(camera_move_worker, 'position'):
            self.message_box.emit("設定檢測樣本", "CameraMoveWorker 不可用！", 3000)
            return

        # 依序讀取正面(0)和反面(1)
        for fount in range(2):
            for i in range(len(camera_move_worker.position)):
                goldens_in_subdir, aoi_rect = self.load_sample_for_index(folder_path, camera_move_worker, fount, i)
                all_goldens.append(goldens_in_subdir)
                all_aoi_rects.append(aoi_rect)

        self.detect_max_index = camera_move_worker.position
        self.goldens = all_goldens
        self.aoi_rect = all_aoi_rects

    def load_sample_for_index(self, folder_path, camera_move_worker, fount, index):
        """為指定的 fount 和 index 載入樣本和 aoi_rect。"""
        x, y, z = camera_move_worker.get_position(index)
        sub_dir_name = f"{fount}_{x}_{y}_{z}"
        sub_dir = os.path.join(folder_path, sub_dir_name)
        return self.sample_load(sub_dir)
    
    def sample_load(self,sub_dir):
        goldens_in_subdir = []
        bmp_files = get_file(sub_dir, extension='.bmp')
        if not bmp_files:
            print(f"{sub_dir} no bmp file, returning empty sample.")
            return goldens_in_subdir, None

        for img_path in bmp_files:
            golden_img = cv2.imread(img_path)
            kp, des = self.aoi_model.get_keypoint_grid(golden_img)
            goldens_in_subdir.append([golden_img, kp, des])
        
        aoi_rect = None
        inifiles = get_file(sub_dir, extension='.ini')
        if inifiles:
            try:
                config = configparser.ConfigParser()
                config.read(inifiles[0])
                if config.has_section('Rect'):
                    x_ini = int(config.get('Rect', 'x'))
                    y_ini = int(config.get('Rect', 'y'))
                    width = int(config.get('Rect', 'width'))
                    height = int(config.get('Rect', 'height'))
                    aoi_rect = [y_ini, y_ini + height, x_ini, x_ini + width]
            except Exception as e:
                self.message_box.emit("讀取 .ini 失敗", f"讀取 {inifiles[0]} 時發生錯誤: {e}", 3000)
        
        return goldens_in_subdir, aoi_rect

    def match(self , index = -1):
        #print(f"self.detect_index = {self.detect_index} self.goldens = {len(self.goldens)} len(self.detect_frame) {len(self.detect_frame)}")
        if self.detect_index >= len(self.detect_frame):
            return None , 0
        frame = self.detect_frame[self.detect_index].copy()

        goldens_index = self.detect_index if self.fount_back == 0 else self.detect_index + self.detect_max_index

        if frame is None or self.goldens[goldens_index] is None:
            return None , 0

        start_time = time.time()
        mask, mask_mean, mask_min, a , maskindex = self.aoi_model.match_template(frame, self.goldens[goldens_index], aoi=self.aoi_rect[goldens_index])
        circle_image , n  = self.match_result(mask_min, a)
        end_time = time.time()
        print(f"frame = {frame.shape} time = {end_time - start_time}  , index = {maskindex} , n = {n}")

        if index == -1:
            self.detect_index += 1
            self.detect_result.append((circle_image , n))
        else:
            self.detect_result[index] = (circle_image , n)

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
        self.multi_position_flag = True

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

            if self.multi_position_flag:
                self.move_to_next_position(0)

            while self.start_move:
                if self.multi_position_flag:
                    # 如果有外部指定 move_camera_get_frame_done_callback ，直接在 worker 執行
                    frame_copy = self.copy_frame()
                    if callable(self.move_camera_get_frame_done_callback):
                        try:
                            self.move_camera_get_frame_done_callback(frame_copy , -1 )
                        except TypeError:
                            print(f"move camera callback error: {e}")

                        try:
                            self.move_to_next_position()
                        except Exception as e:
                            print(f"move_to_next_position error: {e}")

                        if (self.LT_300H_dev.cur_x == self.position[0][0] and 
                            self.LT_300H_dev.cur_y == self.position[0][1]):
                            self.start_move = False
                else:
                    try:
                        self.move_to_next_position(self.current_position_index)
                        frame_copy = self.copy_frame()
                        self.move_camera_get_frame_done_callback(frame_copy , self.current_position_index)
                        self.start_move = False
                    except Exception as e:
                        print(f"move_to_next_position error: {e}")

                time.sleep(1)  # 避免 CPU 佔用過高

            self.camera_move_done.emit()

    def move_to_next_position(self, index=None):
        if not self.position:
            print("⚠️ 位置列表是空的，無法移動。")
            return

        if index is not None:
            if 0 <= index < len(self.position):
                self.current_position_index = index
                new_x, new_y = self.position[self.current_position_index]
            else:
                print(f"⚠️ Index {index} out of range")
                return
        else:
            # 取得下一個位置
            if self.current_position_index < len(self.position) -1 :
                self.current_position_index += 1
                new_x, new_y = self.position[self.current_position_index]
            else:
                self.current_position_index = 0
                new_x = self.LT_300H_dev.start_x
                new_y = self.LT_300H_dev.start_y

        # 回調函數
        def on_position_arrived(context):
            if context == "move_to_next_position":
                self.move_check_wait_cond.wakeOne()

        try:
            # 呼叫裝置移動，並傳遞回調函數
            self.LT_300H_dev.move_to(new_x, new_y, self.LT_300H_dev.start_z, 
                                     callback=lambda ctx: on_position_arrived(ctx),
                                     context="move_to_next_position")
            self.move_check_mutex.lock()
            self.move_check_wait_cond.wait(self.move_check_mutex)
            self.move_check_mutex.unlock()

        except Exception as e:
            print(f"move_to_next_position failed: {e}")

    def copy_frame(self):
        frame_copy = None
        while True:
            if self.camera_app.current_frame is not None:
                try:
                    frame_copy = self.camera_app.current_frame.copy()
                    break
                except Exception:
                    frame_copy = None

            if frame_copy is None or frame_copy.max() == 0:
                time.sleep(0.05)
                continue
        
        return frame_copy

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

    def get_position(self , index):
        new_x, new_y = self.position[index]
        return new_x, new_y, self.LT_300H_dev.start_z

    def close_device(self):
        self.LT_300H_dev.close()

class CameraApp(QWidget):
    message_box = pyqtSignal(str, str, int)
    def __init__(self):
        super().__init__()

        self.camera_setting()

        self.setWindowTitle("Camera Stream (Local UI)")
        
        self.find_contour = FindContour()
        self.find_contour_result = None

        self.display_video = True 
        self.current_frame = None

        self.image_label = AOILabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 讓 image_label 填滿
        self.image_label.video_width = self.video_width  # 傳遞給 AOILabel
        self.image_label.video_height = self.video_height
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

        self.show_detect_result_frame_flag = True
        self.show_detect_result_index = 0

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

        self.control_panel = ControlPanel(self)

        self.last_frame_filepath = [None] * 4

    def on_folder_changed(self, folder_name: str):
        """當控制面板的資料夾下拉選單變更時的回呼函式"""
        start_time = time.time()
        if not folder_name:
            print("資料夾選擇已清除。")
            self.current_folder = ""
            self.AOI_worker.goldens = [] # 清空樣本
            return
        self.current_folder = os.path.join(self.base_path, self.control_panel.folder_combo.currentText())
        self.AOI_worker.set_goldens_sample(self.current_folder , self.camera_move_worker)
        end_time = time.time()
        print(f"選擇的樣本資料夾已變更為: {self.current_folder} , time = {end_time-start_time}")
        self.message_box.emit("設定正樣本", f"正樣本已設定\n{self.current_folder}" , 3000)

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
            if frame.max() == 0:
                self.current_frame = None
                return
            frame = cv2.flip(frame, -1)
            self.current_frame = frame.copy()
            if self.display_video == True:
                display_frame = frame.copy()
                #need to fix Shane
                #if self.aoi_rect:
                #    t, b, l, r = self.aoi_rect
                #    cv2.rectangle(display_frame, (l, t), (r, b), (0, 0, 255), 4)
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
        self.message_box.emit("拍照完成", f"已儲存於：\n{save_path}" , -1)
        return save_path

    def camera_move_done_callback(self ,frame , index):
        today = "AOI_" + time.strftime("%Y%m%d")
        save_dir = self.get_move_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), today) , self.fount_back , self.camera_move_worker.current_position_index)
        self.last_frame_filepath[self.camera_move_worker.current_position_index] = self.snapshot_path(frame , save_dir)
        if index == -1:
            self.AOI_worker.detect_frame.append(frame.copy())
        else:
            if len(self.AOI_worker.detect_frame) <= index:
                print(f"index {index} out of range for AOI_worker.detect_frame")
                return
            self.AOI_worker.detect_frame[index] = frame.copy()
        self.AOI_worker.wake()

    def on_move_done(self):
        #if self.LT_300H_dev.check_current_position(self.LT_300H_dev.start_x , self.LT_300H_dev.start_y ,self.LT_300H_dev.start_z) != True:
        #    return  # 只在回到起點時顯示訊息
        if self.camera_move_worker.move_camera_get_frame_done_callback == None:
            self.message_box.emit("拍照完成", "正樣本拍照完成" , -1 )
        self.display_video = False
        #elif self.camera_move_worker.move_camera_get_frame_done_callback == self.camera_move_done_callback:
        #    self.message_box.emit("檢測完成", "PASS" , -1)

    def snapshot_positive_image(self):
        """啟動正樣本拍照 (在背景 thread 中執行)
        """
        start_time = time.time()
        positive_folder = (self.control_panel.folder_save_path 
                          if hasattr(self.control_panel, 'folder_save_path') 
                          else os.path.dirname(os.path.abspath(__file__)))
        
        if positive_folder is None:
            self.message_box.emit("正樣資料夾錯誤", f'找不到資料夾{positive_folder}', 3000)
            return

        front_samples = get_file(self.get_move_folder(positive_folder , 0 , 0), extension='.bmp')
        back_samples = get_file(self.get_move_folder(positive_folder , 1 , 0), extension='.bmp')

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
                    self.message_box.emit("判斷失敗", f'無法判斷正反面', 3000)
                    return
            else:
                self.fount_back = fount_back
        
        end_time = time.time()
        
        print(f"len(front_samples) = {len(front_samples)} len(back_samples) = {len(back_samples)} time = {end_time - start_time} ")

        def worker_callback(frame_copy):
            folder = self.get_move_folder(positive_folder , fount_back , self.camera_move_worker.current_position_index)
            os.makedirs(folder, exist_ok=True)
            try:
                self.snapshot_path(frame_copy, folder, message=False)
                self.detect_and_save_aoi_rect(frame_copy, folder)
            except Exception as e:
                print(f"snapshot save error: {e}")

        self.camera_move_worker.move_camera_get_frame_done_callback = worker_callback
        self.camera_move_worker.wake()

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

    def snapshot_current_view(self):
        """保存目前顯示的畫面 (F3)，包含縮放裁切"""
        if not hasattr(self, 'image_label') or self.image_label._pixmap is None:
            self.message_box.emit("截圖失敗", "目前沒有畫面可供儲存", 1000)
            return

        try:
            # 取得目前的 pixmap
            pixmap = self.image_label._pixmap
            scale = self.image_label.zoom_scale
            
            image_to_save = None
            
            if scale > 1.0:
                # 參照 AOILabel.paintEvent 的邏輯
                w = pixmap.width()
                h = pixmap.height()
                crop_w = int(w / scale)
                crop_h = int(h / scale)
                
                left = int(max(0, min(self.image_label.left, w - crop_w)))
                top = int(max(0, min(self.image_label.top, h - crop_h)))
                
                image_to_save = pixmap.copy(left, top, crop_w, crop_h)
            else:
                image_to_save = pixmap
            
            if image_to_save:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                today = time.strftime("%Y-%m-%d")
                folder_path = os.path.join(desktop_path, today)
                os.makedirs(folder_path, exist_ok=True)
                
                timestamp = time.strftime("%H%M%S")
                filename = f"snapshot_zoom_{timestamp}.bmp"
                save_path = os.path.join(folder_path, filename)
                
                image_to_save.save(save_path, "BMP")
                self.message_box.emit("截圖完成", f"已儲存於：\n{save_path}", 2000)
                
        except Exception as e:
            print(f"Snapshot error: {e}")
            self.message_box.emit("截圖失敗", f"儲存失敗: {e}", 3000)

    def closeEvent(self, event):
        # 停止 CameraMoveWorker 並關閉其設備
        if hasattr(self, 'camera_move_worker'):
            self.camera_move_worker.stop()
            self.camera_move_worker.close_device()
        if hasattr(self, 'camera_move_thread') and self.camera_move_thread.isRunning():
            self.camera_move_thread.quit()
            self.camera_move_thread.wait()

        self.cap.release()

        if hasattr(self, 'AOI_worker'):
            self.AOI_worker.stop()
        if hasattr(self, 'match_thread') and self.match_thread.isRunning():
            self.match_thread.quit()
            self.match_thread.wait()

        event.accept()

    def check_test_fount_back(self,folder_name: str) -> int:
        start_time = time.time()
        fount_sample = get_file(self.get_move_folder(folder_name , 0 , self.camera_move_worker.current_position_index) , extension='.bmp')
        back_sample = get_file(self.get_move_folder(folder_name , 1 , self.camera_move_worker.current_position_index) , extension='.bmp')
        fount_back = self.aoi_model.get_fount_back_sample(self.current_frame, fount_sample , back_sample)

        end_time = time.time()
        print(f"check_test_fount_back time = {end_time - start_time} s , {fount_back}")
        return fount_back

    def reset_aoi_value(self):
        self.show_detect_result_frame_flag = True
        self.show_detect_result_index += 1
        self.image_label.zoom_scale = 1.0

        threshold = self.control_panel.threshold_spin.value()
        n_erode = self.control_panel.erode_spin.value()
        n_dilate = self.control_panel.dilate_spin.value()
        min_samples = self.control_panel.min_samples_spin.value()
        self.AOI_worker.set_params(threshold, n_erode, n_dilate, min_samples)
        self.image_label.set_overlay_text("辨識中")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.control_panel.close()  # 新增：關閉 ControlPanel
            self.close()
        
        elif event.key() == Qt.Key_P:  # 新增：按下 P 鍵叫出 panel 並顯示在最上層
            self.control_panel.show()
            self.control_panel.raise_()
            self.control_panel.activateWindow()

        elif event.key() == Qt.Key_Enter:
            self.fount_back = self.check_test_fount_back(self.current_folder)

            if self.fount_back < 0 :
                self.message_box.emit("沒有樣本", f"{self.current_folder} 沒有樣本", 3000)
                return
            
            self.display_video = True
            self.camera_move_worker.multi_position_flag = True

            self.AOI_worker.detect_frame = []
            self.AOI_worker.detect_result = []
            self.AOI_worker.detect_index = -1
            self.AOI_worker.detect_max_index = 4
            self.AOI_worker.fount_back = self.fount_back 

            self.reset_aoi_value()
            self.camera_move_worker.move_camera_get_frame_done_callback = self.camera_move_done_callback
            self.camera_move_worker.wake()

        elif event.key() == Qt.Key_D: 
            if len(self.AOI_worker.detect_result) != self.AOI_worker.detect_max_index:
                return
            self.fount_back = self.check_test_fount_back(self.current_folder)

            if self.fount_back < 0 :
                self.message_box.emit("沒有樣本", f"{self.current_folder} 沒有樣本", 3000)
                return
            
            if len(self.AOI_worker.detect_frame) <= self.camera_move_worker.current_position_index:
                self.message_box.emit("沒有測試過", f"neet to fix by Shane", 3000)
                return

            self.display_video = True
            self.camera_move_worker.multi_position_flag = False
            self.camera_move_worker.current_position_index = self.show_detect_result_index

            self.AOI_worker.detect_index = self.show_detect_result_index
            self.AOI_worker.fount_back = self.fount_back 

            self.reset_aoi_value()

            self.camera_move_worker.move_camera_get_frame_done_callback = self.camera_move_done_callback
            self.camera_move_worker.wake()

        elif event.key() == Qt.Key_F1: # F1 新增正樣本
            folder_path = self.control_panel.folder_combo.currentText()
            folder = self.get_move_folder(os.path.join(self.base_path, folder_path) , self.fount_back , self.show_detect_result_index)
            os.makedirs(folder, exist_ok=True)
            if self.last_frame_filepath[self.show_detect_result_index] is None:
                return
            
            filename = os.path.basename(self.last_frame_filepath[self.show_detect_result_index])
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame_filepath[self.show_detect_result_index], new_path)
            message_text = f"已新增正樣本圖片到 {new_path}"
            self.message_box.emit("新增正樣本", message_text, 3000)

            goldens_index = self.show_detect_result_index if self.fount_back == 0 else self.AOI_worker.detect_max_index + self.show_detect_result_index
            self.AOI_worker.goldens[goldens_index], _ = self.AOI_worker.sample_load(folder)
        elif event.key() == Qt.Key_F2: # F2 負樣本
            folder_path = "nagetive_samples"
            folder = self.get_move_folder(os.path.join(self.base_path, folder_path) , self.fount_back , self.show_detect_result_index)
            os.makedirs(folder, exist_ok=True)
            if self.last_frame_filepath[self.show_detect_result_index] is None:
                return
            filename = os.path.basename(self.last_frame_filepath[self.show_detect_result_index])
            new_path = os.path.join(folder, filename)
            shutil.copy(self.last_frame_filepath[self.show_detect_result_index], new_path)
            folder = f"已保存未檢測到瑕疵圖片 {new_path}"
            self.message_box.emit("未檢測到", folder, 3000)
        elif event.key() == Qt.Key_F3: # F3 截圖
            self.snapshot_current_view()
        elif event.key() == Qt.Key_V:  # V for show camera video or AOI result
            if self.show_detect_result_frame_flag == False:
                self.show_detect_result_frame_flag = True
            else:
                self.show_detect_result_frame_flag = False
            self.match_done_to_show_image(self.show_detect_result_index)

        elif event.key() == Qt.Key_S:  # V for show camera video or AOI result
            if self.display_video == False:
                self.display_video = True
            else:
                self.display_video = False
                time.sleep(0.1)
                self.match_done_to_show_image(self.show_detect_result_index)
        elif event.key() == Qt.Key_F:
            if self.image_label.overlay_text == "":
                self.image_label.set_overlay_text(f"角度 {self.show_detect_result_index+1}")
            else:
                self.image_label.set_overlay_text("")
        
        elif event.key() == Qt.Key_Left:
            self.reset_display_frame_for_result()
            self.show_detect_result_index -= 1
            if self.show_detect_result_index < 0 :
                self.show_detect_result_index = len(self.AOI_worker.detect_result) - 1
            self.match_done_to_show_image(self.show_detect_result_index)

        elif event.key() == Qt.Key_Right:
            self.reset_display_frame_for_result()
            self.show_detect_result_index += 1
            if self.show_detect_result_index >= len(self.AOI_worker.detect_result) :
                self.show_detect_result_index = 0
            self.match_done_to_show_image(self.show_detect_result_index)

        elif event.key() in (Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            print(f"Number {event.key() - Qt.Key_0}")

        else:
            super().keyPressEvent(event)

    def match_around_done(self):
        self.reset_display_frame_for_result()
        self.show_detect_result_index = self.camera_move_worker.current_position_index
        self.match_done_to_show_image(self.show_detect_result_index)
        #self.image_label.set_overlay_text("")
        
    def reset_display_frame_for_result(self):
        self.image_label.zoom_scale = 1.0
        self.image_label.zoom_center = None
        self.show_detect_result_frame_flag = True

    def match_done_to_show_image(self,index):
        if index > len(self.AOI_worker.detect_result) - 1:
            return
        if self.AOI_worker.detect_result[index][0] is None:
            return
        self.display_video = False
        if self.show_detect_result_frame_flag:
            self.show_image(self.AOI_worker.detect_result[index][0])
            n = self.AOI_worker.detect_result[index][1]
            if n == 0:
                self.message_box.emit("檢測瑕疵", f"PASS", -1)
            else:
                self.message_box.emit("檢測瑕疵", f"有瑕疵: {n} 個", -1)
        else:
            t, b, l, r = self.AOI_worker.aoi_rect[index]
            self.show_image(self.AOI_worker.detect_frame[index][t:b,l:r])
        self.image_label.set_overlay_text(f"角度 {index+1}")

    @pyqtSlot(str, str, int)
    def show_message_box(self, title, text, close_time=5000, font_size=16, box_width=1000, box_height=400):
        if hasattr(self, 'msg') and self.msg is not None:
            self.msg.close()
            self.msg = None
        self.msg = QMessageBox()
        self.msg.setWindowTitle(title)
        self.msg.setText(text)
        self.msg.setStandardButtons(QMessageBox.Ok)
        self.msg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        
        # 調整文字大小
        font = QFont()
        font.setPointSize(font_size)
        self.msg.setFont(font)
        
        # 調整窗口大小
        self.msg.resize(box_width, box_height)
        
        self.msg.show()

        def close_msg():
            if self.msg is not None:
                self.msg.close()
        if close_time > 0:
            QTimer.singleShot(close_time, close_msg)

    def get_move_folder(self , folder_name: str , fount: int , index : int) -> str:
        x, y ,z= self.camera_move_worker.get_position(index)
        return os.path.join(folder_name, str(fount) + "_" + str(x) + "_" + str(y) + "_" + str(z) + "\\")

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