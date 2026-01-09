import sys
import os
import cv2
import time
import numpy as np

from LT_300H_control import LT300HControl
from aoi_label import AOILabel

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMessageBox
from pygrabber.dshow_graph import FilterGraph

class DeviceControl(AOILabel):
    def __init__(self):
        super().__init__()
        
        self.camera_setting()
        
        self.setWindowTitle("Device Control - Camera & Motion")
        
        # 初始化設備
        self.lt_300h_dev = LT300HControl(port="COM7", baudrate=115200, timeout=1.0)
        #self.lt_300h_dev.set_start_position(35, 48, 21)
        self.lt_300h_dev.set_move_speed(10)
        
        self.current_frame = None
        self.video_width = 3840
        self.video_height = 2160
        
        # 啟動攝像頭更新計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        
        self.showFullScreen()
    
    def camera_setting(self):
        """設定攝像頭（與 CameraApp 相同）"""
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

        if not hasattr(self, 'cap'):
            print("警告：SC0710 找不到")
            self.cap = None
        
        if temp_cap is not None:
            temp_cap.release()
    
    def update_frame(self):
        """更新攝像頭影像"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            if frame.max() == 0:
                self.current_frame = None
                return
            frame = cv2.flip(frame, -1)
            self.current_frame = frame.copy()
            try:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.setPixmap(QPixmap.fromImage(qt_image))
            except Exception as e:
                print(f"show_image error: {e}")
    
    def snapshot(self):
        """拍照並保存到桌面的今天日期資料夾"""
        if self.current_frame is None:
            self.show_message_box("未獲得影像", "無法取得攝像頭影像，請檢查連接", 3000)
            return
        try:
            # 獲取桌面路徑
            desktop_path = os.path.expanduser("~\\Desktop")
            
            # 建立以今天日期命名的資料夾
            today = time.strftime("%Y-%m-%d")
            folder_path = os.path.join(desktop_path, today)
            os.makedirs(folder_path, exist_ok=True)
            
            # 保存圖片
            timestamp = time.strftime("%H%M%S")
            filename = f"snapshot_{timestamp}.bmp"
            file_path = os.path.join(folder_path, filename)
            cv2.imwrite(file_path, self.current_frame)
            
            self.show_message_box("拍照完成", f"已保存於：\n{file_path}", 3000)
            print(f"已保存: {file_path}")
        except Exception as e:
            self.show_message_box("拍照失敗", f"拍照失敗：{str(e)}", 3000)
            print(f"拍照失敗: {e}")
    
    def show_message_box(self, title, text, close_time=5000, font_size=16, box_width=1000, box_height=400):
        """顯示消息框"""
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
    
    def keyPressEvent(self, event):
        """鍵盤按下事件"""
        if event.isAutoRepeat():  # 忽略自動重複事件
            return
        
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.lt_300h_dev.move_left()
        elif event.key() == Qt.Key_Right:
            self.lt_300h_dev.move_right()
        elif event.key() == Qt.Key_Up:
            self.lt_300h_dev.move_top()
        elif event.key() == Qt.Key_Down:
            self.lt_300h_dev.move_down()
        elif  event.key() == Qt.Key_Plus:
            self.lt_300h_dev.move_zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.lt_300h_dev.move_zoom_out()
        elif event.key() == Qt.Key_Enter:
            self.snapshot()
        elif event.key() == Qt.Key_G:
            self.lt_300h_dev.get_current_position()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """鍵盤釋放事件"""
        if event.isAutoRepeat():  # 忽略自動重複事件
            return
        
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_Right or event.key() == Qt.Key_Up or event.key() == Qt.Key_Down or event.key() == Qt.Key_Plus or event.key() == Qt.Key_Minus:
            self.lt_300h_dev.move_stop()
        else:
            super().keyReleaseEvent(event)
    
    def closeEvent(self, event):
        """關閉視窗"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        if self.lt_300h_dev is not None:
            self.lt_300h_dev.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControl()
    sys.exit(app.exec_())
