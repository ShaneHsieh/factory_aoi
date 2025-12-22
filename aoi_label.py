import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
from PyQt5.QtCore import pyqtSignal, Qt


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
        self.video_width = 2560  # 預設值，CameraApp 會設置
        self.video_height = 1440
        # 新增 zoom 狀態
        self.zoom_scale = 1.0
        self.zoom_center = None  # (x, y) in image coordinates
        self.left = 0
        self.top = 0
        # 新增：平移狀態
        self.panning = False
        self.pan_start_pos = None
        # 新增：覆蓋層文字
        self.overlay_text = ""


    def resizeEvent(self, event):
        self.label_width = self.width()   # 更新 label 寬度
        self.label_height = self.height() # 更新 label 高度
        super().resizeEvent(event)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def set_overlay_text(self, text):
        """設定覆蓋層文字"""
        self.overlay_text = text
        self.update()

    def wheelEvent(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x, y = int(pos.x()), int(pos.y())
        delta = event.angleDelta().y()
        #print(f"wheelEvent {x} , {y} {delta}")
        # 呼叫 zoom_at
        self.zoom_at(x, y, delta)
        self.update()

    def zoom_at(self, x, y, delta):
        if self._pixmap is None:
            return
    
        # 1. 將 widget 座標轉換為原始圖片座標
        img_x, img_y = self.widget_to_image_coords(x, y)
        if img_x is None:
            return
    
        # 2. 計算新縮放比例
        factor = 1.25 if delta > 0 else 0.8
        new_scale = self.zoom_scale * factor
        new_scale = max(1.0, min(new_scale, 20.0))
    
        if new_scale == self.zoom_scale:
            return
    
        # 3. 更新 left/top，讓滑鼠指向的點在縮放後位置不變
        if new_scale > 1.0:
            # 新的 left/top = 點的座標 - (點到邊界的距離 / 新縮放比例)
            self.left = int(img_x - (x / self.width()) * (self._pixmap.width() / new_scale))
            self.top = int(img_y - (y / self.height()) * (self._pixmap.height() / new_scale))
        else:
            # 回到初始狀態
            self.left = 0
            self.top = 0
    
        self.zoom_scale = new_scale
        self.update()

    def widget_to_image_coords(self, x, y):
        """將 widget 上的點 (x, y) 轉換為原始圖片的座標"""
        if self._pixmap is None:
            return None, None
    
        widget_w, widget_h = self.width(), self.height()
        original_pixmap_w, original_pixmap_h = self._pixmap.width(), self._pixmap.height()

        # 計算當前顯示的 pixmap 尺寸（考慮了 zoom_scale）
        current_pixmap_w = original_pixmap_w / self.zoom_scale
        current_pixmap_h = original_pixmap_h / self.zoom_scale

        # 計算該 pixmap 在 widget 中為了保持長寬比而縮放後的尺寸和偏移
        scale = min(widget_w / current_pixmap_w, widget_h / current_pixmap_h)
        scaled_w = int(current_pixmap_w * scale)
        scaled_h = int(current_pixmap_h * scale)
        offset_x = (widget_w - scaled_w) // 2
        offset_y = (widget_h - scaled_h) // 2

        # 如果點擊在黑邊上，則不進行任何操作
        if not (offset_x <= x < offset_x + scaled_w and offset_y <= y < offset_y + scaled_h):
            return None, None

        # 將 widget 座標轉換為相對於 scaled_pixmap 的座標
        scaled_x = x - offset_x
        scaled_y = y - offset_y

        # 將 scaled_pixmap 座標轉換為原始圖片座標
        img_x = self.left + (scaled_x / scaled_w) * current_pixmap_w
        img_y = self.top + (scaled_y / scaled_h) * current_pixmap_h
        return img_x, img_y


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._pixmap:
            widget_w, widget_h = self.width(), self.height()
            pixmap_w, pixmap_h = self._pixmap.width(), self._pixmap.height()
            
            if self.zoom_scale > 1.0:
                # --- 縮放模式 ---
                crop_w = int(pixmap_w / self.zoom_scale)
                crop_h = int(pixmap_h / self.zoom_scale)
    
                # 確保裁切框不超過圖片邊界
                self.left = max(0, min(self.left, pixmap_w - crop_w))
                self.top = max(0, min(self.top, pixmap_h - crop_h))
                
                # 再次確保 left/top 不為負
                self.left = max(0, self.left)
                self.top = max(0, self.top)

                cropped = self._pixmap.copy(self.left, self.top, crop_w, crop_h)
                scaled_pixmap = cropped.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # 繪製時置中 (如果 scaled_pixmap 沒有填滿 widget)
                x = (widget_w - scaled_pixmap.width()) // 2
                y = (widget_h - scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, scaled_pixmap)
            else:
                # --- 正常顯示模式 ---
                self.zoom_scale = 1.0
                #self.zoom_center = None
                self.left = 0
                self.top = 0
                scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
                new_w, new_h = int(pixmap_w * scale), int(pixmap_h * scale)
                scaled_pixmap = self._pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (widget_w - scaled_pixmap.width()) // 2
                y = (widget_h - scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, scaled_pixmap)
        
        # 繪製覆蓋層文字（在圖像之上）
        if self.overlay_text:
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))  # 白色文字
            
            # 設定文字矩形（頂部置中）
            text_height = 60
            text_rect = self.rect().adjusted(0, 20, 0, -self.height() + text_height + 20)
            
            # 繪製背景（半透明黑色）
            painter.fillRect(text_rect, QColor(0, 0, 0, 180))
            
            # 繪製文字
            painter.drawText(text_rect, Qt.AlignCenter | Qt.AlignVCenter, self.overlay_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.zoom_scale > 1.0:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.pan_start_pos is not None:
            delta = event.pos() - self.pan_start_pos
            
            # 將 widget 上的移動量轉換為原始圖片上的移動量
            if self._pixmap:
                pixmap_w = self._pixmap.width()
                pixmap_h = self._pixmap.height()
                
                scale_x = (pixmap_w / self.zoom_scale) / self.width()
                scale_y = (pixmap_h / self.zoom_scale) / self.height()
                
                self.left -= delta.x() * scale_x
                self.top -= delta.y() * scale_y
                
                self.pan_start_pos = event.pos()
                self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
