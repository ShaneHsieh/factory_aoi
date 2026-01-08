import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen
from PyQt5.QtCore import pyqtSignal, Qt, QRect


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
        
        # Mask and Drawing
        self.mask = None
        self._mask_pixmap = None
        self.draw_mode = False
        self.drawing_rect = False
        self.rect_start = None
        self.rect_end = None
        
        # 座標轉換用的快取
        self.displayed_rect = None
        self.source_rect = None

    def resizeEvent(self, event):
        self.label_width = self.width()   # 更新 label 寬度
        self.label_height = self.height() # 更新 label 高度
        super().resizeEvent(event)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def set_mask(self, mask):
        self.mask = mask
        self.update_mask_pixmap()
        self.update()

    def update_mask_pixmap(self):
        if self.mask is None:
            self._mask_pixmap = None
            return
        
        h, w = self.mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        # Mask == 0 的區域顯示為半透明灰色 (R=128, G=128, B=128, A=100)
        overlay[self.mask == 0] = [64, 64, 64, 100]
        
        qimg = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
        self._mask_pixmap = QPixmap.fromImage(qimg.copy())

    def set_draw_mode(self, enabled):
        self.draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def save_mask(self, file_path):
        if self.mask is not None:
            # 將 mask 轉換為 0 和 255 (二值化)，以便下次讀取
            binary_mask = (self.mask > 0).astype(np.uint8) * 255
            cv2.imwrite(file_path, binary_mask)

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

    def widget_to_image_coords_clamped(self, x, y):
        """將 widget 座標轉換為影像座標，並限制在影像範圍內"""
        if self._pixmap is None or not hasattr(self, 'displayed_rect') or self.displayed_rect is None:
            return 0, 0
        
        # 限制 x, y 在顯示區域內
        # displayed_rect.right() 是 left + width - 1，所以要加 1 才能包含邊界
        dx = max(self.displayed_rect.left(), min(x, self.displayed_rect.left() + self.displayed_rect.width()))
        dy = max(self.displayed_rect.top(), min(y, self.displayed_rect.top() + self.displayed_rect.height()))
        
        if self.displayed_rect.width() == 0 or self.displayed_rect.height() == 0:
            return 0, 0

        # 計算相對位置 (0.0 ~ 1.0)
        rel_x = (dx - self.displayed_rect.left()) / self.displayed_rect.width()
        rel_y = (dy - self.displayed_rect.top()) / self.displayed_rect.height()

        # 映射回來源影像座標
        img_x = self.source_rect.left() + rel_x * self.source_rect.width()
        img_y = self.source_rect.top() + rel_y * self.source_rect.height()
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
                
                # 確保為整數，避免 copy 報錯
                self.left = int(self.left)
                self.top = int(self.top)

                # 計算目標繪製區域 (維持長寬比)
                scale = min(widget_w / crop_w, widget_h / crop_h)
                dest_w = int(crop_w * scale)
                dest_h = int(crop_h * scale)
                dest_x = (widget_w - dest_w) // 2
                dest_y = (widget_h - dest_h) // 2
                
                target_rect = QRect(dest_x, dest_y, dest_w, dest_h)
                source_rect = QRect(self.left, self.top, crop_w, crop_h)
                
                # 儲存顯示區域與來源區域，供座標轉換使用
                self.displayed_rect = target_rect
                self.source_rect = QRect(self.left, self.top, crop_w, crop_h)
                
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                painter.drawPixmap(target_rect, self._pixmap, source_rect)
                
                if self._mask_pixmap:
                    painter.drawPixmap(target_rect, self._mask_pixmap, source_rect)
            else:
                # --- 正常顯示模式 ---
                self.zoom_scale = 1.0
                #self.zoom_center = None
                self.left = 0
                self.top = 0
                scale = min(widget_w / pixmap_w, widget_h / pixmap_h)
                new_w, new_h = int(pixmap_w * scale), int(pixmap_h * scale)

                x = (widget_w - new_w) // 2
                y = (widget_h - new_h) // 2
                
                target_rect = QRect(x, y, new_w, new_h)
                source_rect = QRect(0, 0, pixmap_w, pixmap_h)
                
                # 儲存顯示區域與來源區域，供座標轉換使用
                self.displayed_rect = target_rect
                self.source_rect = source_rect
                
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                painter.drawPixmap(target_rect, self._pixmap, source_rect)
                
                if self._mask_pixmap:
                    painter.drawPixmap(target_rect, self._mask_pixmap, source_rect)
        
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

        if self.drawing_rect and self.rect_start and self.rect_end:
            painter.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            rect = QRect(self.rect_start, self.rect_end)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self.draw_mode and (event.button() == Qt.LeftButton or event.button() == Qt.RightButton):
            self.drawing_rect = True
            self.rect_start = event.pos()
            self.rect_end = event.pos()
            self.update()
        elif event.button() == Qt.RightButton and self.zoom_scale > 1.0:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_rect:
            self.rect_end = event.pos()
            self.update()
        elif self.panning and self.pan_start_pos is not None:
            delta = event.pos() - self.pan_start_pos
            
            # 將 widget 上的移動量轉換為原始圖片上的移動量
            if self._pixmap:
                pixmap_w = self._pixmap.width()
                pixmap_h = self._pixmap.height()
                
                # 計算裁切區域大小
                crop_w = pixmap_w / self.zoom_scale
                crop_h = pixmap_h / self.zoom_scale
                
                # 計算實際顯示的縮放比例 (考慮 KeepAspectRatio)
                scale = min(self.width() / crop_w, self.height() / crop_h)
                
                if scale > 0:
                    self.left -= delta.x() / scale
                    self.top -= delta.y() / scale
                    
                    self.pan_start_pos = event.pos()
                    self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_rect :
            self.drawing_rect = False
            self.rect_end = event.pos()
            if event.button() == Qt.LeftButton:
                self.apply_mask_rect(0)
            elif event.button() == Qt.RightButton:
                self.apply_mask_rect(1)
            self.update()
        elif event.button() == Qt.RightButton:
            self.panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def apply_mask_rect(self, value=0):
        print("apply_mask_rect called")
        if self.rect_start is None or self.rect_end is None or self._pixmap is None:
            return

        if self.mask is None:
            print("No existing mask, creating new one")
            self.mask = np.ones((self.video_height, self.video_width), dtype=np.uint8)
            #self.update_mask_pixmap()
        print("Current mask shape: ", self.mask.shape)
        #x1, y1 = self.widget_to_image_coords_clamped(self.rect_start.x(), self.rect_start.y())
        #x2, y2 = self.widget_to_image_coords_clamped(self.rect_end.x(), self.rect_end.y())

        x1, y1 = self.widget_to_image_coords(self.rect_start.x(), self.rect_start.y())
        x2, y2 = self.widget_to_image_coords(self.rect_end.x(), self.rect_end.y())

        ix1, ix2 = sorted([int(x1), int(x2)])
        iy1, iy2 = sorted([int(y1), int(y2)])

        h, w = self.mask.shape
        ix1 = max(0, min(ix1, w))
        ix2 = max(0, min(ix2, w))
        iy1 = max(0, min(iy1, h))
        iy2 = max(0, min(iy2, h))

        self.mask[iy1:iy2, ix1:ix2] = value
        self.update_mask_pixmap()
