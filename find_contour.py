import cv2
import numpy as np


class FindContour:
    """輪廓檢測類，用於檢測圖像中的最大輪廓並計算最小外接矩形"""
    
    def __init__(self, blur_kernel_size=(5, 5), canny_low=50, canny_high=150):
        """
        初始化 FindContour 類
        
        Args:
            blur_kernel_size: 高斯模糊的核大小，預設為 (5, 5)
            canny_low: Canny 邊緣檢測的低閾值，預設為 50
            canny_high: Canny 邊緣檢測的高閾值，預設為 150
        """
        self.blur_kernel_size = blur_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high

    def _box_to_rect2(self, box, frame_shape):
        """
        將 box (4 頂點) 轉換為 [top, bottom, left, right]，並裁切到影像邊界
        box: numpy array shape (4,2) 各點為 (x,y)
        frame_shape: frame.shape (h, w, ...)
        返回整數列表 [top, bottom, left, right]
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])
        # 確保為整數陣列
        xs = box[:, 0].astype(int)
        ys = box[:, 1].astype(int)
        top = max(0, int(np.min(ys)))
        bottom = min(h - 1, int(np.max(ys)))
        left = max(0, int(np.min(xs)))
        right = min(w - 1, int(np.max(xs)))
        return [top, bottom, left, right]
    
    def detect(self, frame, draw_result=False, draw_color=(0, 255, 0), draw_thickness=2):
        """
        檢測 frame 中的最大輪廓並計算最小外接矩形
        返回包含 'rect2' = [top, bottom, left, right]
        """
        if frame is None:
            return {
                'contour': None,
                'box': None,
                'rect': None,
                'result_image': None,
                'success': False,
                'rect2': None
            }
        
        # 1. Gray + blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        
        # 2. Canny
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        # 3. Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'contour': None,
                'box': None,
                'rect': None,
                'result_image': frame.copy() if draw_result else None,
                'success': False,
                'rect2': None
            }
        
        # 找到面積最大的輪廓
        cnt = max(contours, key=cv2.contourArea)
        
        # 4. Rotated bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        # rect2: [top, bottom, left, right] 並裁切到影像邊界
        rect2 = self._box_to_rect2(box, frame.shape)

        # 繪製結果
        result_image = None
        if draw_result:
            result_image = frame.copy()
            cv2.drawContours(result_image, [box], 0, draw_color, draw_thickness)
        
        return {
            'contour': cnt,
            'box': box,
            'rect': rect,
            'rect2': rect2,
            'result_image': result_image,
            'success': True
        }
    
    def detect_by_color(self, frame, lower_bound , upper_bound , draw_result=False, draw_color=(0, 255, 0), draw_thickness=2, merge_all=True):
        """
        根據指定的顏色檢測區域並計算最小外接矩形
        merge_all: 若為 True，會將所有區域合併成一個大 box（忽略中間縫隙）
        """
        if frame is None:
            return {
                'mask': None,
                'contour': None,
                'box': None,
                'rect': None,
                'rect2': None,
                'result_image': None,
                'success': False
            }
        
        # 轉換為 HSV 色彩空間（更適合顏色檢測）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)

        # 創建遮罩
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 形態學操作：去除雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {
                'mask': mask,
                'contour': None,
                'box': None,
                'rect': None,
                'rect2': None,
                'result_image': frame.copy() if draw_result else None,
                'success': False
            }
        # 合併所有輪廓點
        if merge_all and len(contours) > 1:
            all_points = np.vstack(contours)
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            rect2 = self._box_to_rect2(box, frame.shape)
            cnt = all_points
        else:
            # 找到面積最大的輪廓
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            rect2 = self._box_to_rect2(box, frame.shape)

        # 繪製結果
        result_image = None
        if draw_result:
            result_image = frame.copy()
            cv2.drawContours(result_image, [box], 0, draw_color, draw_thickness)
        
        return {
            'mask': mask,
            'contour': cnt,
            'box': box,
            'rect': rect,
            'rect2': rect2,
            'result_image': result_image,
            'success': True
        }
