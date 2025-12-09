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
    
    def detect(self, frame, draw_result=False, draw_color=(0, 255, 0), draw_thickness=2):
        """
        檢測 frame 中的最大輪廓並計算最小外接矩形
        
        Args:
            frame: 輸入的圖像 frame (numpy array, BGR 格式)
            draw_result: 是否在結果圖像上繪製邊界框，預設為 False
            draw_color: 繪製邊界框的顏色 (B, G, R)，預設為綠色 (0, 255, 0)
            draw_thickness: 繪製邊界框的線條粗細，預設為 2
        
        Returns:
            dict: 包含以下鍵值的字典
                - 'contour': 最大輪廓 (numpy array)
                - 'box': 最小外接矩形的四個頂點座標 (numpy array, shape: (4, 2))
                - 'rect': 最小外接矩形的資訊 ((中心x, 中心y), (寬, 高), 旋轉角度)
                - 'result_image': 繪製結果的圖像 (如果 draw_result=True，否則為 None)
                - 'success': 是否成功檢測到輪廓 (bool)
        """
        if frame is None:
            return {
                'contour': None,
                'box': None,
                'rect': None,
                'result_image': None,
                'success': False
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
                'success': False
            }
        
        # 找到面積最大的輪廓
        cnt = max(contours, key=cv2.contourArea)
        
        # 4. Rotated bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 繪製結果
        result_image = None
        if draw_result:
            result_image = frame.copy()
            cv2.drawContours(result_image, [box], 0, draw_color, draw_thickness)
        
        return {
            'contour': cnt,
            'box': box,
            'rect': rect,
            'result_image': result_image,
            'success': True
        }
