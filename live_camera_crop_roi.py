# import os
# import cv2

# class CameraApp():
#     def __init__(self):
#         super().__init__()
        
#         self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         self.display_video = True 
#         self.current_frame = None

import cv2

from pygrabber.dshow_graph import FilterGraph

class NetworkCameraApp():
    def __init__(self, camera_url):
        self.display_video = True
        graph = FilterGraph()
        self.video_width = 3840
        self.video_height = 2160
        for i, name in enumerate(graph.get_input_devices()):
            if "SC0710 PCI, Video 01 Capture" in name:
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width):
                    print("Can not set frame width to", self.video_width)
                if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height):
                    print("Can not set frame height to", self.video_height)
            elif "NeuroEye" in name:
                temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not temp_cap.set(cv2.CAP_PROP_EXPOSURE, -8):
                    print("Can not set exposure to -8")
                if not temp_cap.set(cv2.CAP_PROP_AUTO_WB, 0):
                    print("Can not set auto white balance to 0")
                if not temp_cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000):
                    print("Can not set white balance blue U to 4000")
                if not temp_cap.set(cv2.CAP_PROP_SETTINGS, 1):
                    print("Can not open camera settings")
                if not temp_cap.set(cv2.CAP_PROP_BRIGHTNESS, 100):
                    print("Can not set brightness to 10")
                if not temp_cap.set(cv2.CAP_PROP_CONTRAST, 100):
                    print("Can not set contrast to 100")
                if not temp_cap.set(cv2.CAP_PROP_SHARPNESS, 0):
                    print("Can not set sharpness to 0")

                temp_cap.release()


        self.roi_points = []
        self.frame = None

        cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)  # 可調整視窗大小


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.roi_points) < 2:
                self.roi_points.append((x, y))
                print(f"點擊座標: {x}, {y}")


    def run(self):
        cv2.namedWindow('Camera')
        cv2.setMouseCallback('Camera', self.mouse_callback)
        while self.display_video:
            ret, frame = self.cap.read()
            if not ret:
                print("無法取得影像")
                break
            #print("frame shape:", frame.shape)
            cv2.imshow('Camera', frame)
            if len(self.roi_points) == 2:
                x1, y1 = self.roi_points[0]
                x2, y2 = self.roi_points[1]
                self.frame = frame.copy()
                roi = self.frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

                if roi.size > 0:
                    #roi_big = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('ROI', roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.roi_points = []
                #cv2.destroyWindow('ROI')
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_url = 2  # 例如 rtsp://xxx 或 http://xxx
    app = NetworkCameraApp(camera_url)
    app.run()