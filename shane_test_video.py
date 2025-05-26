from match_template import cv_aoi
import cv2
import numpy as np
import threading
import queue

def defect_detection_worker(frame_queue, stop_event):

    #aoi_model = cv_aoi()

    # img1 = cv2.imread('unname_16.bmp')
    # img1_GRAY = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # golden_img = cv2.imread('unname_25.bmp')

    # goldens = []
    # for img in [golden_img, golden_img, golden_img, golden_img, golden_img]:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     kp, des = aoi_model.get_keypoint(img)
    #     goldens.append([img, kp, des])

    # mask, M, a = aoi_model(img1_GRAY, goldens , threshold=25, n_erode=1, n_dilate=1, aoi=[500, 1700, 600, 3000] )


    # res = (np.stack([np.maximum(mask,a),a*(mask==0),a*(mask==0)], axis=-1))

    # # Display the results
    # cv2.imshow('Mask', mask)

    # cv2.imshow('res', res)
    
    # cv2.imshow('output', aoi_model.draw_circle(mask, res, min_samples=50))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            # 在這裡做瑕疵檢測
            # result = your_defect_detection_function(frame)
            # 可以將結果存到某個地方或進行後續處理
        except queue.Empty:
            continue

def main():

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video.")
        return

    width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))

    print("width: ", width, " height: ", height, " fps: ", fps)

    # fps = 60
    # width = 1920 
    # height = 1080
    # camera.set(cv2.CAP_PROP_FPS, fps)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)    
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 

    sleeptime = 1000/fps
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter( str(icol)+ '.mp4',fourcc, fps, (frame2.shape[1],frame.shape[0]+roi[3]+200))

    grabbed, frame = camera.read()

    if frame is None:
        print("can not open video")
        #return
    #count = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    worker = threading.Thread(target=defect_detection_worker, args=(frame_queue, stop_event))
    worker.start()

    while not (frame is None):    
        cv2.imshow('frame', frame)
        # 將 frame 放入 queue，若 queue 滿則略過
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass
        grabbed, frame = camera.read() 
        if cv2.waitKey(int(sleeptime)) & 0xFF == 27:
            break
        
    print("video end")
    stop_event.set()
    worker.join()
    #out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()