from flask import Flask, Response, render_template_string, request, redirect, url_for
import cv2
import time
import threading
import queue

app = Flask(__name__)

# 開啟預設攝影機（索引 0）
camera = cv2.VideoCapture(0)
latest_frame = None  # 儲存最新影格
frame_lock = threading.Lock()  # 保護 latest_frame

def camera_reader(stop_event):
    global latest_frame
    while not stop_event.is_set():
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()
                try:
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
        else:
            time.sleep(0.05)  # 若失敗則稍等

def generate_frames():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        # 只在這裡 encode
        ret, buffer = cv2.imencode('.jpg', frame)
        jpg_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        time.sleep(0.03)  # 控制 FPS

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>Camera Stream</title>
        </head>
        <body>
            <h1>Live Camera Feed</h1>
            <img src="/video_feed" width="1280" height="720"><br>
            <form action="/capture" method="post">
                <button type="submit">📸 拍照</button>
            </form>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global latest_frame
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
    return redirect(url_for('index'))

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
            print("Processing frame for defect detection...")
            # 在這裡做瑕疵檢測
            # result = your_defect_detection_function(frame)
            # 可以將結果存到某個地方或進行後續處理
        except queue.Empty:
            continue

if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    # 啟動 camera reader thread
    camera_thread = threading.Thread(target=camera_reader, args=(stop_event,))
    camera_thread.start()
    worker = threading.Thread(target=defect_detection_worker, args=(frame_queue, stop_event))
    worker.start()

    app.run(host='0.0.0.0', port=8889)

    stop_event.set()
    camera_thread.join()
    worker.join()
