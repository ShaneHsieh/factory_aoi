from skimage.feature import match_template
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor, as_completed

class cv_aoi:
    def __init__(self):
        # 強制啟用 OpenCL 以支援 Intel GPU 加速
        cv2.ocl.setUseOpenCL(True)
        
    def __call__(self, img1, goldens, aoi = None):
        mask, mask_mean, mask_min, a = self.match_template(img1, goldens, aoi = aoi)
        return mask, mask_mean, mask_min, a
    
    def get_keypoint(self, img, nfeatures = 500):
        orb = cv2.ORB_create(nfeatures) 
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
    
    def get_front_back_sample(self, camera_img , front_files , back_files):
        orb_sample = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def compute_score(img1, img2):
            """
            計算兩張影像的特徵匹配分數
            回傳：match 數量（越高越相似）
            """
            # 灰階
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # ORB keypoints + descriptors
            kp1, des1 = orb_sample.detectAndCompute(g1, None)
            kp2, des2 = orb_sample.detectAndCompute(g2, None)

            if des1 is None or des2 is None:
                return 0

            matches = bf.match(des1, des2)
            # 用較短的前 N 個 matches 代表相似度（避免背景雜訊）
            matches = sorted(matches, key=lambda x: x.distance)
            N = min(len(matches), 100)

            score = sum(1.0 / (m.distance + 1e-6) for m in matches[:N])
            return score

        def classify_pcb_face(frame, front_samples, back_samples):
            """
            return:
            "front" 或 "back"
            """

            # 計算與每張 sample 的匹配分數
            front_scores = [compute_score(frame, fs) for fs in front_samples]
            back_scores = [compute_score(frame, bs) for bs in back_samples]

            avg_front = np.mean(front_scores)
            avg_back = np.mean(back_scores)

            print(f"front_scores: {front_scores}")
            print(f"back_scores: {back_scores}")

            # 檢查是否為有效值（非 NaN）
            #front_valid = not np.isnan(avg_front) and avg_front > 2
            #back_valid = not np.isnan(avg_back) and avg_back > 2

            print(f"avg_front {avg_front} , avg_back {avg_back} avg_front > avg_back {(not np.isnan(avg_front) and not np.isnan(avg_back)) and (avg_front > avg_back)}")
            print(f"front {np.max(front_scores)} back {np.max(back_scores)} ")


            if np.isnan(avg_back) and np.isnan(avg_front):
                return -1
            elif np.isnan(avg_back) and any(x > 4 for x in front_scores):
                return 0
            elif np.isnan(avg_front) and any(x > 4 for x in back_scores):
                return 1
            elif avg_front > avg_back:
                return 0
            else:
                return 1
             

        front_samples = [cv2.imread(f) for f in front_files]
        back_samples = [cv2.imread(f) for f in back_files]

        result = classify_pcb_face(camera_img, front_samples, back_samples)
        return result
    
    def get_keypoint_grid(self, img, nfeatures=6000, grid=(4,4)):
        h, w = img.shape[:2]
        gh, gw = grid
        orb = cv2.ORB_create(int(nfeatures/(gh*gw)))

        all_kp = []
        all_des = []

        for r in range(gh):
            for c in range(gw):
                y1 = int(r * h / gh)
                y2 = int((r+1) * h / gh)
                x1 = int(c * w / gw)
                x2 = int((c+1) * w / gw)

                tile = img[y1:y2, x1:x2]
                kp, des = orb.detectAndCompute(tile, None)
                if kp is not None:
                    # 把座標從區塊位置轉回全圖座標
                    for k in kp:
                        k.pt = (k.pt[0] + x1, k.pt[1] + y1)
                    all_kp.extend(kp)
                    if des is not None:
                        all_des.append(des)

        if len(all_des) > 0:
            all_des = np.vstack(all_des)
        else:
            all_des = None

        return all_kp, all_des

    def get_diff(self,img1, img2, kp1, kp2, des1, des2, aoi):
        # img1 is test 
        # img2 is golden
        # golden 去做仿射變換
        # 去找 golden 在 test 的位置
        # 最後去看 diff 差異
    

        start_time = time.time()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des2, des1, k=2)
        
        good = []
        for m in matches:
            if len(m)>1:
                m, n = m
                if m.distance < 0.7*n.distance:
                    #print(m.distance,n.distance)
                    good.append(m)
        #print(len(good))
        MIN_MATCH_COUNT = 10
        c = img2.copy()

        golden_crop = round(aoi[0]/2), round(aoi[1] + (c.shape[0] - aoi[1]) / 2), round(aoi[2] / 2), round(aoi[3] + (c.shape[1] - aoi[3]) / 2)

        #if c.shape != img1.shape:
        if aoi is not None:
            c = c[ golden_crop[0] : golden_crop[1], golden_crop[2] : golden_crop[3]]
        else:
            c = cv2.warpPerspective(c, np.eye(3), img1.shape[:2][::-1])

        
        start_time2 = time.time()

        if len(good)>MIN_MATCH_COUNT:
            try:
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                if M is None:
                    print("M is None")
                a = cv2.warpPerspective(img2, M, img1.shape[:2][::-1])
            except:
                a = c
        else:
            a = c
        
        start_time3 = time.time()

        b = img1.copy()[ aoi[0]:aoi[1], aoi[2]:aoi[3] ]
        a = a[ aoi[0]:aoi[1], aoi[2]:aoi[3] ]
        c = c[ aoi[0] - golden_crop[0] : aoi[1] - golden_crop[1] , aoi[2] - golden_crop[2] : aoi[3] - golden_crop[3] ]

        # --- Intel GPU (OpenCL) 加速區段 ---
        if cv2.ocl.haveOpenCL():
            try:
                # 定義計算全域 mean/std 的輔助函式 (符合 numpy 行為)
                def get_stats_umat(u):
                    m, s = cv2.meanStdDev(u)
                    if hasattr(m, 'get'):
                        m = m.get()
                    if hasattr(s, 'get'):
                        s = s.get()
                    m = m.flatten()
                    s = s.flatten()
                    g_m = np.mean(m)
                    # Global variance = mean(sigma^2 + mu^2) - global_mu^2
                    g_v = np.mean(s**2 + m**2) - g_m**2
                    g_s = np.sqrt(max(0, g_v))
                    return g_m, g_s

                # 將 numpy array 轉為 UMat (自動使用 GPU 記憶體)
                # 使用 float32 進行運算以保持精度
                b_u = cv2.UMat(b)
                # 使用 addWeighted 替代 convertTo 進行型別轉換 (uint8 -> float32)
                b_f = cv2.addWeighted(b_u, 1.0, b_u, 0, 0.0, dtype=cv2.CV_32F)
                mean_b, std_b = get_stats_umat(b_u)

                def normalize_and_diff(img_np, mean_target, std_target, img_target_f):
                    img_u = cv2.UMat(img_np)
                    mean_src, std_src = get_stats_umat(img_u)
                    
                    scale = 1.0
                    offset = 0.0
                    if std_src > 1e-6:
                        scale = std_target / std_src
                        offset = mean_target - mean_src * scale
                    
                    # 轉換為 float32 並同時應用線性變換 (alpha=scale, beta=offset)
                    # 使用 addWeighted 替代 convertTo，同時完成線性變換與型別轉換
                    img_f = cv2.addWeighted(img_u, scale, img_u, 0, offset, dtype=cv2.CV_32F)
                    
                    # 計算差異: abs(img - target)
                    diff_f = cv2.absdiff(img_f, img_target_f)
                    # 加總 Channel: (H,W,3) -> (H,W,1)
                    kernel = np.ones((1, 3), dtype=np.float32)
                    diff_sum = cv2.transform(diff_f, kernel)
                    return diff_sum

                diff_a_u = normalize_and_diff(a, mean_b, std_b, b_f)
                diff_c_u = normalize_and_diff(c, mean_b, std_b, b_f)

                # 比較總差異
                sum_a = cv2.sumElems(diff_a_u)[0]
                sum_c = cv2.sumElems(diff_c_u)[0]

                best_diff_u = diff_a_u if sum_a < sum_c else diff_c_u
                # 轉回 CPU numpy array 並移除最後一個維度 (H,W,1) -> (H,W)
                best_diff_np = best_diff_u.get()
                return best_diff_np.reshape(best_diff_np.shape[:2]), b
            except cv2.error as e:
                print(f"OpenCL error: {e}, falling back to CPU")
        
        # --- 原始 CPU 邏輯 (Fallback) ---
        else:
            a = (a-a.mean())*b.std()/a.std()+b.mean()
            c = (c-c.mean())*b.std()/c.std()+b.mean()

            diff_a = np.abs(a-b).sum(-1)
            diff_c = np.abs(c-b).sum(-1)

            
            end_time = time.time()

            print(f"get diff match time2 = {start_time2 - start_time} time3 = {start_time3 - start_time2} end = {end_time - start_time3} ")
            
            print(f"get diff pre time = {end_time - start_time} ")

            if diff_a.sum()<diff_c.sum():
                return diff_a , b
            else:
                return diff_c , b

    def match_template(self,img1, goldens, aoi = None ):

        kp1, des1 = self.get_keypoint_grid(img1)

        diff = []
        image_results = []

        with ThreadPoolExecutor(max_workers=len(goldens)) as executor:
            future_to_diff = []

            for img2, kp2, des2 in goldens:
                future = executor.submit(self.get_diff, img1, img2, kp1, kp2, des1, des2, aoi)
                future_to_diff.append(future)

            for future in as_completed(future_to_diff):
                future_data = future.result()
                diff.append(future_data[0])
                image_results.append(future_data[1])
                del future

            del future_to_diff 

        index = np.argmin(np.sum(diff, axis=(1,2)))

        return diff[index], np.mean(diff, axis=0), np.min(diff, axis=0), image_results[index] , index

    def post_proc(self,mask,  threshold = 25, n_erode = 1, n_dilate = 1):
        kernel = np.ones((3,3), np.uint8)
        mask = ((mask>threshold)*255).astype('uint8')
        mask = cv2.erode(mask, kernel, iterations = n_erode)
        mask = cv2.dilate(mask, kernel, iterations = n_dilate)
        return mask

    def draw_circle(self,mask, res, eps = 10, min_samples=50):
        
        if np.sum(mask)==0:
            return res , 0

        p = np.stack(np.where(mask)).T
        if len(p) > 10000:
            return res , -1
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(p)
        n,c = np.unique(clustering.labels_, return_counts=1)
        n_size = len(n)
        #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        for i in n:
            if i>=0:
                pp = p[clustering.labels_==i]
                y,x = np.round(np.mean(pp, 0))
                res = cv2.circle(res, (int(x),int(y)), int((100*len(pp)/min_samples)**.8), (255, 0, 255), 4)
            else:
                n_size = n_size-1

        return res , n_size
