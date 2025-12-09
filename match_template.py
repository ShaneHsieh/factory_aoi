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
        pass
        
    def __call__(self, img1, goldens, aoi = None):
        mask, mask_mean, mask_min, a = self.match_template(img1, goldens, aoi = aoi)
        return mask, mask_mean, mask_min, a
    
    def get_keypoint(self,img, nfeatures = 500):
        orb = cv2.ORB_create(nfeatures) 
        kp, des = orb.detectAndCompute(img, None)
        return kp, des

    def get_fount_back_sample(self, camera_img , fount_files , back_files):
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

            # 檢查是否為有效值（非 NaN）
            front_valid = not np.isnan(avg_front) and avg_front > 2
            back_valid = not np.isnan(avg_back) and avg_back > 2

            print(f"avg_front {avg_front} , avg_back {avg_back} avg_front > avg_back {(not np.isnan(avg_front) and not np.isnan(avg_back)) and (avg_front > avg_back)}")

            if front_valid and back_valid:
                if avg_front > avg_back:
                    return 0
                else:
                    return 1
            elif front_valid:
                return 0
            elif back_valid:
                return 1
            else:
                return -1

        front_samples = [cv2.imread(f) for f in fount_files]
        back_samples = [cv2.imread(f) for f in back_files]

        result = classify_pcb_face(camera_img, front_samples, back_samples)
        return result


    def get_diff(self, img1, img2, kp1, kp2, des1, des2, aoi):
        index_params= dict( algorithm = 6,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 2) #2
        
        search_params = dict(checks = 100)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des2, des1, k=2)
        
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
        if c.shape != img1.shape:
            if aoi is not None:
                c = c[aoi[0]:aoi[1], aoi[2]:aoi[3]]
            else:
                c = cv2.warpPerspective(c, np.eye(3), img1.shape[:2][::-1])
        
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

        b = img1.copy()        
        a = (a-a.mean())*b.std()/a.std()+b.mean()
        c = (c-c.mean())*b.std()/c.std()+b.mean()

        diff_a = np.abs(a-b).sum(-1)
        diff_c = np.abs(c-b).sum(-1)

        if diff_a.sum()<diff_c.sum():
            return diff_a
        else:
            #print('no transform')
            return diff_c

    def match_template(self,img1, goldens, aoi = None ):

        kp1, des1 = self.get_keypoint(img1)

        diff = []

        with ThreadPoolExecutor(max_workers=len(goldens)) as executor:
            future_to_diff = []

            for img2, kp2, des2 in goldens:
                future = executor.submit(self.get_diff, img1, img2, kp1, kp2, des1, des2, aoi)
                future_to_diff.append(future)

            for future in as_completed(future_to_diff):
                diff.append(future.result())
                del future

            del future_to_diff 

        index = np.argmin(np.sum(diff, axis=(1,2)))

        return diff[index], np.mean(diff, axis=0), np.min(diff, axis=0), img1 , index

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
                res = cv2.circle(res, (int(x),int(y)), int((100*len(pp)/min_samples)**.6), (255, 0, 255), 2)
            else:
                n_size = n_size-1

        return res , n_size
