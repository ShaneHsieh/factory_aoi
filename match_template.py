from skimage.feature import match_template
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor, as_completed

class cv_aoi:
    def __init__(self):
        pass
        
    def __call__(self, img1, goldens, aoi = None):
        mask, mask_mean, mask_min, a = self.match_template(img1, goldens, aoi = aoi)
        return mask, mask_mean, mask_min, a
    
    def get_keypoint(self, img, nfeatures = 500):
        orb = cv2.ORB_create(nfeatures) 
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
    
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
    
    def get_diff(img1, img2, kp1, kp2, des1, des2, aoi):
        # img1 is test 
        # img2 is golden
        # golden 去做仿射變換
        # 去找 golden 在 test 的位置
        # 最後去看 diff 差異
    
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

        b = img1.copy()[ aoi[0]:aoi[1], aoi[2]:aoi[3] ]
        a = a[ aoi[0]:aoi[1], aoi[2]:aoi[3] ]
        c = c[ aoi[0] - golden_crop[0] : aoi[1] - golden_crop[1] , aoi[2] - golden_crop[2] : aoi[3] - golden_crop[3] ]

        a = (a-a.mean())*b.std()/a.std()+b.mean()
        c = (c-c.mean())*b.std()/c.std()+b.mean()

        diff_a = np.abs(a-b).sum(-1)
        diff_c = np.abs(c-b).sum(-1)

        if diff_a.sum()<diff_c.sum():
            return diff_a , b
        else:
            #print('no transform')
            return diff_c , b

    def match_template(self,img1, goldens, aoi = None ):

        kp1, des1 = self.get_keypoint(img1)

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
        #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        for i in n:
            if i>=0:
                pp = p[clustering.labels_==i]
                y,x = np.round(np.mean(pp, 0))
                res = cv2.circle(res, (int(x),int(y)), int((100*len(pp)/min_samples)**.6), (255, 0, 255), 2)

        return res , len(n)
