from skimage.feature import match_template
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from sklearn.cluster import DBSCAN

class cv_aoi:
    def __init__(self):
        pass
        
    def __call__(self, img1, goldens,   threshold = 25,
                                        n_erode = 1,
                                        n_dilate = 1,
                                        aoi = [0, -1, 0, -1]):
        mask, M, a = self.match_template(img1, goldens,
                                         threshold = threshold,
                                         n_erode = n_erode,
                                         n_dilate = n_dilate,
                                         aoi = aoi)
        return mask, M, a
    
    def get_keypoint(self,img, nfeatures = 500):
        orb = cv2.ORB_create(nfeatures) 
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
    
    def diff_compare(self, img1, img2, threshold = 25, n_erode = 1, n_dilate = 1, aoi = [0, -1, 0, -1]):
        a = img1.copy()
        b = img2.copy()
        a = a[aoi[0]:aoi[1], aoi[2]:aoi[3]]
        b = b[aoi[0]:aoi[1], aoi[2]:aoi[3]]
        b = (b-b.mean())*a.std()/b.std()+a.mean()
        mask = ((np.abs(a-b)>threshold)*255).astype('uint8')

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = n_erode)
        mask = cv2.dilate(mask, kernel, iterations = n_dilate)

        return mask , a

    def match_template(self,img1, goldens
                                    , threshold = 25
                                    , n_erode = 1
                                    , n_dilate = 1
                                    , aoi = [0, -1, 0, -1] ):

        kp1, des1 = self.get_keypoint(img1)

        index_params= dict(algorithm = 6,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 2) #2
        search_params = dict(checks = 100)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        masks = []

        for img2, kp2, des2 in goldens:

            matches = flann.knnMatch(des1, des2, k=2)

            good = []
            for m in matches:
                if len(m)>1:
                    m, n = m
                    if m.distance < 0.7*n.distance:
                        good.append(m)

            MIN_MATCH_COUNT = 10

            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            else:
                print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                M = None

            kernel = np.ones((3,3), np.uint8)

            if M is not None:
                a = cv2.warpPerspective(img1, M, img2.shape[:2][::-1])
                mask , a= self.diff_compare(a, img2, threshold, n_erode, n_dilate, aoi)

                masks.append([mask, M, a])
            
            mask , a= self.diff_compare(img1, img2, threshold, n_erode, n_dilate, aoi)
            masks.append([mask, None, a])

        idx = np.argmin([np.sum(mask) for mask, _, _ in masks])

        return masks[idx]

    def draw_circle(self,mask, res, eps = 10, min_samples=50):
        if np.sum(mask)==0:
            return res

        p = np.stack(np.where(mask)).T
        #print(len(p))

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(p)
        n,c = np.unique(clustering.labels_, return_counts=1)

        for i in n:
            if i>=0:
                pp = p[clustering.labels_==i]
                y,x = np.round(np.mean(pp, 0))
                res = cv2.circle(res, (int(x),int(y)), int((100*len(pp)/min_samples)**.6), (255, 255, 0), 2)

        return res
