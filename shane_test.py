from match_template import cv_aoi
import cv2
import numpy as np

def main():
    aoi_model = cv_aoi()
    # Load the image
    img1 = cv2.imread('unname_16.bmp')
    img1_GRAY = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    golden_img = cv2.imread('unname_25.bmp')

    goldens = []
    for img in [golden_img, golden_img, golden_img, golden_img, golden_img]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = aoi_model.get_keypoint(img)
        goldens.append([img, kp, des])

    mask, M, a = aoi_model(img1_GRAY, goldens , threshold=25, n_erode=1, n_dilate=1, aoi=[500, 1700, 600, 3000] )


    res = (np.stack([np.maximum(mask,a),a*(mask==0),a*(mask==0)], axis=-1))

    # Display the results
    cv2.imshow('Mask', mask)

    cv2.imshow('res', res)
    
    cv2.imshow('output', aoi_model.draw_circle(mask, res, min_samples=50))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()