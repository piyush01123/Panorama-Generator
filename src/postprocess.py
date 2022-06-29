
import cv2, numpy as np, os

def dilate_and_save(fp):
    """Post processing step to remove lines obtained after warping."""
    assert os.path.isfile(fp)
    fp_dash, ext = os.path.splitext(fp)
    img = cv2.imread(fp)
    kernel = np.ones((5,5),np.uint8)
    dilated_img = cv2.dilate(img,kernel,iterations = 1)
    fp_dilated = "{}_dilated{}".format(fp_dash,ext)
    cv2.imwrite(fp_dilated, dilated_img)
    return fp_dilated

