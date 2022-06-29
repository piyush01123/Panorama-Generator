
import os
import glob
import cv2
from stitch import stitch_multiple_images_fast

folder_path = "data/images/Image Mosaicing"
image_paths = [os.path.join(folder_path,file_name) for file_name in ["1_2.jpg", "1_4.jpg", "1_3.jpg", "1_1.jpg"]]
img_mosaic = stitch_multiple_images_fast(image_paths, prefix="set1")
cv2.imwrite("results/mosaic_set1.jpg", img_mosaic)