
import os
import glob
import cv2
from stitch import stitch_multiple_images_fast,stitch_multiple_images
from postprocess import dilate_and_save
from utils import plot_image

folder_path = "data/images/Image Mosaicing"
# image_paths = [os.path.join(folder_path,file_name) for file_name in ["1_2.jpg", "1_4.jpg", "1_3.jpg", "1_1.jpg"]]
# img_mosaic = stitch_multiple_images_fast(image_paths, prefix="set1")
# cv2.imwrite("results/mosaic_set1.jpg", img_mosaic)

# plot_image(dilate_and_save("results/mosaic_set1.jpg"))


# image_paths = glob.glob("data/images/Image Mosaicing/2_*.*")
# img_mosaic = stitch_multiple_images(image_paths, prefix="set2", max_iter=5000, threshold=2)
# cv2.imwrite("results/mosaic_set2.jpg", img_mosaic)

# image_paths = [os.path.join(folder_path,file_name) for file_name in ["3_1.png", "3_2.png"]]
# img_mosaic = stitch_multiple_images_fast(image_paths, prefix="set3", max_iter=5000, threshold=2)
# cv2.imwrite("results/mosaic_set3.jpg", img_mosaic)
plot_image(dilate_and_save("results/mosaic_set3.jpg"))
