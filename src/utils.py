
import cv2, numpy as np, matplotlib.pyplot as plt, os, glob


def draw_common_points(img_A, img_B, points_used_A, points_used_B):
    """
    Given two images and the coordinates of the matching points, returns an image 
    having both images side by side and lines between each pair of matching points 
    from image A to image B

    Arguments
    ---------
    img_A: numpy array of shape (H_A,W_A,3) having the image data of image A
    img_B: numpy array of shape (H_B,W_B,3) having the image data of image B
    points_used_A: numpy array of shape (m,2) denoting the coordinates of points in image A
    points_used_B: numpy array of shape (m,2) denoting the coordinates of points in image B

    Returns
    -------
    line_img: numpy array of shape (max(H_A+H_B), W_A+W_B, 3) having the required image
    """
    h1,w1,_ = img_A.shape
    h2,w2,_ = img_B.shape
    line_img = np.zeros((max(h1,h2),w1+w2,3),dtype=np.float32)
    line_img[0:h1,0:w1,:] = img_A
    line_img[0:h2,w1:w1+w2,:] = img_B
    for pfa,pfb in zip(points_used_A, points_used_B):
        x1,y1=pfa
        x2,y2=pfb
        line_img = cv2.line(line_img, (x1,y1), (x2+w1,y2), (0,0,225), 5)
        line_img = cv2.circle(line_img, (x1,y1),8,(0,0,255),-1)
        line_img = cv2.circle(line_img, (x2+w1,y2),8,(0,0,255),-1)
    return line_img

def plot_image(fp):
    """Utility function to plot an image in notebook"""
    assert os.path.isfile(fp)
    img = cv2.imread(fp)
    plt.figure(figsize=(20,16))
    plt.imshow(img[:,:,[2,1,0]])
    plt.axis("off")

def plot_multiple_images(file_paths):
    """Utility function to plot multiple images vertically stacked"""
    assert all([os.path.isfile(fp) for fp in file_paths])
    n = len(file_paths)
    plt.figure(figsize=(20,16))

    for idx in range(n):
        ax = plt.subplot(n,1,idx+1)
        fp = file_paths[idx]
        img = cv2.imread(fp)
        ax.imshow(img[:,:,[2,1,0]])
        plt.axis("off")

