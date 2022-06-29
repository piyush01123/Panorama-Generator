
import cv2
import numpy as np
import os
from homography import get_points_and_features, get_matches, estimate_homography_RANSAC
from utils import draw_common_points


def get_projected_points_on_img_A(points_B, H):
    """
    Projects points in image B on image A using homography matrix H.
    IMP: This funcion is the same as get_projected_points except it does the projection only one way.

    Arguments
    ---------
    points_B: numpy array of shape (m,2) denoting some points in image B that need to projected on image A
    H:  numpy array of shape (3,3) denoting the homography matrix H

    Returns
    -------
    points_B_proj_on_img_A: numpy array of shape (m,2) denoting projections of points_B on image A
                            Also we are casting these from float to int before returning.
    """
    ones = np.ones((len(points_B),1))
    points_B = np.concatenate([points_B, ones], axis=1)   
    
    H_inv = np.linalg.inv(H)
    points_B_proj_on_img_A = points_B @ H_inv.T
    points_B_proj_on_img_A = points_B_proj_on_img_A / points_B_proj_on_img_A[:,-1].reshape(-1,1)

    points_B_proj_on_img_A = points_B_proj_on_img_A[:,:2]

    return points_B_proj_on_img_A.astype(int)
    

def stitch(img_A, img_B, H):
    """
    Stitches img_B on img_A using homography matrix H. The images can be of different sizes.
    Algorithm:
    1. Find the coordinates of the four corners of image B and find their projections on image A using H.
    2. Find the expected height (H') and width (W') of the stitched image using the coordinates of corners of 
       image A and the projected corners of image B found in step 1.
    3. Find the offset at which the original image A must start in the stitched image. This will be non zero
       if one of projected corners of image B has a negative coordinate.
    4. Copy paste image A on the stitched image starting from the offset point
    5. Find the projection of each point in image B on image A using H and find the coordinate of that point 
       in the stitched image using the offset.

    Arguments
    ---------
    img_A: numpy array of shape (H_A,W_A,3) having the image data of image A
    img_B: numpy array of shape (H_B,W_B,3) having the image data of image B
    H:  numpy array of shape (3,3) denoting the homography matrix H

    Returns
    -------
    img_new: numpy array of shape (H',W',3) having the image data of the stitched image
    """
    h1,w1,_ = img_A.shape
    h2,w2,_ = img_B.shape

    # bounds = np.array([[0,0],[0,h2],[w2,h2],[w2,0]])
    bounds = np.array([[0,0],[h2,0],[h2,w2],[0,w2]])
    bounds_proj = get_projected_points_on_img_A(bounds[:,[1,0]],H)[:,[1,0]]
    # print("Bounds: \n{} \nBounds projected: \n{}".format(bounds,bounds_proj))
    (hmin,wmin),(hmax,wmax) = bounds_proj.min(0),bounds_proj.max(0)

    h_new, w_new = max(hmax,h1)+max(0,-hmin), max(wmax,w1)+max(0,-wmin)
    print("New Image size: H=",h_new, ", W=",w_new)

    img_new = np.zeros((h_new, w_new,3),dtype=np.float32)
    # img_new = np.full((h_new, w_new,3),255,dtype=np.float32)
    h_a,w_a = 0+max(-hmin,0),0+max(-wmin,0)
    img_new[h_a:h_a+h1,w_a:w_a+w1,:] = img_A


    hgrid,wgrid=np.meshgrid(range(h2),range(w2))
    points = np.vstack([hgrid.ravel(),wgrid.ravel()]).T 
    points_proj = get_projected_points_on_img_A(points[:,[1,0]],H) [:,[1,0]]
    for pt,pt_proj in zip(points,points_proj):
        (py,px),(py_proj,px_proj) = pt, pt_proj
        py_proj_fin = min(max(0,h_a+py_proj),h_new-1)
        px_proj_fin = min(max(0,w_a+px_proj),w_new-1)
        img_new[py_proj_fin,px_proj_fin] = img_B[py,px]

    return img_new

def stitch_pipeline(img_A, img_B, **kwargs):
    """
    Runs the entire stitching pipeline to stitch img_B on img_A. The images can be of different sizes.
    Steps:
    1. Find points and features of both images
    2. Find best matching key points that are common in both images
    3. Run RANSAC to estimate homography H.
    4. Use H to stitch image B on image A.

    Arguments
    ---------
    img_A: numpy array of shape (H_A,W_A,3) having the image data of image A
    img_B: numpy array of shape (H_B,W_B,3) having the image data of image B
    kwargs: keyword arguments for estimate_homography_RANSAC function

    Returns
    -------
    img_pano: Stitched image
    img_common_pts_used: Image containing both images side by side and line between the best set of points
                         that the RANSAC found to estimate homography
    pt_comb: All good matches between images A and B
    """
    hA,wA,_ = img_A.shape
    points_A_all, features_A_all = get_points_and_features(img_A)
    points_B_all, features_B_all = get_points_and_features(img_B)
    points_A, points_B, num_points = get_matches(points_A_all, features_A_all, points_B_all, features_B_all)
    print("Points matched: ", num_points)

    H, max_inliers, error, pfinalA, pfinalB = estimate_homography_RANSAC(points_A, points_B, **kwargs)
    print("Homography estimated:\n{}\nFraction of Inliers from RANSAC:{}".format(H,max_inliers/num_points))

    img_common_pts_used = draw_common_points(img_A, img_B, pfinalA, pfinalB)
    img_pano = stitch(img_A,img_B,H)
    return img_pano, img_common_pts_used

def find_best_img_idx_to_stitch(img_A, images):
    """
    Given image A and a list of images, returns the index of best image to stitch
    based on the number of points matched

    Arguments
    ---------
    img_A: numpy array of shape (H_A,W_A,3) having the image data of image A
    images: list of images of varying sizes. Each entity is a numpy array of shape (h,w,3)

    Returns
    -------
    idx[int]: Index of image in `images` with maximum number of matching points with img_A
    """
    p1, f1 = get_points_and_features(img_A)
    num_points_matches = []
    for img in images:
        p2, f2 = get_points_and_features(img)
        points_A, points_B, num_points = get_matches(p1, f1, p2, f2)
        num_points_matches.append(num_points)
    idx = max(range(len(images)), key=lambda i: num_points_matches[i])
    return idx

def stitch_multiple_images(image_paths, prefix, **kwargs):
    """
    Given a list of image paths of images of varying sizes in any permutation, 
    stitches them to produce a single mosaic image

    Arguments
    ---------
    image_paths (List[str]): list of image paths
    kwargs: keyword arguments for estimate_homography_RANSAC function
    prefix[str]: Prefix used in file names of intermediate step images [Useful for debugging]

    Returns
    -------
    Mosaic image: numpy array of shape (H',W',3)
    """
    assert all([os.path.isfile(fp) for fp in image_paths]), "One or more files missing."
    image_paths = sorted(image_paths)

    images = [cv2.imread(fp) for fp in image_paths]
    ctr = 0
    while len(images)>1:
        img_A = images[0]
        img_B_idx = find_best_img_idx_to_stitch(img_A, images[1:])
        img_B = images[1+img_B_idx]
        img_A,img_pt = stitch_pipeline(img_A, img_B, **kwargs)
        cv2.imwrite("../images/output/mosaicing/{}_common_pts_step_{}.jpg".format(prefix, ctr), img_A)
        cv2.imwrite("../images/output/mosaicing/{}_mosaic_step_{}.jpg".format(prefix, ctr), img_pt)
        images.pop(1+img_B_idx); images.pop(0); images.insert(0, img_A)
        ctr += 1
    return img_A

def stitch_multiple_images_fast(image_paths, prefix, **kwargs):
    """
    Given a list of image paths of images of varying sizes, stitches them to produce 
    a single mosaic. May not produce very good result with all random orderings but 
    it is much faster than the above order agnostic function.
    Gives good result when given in an order such that the center image is passed first.

    Arguments
    ---------
    image_paths (List[str]): list of image paths
    kwargs: keyword arguments for estimate_homography_RANSAC function
    prefix[str]: Prefix used in file names of intermediate step images [Useful for debugging]

    Returns
    -------
    Mosaic image: numpy array of shape (H',W',3)
    """
    assert all([os.path.isfile(fp) for fp in image_paths]), "One or more files missing."

    images = [cv2.imread(fp) for fp in image_paths]
    img_A = images[0]
    for ctr, img_B in enumerate(images[1:]):
        img_A, img_common_pts_used = stitch_pipeline(img_A, img_B, **kwargs)
        cv2.imwrite("../images/output/mosaicing/{}_common_pts_step_{}.jpg".format(prefix, ctr), img_common_pts_used)
        cv2.imwrite("../images/output/mosaicing/{}_mosaic_step_{}.jpg".format(prefix, ctr), img_A)
    return img_A

