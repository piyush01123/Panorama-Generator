
import cv2
import numpy as np


def get_points_and_features(img):
    """
    Returns keypoints and features obtained using SIFT from OpenCV.
    Requires opencv-python==4.5.1.48 for `SIFT_create` API. 
    For earlier versions the API might be slightly different

    Arguments:
    ----------
    img: numpy array of shape (H,W,3) having the image data 

    Returns:
    --------
    key_points: numpy array of shape (m,2) denoting the X,Y coordinates of key points
    features: numpy array of shape (m,128) denoting the SIFT features of key points
    (m is the number of detected keypoints)
    """
    sift = cv2.SIFT_create()
    key_points, features = sift.detectAndCompute(img.astype(np.uint8),None)
    return key_points, features

def get_matches( key_points_A, features_A, key_points_B, features_B, test_ratio=0.75):
    """
    Performs feature matching between a pair of key points in 2 steps. In step 1, it 
    finds the 2 best matches for each key point in set 1 from the key points in set 2 where 
    best matches means the ones with the smallest distance between their features. Then in step 2, 
    we apply Lowe's ratio test to check that the two distances are different by at least 
    a factor. If not, we discard that match and return the remaining matches.
    Reference: https://stackoverflow.com/a/60343973/9563006    

    Arguments
    ---------
    key_points_A: numpy array of shape (m1,2) denoting the key points in image A
    features_A: numpy array of shape (m1,128) denoting the features in image A
    key_points_B: numpy array of shape (m2,2) denoting the key points in image B
    features_B: numpy array of shape (m2,128) denoting the features in image B
    test_ratio [float]: The fraction of distances used in Lowe's ratio test

    Returns
    -------
    good_points_A: numpy array of shape (m,2) denoting chosen key points in image A
    good_points_B: numpy array of shape (m,2) denoting chosen key points in image B
    num_points [int]: Number of good matches (m)
    """
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(features_A, features_B, k=2)    
    good_matches_A, good_matches_B = [], []
    for m,n in matches:
        if m.distance < n.distance * test_ratio:
            good_matches_A.append(m.queryIdx)
            good_matches_B.append(m.trainIdx)

    all_points_A = np.array([kp.pt for kp in key_points_A]).astype(int)
    all_points_B = np.array([kp.pt for kp in key_points_B]).astype(int)

    good_points_A = all_points_A[good_matches_A]
    good_points_B = all_points_B[good_matches_B]
    assert good_points_A.shape == good_points_B.shape
    return good_points_A, good_points_B, len(good_points_A)

def estimate_homography(points_A, points_B):
    """
    Estimates the homography matrix H using DLT method.

    Arguments
    ---------
    points_A: numpy array of shape (m,2) denoting key points in image A used in DLT to estimate H
    points_B: numpy array of shape (m,2) denoting key points in image B used in DLT to estimate H

    Returns
    -------
    H:  numpy array of shape (3,3) denoting the homography matrix H
    Note that the H matrix is normalized such that last row last column element of H becomes 1. 
    """
    assert points_A.shape == points_B.shape
    G = np.zeros((len(points_A)*2, 9))
    for i, (pt_img_A, pt_img_B) in enumerate(zip(points_A, points_B)):
        x1,y1 = pt_img_A
        x2, y2 = pt_img_B
        G[2*i] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        G[2*i+1] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
    U, D, V_T = np.linalg.svd(G)
#     assert np.allclose(G, U[:,:9]@np.diag(D)@V_T, atol=1e-5)
    H = V_T[-1].reshape(3,3)
    H = H/H[-1,-1]
    return H    

def get_projected_points(points_A, points_B, H):
    """
    Projects points in image A on image B and points in image B on image A using homography matrix H.

    Arguments
    ---------
    points_A: numpy array of shape (m,2) denoting some points in image A that need to projected on image B
    points_B: numpy array of shape (m,2) denoting some points in image B that need to projected on image A
    H:  numpy array of shape (3,3) denoting the homography matrix H

    Returns
    -------
    points_A_proj_on_img_B: numpy array of shape (m,2) denoting projections of points_A on image B
    points_B_proj_on_img_A: numpy array of shape (m,2) denoting projections of points_B on image A
    """
    ones = np.ones((len(points_A),1))
    points_A = np.concatenate([points_A, ones], axis=1)
    points_B = np.concatenate([points_B, ones], axis=1)   
    
    points_A_proj_on_img_B = points_A @ H.T
    points_A_proj_on_img_B = points_A_proj_on_img_B / points_A_proj_on_img_B[:,-1].reshape(-1,1)

    H_inv = np.linalg.inv(H)
    points_B_proj_on_img_A = points_B @ H_inv.T
    points_B_proj_on_img_A = points_B_proj_on_img_A / points_B_proj_on_img_A[:,-1].reshape(-1,1)

    points_A_proj_on_img_B = points_A_proj_on_img_B[:,:2]
    points_B_proj_on_img_A = points_B_proj_on_img_A[:,:2]

    return points_A_proj_on_img_B, points_B_proj_on_img_A
    

def projection_error(points_A, points_B, points_A_proj_on_img_B, points_B_proj_on_img_A):
    """
    Calculates projection error given some points in images A and B and their projections in images B and A respectively.

    Arguments
    ---------
    points_A: numpy array of shape (m,2) denoting some points in image A that need to projected on image B
    points_B: numpy array of shape (m,2) denoting some points in image B that need to projected on image A
    points_A_proj_on_img_B: numpy array of shape (m,2) denoting projections of points_A on image B
    points_B_proj_on_img_A: numpy array of shape (m,2) denoting projections of points_B on image A

    Returns
    -------
    sum_errors [float]: Sum of L2 norms of errors of all point pairs
    errors:  numpy array (1-D) of shape (m,) denoting L2 norms of errors of each point pairs summed over 
             both directions A to B and B to A
    """
    error_A = np.linalg.norm(points_A_proj_on_img_B-points_B, axis=1)
    error_B = np.linalg.norm(points_B_proj_on_img_A-points_A, axis=1)
    errors = error_A + error_B
    return errors.sum(), errors

def estimate_homography_RANSAC(points_A, points_B, max_iter=5000, threshold=50, points_to_sample=4, fraction_inliers=0.99):
    """
    Estimates homography matrix H using RANSAC + DLT.

    Arguments
    ---------
    points_A: numpy array of shape (m,2) denoting key points in image A
    points_B: numpy array of shape (m,2) denoting key points in image B
    max_iter[int]: Maximum number of iterations to run RANSAC
    threshold[int]: Number of pixels that the projections need to be within to be considered as an inlier
    points_to_sample[int]: Number of points being sampled in each iteration of RANSAC to run DLT (Should be at least 4)
    fraction_inliers[float]: Fraction of inliers used as a termination criterion to stop RANSAC

    Returns
    -------
    H_best:  numpy array of shape (3,3) denoting the best estimated homography matrix H 
             (Here best means having the maximum number of inliers)
    max_inliers[int]: Maximum number of inliers that the RANSAC was able to find while running RANSAC
    min_recon_error[float]: Total reconstruction error value in the  iteration where H_best is chosen
    pfinalA: numpy array of shape (points_to_sample,2) denoting the points in image A that gave H_best
    pfinalB: numpy array of shape (points_to_sample,2) denoting the points in image A that gave H_best
    """
    assert points_A.shape == points_B.shape, "Points in A and B need to have the same shape."
    assert points_to_sample>=4, "Homography estimation needs at least 4 points"
    num_points = len(points_A)
    assert points_to_sample<=num_points, "Cannot sample more points than what is provided"

    H_best = np.random.randn(3,3)
    min_recon_error = np.inf
    curr_max_inliers = 0

    for iter in range(max_iter):
        indices_selected = np.random.choice(num_points, points_to_sample, replace=False)
        points_A_selected = points_A[indices_selected]
        points_B_selected = points_B[indices_selected]
        H = estimate_homography(points_A_selected, points_B_selected)

        points_A_proj_on_img_B, points_B_proj_on_img_A = get_projected_points(points_A, points_B, H)
        total_error, errors = projection_error(points_A, points_B, points_A_proj_on_img_B, points_B_proj_on_img_A)
        num_inliers = np.sum(errors<threshold)

        if iter%10==0:
            # print("Iteration: {} # Inliers: {}".format(iter, num_inliers))
            pass


        if num_inliers > curr_max_inliers:
            curr_max_inliers = num_inliers
            H_best = H
            min_recon_error = total_error
            pfinalA, pfinalB = points_A_selected, points_B_selected
        if num_inliers > fraction_inliers*num_points:
            break
    return H_best, curr_max_inliers, min_recon_error, pfinalA,  pfinalB
