from typing import Tuple
import numpy as np
import random

import utils


def q1_a(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a least squares plane by taking the Eigen values and vectors
    of the sample covariance matrix

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''

    # get mean value of the Nx3 matrix
    center = P.mean(0)
    # get covariance matrix
    conv_mat = np.cov(P.T)
    # update normal with the smallest eigenvector
    w, v = np.linalg.eig(conv_mat)
    normal = v[:, w.argmin()]

    return normal, center


def ransac_plane(p, threshold=0.02, iterations=1000):
    inliers = []
    normal = []
    i = 1

    while i < iterations:
        idx_samples = random.sample(range(len(p)), 3)
        points = p[idx_samples]
        normal = np.cross(points[1] - points[0], points[2] - points[0])

        # get distance from surface to point
        a, b, c = normal / np.linalg.norm(normal)
        d = -np.sum(normal * points[1])
        distance = (a * p[:, 0] + b * p[:, 1] + c * p[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]

        if len(idx_candidates) > len(inliers):
            normal = [a, b, c]
            inliers = idx_candidates

        i += 1
    return normal, inliers


def q1_c(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a plane using RANSAC

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''

    normal, inliner = ransac_plane(P)
    p_list = P[inliner]
    center = p_list.mean(0)

    return normal, center



def q2(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Localize a sphere in the point cloud. Given a point cloud as
    input, this function should locate the position and radius
    of a sphere

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting sphere center
    radius : float
        scalar radius of sphere
    '''
    i = 1
    iterations = 500
    center = []
    radius = 0
    threshold = 0.02
    inliers = np.empty(0)

    while i < iterations:
        p_size, _ = P.shape
        # sample a point from point cloud
        random_index = random.randint(0, p_size - 1)
        point = P[random_index]
        # sample a radius in [5cm, 11cm] --> [0.05, 0.11]
        tmp_radius = random.uniform(0.05, 0.11)
        # get the center of the sphere
        tmp_center = point + N[random_index] * radius

        # update the inlier
        dist = np.array([np.linalg.norm(tmp_center - p) for p in P])
        idx_candidates = np.array(np.where(np.logical_and(np.abs(dist) >= tmp_radius - threshold,
                                                          np.abs(dist) <= tmp_radius + threshold)))

        if idx_candidates.size > inliers.size:
            center = tmp_center
            radius = tmp_radius
            inliers = idx_candidates

        i += 1

    return center, radius


def q3(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Localize a cylinder in the point cloud. Given a point cloud as
    input, this function should locate the position, orientation,
    and radius of the cylinder

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting 100 points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting cylinder center
    axis : np.ndarray
        array of shape (3,) pointing along cylinder axis
    radius : float
        scalar radius of cylinder
    '''
    i = 1
    iterations = 5000
    center = []
    axis = []
    radius = 0
    threshold = 0.01
    inliers = np.empty(0)

    while i < iterations:
        p_size, _ = P.shape

        # sample a radius in [5cm, 11cm] --> [0.05, 0.11]
        tmp_radius = random.uniform(0.05, 0.11)
        # sample two points and calculate the axis direction
        idx_samples = random.sample(range(p_size), 2)
        tmp_axis = np.cross(N[idx_samples[0]], N[idx_samples[1]])

        # sample a point and calculate the center
        point = P[idx_samples[0]]
        tmp_center = point + N[idx_samples[0]] * radius

        # project the points onto the orthogonal plane
        tmp_projection = np.identity(3) - np.outer(tmp_axis, np.transpose(tmp_axis))
        tmp_point_proj = np.array([np.matmul(tmp_projection, p) for p in P])
        tmp_center_proj = np.matmul(tmp_projection, tmp_center)

        # update the inliers
        dist = np.array([np.linalg.norm(tmp_p_proj - tmp_center_proj) for tmp_p_proj in tmp_point_proj])
        idx_candidates = np.array(np.where(np.logical_and(np.abs(dist) >= tmp_radius - threshold,
                                                          np.abs(dist) <= tmp_radius + threshold)))

        if idx_candidates.size > inliers.size:
            center = tmp_center_proj
            axis = tmp_axis
            radius = tmp_radius
            inliers = idx_candidates

        i += 1

    return center, axis, radius


def q4_a(M: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Find transformation T such that D = T @ M. This assumes that M and D are
    corresponding (i.e. M[i] and D[i] correspond to same point)

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    use `np.linalg.svd` to perform singular value decomposition
    '''

    m, n = M.shape
    # initialize a homogeneous transformation
    T = np.identity(n + 1)

    M_mean = M.mean(0)
    D_mean = D.mean(0)
    m_hat = np.subtract(M, M_mean)
    d_hat = np.subtract(D, D_mean)

    # get s, v, d
    W = np.array([np.outer(m_hat[i], d_hat[i].T) for i in range(m)])
    W = np.sum(W, axis=0)
    U, s, Vt = np.linalg.svd(W)

    # rotation matrix
    R = np.matmul(U, Vt).T
    # translation vector
    t = D_mean.T - np.matmul(R, M_mean.T)
    # update the transformation matrix
    T[:n, :n] = R
    T[:n, n] = t

    return T


def q4_c(M: np.ndarray, D: np.ndarray) -> np.ndarray:
    '''
    Solves iterative closest point (ICP) to generate transformation T to best
    align the points clouds: D = T @ M

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    you should make use of the function `q4_a`
    '''

    t_range = 0.001
    r_range = 0.1
    t_error = 100
    r_error = 100
    n, m = M.shape
    D_tmp = np.copy(D)

    # loop until small
    while (r_error > r_range) or (t_error > t_range):
        D_new = np.empty((n, m))

        # get nearest neighbor and update to D_new
        for i in range(n):
            i_min = np.argmin(np.square(D_tmp - M[i, :]).sum(1))
            D_new[i, :] = D_tmp[i_min, :]

        # get the new transformation, and calculate the updated error
        T = q4_a(D_new, M)
        R = T[:m, :m]
        t = T[:m, -1]
        r_error = np.arccos(np.clip((np.trace(R.dot(R.T)) - 1) / 2, -1, 1))
        t_error = np.linalg.norm(t)
        # apply transform to the D
        D_tmp = utils.apply_transform(D_tmp, T)

    # computer T for nearest neighbor for original D
    for i in range(n):
        i_min = np.argmin(np.square(D_tmp - M[i, :]).sum(1))
        D_new[i, :] = D[i_min, :]

    return q4_a(M, D_new)
    # original scene
    # return q4_a(M, D)