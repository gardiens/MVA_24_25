#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    """
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    """

    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0], 1))

    # compute barycenter p_m and q_m
    p_m = np.mean(data, axis=1).reshape(-1, 1)
    q_m = np.mean(ref, axis=1).reshape(-1, 1)
    Q = data - p_m
    Qprime = ref - q_m

    # compute the covariance matrix H
    H = Qprime @ Q.T

    # Find the SVD of H
    U, S, Vt = np.linalg.svd(H)

    # Compute R
    R = Vt.T @ U.T
    T = q_m - R @ p_m
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]

    return R, T


from scipy.spatial import cKDTree


def Query_CPU(
    xyz_query,
    xyz_search,
    K,
):
    # Use scipy Kdtree to perform the query
    # XYZ MUST BE OF dimension 2 not 3
    kdtree = cKDTree(xyz_search)
    # TODO: you may need to tweak the number of num_workers, 1 is really long but -1 is can OOM
    distances, neighbors = kdtree.query(xyz_query, k=K, workers=-1)
    return distances, neighbors


def compute_nearest_neighbor(X, Y):
    """Compute the nearest neighbor in Y for each point in X.

    Parameters:
    -----------
    X : (n, d) array of points
    Y : (m, d) array of points

    Returns:
    --------
    nearst_neighbor : (n,) array of indices of the nearest neighbor in Y for X
    """
    return Query_CPU(xyz_query=X, xyz_search=Y, K=1)[1]


import scipy


def compute_rigid_transform(X_source, X_target):
    """Compute the optimal rotation matrix and translation that aligns two point clouds of the same
    size X_source and X_target. This rotation should be applied to X_source.

    Parameters:
    -----------
    X_source : (n, d) array of points
    Y_target : (n, d) array of points

    Returns:
    --------
    R : (d, d) rotation matrix
    t : (d,) translation vector
    """
    cardX = X_source.shape[0]
    t = 1 / cardX * (X_target - X_source).sum(axis=0)

    A = np.zeros((X_source.shape[1], X_source.shape[1]))
    for i in range(cardX):
        A += np.outer(X_target[i] - t, X_source[i])
    U, S, Vt = np.linalg.svd(A)
    D = np.eye(X_source.shape[1])
    D[-1, -1] = np.linalg.det(np.dot(U, Vt))
    R = np.dot(np.dot(U, D), Vt)
    return R, t


def transform_pointcloud(X, R, t):
    """Transform a point cloud X by a rotation matrix R and a translation vector t.

    Parameters:
    -----------
    X : (n, d) array of points
    R : (d, d) rotation matrix
    t : (d,) translation vector

    Returns:
    --------
    X_transformed : (n, d) array of transformed points
    """

    return X @ R.T + t


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    """Iterative closest point algorithm with a point to point strategy.

    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
    """

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    k = 0
    loss = np.inf
    # TODO compute ICP alignement
    X_source = data_aligned.copy().T
    Y = ref.copy().T
    # loss> np.linalg.norm(X_source-Y_target,ord=2) and
    while k < max_iter and loss > RMS_threshold:
        pi = compute_nearest_neighbor(X=X_source, Y=Y)
        X_target = Y[pi]
        R, t = compute_rigid_transform(X_source=X_source, X_target=X_target)
        R_list.append(R)
        T_list.append(t)
        neighbors_list.append(pi)
        RMS_list.append(np.linalg.norm(X_source - X_target, ord=2))
        X_source = transform_pointcloud(X_source, R, t)

        loss = np.linalg.norm(X_source - X_target, ord=2)
        print("iteration", k, "loss", loss)
        k += 1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_sampled(data, ref, max_iter, RMS_threshold, sampling_limit):
    """Iterative closest point algorithm with a point to point strategy.

    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
    """

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    k = 0
    loss = np.inf
    # TODO compute ICP alignement
    X_source = data_aligned.copy().T
    Y = ref.copy().T
    # loss> np.linalg.norm(X_source-Y_target,ord=2) and
    # We store here all the neighbors to speed up the computation
    idx_ref = compute_nearest_neighbor(X=X_source, Y=Y)
    while k < max_iter and loss > RMS_threshold:
        sampled_point = np.random.choice(
            X_source.shape[0], sampling_limit, replace=False
        )
        X_source_sampled = X_source[sampled_point]

        # compute the nearest neighbor of the sampled point
        pi = compute_nearest_neighbor(X=X_source_sampled, Y=Y)
        X_target = Y[pi]  #! It has been sampled
        idx_ref[sampled_point] = pi
        R, t = compute_rigid_transform(X_source=X_source_sampled, X_target=X_target)
        R_list.append(R)
        T_list.append(t)
        neighbors_list.append(pi)
        X_source[sampled_point] = transform_pointcloud(X_source_sampled, R, t)

        #! Here a weird thing is gonna happen as we gonna Comput the RMS ON THE WHOLE DATA

        X_target = Y[idx_ref]

        RMS_list.append(np.linalg.norm(X_source - X_target, ord=2))

        loss = np.linalg.norm(X_source - X_target, ord=2)
        print("iteration", k, "loss", loss)
        k += 1

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == "__main__":
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if True:
        # Cloud paths
        bunny_o_path = "../data/bunny_original.ply"
        bunny_r_path = "../data/bunny_returned.ply"

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply["x"], bunny_o_ply["y"], bunny_o_ply["z"]))
        bunny_r = np.vstack((bunny_r_ply["x"], bunny_r_ply["y"], bunny_r_ply["z"]))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloudidx_ref
        write_ply("../bunny_r_opt", [bunny_r_opt.T], ["x", "y", "z"])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print("Average RMS between points :")
        print("Before = {:.3f}".format(RMS_before))
        print(" After = {:.3f}".format(RMS_after))

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if True:
        # Cloud paths
        ref2D_path = "../data/ref2D.ply"
        data2D_path = "../data/data2D.ply"

        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply["x"], ref2D_ply["y"]))
        data2D = np.vstack((data2D_ply["x"], data2D_ply["y"]))

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(
            data2D, ref2D, 10, 1e-4
        )

        # Show ICP
        # show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        # clear the plot
        plt.clf()
        # Plot RMS
        print("RMS_list", RMS_list)
        plt.plot(RMS_list)
        plt.show()
        plt.title("RMS for 2D data")
        plt.xlabel("Iterations")
        plt.ylabel("RMS")

        # save the img
        plt.savefig("RMS_2.png")
        print("Saved RMS 2 ")

    # If statement to skip this part if wanted
    if True:
        # Cloud paths
        bunny_o_path = "../data/bunny_original.ply"
        bunny_p_path = "../data/bunny_perturbed.ply"

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply["x"], bunny_o_ply["y"], bunny_o_ply["z"]))
        bunny_p = np.vstack((bunny_p_ply["x"], bunny_p_ply["y"], bunny_p_ply["z"]))

        # Apply ICP
        iteration = 25
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list1 = icp_point_to_point(
            bunny_p, bunny_o, iteration, 1e-4
        )

        # Show ICP
        # show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # save the img
        print("RMS list", RMS_list1)
        # x = range(len(RMS_list1))  # Indices for x-axis
        plt.clf()
        plt.plot(RMS_list1)  # Provide x and y explicitly
        plt.title("RMS for Bunny Data")
        plt.xlabel("Iterations")
        plt.ylabel("RMS")
        plt.show()

        # save the img
        plt.savefig("RMS_bunny.png")

    # If statement to skip this part if wanted
    if True:
        # Cloud paths
        ND_o_path = "../data/Notre_Dame_Des_Champs_1.ply"
        ND_p_path = "../data/Notre_Dame_Des_Champs_2.ply"

        # Load clouds
        ND_o_ply = read_ply(ND_o_path)
        ND_p_ply = read_ply(ND_p_path)
        ND_o = np.vstack((ND_o_ply["x"], ND_o_ply["y"], ND_o_ply["z"]))
        ND_p = np.vstack((ND_p_ply["x"], ND_p_ply["y"], ND_p_ply["z"]))

        # Apply ICP
        iteration = 50
        sampling_points = 1000
        ND_p_opt, R_list, T_list, neighbors_list, RMS_list1 = (
            icp_point_to_point_sampled(
                ND_p, ND_o, iteration, 1e-4, sampling_limit=sampling_points
            )
        )

        # Show ICP
        # show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # save the img
        print("RMS list", RMS_list1)
        plt.clf()
        # x = range(len(RMS_list1))  # Indices for x-axis
        plt.plot(RMS_list1)  # Provide x and y explicitly
        plt.title("RMS for ND with 1000 points used")
        plt.xlabel("Iterations")
        plt.ylabel("RMS")
        plt.show()

        # save the img
        plt.savefig("1000_RMS_ND_samplingpoint1000.png")

        sampling_points = 10000
        ND_p_opt, R_list, T_list, neighbors_list, RMS_list1 = (
            icp_point_to_point_sampled(
                ND_p, ND_o, iteration, 1e-4, sampling_limit=sampling_points
            )
        )

        # Show ICP
        # show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # save the img
        # x = range(len(RMS_list1))  # Indices for x-axis
        plt.clf()
        plt.plot(RMS_list1)  # Provide x and y explicitly
        plt.title("RMS for 10000 sampled points")
        plt.xlabel("Iterations")
        plt.ylabel("RMS")
        plt.show()

        # save the img
        plt.savefig("1000_RMS_ND_samplingpoint10000.png")
