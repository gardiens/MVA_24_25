#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
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

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

import numpy as np


def PCA(points):
    eigenvalues = None
    eigenvectors = None

    # Compute the barycenter of the points
    points = points - np.mean(points, axis=0)
    # Compute the covariance matrix
    Cov = np.dot(points.T, points) / points.shape[0]

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(Cov)
    # Sort the eigenvalues in ascending order
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_local_PCA_neighbors(query_points, cloud_points, neighbors):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

    # Create a KDTree
    tree = KDTree(cloud_points)

    # Find neighborhoods for each query point
    distance, idx = tree.query(query_points, k=neighbors)
    # For each query point
    for i, neighborhood_indices in enumerate(idx):
        # Get the neighborhood points
        neighborhood_points = cloud_points[neighborhood_indices]

        # Perform PCA if the neighborhood has enough points
        if len(neighborhood_points) >= 3:
            eigenvalues, eigenvectors = PCA(neighborhood_points)

            # Store eigenvalues and eigenvectors
            all_eigenvalues[i] = eigenvalues
            all_eigenvectors[i] = eigenvectors
        else:
            # Handle cases with insufficient points (e.g., padding with zeros)
            all_eigenvalues[i] = [1e-3, 1e-3, 1e-3]
            all_eigenvectors[i] = np.eye(3)  # Identity matrix

    return all_eigenvalues, all_eigenvectors


def compute_local_PCA(query_points, cloud_points, radius):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

    # Create a KDTree
    tree = KDTree(cloud_points)

    # Find neighborhoods for each query point
    neighborhoods = tree.query_radius(query_points, r=radius)

    # For each query point
    for i, neighborhood_indices in enumerate(neighborhoods):
        # Get the neighborhood points
        neighborhood_points = cloud_points[neighborhood_indices]

        # Perform PCA if the neighborhood has enough points
        if len(neighborhood_points) >= 3:
            eigenvalues, eigenvectors = PCA(neighborhood_points)

            # Store eigenvalues and eigenvectors
            all_eigenvalues[i] = eigenvalues
            all_eigenvectors[i] = eigenvectors
        else:
            # Handle cases with insufficient points (e.g., padding with zeros)
            all_eigenvalues[i] = [1e-3, 1e-3, 1e-3]
            all_eigenvectors[i] = np.eye(3)  # Identity matrix

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    all_eigenvalues, all_eigenvectors = compute_local_PCA(
        query_points, cloud_points, radius
    )

    verticality = np.zeros((cloud.shape[0], 1))
    linearity = np.zeros((cloud.shape[0], 1))

    planarity = np.zeros((cloud.shape[0], 1))
    sphericity = np.zeros((cloud.shape[0], 1))

    # Compute features for each point
    for i, (eigenvalues, eigenvectors) in enumerate(
        zip(all_eigenvalues, all_eigenvectors)
    ):
        # Compute features
        linearity[i] = 1 - eigenvalues[1] / eigenvalues[2]
        planarity[i] = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
        sphericity[i] = eigenvalues[0] / eigenvalues[2]
        verticality[i] = 2 * np.arcsin(np.abs(eigenvectors[2, 2])) / np.pi
    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == "__main__":
    # PCA verification
    # ****************
    if True:
        # Load cloud as a [N x 3] matrix
        cloud_path = "../data/Lille_street_small.ply"
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        assert np.allclose(
            eigenvalues, [5.25050177, 21.7893201, 89.58924003], atol=1e-2
        )
        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    if True:
        # Load cloud as a [N x 3] matrix
        cloud_path = "../data/Lille_street_small.ply"
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply(
            "../Lille_street_small_normals.ply",
            (cloud, normals),
            ["x", "y", "z", "nx", "ny", "nz"],
        )

    #  Normal computation with neighbors
    # ******************
    if True:
        # Load cloud as a [N x 3] matrix
        cloud_path = "../data/Lille_street_small.ply"
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA_neighbors(
            cloud, cloud, 30
        )
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply(
            "../Lille_street_small_normals_30neighbors.ply",
            (cloud, normals),
            ["x", "y", "z", "nx", "ny", "nz"],
        )

    # Normal computation with Feature s
    # ******************
    if True:
        # Load cloud as a [N x 3] matrix
        cloud_path = "../data/Lille_street_small.ply"
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

        # Compute PCA on the whole cloud
        verticality, linearity, planarity, sphericity = compute_features(
            cloud, cloud, 0.5
        )
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply(
            "../Lille_street_small_normals_features.ply",
            (cloud, normals, verticality, linearity, planarity, sphericity),
            [
                "x",
                "y",
                "z",
                "nx",
                "ny",
                "nz",
                "verticality",
                "linearity",
                "planarity",
                "sphericity",
            ],
        )
