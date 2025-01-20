#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # YOUR CODE
    neighborhoods = []
    for point in queries:
        neighborhood = []
        for i, support in enumerate(supports):
            if np.linalg.norm(support - point) < radius:
                neighborhood.append(i)
        neighborhoods.append(neighborhood)

    return neighborhoods


def brute_force_KNN(queries, supports, k):
    # YOUR CODE
    neighborhoods = []
    for point in queries:
        neighborhood = np.argsort(np.linalg.norm(supports - point, axis=1))[:k]
        neighborhoods.append(neighborhood)

    return neighborhoods


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == "__main__":
    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = "../data/indoor_scan.ply"

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data["x"], data["y"], data["z"])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:
        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        print("search spherical")
        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()
        print("KNN?", neighbors_num)
        # Search KNN
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print(
            "{:d} spherical neighborhoods computed in {:.3f} seconds".format(
                num_queries, t1 - t0
            )
        )
        print("{:d} KNN computed in {:.3f} seconds".format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print(
            "Computing spherical neighborhoods on whole cloud : {:.0f} hours".format(
                total_spherical_time / 3600
            )
        )
        print(
            "Computing KNN on whole cloud : {:.0f} hours".format(total_KNN_time / 3600)
        )

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
        from scipy.spatial import cKDTree

        # Define the search parameters
        kdtree = cKDTree(points)
        distances, eighbors = kdtree.query(
            points[:num_queries], k=neighbors_num, distance_upper_bound=radius
        )
        # YOUR CODE
        #! Question 4.A)
        num_queries = 10
        num_neighbors = 1000000  #! really big
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        radius = 0.2
        num_queries = 1000
        best_leaf_size = 128
        print("search spherical with same parameters")
        for leaf_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            kdtree = cKDTree(points, leafsize=leaf_size)

            t1 = time.time()

            # Search KNN
            distances, eighbors = kdtree.query(
                queries, k=num_neighbors, distance_upper_bound=radius
            )

            t2 = time.time()
            print("leaf size", leaf_size)
            print("KNN computed in {:.3f} seconds".format(t2 - t1))

        #! Question 4.a2)
        print("search KNN with best parameters")
        # Question 4.B)

        radius = 0.2
        num_queries = 1000
        best_leaf_size = 128
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        t1 = time.time()
        # select random num_queries queries

        # Search KNN
        kdtree = cKDTree(points, leafsize=best_leaf_size)
        distances, eighbors = kdtree.query(
            queries, k=neighbors_num, distance_upper_bound=radius
        )
        t2 = time.time()
        print("leaf size", best_leaf_size)
        print(
            "KNN computed in {:.3f} seconds with 1000 random queries ".format(t2 - t1)
        )

        print("start computing for 20cm back")
        num_queries = 100000
        num_neighbors = 1000000  #! really big
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        radius = 0.2
        num_queries = 1000
        best_leaf_size = 128

        t1 = time.time()
        # Search KNN
        distances, eighbors = kdtree.query(
            queries, k=neighbors_num, distance_upper_bound=radius
        )
        t2 = time.time()

        # Print timing results
        print(
            "{:d} KNN computed back with the best leaves  in {:.3f} seconds".format(
                num_queries, t2 - t1
            )
        )

        # Time to compute all neighborhoods in the cloud
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print(
            "Computing KNN on whole cloud : {:.0f} hours".format(total_KNN_time / 3600)
        )

        #! Question 4.C)
        # Benchmark with radius
        neighbors_num = 50000
        num_queries = 1000
        best_leaf_size = 128
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        kdtree = cKDTree(points, leafsize=best_leaf_size)
        print("start benchmark radius")
        for radius in np.logspace(0, 2, 20, base=10):
            t1 = time.time()
            # Search KNN
            distances, eighbors = kdtree.query(
                queries, k=neighbors_num, distance_upper_bound=radius
            )

            t2 = time.time()
            print("radius", radius)
            print("KNN computed in {:.3f} seconds".format(t2 - t1))
        # distances,eighbors=kdtree.query(points[:num_queries],k=neighbors_num,distance_upper_bound=radius)
