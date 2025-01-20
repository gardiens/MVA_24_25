#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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


def cloud_decimation(
    points: np.array, colors: np.array, labels: np.array, factor: float
):
    n, m = points.shape

    # YOUR CODE
    decimated_points = np.zeros((n // factor, m))
    decimated_colors = np.zeros((n // factor, 3))
    decimated_labels = np.zeros(n // factor)

    for i in range(n // factor):
        decimated_points[i] = points[i * factor]
        decimated_colors[i] = colors[i * factor]
        decimated_labels[i] = labels[i * factor]

    # convert decimated colors to uint8
    decimated_colors = decimated_colors.astype(np.uint8)

    return decimated_points, decimated_colors, decimated_labels


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
    colors = np.vstack((data["red"], data["green"], data["blue"])).T
    labels = data["label"]
    print("the shape of colors", colors.shape)

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(
        points, colors, labels, factor
    )
    print("decimated colors", decimated_colors.shape)
    t1 = time.time()
    print("decimation done in {:.3f} seconds".format(t1 - t0))

    # Save
    write_ply(
        "../decimated.ply",
        [decimated_points, decimated_colors, decimated_labels],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )

    print("Done")
