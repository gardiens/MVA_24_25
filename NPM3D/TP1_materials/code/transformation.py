#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


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

    # Path of the file
    file_path = "../data/bunny.ply"

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data["x"], data["y"], data["z"])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data["red"], data["green"], data["blue"])).T

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #
    # center the point clouds
    center = np.mean(points, axis=0)
    points = points - center
    print("added center ")

    # divide the scale by 2
    points = points / 2
    # recenter the cloud
    points = points + center
    # add a 10 cm translation in the y axis
    points[:, 1] = points[:, 1] - 0.1

    # Replace this line by your code
    transformed_points = points

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply(
        "../little_bunny.ply",
        [transformed_points, colors],
        ["x", "y", "z", "red", "green", "blue"],
    )

    print("Done")
