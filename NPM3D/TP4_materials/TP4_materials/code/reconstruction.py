#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    """Hoppe Surface Reconstruction Algorithm Computes the signed distance field using point cloud
    normals.

    :param points: (N, 3) array of input point cloud
    :param normals: (N, 3) array of normal vectors associated with the points
    :param scalar_field: (grid_resolution, grid_resolution, grid_resolution) empty array to store
        scalar field
    :param grid_resolution: int, number of voxels per dimension
    :param min_grid: (3,) array, minimum bounding box coordinates
    :param size_voxel: float, voxel size
    :return: Updated scalar field
    """
    # Construct KDTree for nearest neighbor search
    tree = KDTree(points)

    # Generate grid coordinates
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(
            min_grid[0],
            min_grid[0] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        np.linspace(
            min_grid[1],
            min_grid[1] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        np.linspace(
            min_grid[2],
            min_grid[2] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        indexing="ij",
    )

    grid_points = np.vstack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
    ).T  # Reshape into (M, 3)

    # Find nearest neighbors for each grid point
    dists, idxs = tree.query(grid_points, k=1)
    nearest_points = points[idxs.flatten()]
    nearest_normals = normals[idxs.flatten()]

    # Compute signed distance function (SDF)
    voxel_to_point = grid_points - nearest_points
    signed_distances = np.einsum(
        "ij,ij->i", voxel_to_point, nearest_normals
    )  # Dot product for sign

    # Reshape into the scalar field grid
    scalar_field = signed_distances.reshape(
        (grid_resolution, grid_resolution, grid_resolution)
    )

    return scalar_field


# IMLS surface reconstruction
def compute_imls(
    points,
    normals,
    scalar_field,
    grid_resolution,
    min_grid,
    size_voxel,
    k: int = 30,
    h: float = 0.01,
):
    """Implicit Moving Least Squares (IMLS) Surface Reconstruction.

    :param points: (N, 3) array of input point cloud
    :param normals: (N, 3) array of normal vectors associated with the points
    :param scalar_field: (grid_resolution, grid_resolution, grid_resolution) empty array to store
        scalar field
    :param grid_resolution: int, number of voxels per dimension
    :param min_grid: (3,) array, minimum bounding box coordinates
    :param size_voxel: float, voxel size
    :param k: int, number of nearest neighbors to use
    :param h: float, smoothing parameter for weight function
    :return: Updated scalar field
    """

    # Construct KDTree for nearest neighbor search
    tree = KDTree(points)

    # Generate grid coordinates
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(
            min_grid[0],
            min_grid[0] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        np.linspace(
            min_grid[1],
            min_grid[1] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        np.linspace(
            min_grid[2],
            min_grid[2] + size_voxel * (grid_resolution - 1),
            grid_resolution,
        ),
        indexing="ij",
    )

    grid_points = np.vstack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
    ).T  # Reshape into (M, 3)

    # Find k nearest neighbors for each voxel center
    dists, idxs = tree.query(grid_points, k=k)  #
    nearest_points = points[idxs]  #
    nearest_normals = normals[idxs]  #

    # Compute signed distances
    voxel_to_point = grid_points[:, None, :] - nearest_points  #

    # Compute weights using Gaussian function
    weights = np.exp(-(dists**2) / h**2)  #
    weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights

    # Compute weighted normal directions
    weighted_normals = np.sum(weights[:, :, None] * nearest_normals, axis=1)  # (M, 3)
    weighted_normals /= np.linalg.norm(
        weighted_normals, axis=1, keepdims=True
    )  # Normalize weighted normals

    # Compute final signed distance (dot product)
    signed_distances = np.sum(voxel_to_point * weighted_normals[:, None, :], axis=2)  #
    signed_distances = np.sum(
        weights * signed_distances, axis=1
    )  # Weighted sum -> (M,)

    # Reshape scalar field back into grid
    scalar_field[:] = signed_distances.reshape(
        (grid_resolution, grid_resolution, grid_resolution)
    )
    print("scalar_field: ", scalar_field.shape)
    return scalar_field


if __name__ == "__main__":
    if False:  # Hoppe computation
        t0 = time.time()

        # Path of the file
        file_path = "../data/bunny_normals.ply"

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data["x"], data["y"], data["z"])).T
        normals = np.vstack((data["nx"], data["ny"], data["nz"])).T

        # Compute the min and max of the data points
        min_grid = np.amin(points, axis=0)
        max_grid = np.amax(points, axis=0)

        # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
        min_grid = min_grid - 0.10 * (max_grid - min_grid)
        max_grid = max_grid + 0.10 * (max_grid - min_grid)

        # grid_resolution is the number of voxels in the grid in x, y, z axis
        grid_resolution = 16  # 128
        # grid_resolution=128
        size_voxel = max(
            [
                (max_grid[0] - min_grid[0]) / (grid_resolution - 1),
                (max_grid[1] - min_grid[1]) / (grid_resolution - 1),
                (max_grid[2] - min_grid[2]) / (grid_resolution - 1),
            ]
        )
        print("size_voxel: ", size_voxel)

        # Create a volume grid to compute the scalar field for surface reconstruction
        scalar_field = np.zeros(
            (grid_resolution, grid_resolution, grid_resolution), dtype=np.float32
        )

        # Compute the scalar field in the grid
        scalar_field = compute_hoppe(
            points, normals, scalar_field, grid_resolution, min_grid, size_voxel
        )
        # compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

        # Compute the mesh from the scalar field based on marching cubes algorithm
        verts, faces, normals_tri, values_tri = measure.marching_cubes(
            scalar_field, level=0.0, spacing=(size_voxel, size_voxel, size_voxel)
        )
        verts += min_grid

        # Export the mesh in ply using trimesh lib
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(
            file_obj=f"../bunny_mesh_hoppe_{grid_resolution}.ply", file_type="ply"
        )

        print("Total time for surface reconstruction : ", time.time() - t0)

    if True:
        print("COMPUTE IMLS")
        t0 = time.time()

        # Path of the file
        file_path = "../data/bunny_normals.ply"

        # Load point cloud
        data = read_ply(file_path)

        # Concatenate data
        points = np.vstack((data["x"], data["y"], data["z"])).T
        normals = np.vstack((data["nx"], data["ny"], data["nz"])).T

        # Compute the min and max of the data points
        min_grid = np.amin(points, axis=0)
        max_grid = np.amax(points, axis=0)

        # Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
        min_grid = min_grid - 0.10 * (max_grid - min_grid)
        max_grid = max_grid + 0.10 * (max_grid - min_grid)

        # grid_resolution is the number of voxels in the grid in x, y, z axis
        # grid_resolution = 16 #128
        grid_resolution = 128
        size_voxel = max(
            [
                (max_grid[0] - min_grid[0]) / (grid_resolution - 1),
                (max_grid[1] - min_grid[1]) / (grid_resolution - 1),
                (max_grid[2] - min_grid[2]) / (grid_resolution - 1),
            ]
        )
        print("size_voxel: ", size_voxel)

        # Create a volume grid to compute the scalar field for surface reconstruction
        scalar_field = np.zeros(
            (grid_resolution, grid_resolution, grid_resolution), dtype=np.float32
        )

        # Compute the scalar field in the grid
        compute_imls(
            points, normals, scalar_field, grid_resolution, min_grid, size_voxel, 30
        )

        # Compute the mesh from the scalar field based on marching cubes algorithm
        verts, faces, normals_tri, values_tri = measure.marching_cubes(
            scalar_field, level=0.0, spacing=(size_voxel, size_voxel, size_voxel)
        )
        verts += min_grid

        # Export the mesh in ply using trimesh lib
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(
            file_obj=f"../bunny_mesh_imls_{grid_resolution}.ply", file_type="ply"
        )

        print("Total time for surface reconstruction : ", time.time() - t0)
