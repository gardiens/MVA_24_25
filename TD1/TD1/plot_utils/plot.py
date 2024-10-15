import numpy as np
import meshplot as mp
import matplotlib.colors
from . import utils


def plot(mesh1, cmap=None, points=None, cmap_p=None, rotation=None, colormap='viridis',
         shading=None, shading_p=None, flat=True, pretty=False, point_size=None):
    """
    Plot a mesh (or a point cloud) with a pointcloud on top of it

    Parameters
    -----------------------------
    mesh1      : TriMesh - mesh to plot
    cmap       : (n,3) RGB values or (n,) scalar values for each face or vertex
    points     : int or (m,) for single or multiple landmarks, or (m,3) coordinates of an eventual
                 point cloud to plot
    cmap_p     : (m,3) RGB values or (m,) scalar values for each point
    rotation   : (3,3) rotation matrix
    colormap   : str - matplotlib name of a colormap, or "binary" if cmap only takes values 0 and 1
    shading    : dict - shading values of the mesh for meshplot plots
    shading_p  : dict - shading values of the point cloud for meshplot plots
    flat       : bool - whether to use flat shading
    pretty     : sets some Matlab-like shading parameters
    point_size : point size of the point cloud. Will be automatically computed else.

    Output
    ------------------------------
    Viewer - meshplot.Viewer class
    """
    # Obtain shadings
    shading_m = {}
    if pretty:
        shading_m = utils.get_smooth_shading(flat=flat)
    if shading is not None:
        shading_m.update(shading)
        if 'flat' not in shading.keys():
            shading_m['flat'] = flat
    if shading is None and not pretty:
        shading_m['flat'] = flat

    # Compute RGB colormap
    if cmap is None:
        cmap = utils.base_cmap(mesh1.vertices.shape[0], pc=(mesh1.faces.shape[0] == 0))
    elif cmap.ndim == 1:  # Scalar quantity
        cmap = utils.get_cmap(cmap, colormap=colormap)

    # Rotate vertices
    vertices = utils.rotate(mesh1.vertices, rotation=rotation)
    faces = mesh1.faces

    p = mp.plot(vertices, faces, c=cmap, shading=shading_m)

    # When adding a point_cloud
    if points is not None:
        # Obtain shading
        if shading_p is None:
            shading_p = {}
        if point_size is not None:
            shading_p['point_size'] = point_size
        elif 'point_size' not in shading_p.keys():
            mesh_area = mesh1.A.sum()
            if mesh_area is not None:
                shading_p['point_size'] = .1*np.sqrt(mesh_area)

        # Get points coordinates
        if type(points) is list or np.issubdtype(type(points), np.integer) or (points.ndim == 1):
            points = vertices[points]
            if points.ndim == 1:
                points = points[None, :]
        else:
            points = utils.rotate(points, rotation=rotation)

        # Get the point cloud colormap
        if cmap_p is not None and cmap_p.ndim == 1:
            cmap_p = utils.get_cmap(cmap_p, colormap=colormap)

        p.add_points(points, c=cmap_p, shading=shading_p)

    return p


def plot_lines(mesh1, mesh2, p2p, n_points=10, pretty=False, dist_ratio=1, rotation=None,
               rev=False, shading=None, param=[-2, -1, 3], flat=False):
    """
    Plot two meshes in correspondences with mathcing colors and some random lines between
    correspondences

    Parameters
    -----------------------------
    mesh1      : TriMesh - first mesh to plot
    mesh2      : TriMesh - second mesh to plot
    p2p        : (n2,) assignment from vertices of mesh2 to vertices on mesh1. Can be a (n,3) matrix
                 of barycentric coordinates
    n_points   : int number of lines to plot
    pretty     : sets some Matlab-like shading parameters
    dist_ratio : float - space between the two meshes is set as 1/dist_ratio* width1
    rotation   : (3,3) rotation matrix
    rev        : bool - reverse the colormap
    colormap   : str - matplotlib name of a colormap, or "binary" if cmap only takes values 0 and 1
    shading    : dict - shading values of the mesh for meshplot plots
    param      : transformation of [1,2,3] up to swap and sign flips.
    flat       : bool - whether to use flat shading

    Output
    ------------------------------
    Viewer - meshplot.Viewer class
    """
    # Compute colormap from RGB coordinates and parameters
    cmap1 = utils.vert2rgb(utils.rotate(mesh1.vertlist, rotation=rotation),
                           pretty=pretty, param=param)
    if rev:
        cmap1 = 1 - cmap1

    # Set shading
    shading_m = {}
    if pretty:
        shading_m = utils.get_smooth_shading(flat=flat)
    if shading is not None:
        shading_m.update(shading)

    # Set the colormaps using the pointwise
    if p2p.ndim == 1:
        cmap2 = cmap1[p2p]
    else:
        assert p2p.shape == (mesh2.n_vertices, mesh1.n_vertices), 'Pb with p2p dimension'
        cmap2 = p2p @ cmap1

    # Compute the center of mass
    if mesh1.vertex_areas is not None:
        cm1 = np.average(mesh1.vertlist, axis=0, weights=mesh1.vertex_areas)
    else:
        cm1 = mesh1.mean(axis=0)
    if mesh2.vertex_areas is not None:
        cm2 = np.average(mesh2.vertlist, axis=0, weights=mesh2.vertex_areas)
    else:
        cm2 = mesh2.mean(axis=0)

    # Center both mesg
    vert1 = mesh1.vertlist - cm1[None,:]
    vert1 = utils.rotate(vert1, rotation=rotation)

    vert2 = mesh2.vertlist - cm2[None,:]
    vert2 = utils.rotate(vert2, rotation=rotation)

    # Translate second mesh
    xmin1, xmax1 = vert1[:, 0].min(),  vert1[:, 0].max()
    xmin2 = vert2[:, 0].min()

    transl_x = xmax1 - xmin2 + (xmax1-xmin1)/dist_ratio
    transl = np.array([transl_x, 0, 0])

    vert2 += transl[None, :]

    # Extract poins from which to draw lines
    fps2 = mesh2.extract_fps(n_points, geodesic=False)

    p = mp.plot(vert1, mesh1.facelist, c=cmap1, shading=shading_m)
    p.add_mesh(vert2, mesh2.facelist, c=cmap2, shading=shading_m)

    for ind in fps2:
        p.add_lines(vert1[p2p[ind]], vert2[ind],
                    shading={"line_color": matplotlib.colors.to_hex(cmap2[ind])})


def double_plot(mesh1, mesh2, cmap1=None, cmap2=None, pretty=False, rotation=None, shading=None,
                colormap='viridis', colormap2=None, flat=False, shading2=None):
    """
    Plot two meshes (or a point clouds) on two different windows

    Parameters
    -----------------------------
    mesh1      : TriMesh - first mesh to plot
    mesh1      : TriMesh - second mesh to plot
    cmap1      : (n,3) RGB values or (n,) scalar values for each face or vertex of the first mesh
    cmap2      : (n,3) RGB values or (n,) scalar values for each face or vertex of the second mesh
    rotation   : (3,3) rotation matrix
    colormap   : str - matplotlib name of a colormap, or "binary" if cmap only takes values 0 and 1
    colormap2  : str - matplotlib name of a colormap, or "binary" if cmap only takes values 0 and 1.
                 If specified will be applied only to cmap2
    shading    : dict - shading values of the first mesh for meshplot plots
    shading2   : dict - shading values of the second mesh for meshplot plots
    flat       : bool - whether to use flat shading
    pretty     : sets some Matlab-like shading parameters

    Output
    ------------------------------
    Viewer - meshplot.Subplot class
    """
    # Get colormaps parameters and shadings
    if colormap2 is None:
        colormap2 = colormap

    shading_m1 = utils._find_shading(shading=shading, pretty=pretty, flat=flat)
    if shading2 is None:
        shading_m2 = shading_m1
    else:
        shading_m2 = utils._find_shading(shading=shading2, pretty=pretty, flat=flat)

    # Obtain RGB colormaps
    cmap1 = utils._find_cmap(mesh1, cmap=cmap1, colormap=colormap)
    cmap2 = utils._find_cmap(mesh2, cmap=cmap2, colormap=colormap2)

    vertices_1 = utils.rotate(mesh1.vertices, rotation=rotation)
    faces_1 = mesh1.faces

    vertices_2 = utils.rotate(mesh2.vertices, rotation=rotation)
    faces_2 = mesh2.faces

    d = mp.subplot(vertices_1, faces_1, c=cmap1, s=[2, 2, 0], shading=shading_m1)
    p = mp.subplot(vertices_2, faces_2, c=cmap2, s=[2, 2, 1], shading=shading_m2, data=d)

    return p


def plot_p2p(mesh1, mesh2, p2p, rotation=None, pretty=True, rev=False, shading=None, n_colors=-1,
             param=[-2, -1, 3], flat=False):
    """
    Plot two meshes in correspondences on two separate Viewers.

    Parameters
    -----------------------------
    mesh1      : TriMesh - first mesh to plot
    mesh2      : TriMesh - second mesh to plot
    p2p        : (n2,) assignment from vertices of mesh2 to vertices on mesh1. Can be a (n,3) matrix
                 of barycentric coordinates
    pretty     : sets some Matlab-like shading parameters
    rotation   : (3,3) rotation matrix
    rev        : bool - reverse the colormap
    shading    : dict - shading values of the mesh for meshplot plots
    param      : transformation of [1,2,3] up to swap and sign flips - used if `pretty`
    flat       : bool - whether to use flat shading
    n_colors   : int - if positive, restricts the number of colors per x,y and z coordinates

    Output
    ------------------------------
    Viewer - meshplot.Viewer class
    """
    # Obtain the RGB values from coordinates
    cmap1 = utils.vert2rgb(utils.rotate(mesh1.vertices, rotation=rotation),
                           pretty=pretty, param=param, n_colors=n_colors)

    if rev:
        cmap1 = 1 - cmap1

    # Set shadings
    shading_m = {}
    if pretty:
        shading_m = utils.get_smooth_shading(flat=flat)
    if shading is not None:
        shading_m.update(shading)

    # Transfer the colormap using the pointwise map
    if p2p.ndim == 1:
        cmap2 = cmap1[p2p]
    else:
        assert p2p.shape == (mesh2.n_vertices, mesh1.n_vertices), 'Pb with p2p dimension'
        cmap2 = p2p @ cmap1

    return double_plot(mesh1, mesh2, cmap1=cmap1, cmap2=cmap2, rotation=rotation, shading=shading_m)
