import numpy as np
import matplotlib.pyplot as plt


def base_cmap(n_vert, pc=False):
    """
    Outputs a white or red colormap for each point. Red one is usually used for point clouds.

    Parameters
    ----------------------------
    n_vert : int - number of vertices
    pc     : bool - Specify if the colormap concerns a point cloud.

    Output
    ---------------------------
    cmap : (n_vert,3) with each line filled with [1,1,1] or [1,0,0]
    """
    if pc:
        return np.tile([1, 0, 0], (n_vert, 1))

    return np.ones((n_vert, 3))


def get_cmap(cmap, colormap='viridis'):
    """
    Outputs a RGB-valued colormap from scalar values. Uses all matplotlib colormap and a binary
    red-white one.

    Parameters
    ----------------------------
    cmap     : (n,) scalar values for quantities
    colormap : str - Name of the colormap. 'binary' sets positive values to red and others to white.

    Output
    ---------------------------
    cmap : (n,3) colormap
    """
    if colormap == 'binary':
        inds = cmap > 0
        cmap = np.ones((len(cmap), 3))
        cmap[inds, 1:] = 0

    else:
        # normalize and transform with matplotlib
        cmap = (cmap - np.min(cmap)).astype(float)
        cmap /= np.max(cmap)
        cmap = plt.get_cmap(colormap)(cmap)[:,:3]

    return cmap


def color_list(n):
    """
    Outputs a list of n evenly spaced colors.

    Parameters
    ----------------------------
    n : int - number of colors

    Output
    ---------------------------
    colors : (n,3) - n different colors
    """
    cm = plt.get_cmap('rainbow')
    colors = np.array([cm(1.*i/(n-1))[:3] for i in range(n)])
    return colors


def base_colors():
    """
    Outputs the list of base colors from matplotlib when doing plots

    Output
    -------------------------
    base_colors : list of standard colors of matplotlib.
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def normalize(f, vmin=0, vmax=1):
    """
    Normalize a function or a set of functions between vmin and vmax

    Parameters
    ----------------------------
    f : (n,) or (n,p) - one or multiple functions
    vmin : minimum value for the normalized function(s)
    vmax : maximum value for the normalized function(s)

    Output
    ---------------------------
    f_normalized : (n,) or (n,p) - normalized function(s)
    """
    if f.ndim == 1:
        f_norm = f - np.min(f)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm)

    else:
        f_norm = f - np.min(f, axis=0, keepdims=True)
        f_norm = vmin + (vmax - vmin) * f_norm / np.max(f_norm, axis=0, keepdims=True)

    return f_norm


def vertices_to_rgb_basic(vertices):
    """
    Transforms (x,y,z) coordinates into RGB values using normalization

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    cmap = normalize(vertices, vmin=0, vmax=1)
    return cmap


def vertices_to_rgb_pretty(vertices, param=[-2, -1, 3]):
    """
    Transforms (x,y,z) coordinates into RGB values using a prettier procedure, parametrized by
    a rearangement of the array [1,2,3] up to sign flip and column reordering.
    Default parameter value is [-2, -1, 3].

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices
    param    : (3,) - rearangement of the [1,2,3] array up to sign flip and column reordering.

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    param = np.asarray(param)

    if np.any(np.sort(np.abs(param)) != np.arange(1, 4)):
        raise ValueError("'param' should use a reorganization of \
                          [1,2,3] up to sign flip and column switch")

    # Invert some colors and switch some channels
    cmap = np.sign(param)[None, :] * vertices
    cmap = cmap[:, np.abs(param)-1]

    cmap = normalize(np.cos(normalize(cmap)))

    return cmap


def digitize_cmap(cmap, n_colors):
    """
    Digitize a colormap into a finite number of colors for each channel

    Parameters
    ----------------------------
    cmap     : (n,) or (n,3) - scalar or RGB values.
    n_colors : int - number of colors for each coordinate.

    Output
    ----------------------------
    cmap_new : (n,) or (n,3) quantized values
    """
    # Trick is to use the midpoints as bins
    bins = np.linspace(0, 1, n_colors)
    midpoints = (bins[1:] + bins[:-1])/2

    cmap = bins[np.digitize(cmap, midpoints)]

    return cmap


def vert2rgb(vertices, pretty=False, param=[-2, -1, 3], n_colors=-1):
    """
    Convert (x,y,z) coordinates of vertices to RGB values, using either `vertices_to_rgb_basic` or
    `vertices_to_rgb_basic`.

    If n_colors is specified as a positive integer, each channel only supports the given number of
    colors.

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices
    pretty   : bool - whether to use the 'pretty' procedure.
    param    : (3,) - rearangement of the [1,2,3] array up to sign flip and column reordering. Only
               used if `pretty` is set to "True".
    n_colors : int - If positive, limits the number of possible colors per channel

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex
    """
    if pretty:
        cmap = vertices_to_rgb_pretty(vertices, param=param)
    else:
        cmap = vertices_to_rgb_basic(vertices)

    if n_colors > 0:
        cmap = digitize_cmap(cmap, n_colors)

    return cmap


def vert2_rgb_mesh(mesh, pretty=False, param=[-2, -1, 3], n_colors=-1):
    """
    Similar to `vert2rgb` but takes a TriMesh as an input.
    """
    cmap = vert2rgb(mesh.vertlist, pretty=pretty, param=param, n_colors=n_colors)
    return cmap


def rotate(vertices, rotation=None):
    """
    rotate one vertex or multiple vertices with a given rotation

    Parameters
    ----------------------------
    vertices : (n,3) or (3,) - x,y,z coordinates of vertices
    rotation : (3,3) - matrix of rotation

    Output
    ----------------------------
    rotated_vertices : (n,3) or (3,) rotated vertices
    """
    if rotation is None:
        return vertices

    return vertices @ rotation.T


def get_smooth_shading(flat=False):
    """
    Some basic shading parameters for rough texture

    Parameters
    ----------------------------
    flat : bool - whether to use flat faces or not

    Output
    ----------------------------
    shading : shading parameters for meshplot
    """
    shading = {"flat": flat, "metalness": 0.25, "reflectivity": 0, "roughness": .9}

    return shading


def _find_shading(shading=None, pretty=False, flat=False):
    """
    Compute some basic shading and update it with given values

    Parameters
    ---------------------------
    shading : dict - shading parameters to enforce
    pretty  : bool - if True, uses `get_smooth_shading`
    flat    : bool - whether to use flat shading or smooth
    """
    new_shading = dict()
    if pretty:
        new_shading = get_smooth_shading(flat=flat)

    if shading is not None:
        new_shading.update(shading)

    return new_shading


def _find_cmap(mesh1, cmap=None, colormap='viridis'):
    """
    Returns either the cmap itself, white values or transforms
    scalar values into RGB values.

    Parameters
    ---------------------------
    mesh1     : TriMesh - must have n_vertices and n_faces attributes
    cmap      : None, or (n,3) or (n,) - nothing, RGB or scalar values
    colormap  : colormap to use when converting scalar to RGB
    """
    if cmap is None:
        cmap = base_cmap(mesh1.n_vertices, pc=(mesh1.n_faces == 0))
    elif cmap.ndim == 1:  # Scalar quantity
        cmap = get_cmap(cmap, colormap=colormap)

    return cmap