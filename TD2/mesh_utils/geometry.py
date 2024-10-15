import numpy as np
import scipy.sparse as sparse

from tqdm.auto import tqdm

def edges_from_faces(faces):
    """
    Compute all edges in the mesh

    Parameters
    --------------------------------
    faces : (m,3) array defining faces with vertex indices

    Output
    --------------------------
    edges : (p,2) array of all edges defined by vertex indices
            with no particular order
    """
    # Number of verties
    N = 1 + np.max(faces)

    # Use a sparse matrix and find non-zero elements
    # This is way faster than a np.unique somehow
    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
    # V = np.ones_likeke(I)

    In = np.concatenate([I, J])
    Jn = np.concatenate([J, I])
    Vn = np.ones_like(In)

    M = sparse.csr_matrix((Vn, (In, Jn)), shape=(N, N)).tocoo(copy=False)

    edges0 = M.row
    edges1 = M.col

    indices = M.col > M.row

    edges = np.concatenate([edges0[indices,None], edges1[indices, None]], axis=1)
    return edges


def compute_faces_areas(vertices, faces):
    """
    Compute per-face areas of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    faces_areas : (m,) array of per-face areas
    """

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

    return faces_areas


def compute_vertex_areas(vertices, faces, faces_areas=None):
    """
    Compute per-vertex areas of a triangular mesh.
    Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

    Parameters
    -----------------------------
    vertices    : (n,3) array of vertices coordinates
    faces       : (m,3) array of vertex indices defining faces
    faces_areas :

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """
    N = vertices.shape[0]

    if faces_areas is None:
        faces_areas = compute_faces_areas(vertices,faces)  # (m,)

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.zeros_like(I)

    V = np.tile(faces_areas / 3, 3)

    # Get the (n,) array of vertex areas
    vertex_areas = np.array(sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

    return vertex_areas


def compute_normals(vertices, faces):
    """
    Compute face normals of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    normals : (m,3) array of normalized per-face normals
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    normals = np.cross(v2-v1, v3-v1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


def per_vertex_normal(vertices, faces, face_normals=None, weighting='uniform'):
    """
    Compute per-vertex normals of a triangular mesh, with a chosen weighting scheme.

    Parameters
    -----------------------------
    vertices     : (n,3) array of vertices coordinates
    faces        : (m,3) array of vertex indices defining faces
    face_normals : (m,3) array of per-face normals
    weighting    : str - 'area' or 'uniform'.

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """
    if weighting.lower() == 'uniform':
        vert_normals = per_vertex_normal_uniform(vertices, faces, face_normals=face_normals)

    elif weighting.lower() == 'area':
        vert_normals = per_vertex_normal_area(vertices, faces)

    else:
        raise ValueError("Did not implement other weighting scheme for vertex-normals")

    return vert_normals


def per_vertex_normal_area(vertices, faces):
    """
    Compute per-vertex normals of a triangular mesh, weighted by the area of adjacent faces.

    Parameters
    -----------------------------
    vertices     : (n,3) array of vertices coordinates
    faces        : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """

    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # That is 2* A(T) n(T) with A(T) area of face T
    face_normals_weighted = np.cross(1e3*(v2-v1), 1e3*(v3-v1))  # (m,3)

    # A simple version should be :
    # vert_normals = np.zeros((n_vertices,3))
    # np.add.at(vert_normals, faces.flatten(),np.repeat(face_normals_weighted,3,axis=0))
    # But this code is way faster in practice

    In = np.repeat(faces.flatten(), 3)  # (9m,)
    Jn = np.tile(np.arange(3), 3*n_faces)  # (9m,)
    Vn = np.tile(face_normals_weighted, (1,3)).flatten()  # (9m,)

    vert_normals = sparse.coo_matrix((Vn, (In, Jn)), shape=(n_vertices, 3))
    vert_normals = np.asarray(vert_normals.todense())
    vert_normals /= (1e-6 + np.linalg.norm(vert_normals, axis=1, keepdims=True))

    return vert_normals


def per_vertex_normal_uniform(vertices, faces, face_normals=None):
    """
    Compute per-vertex normals of a triangular mesh, weighted by the area of adjacent faces.

    Parameters
    -----------------------------
    vertices     : (n,3) array of vertices coordinates
    faces        : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """

    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    if face_normals is None:
        face_normals = np.cross(1e3*(v2-v1), 1e3*(v3-v1))  # (m,3)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # A simple version should be :
    # vert_normals = np.zeros((n_vertices,3))
    # np.add.at(vert_normals, faces.flatten(),np.repeat(face_normals,3,axis=0))
    # But this code is way faster in practice

    In = np.repeat(faces.flatten(), 3)  # (9m,)
    Jn = np.tile(np.arange(3), 3*n_faces)  # (9m,)
    Vn = np.tile(face_normals, (1, 3)).flatten()  # (9m,)

    vert_normals = sparse.coo_matrix((Vn, (In, Jn)), shape=(n_vertices, 3))
    vert_normals = np.asarray(vert_normals.todense())
    vert_normals /= (1e-6 + np.linalg.norm(vert_normals, axis=1, keepdims=True))

    return vert_normals


def neigh_faces(faces):
    """
    Return the indices of neighbor faces for each vertex. This supposed all vertices appear in
    the face list.

    Parameters
    --------------------
    faces : (m,3) list of faces

    Output
    --------------------
    neighbors : (n,) list of indices of neighbor faces for each vertex
    """
    n_vertices = 1+faces.max()

    neighbors = [[] for i in range(n_vertices)]

    for face_ind, (i,j,k) in enumerate(faces):
        neighbors[i].append(face_ind)
        neighbors[j].append(face_ind)
        neighbors[k].append(face_ind)

    neighbors = [np.unique(x) for x in neighbors]

    return neighbors
