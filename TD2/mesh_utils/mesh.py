import os
import time

import numpy as np

from . import file_utils
from . import geometry as geom
from . import laplacian
import scipy.linalg
import scipy.sparse as sparse



class TriMesh:
    """
    Mesh Class (can also represent point clouds)
    ________

    Attributes
    ------------------
    # FILE INFO
    path         : str - path the the loaded .off file. Set to None if the geometry is modified.
    meshname     : str - name of the .off file. Remains even when geometry is modified. '_n' is
                   added at the end if the mesh was normalized.

    # GEOMETRY
    vertlist       : (n,3) array of n vertices coordinates
    facelist       : (m,3) array of m triangle indices
    normals        : (m,3) array of normals
    vertex_normals : (n,3) array of vertex normals
                     (change weighting type with self.set_vertex_normal_weighting)

    # SPECTRAL INFORMATION
    W            : (n,n) sparse cotangent weight matrix
    A            : (n,n) sparse area matrix (either diagonal or computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    Properties
    ------------------
    area         : float - area of the mesh
    face_areas   : (m,) per face area
    vertex_areas : (n,) per vertex area
    center_mass  : (3,) center of mass

    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    edges        : (p,2) edges defined by vertex indices
    """
    def __init__(self, *args, **kwargs):
        # area_normalize=False, center=False, rotation=None, translation=None):
        """
        Read the mesh. Give either the path to a .off file or a list of vertices
        and corrresponding triangles

        Parameters
        ----------------------
        path           : path to a .off file
        vertices       : (n,3) vertices coordinates
        faces          : (m,3) list of indices of triangles
        area_normalize : If True, normalize the mesh
        """
        self._init_all_attributes()
        assert 0 < len(args) < 3, "Provide a path or vertices / faces"

        rotation, translation, area_normalize, center = self._read_init_kwargs(kwargs)

        # Differnetiate between [path] or [vertex] or [vertex, faces]
        if len(args) == 1 and type(args[0]) is str:
            self._load_mesh(args[0])
        elif len(args) == 1:
            self.vertlist = args[0]
            self.facelist = None
        else:
            self.vertlist = args[0]
            self.facelist = args[1]

        if rotation is not None:
            self.rotate(rotation)
        if translation is not None:
            self.translate(translation)

        if area_normalize:
            self.area_normalize()

        if center:
            self.translate(-self.center_mass)

    @property
    def vertlist(self):
        """
        Get or set the vertices.
        Checks the format when setting
        """
        return self._vertlist

    @vertlist.setter
    def vertlist(self, vertlist):
        vertlist = np.asarray(vertlist, dtype=float)
        if vertlist.ndim != 2:
            raise ValueError('Vertex list has to be 2D')
        elif vertlist.shape[1] != 3:
            raise ValueError('Vertex list requires 3D coordinates')

        self._reset_vertex_attributes()
        if hasattr(self, "_vertlist") and self._vertlist is not None:
            self._modified = True
            self._normalized = False
        self.path = None
        self._vertlist = vertlist

    @property
    def facelist(self):
        """
        Get or set the faces.
        Checks the format when setting
        """
        return self._facelist

    @facelist.setter
    def facelist(self, facelist):
        facelist = np.asarray(facelist) if facelist is not None else None
        if facelist is not None:
            if facelist.ndim != 2:
                raise ValueError('Faces list has to be 2D')
            elif facelist.shape[1] != 3:
                raise ValueError('Each face is made of 3 points')
            self._facelist = np.asarray(facelist)
        else:
            self._facelist = None
        self.path = None

    @property
    def vertices(self):
        "alias for vertlist"
        return self.vertlist

    @property
    def faces(self):
        "alias for facelist"
        return self.facelist

    @property
    def n_vertices(self):
        """
        return the number of vertices in the mesh
        """
        return self.vertlist.shape[0]

    @property
    def n_faces(self):
        """
        return the number of faces in the mesh
        """
        if self.facelist is None:
            return 0
        return self.facelist.shape[0]

    @property
    def area(self):
        """
        Returns the area of the mesh
        """
        if self.A is None:
            if self.facelist is None:
                return None
            faces_areas = geom.compute_faces_areas(self.vertlist, self.facelist)
            return faces_areas.sum()

        return self.A.sum()

    @property
    def sqrtarea(self):
        """
        square root of the area
        """
        return np.sqrt(self.area)

    @property
    def edges(self):
        """
        return a (p,2) array of edges defined by vertex indices.
        """
        if self._edges is None:
            self.compute_edges()
        return self._edges

    @property
    def normals(self):
        """
        face normals
        """
        if self._normals is None:
            self.compute_normals()
        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    @property
    def vertex_normals(self):
        """
        per vertex_normal
        """
        if self._vertex_normals is None:
            self.compute_vertex_normals()
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, vertex_normals):
        self._vertex_normals = vertex_normals

    @property
    def vertex_areas(self):
        """
        per vertex area
        """
        if self.A is None:
            return geom.compute_vertex_areas(self.vertlist, self.facelist)

        return np.asarray(self.A.sum(1)).squeeze()

    @property
    def faces_areas(self):
        """
        per face area
        """
        if self._faces_areas is None:
            self._faces_areas = geom.compute_faces_areas(self.vertlist, self.facelist)
        return self._faces_areas

    @faces_areas.setter
    def face_areas(self, face_areas):
        self._faces_areas = face_areas

    @property
    def center_mass(self):
        """
        center of mass
        """
        return np.average(self.vertlist, axis=0, weights=self.vertex_areas)

    @property
    def is_normalized(self):
        """
        Whether the mash has been manually normalized using the self.area_normalize method
        """
        if not hasattr(self, "_normalized"):
            self._normalized = False
        return self._normalized

    @property
    def is_modified(self):
        """
        Whether the mash has been modified from path with
        non-isometric deformations
        """
        if not hasattr(self, "_modified"):
            self._modified = False
        return self._modified

    def area_normalize(self):
        self.scale(1/self.sqrtarea)
        self._normalized = True
        return self

    def rotate(self, R):
        """
        Rotate mesh and normals
        """
        if R.shape != (3, 3) or not np.isclose(scipy.linalg.det(R), 1):
            raise ValueError("Rotation should be a 3x3 matrix with unit determinant")

        self._vertlist = self.vertlist @ R.T
        if self._normals is not None:
            self.normals = self.normals @ R.T

        if self._vertex_normals is not None:
            self._vertex_normals = self._vertex_normals @ R.T

        return self

    def translate(self, t):
        """
        translate mesh
        """
        self._vertlist += np.asarray(t).squeeze()[None, :]
        return self

    def scale(self, alpha):
        """
        Multiply mesh by alpha.
        modify vertices, area, spectrum, geodesic distances
        """
        self._vertlist *= alpha

        if self.A is not None:
            self.A = alpha**2 * self.A

        if self._faces_areas is not None:
            self._faces_area *= alpha

        if self.eigenvalues is not None:
            self.eigenvalues = 1 / alpha**2 * self.eigenvalues

        if self.eigenvectors is not None:
            self.eigenvectors = 1 / alpha * self.eigenvectors

        self._solver_heat = None
        self._solver_lap = None
        self._solver_geod = None

        self._modified = True
        self._normalized = False
        return self

    def center(self):
        """
        center the mesh
        """
        self.translate(-self.center_mass)
        return self

    def laplacian_spectrum(self, k, verbose=False):
        """
        Compute the Laplace Beltrami Operator and its spectrum.
        Consider using the .process() function for easier use !

        Parameters
        -------------------------
        K               : int - number of eigenvalues to compute
        intrinsic       : bool - Use intrinsic triangulation
        robust          : bool - use tufted laplacian
        return_spectrum : bool - Whether to return the computed spectrum

        Output
        -------------------------
        eigenvalues, eigenvectors : (k,), (n,k) - Only if return_spectrum is True.
        """
        self.W = laplacian.cotangent_weights(self.vertlist, self.facelist)
        self.A = laplacian.dia_area_mat(self.vertlist, self.facelist)

        # If k is 0, stop here
        if k > 0:
            if verbose:
                print(f"Computing {k} eigenvectors")
                start_time = time.time()
            self.eigenvalues, self.eigenvectors = laplacian.laplacian_spectrum(self.W, self.A,
                                                                               spectrum_size=k)

            if verbose:
                print(f"\tDone in {time.time()-start_time:.2f} s")

    def process(self, k=0, skip_normals=True, verbose=False):
        """
        Process the LB spectrum and saves it.
        Additionnaly computes per-face normals

        Parameters:
        -----------------------
        k            : int - (default = 200) Number of eigenvalues to compute
        skip_normals : bool - If set to True, skip normals computation
        intrinsic    : bool - Use intrinsic triangulation
        robust       : bool - use tufted laplacian
        """
        if not skip_normals and self._normals is None:
            self.compute_normals()

        if (self.eigenvectors is not None) and (self.eigenvalues is not None)\
           and (len(self.eigenvalues) >= k):
            self.eigenvectors = self.eigenvectors[:,:k]
            self.eigenvalues = self.eigenvalues[:k]

        else:
            if self.facelist is None:
                robust = True
            self.laplacian_spectrum(k, verbose=verbose)

        return self

    def l2_sqnorm(self, func):
        """
        Return the squared L2 norm of one or multiple functions on the mesh.
        For a single function f, this returns f.T @ A @ f with A the area matrix.

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of squared l2 norms or a float only one function was provided.
        """
        return self.l2_inner(func, func)

    def l2_inner(self, func1, func2):
        """
        Return the L2 inner product of two functions, or pairwise inner products if lists
        of function is given.

        For two functions f1 and f2, this returns f1.T @ A @ f2 with A the area matrix.

        Parameters
        -----------------
        func1 : (n,p) or (n,) functions on the mesh
        func2 : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of L2 inner product or a float only one function per argument
                  was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.A @ func2

        return np.einsum('np,np->p', func1, self.A@func2)

    def h1_sqnorm(self, func):
        """
        Return the squared H^1_0 norm (L2 norm of the gradient) of one or multiple functions
        on the mesh.
        For a single function f, this returns f.T @ W @ f with W the stiffness matrix.

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of squared H1 norms or a float only one function was provided.
        """
        return self.h1_inner(func, func)

    def h1_inner(self, func1, func2):
        """
        Return the H1 inner product of two functions, or pairwise inner products if lists
        of function is given.

        For two functions f1 and f2, this returns f1.T @ W @ f2 with W the stiffness matrix.

        Parameters
        -----------------
        func1 : (n,p) or (n,) functions on the mesh
        func2 : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of H1 inner product or a float only one function per argument
                  was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.W @ func2

        return np.einsum('np,np->p', func1, self.W@func2)

    def integrate(self, func):
        """
        Integrate a function or a set of function on the mesh

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        integral : (p,) array of integrals or a float only one function was provided.
        """
        if func.ndim == 1:
            return np.sum(self.A @ func)
        return np.sum(self.A @ func, axis=0)
    
    def export(self, filename, precision=None):
        """
        Write the mesh in a .off file

        Parameters
        -----------------------------
        filename  : path to the file to write
        precision : floating point precision
        """
        # assert os.path.splitext(filename)[1] in ['.off',''], "Can only export .off files"
        file_ext = os.path.splitext(filename)[1]
        if file_ext == '':
            filename += '.off'
            file_ext = '.off'

        if file_ext == '.off':
            file_utils.write_off(filename, self.vertlist, self.facelist, precision=precision)

        elif file_ext == '.obj':
            file_utils.write_obj(filename, self.vertlist, self.facelist, precision=precision)

        return self

    def get_uv(self, ind1, ind2, mult_const, rotation=None):
        """
        Extracts UV coordinates for each vertices

        Parameters
        -----------------------------
        ind1       : int - column index to use as first coordinate
        ind2       : int - column index to use as second coordinate
        mult_const : float - number of time to repeat the pattern

        Output
        ------------------------------
        uv : (n,2) UV coordinates of each vertex
        """
        vert = self.vertlist if rotation is None else self.vertlist @ rotation.T
        return file_utils.get_uv(vert, ind1, ind2, mult_const=mult_const)

    def export_texture(self, filename, uv, mtl_file='material.mtl', texture_im='texture_1.jpg',
                       precision=None, verbose=False):
        """
        Write a .obj file with texture using uv coordinates

        Parameters
        ------------------------------
        filename   : str - path to the .obj file to write
        uv         : (n,2) uv coordinates of each vertex
        mtl_file   : str - name of the .mtl file
        texture_im : str - name of the .jpg file definig texture
        """
        if os.path.splitext(filename)[1] != '.obj':
            filename += '.obj'

        file_utils.write_obj(filename, self.vertlist, self.facelist, uv=uv,
                             mtl_file=mtl_file, texture_im=texture_im, verbose=verbose)

        return self

    def compute_normals(self):
        """
        Compute normal vectors for each face
        """
        self.normals = geom.compute_normals(self.vertlist, self.facelist)

    def set_vertex_normal_weighting(self, weight_type):
        """
        Set weighting type for vertex normals between 'area' and 'uniform'
        """
        weight_type = weight_type.lower()
        assert weight_type in ['uniform', 'area'], "Only implemented uniform and area weighting"

        if weight_type != self._vertex_normals_weighting:
            self._vertex_normals_weighting = weight_type
            self._vertex_normals = None

    def compute_vertex_normals(self):
        """
        computes vertex normals in self.vertex_normals
        """
        self.vertex_normals = geom.per_vertex_normal(self.vertlist, self.facelist,
                                                     weighting=self._vertex_normals_weighting)

    def compute_edges(self):
        """
        computes edges in self.edges
        """
        self._edges = geom.edges_from_faces(self.facelist)

    def _reset_vertex_attributes(self):
        """
        Resets attributes which depend on the vertex positions
        in the case of nonisometric deformation
        """
        self._face_areas = None

        self._normals = None
        self._vertex_normals = None

        self._intrinsic = False

        self.W = None
        self.A = None

        self.eigenvalues = None
        self.eigenvectors = None

        self._solver_heat = None
        self._solver_lap = None
        self._solver_geod = None

    def _load_mesh(self, meshpath):
        """
        Load a mesh from a file

        Parameters:
        --------------------------
        meshpath : path to file
        """

        if os.path.splitext(meshpath)[1] == '.off':
            self.vertlist, self.facelist = file_utils.read_off(meshpath)
        elif os.path.splitext(meshpath)[1] == '.obj':
            self.vertlist, self.facelist = file_utils.read_obj(meshpath)

        else:
            raise ValueError('Provide file in .off or .obj format')

        self.path = meshpath
        self.meshname = os.path.splitext(os.path.basename(meshpath))[0]

    def _read_init_kwargs(self, kwargs):
        rotation = kwargs['rotation'] if 'rotation' in kwargs.keys() else None
        translation = kwargs['translation'] if 'translation' in kwargs.keys() else None
        area_normalize = kwargs['area_normalize'] if 'area_normalize' in kwargs.keys() else False
        center = kwargs['center'] if 'center' in kwargs.keys() else False
        return rotation, translation, area_normalize, center

    def _init_all_attributes(self):

        self.path = None
        self.meshname = None

        self._vertlist = None
        self._facelist = None

        self._modified = False
        self._normalized = False

        self._edges = None
        self._normals = None

        self._vertex_normals_weighting = 'area'
        self._vertex_normals = None

        self.W = None
        self.A = None
        self._intrinsic = False

        self._faces_areas = None

        self.eigenvalues = None
        self.eigenvectors = None

        self._solver_geod = None
        self._solver_heat = None
        self._solver_lap = None
