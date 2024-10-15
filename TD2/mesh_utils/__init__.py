import numpy as np
def read_obj_texture(path):
    """
    read a .obj file with vertex faces and uv coordinates
    """

    with open(path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    uv = []
    for line in lines:
        if line.startswith('v '):
            vertices.append([float(x) for x in line[2:].split()])
        elif line.startswith('vt '):
            uv.append([float(x) for x in line[3:].split()])
        elif line.startswith('f '):
            faces.append([int(x.split('/')[0])-1 for x in line[2:].split()])

    return np.array(vertices), np.array(faces), np.array(uv)