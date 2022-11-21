import numpy as np
import matplotlib.pyplot as plt
from typing import List
from Rasterizer import Rasterizer


def box_faces(center, size, R=np.eye(3)):
    import itertools
    for i in range(3):
        vertices = list(np.column_stack((-size, size)))

        for j in [-1, 1]:
            face_verts = []
            vertices[i] = [j * size[i]]
            for v in itertools.product(*vertices):
                v = R @ np.array(v)
                face_verts.append((center + v).astype(int))
            face_verts[-2], face_verts[-1] = face_verts[-1], face_verts[-2]
            yield [
                (face_verts[k-1], face_verts[k])
                for k in range(len(face_verts))
            ]



def test_line():
    import matplotlib.pyplot as plt

    a = np.array((20, -20))
    b = np.array((10, -7))

    plt.scatter(*Rasterizer.line(a, b).T)
    plt.show()


def test_line3d():
    import matplotlib.pyplot as plt

    a = np.array((5, 8, 10))
    b = np.array((10, -5, 5))

    ax = plt.subplot(projection='3d')
    ax.scatter(*Rasterizer.line(a, b).T)

    plt.show()


def test_polygon():
    grid = np.zeros((50, 50))
    shape = np.array([
        (10, 10),
        (20, 23),
        (30, 20),
        (15, 5),
    ])
    for i in Rasterizer.polygon(shape):
        grid[i] += 1
    plt.imshow(grid.T, origin='lower')
    plt.show()

def test_face():
    vertices = [
        (1, 1, 5),
        (1, 5, 5),
        (5, 5, 10),
        (5, 1, 10)
    ]

    edges = [
        (vertices[i-1], vertices[i])
        for i in range(len(vertices))
    ]

    ax = plt.subplot(projection='3d')
    ax.scatter(*Rasterizer.face(edges).T)
    plt.show()


def test_volume():
    from matplotlib.animation import FuncAnimation
    from scipy.spatial.transform import Rotation

    center = np.array((0, 0, 0))
    size = np.array((3, 4, 5))

    rot_vec = np.random.rand(3) - 0.5
    rot_vec /= np.linalg.norm(rot_vec)
    ax = plt.subplot(projection='3d')

    import time
    s = time.perf_counter()
    for t in np.linspace(0, 2 * np.pi, 200):
        R = Rotation.from_rotvec(t * rot_vec).as_matrix()
        faces = list(box_faces(center, size, R))
        lb = np.min(np.array(faces), axis=(0, 1, 2)).astype(int)
        ub = np.max(np.array(faces), axis=(0, 1, 2)).astype(int)
        points = Rasterizer.polyhedron(faces, lb, ub)
    e = time.perf_counter()
    print((e - s) / 200)
    return 
    def update(t):
        R = Rotation.from_rotvec(t * rot_vec).as_matrix()
        
        faces = list(box_faces(center, size, R))
        lb = np.min(np.array(faces), axis=(0, 1, 2)).astype(int)
        ub = np.max(np.array(faces), axis=(0, 1, 2)).astype(int)

        points = Rasterizer.polyhedron(faces, lb, ub)

        ax.clear()
        ax.scatter(*np.array(points).T)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])

    ani = FuncAnimation(plt.gcf(), update, np.linspace(0, 2 * np.pi, 200))
    plt.show()


if __name__ == '__main__':
    test_polygon()