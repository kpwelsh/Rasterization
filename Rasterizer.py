import numpy as np


class Rasterizer:
    MAX_INT = 100_00
    MIN_INT = -MAX_INT

    @staticmethod
    def line(a, b):
        d = b - a
        k = np.where(d >= 0, 1, -1)
        d = np.abs(d)

        order = np.flip(np.argsort(d))
        unorder = np.argsort(order)
        res = Rasterizer.__line((a * k)[order], d[order])[:, unorder] * k
        return res

    @staticmethod
    def __line(a, d):
        indices = np.zeros((d[0] + 1, len(a)), dtype=int)
        indices[:, 0] = np.arange(a[0], a[0] + d[0] + 1)
        for i in range(1, len(a)):
            indices[:, i] = Rasterizer.intlerp(a[i], a[i] + d[i], d[0] + 1)
        return indices
    

    @staticmethod
    def intlerp(a, b, n):
        v = []
        d = b - a

        for i in range(n):
            if i == 0:
                j = 0
            else:
                j = int((i * d) / (n - 1) + 0.5)
            v.append(a + j)
        return v


    @staticmethod
    def polygon(vertices):
        xl = min(vertices, key=lambda v: v[0])[0]
        xr = max(vertices, key=lambda v: v[0])[0]
        buffer_size = xr + 1 - xl

        min_buffer = np.full(buffer_size, Rasterizer.MAX_INT)
        max_buffer = np.full(buffer_size, Rasterizer.MIN_INT)

        lines = [
            (vertices[i-1], vertices[i])
            for i in range(len(vertices))
        ]
        for line in lines:
            for x, y in Rasterizer.line(*line):
                i = x - xl
                min_buffer[i] = min(y, min_buffer[i])
                max_buffer[i] = max(y, max_buffer[i])
        
        for i in range(len(min_buffer)):
            x = xl + i
            for y in range(min_buffer[i], max_buffer[i]+1):
                yield x, y
    
    @staticmethod
    def polyhedron(faces, lb, ub):
        buffer_size = 1 + (ub - lb)[:2]
        min_buffer = np.full(buffer_size, Rasterizer.MAX_INT)
        max_buffer = np.full(buffer_size, Rasterizer.MIN_INT)

        for f in faces:
            for x, y, z in Rasterizer.face(f):
                i = x - lb[0]
                j = y - lb[1]
                min_buffer[i, j] = min(z, min_buffer[i, j])
                max_buffer[i, j] = max(z, max_buffer[i, j])
        
        indices = []
        for i, j in np.column_stack(np.where(min_buffer > Rasterizer.MIN_INT)):
            x, y = lb[0] + i, lb[1] + j
            for z in range(min_buffer[i, j], max_buffer[i, j] + 1):
                indices.append((x, y, z))
        return np.array(indices)

    @staticmethod
    #@profile
    def face(edges):
        edges = np.array(edges)
        lb, ub = np.min(edges, axis=(0, 1)), np.max(edges, axis=(0, 1))
        order = np.flip(np.argsort(ub - lb))
        unorder = np.argsort(order)
        return Rasterizer.__face(edges[:, :, order], lb[order], ub[order])[:, unorder]

    @staticmethod
    #@profile
    def __face(edges, lb, ub):
        height = np.full(1 + (ub - lb)[:2], np.nan)

        for a, b in edges:
            for x, y, z in Rasterizer.line(a, b):
                height[x - lb[0], y - lb[1]] = z
        
        indices = []

        for i, col in enumerate(height):
            start_j = start_z = None
            end_j = end_z = None

            for j in range(len(col)):
                v = col[j]
                if start_z is None and not np.isnan(v):
                    start_j, start_z = j, int(v)
                    break
            
            for j in range(len(col)-1, -1, -1):
                v = col[j]
                if end_z is None and not np.isnan(v):
                    end_j, end_z = j, int(v)
                    break
            
            indices.append(Rasterizer.line(
                np.array((lb[0] + i, lb[1] + start_j, start_z)),
                np.array((lb[0] + i, lb[1] + end_j, end_z))
            ))
        return np.concatenate(indices, axis=0)
