"""Exact closest triangle distance with barycentrically interpolated vertex normals.

The sign is local, not an inside/outside test for open meshes. Boundary and
tangential signs are reported as uncertain; distances are never zeroed to hide it.
"""
import numpy as np
import trimesh
import igl


def surface_distance(case, points, chunk_size=4096):
    if 'lower_faces' not in case:
        raise ValueError('nearest_surface requires lower_faces; rerun prepare-batch from STL')
    if '_surface_mesh' not in case:
        faces = np.asarray(case['lower_faces'])
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(case['lower_xyz'])):
            raise ValueError('Invalid lower triangle vertex indices')
        mesh = trimesh.Trimesh(vertices=case['lower_xyz'], faces=faces, process=False)
        valid = mesh.area_faces > 1e-12
        case['_surface_degenerate_faces'] = int((~valid).sum())
        if not valid.any():
            raise ValueError('Lower mesh has no nondegenerate triangles')
        if not valid.all():
            mesh = trimesh.Trimesh(vertices=case['lower_xyz'], faces=faces[valid], process=False)
        if not mesh.is_winding_consistent:
            raise ValueError('Lower mesh winding is inconsistent')
        edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
        boundary = np.zeros(len(mesh.vertices), dtype=bool)
        boundary[edges[counts != 2].reshape(-1)] = True
        tree = igl.AABB()
        tree.init(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        case['_surface_tree'] = tree
        case['_surface_mesh'] = mesh
        case['_surface_boundary'] = boundary
    mesh = case['_surface_mesh']
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite query points with shape [N,3]')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if len(points) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
    signed, confidence = [], []
    for start in range(0, len(points), chunk_size):
        xyz = np.asarray(points[start:start + chunk_size], dtype=np.float64)
        squared, triangle, closest = case['_surface_tree'].squared_distance(
            np.asarray(mesh.vertices), np.asarray(mesh.faces), xyz)
        distance = np.sqrt(np.maximum(squared, 0))
        faces = np.asarray(mesh.faces)[triangle]
        weights = trimesh.triangles.points_to_barycentric(mesh.triangles[triangle], closest)
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum(axis=1, keepdims=True)
        normals = np.einsum('ni,nij->nj', weights, case['lower_normal'][faces])
        norm = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(norm[:, None], 1e-12)
        direction = np.einsum('ij,ij->i', xyz - closest, normals)
        signed.append(np.where(direction >= 0, distance, -distance))
        boundary = (case['_surface_boundary'][faces] & (weights > 1e-6)).any(axis=1)
        confidence.append((~boundary) & (norm > 1e-6) &
                          (np.abs(direction) > 1e-4 * np.maximum(distance, 1e-8)))
    return np.concatenate(signed).astype(np.float32), np.concatenate(confidence)
