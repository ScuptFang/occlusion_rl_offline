"""Prepare original upper/lower mesh vertices for nearest-point signed distance."""
import json
from pathlib import Path

import numpy as np
import trimesh

from .data import load_case, sample_sdf


def _mesh(path):
    mesh = trimesh.load(str(path), force='mesh', process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('Expected a nonempty STL mesh: %s' % path)
    if not mesh.is_winding_consistent:
        raise ValueError('Mesh face winding is inconsistent: %s' % path)
    return mesh


def _upper_features(mesh):
    # Keep every vertex from the loaded mesh. STL has no shared-vertex table;
    # trimesh merges coincident triangle corners when process=True in _mesh.
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    normal = np.asarray(mesh.vertex_normals)
    faces = np.asarray(mesh.faces)
    a, b, c = (xyz[faces[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    twice_area = np.linalg.norm(cross, axis=1)
    valid = twice_area > 1e-12
    area = np.zeros(len(xyz), dtype=np.float64)
    laplacian = np.zeros_like(xyz)
    for vertex_ids in faces.T:
        np.add.at(area, vertex_ids[valid], twice_area[valid] / 6)
    # Each triangle angle contributes a cotangent weight to its opposite edge.
    # The resulting mesh Laplacian uses vertex-edge-face adjacency, not a
    # Euclidean k-nearest-neighbor normal-difference proxy.
    for i, j, k in ((1, 2, 0), (2, 0, 1), (0, 1, 2)):
        edge_i, edge_j = faces[valid, i], faces[valid, j]
        opposite = (a, b, c)[k][valid]
        vi, vj = xyz[edge_i], xyz[edge_j]
        cotangent = np.einsum('ij,ij->i', vi - opposite, vj - opposite) / twice_area[valid]
        contribution = cotangent[:, None] * (vj - vi)
        np.add.at(laplacian, edge_i, contribution)
        np.add.at(laplacian, edge_j, -contribution)
    # |sum_j (cot alpha + cot beta)(v_j-v_i)| / (4 A_i)
    curvature = np.linalg.norm(laplacian, axis=1) / (4 * np.maximum(area, 1e-12))
    scale = max(float(np.quantile(curvature, .99)), 1e-8)
    curvature = np.clip(curvature / scale, 0, 1)
    return xyz.astype(np.float32), normal.astype(np.float32), curvature.astype(np.float32)


def prepare_case(upper_path, lower_path, out_path, cfg):
    upper = _mesh(upper_path)
    lower = _mesh(lower_path)
    xyz, normal, curvature = _upper_features(upper)
    lower_xyz = np.asarray(lower.vertices, dtype=np.float32)
    lower_normal = np.asarray(lower.vertex_normals, dtype=np.float32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xyz=xyz, normal=normal, curvature=curvature,
                        lower_xyz=lower_xyz, lower_normal=lower_normal)
    case = load_case(out)
    probe = sample_sdf(case, xyz[:min(2048, len(xyz))])
    audit = {'upper_stl': str(upper_path), 'lower_stl': str(lower_path),
             'upper_watertight': bool(upper.is_watertight),
             'lower_watertight': bool(lower.is_watertight),
             'upper_points': len(xyz), 'upper_point_source': 'mesh_vertices',
             'upper_normal_source': 'mesh_vertex_normals',
             'upper_curvature_method': 'cotangent_laplacian_mean_curvature',
             'lower_points': len(lower_xyz),
             'probe_sdf_negative_fraction': float((probe < 0).mean()),
             'probe_sdf_near_0_6mm_fraction': float((np.abs(probe) < .6).mean()),
             'sdf_method': 'nearest original lower mesh vertex; sign from its vertex normal; no grid',
             'config': cfg}
    (out.parent / (out.stem + '.audit.json')).write_text(json.dumps(audit, indent=2))
    return case, audit
