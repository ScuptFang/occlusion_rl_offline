"""Prepare upper-surface point features and a lower signed-distance grid."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .data import load_case, sample_sdf


def _mesh(path):
    mesh = trimesh.load(str(path), force='mesh', process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError('Expected a nonempty STL mesh: %s' % path)
    if not mesh.is_winding_consistent:
        raise ValueError('Mesh face winding is inconsistent: %s' % path)
    return mesh


def _upper_features(mesh, cfg):
    axis = int(cfg['occlusal_axis'])
    direction = int(cfg['occlusal_direction'])
    if axis not in (0, 1, 2) or direction not in (-1, 1):
        raise ValueError('occlusal_axis must be 0..2 and direction must be -1 or 1')
    centers = mesh.triangles_center[:, axis]
    extreme = mesh.bounds[0, axis] if direction == -1 else mesh.bounds[1, axis]
    selected = (centers - extreme) * (-direction) <= cfg['occlusal_band_mm']
    weights = mesh.area_faces * selected
    if weights.sum() <= 0:
        raise ValueError('Occlusal band selected no upper mesh faces')
    count = int(cfg['upper_surface_points'])
    xyz, face = trimesh.sample.sample_surface(mesh, count, face_weight=weights,
                                              seed=int(cfg['seed']))
    normal = mesh.face_normals[face]
    tree = cKDTree(xyz)
    neighbors = min(int(cfg['curvature_neighbors']) + 1, count)
    _, ids = tree.query(xyz, k=neighbors, workers=-1)
    if ids.ndim == 1:
        ids = ids[:, None]
    agreement = np.einsum('ij,ikj->ik', normal, normal[ids])
    curvature = (1 - np.clip(agreement[:, 1:], -1, 1)).mean(axis=1)
    scale = max(float(np.quantile(curvature, .99)), 1e-8)
    curvature = np.clip(curvature / scale, 0, 1)
    return xyz.astype(np.float32), normal.astype(np.float32), curvature.astype(np.float32)


def _lower_sdf(lower, upper, cfg):
    pitch = float(cfg['sdf_pitch_mm'])
    margin = float(cfg['sdf_margin_mm'])
    if pitch <= 0 or margin <= 0:
        raise ValueError('sdf_pitch_mm and sdf_margin_mm must be positive')
    bounds = np.array([np.minimum(lower.bounds[0], upper.bounds[0]),
                       np.maximum(lower.bounds[1], upper.bounds[1])])
    origin = np.floor((bounds[0] - margin) / pitch) * pitch
    end = np.ceil((bounds[1] + margin) / pitch) * pitch
    shape = np.rint((end - origin) / pitch).astype(int) + 1
    if np.prod(shape) > 80000000:
        raise ValueError('SDF grid would exceed 80 million cells; increase pitch or reduce margin')
    sampled, faces = trimesh.sample.sample_surface(lower, int(cfg['lower_surface_samples']),
                                                    seed=int(cfg['seed']) + 1)
    normals = lower.face_normals[faces]
    tree = cKDTree(sampled)
    grid = np.empty(tuple(shape), dtype=np.float32)
    ys = origin[1] + np.arange(shape[1]) * pitch
    zs = origin[2] + np.arange(shape[2]) * pitch
    for first in range(0, shape[0], 8):
        xs = origin[0] + np.arange(first, min(first + 8, shape[0])) * pitch
        xyz = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
        distance, ids = tree.query(xyz, workers=-1)
        outside = np.einsum('ij,ij->i', xyz - sampled[ids], normals[ids]) >= 0
        signed = np.where(outside, distance, -distance)
        grid[first:first + len(xs)] = signed.reshape(len(xs), shape[1], shape[2]).astype(np.float32)
    return grid, origin.astype(np.float32), np.full(3, pitch, dtype=np.float32)


def prepare_case(upper_path, lower_path, out_path, cfg):
    upper = _mesh(upper_path)
    lower = _mesh(lower_path)
    xyz, normal, curvature = _upper_features(upper, cfg)
    grid, origin, spacing = _lower_sdf(lower, upper, cfg)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xyz=xyz, normal=normal, curvature=curvature,
                        sdf_grid=grid, origin=origin, spacing=spacing)
    case = load_case(out)
    probe = sample_sdf(grid, origin, spacing, xyz[:min(2048, len(xyz))])
    audit = {'upper_stl': str(upper_path), 'lower_stl': str(lower_path),
             'upper_watertight': bool(upper.is_watertight),
             'lower_watertight': bool(lower.is_watertight),
             'upper_points': len(xyz), 'grid_shape': list(grid.shape),
             'grid_pitch_mm': float(spacing[0]),
             'probe_sdf_negative_fraction': float((probe < 0).mean()),
             'probe_sdf_near_0_6mm_fraction': float((np.abs(probe) < .6).mean()),
             'sdf_method': 'nearest sampled lower surface point; sign from its face normal',
             'config': cfg}
    (out.parent / (out.stem + '.audit.json')).write_text(json.dumps(audit, indent=2))
    return case, audit
