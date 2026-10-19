"""Recover original mesh topology before teacher/agent candidate sampling."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .region_segmentation import predictor, sample_ids, pack_features


def enabled(options):
    return bool(((options or {}).get('segmentation_region') or {}).get('mesh_before_sampling', False))


@torch.no_grad()
def mesh_probabilities(case, jaw, options, rotation=None, translation=None, xyz=None, normal=None):
    prefix = 'lower_' if jaw else ''
    original = case[prefix + 'xyz']
    model, sampling = predictor(options)
    key = (jaw, options['sha256'], options.get('interpolation_neighbors', 3))
    cache = case.setdefault('_mesh_interpolation', {})
    if key not in cache:
        ids = sample_ids(original, sampling['points'], sampling['candidates'], sampling['seed'])
        distance, index = cKDTree(original[ids]).query(original, k=min(key[2], len(ids)))
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        weight = 1 / np.maximum(distance, 1e-8) ** 2
        weight /= weight.sum(1, keepdims=True)
        cache[key] = ids, index, weight
    ids, index, weight = cache[key]
    # Only rigid-invariant interpolation weights/IDs are cached. Predictions are
    # recomputed from the CURRENT pose and normals on every upper-jaw call.
    if xyz is not None:
        points, normals = xyz[ids], normal[ids]
    elif rotation is not None:
        pivot = original.mean(0)
        points = (original[ids] - pivot) @ rotation.T + pivot + translation
        normals = case[prefix + 'normal'][ids] @ rotation.T
    else:
        points, normals = original[ids], case[prefix + 'normal'][ids]
    features = pack_features(points, normals, case[prefix + 'curvature'][ids], jaw)
    probability = model(torch.from_numpy(features)[None].to(next(model.parameters()).device)).sigmoid()[0].cpu().numpy()
    return (probability[index] * weight).sum(1).astype(np.float32)


def recover_mesh(case, jaw, probability, options):
    """Select original triangles, compact their vertices, retain both ID maps.

    STL has triangle corners, not intrinsic shared vertex IDs. Source vertex IDs
    refer to the prepared STL's welded vertex table; face order is preserved.
    Normals/curvature come from the original complete mesh, not a cut boundary.
    """
    prefix = 'lower_' if jaw else ''
    faces_key = 'lower_faces' if jaw else 'upper_faces'
    if faces_key not in case:
        raise ValueError('Original upper/lower triangles required; rerun prepare')
    faces = case[faces_key]
    rule = options.get('face_rule', 'all')
    if rule not in ('all', 'mean'):
        raise ValueError('Unknown region face rule: ' + rule)
    face_score = probability[faces].min(1) if rule == 'all' else probability[faces].mean(1)
    keep = face_score >= options.get('threshold', .5)
    minimum = options.get('minimum_lower_faces' if jaw else 'minimum_upper_faces', 128)
    fallback = int(keep.sum()) < minimum
    if fallback:
        keep[np.argsort(face_score)[-min(minimum, len(faces)):]] = True
    face_ids = np.flatnonzero(keep)
    selected_faces = faces[face_ids]
    vertex_mask = np.zeros(len(probability), dtype=bool)
    vertex_mask[selected_faces] = True
    vertex_ids = np.flatnonzero(vertex_mask)
    inverse = np.full(len(probability), -1, dtype=np.int64)
    inverse[vertex_ids] = np.arange(len(vertex_ids))
    compact = inverse[selected_faces]
    return dict(xyz=case[prefix + 'xyz'][vertex_ids], normal=case[prefix + 'normal'][vertex_ids],
                curvature=case[prefix + 'curvature'][vertex_ids], faces=compact,
                vertex_ids=vertex_ids, face_ids=face_ids, fallback=fallback)


def upper_mesh_candidates(case, options, rotation=None, translation=None, xyz=None, normal=None):
    region_options = options['segmentation_region']
    probability = mesh_probabilities(case, 0, region_options, rotation, translation, xyz, normal)
    mesh = recover_mesh(case, 0, probability, region_options)
    count = options['candidate_points']
    if len(mesh['vertex_ids']) < count:
        raise ValueError('Predicted upper mesh has %d vertices, fewer than %d candidates; no duplication or outside-region padding' %
                         (len(mesh['vertex_ids']), count))
    local_ids = np.random.RandomState(options.get('point_seed', 42)).choice(len(mesh['vertex_ids']), count, replace=False)
    ids = mesh['vertex_ids'][local_ids]
    case['_upper_region_mesh'] = mesh
    case['_upper_candidate_ids'] = ids
    case['_upper_region_audit'] = dict(vertices=len(mesh['vertex_ids']), faces=len(mesh['face_ids']),
        candidates=len(ids), fallback=bool(mesh['fallback']), sampling_order='segment->original_mesh->uniform_candidates->distance_fps')
    return ids


def lower_mesh_region(case, options):
    key = (options['sha256'], options.get('threshold', .5), options.get('face_rule', 'all'),
           options.get('minimum_lower_faces', 128), options.get('interpolation_neighbors', 3))
    cached = case.get('_lower_mesh_region')
    if cached is not None and cached[0] == key:
        return cached[1]
    mesh = recover_mesh(case, 1, mesh_probabilities(case, 1, options), options)
    region = dict(lower_xyz=mesh['xyz'], lower_normal=mesh['normal'], lower_faces=mesh['faces'])
    case['_lower_region_mesh'] = mesh
    case['_lower_region_audit'] = dict(vertices=len(mesh['vertex_ids']), faces=len(mesh['face_ids']),
                                       total_faces=len(case['lower_faces']), fallback=bool(mesh['fallback']))
    case['_lower_mesh_region'] = key, region
    return region
