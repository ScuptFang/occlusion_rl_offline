from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from scipy.spatial import cKDTree

def sample_sdf(case, xyz, options=None):
    """Configured signed vertex/surface distance; no volumetric inside/outside test."""
    if (options or {}).get('segmentation_region'):
        from .region_mesh import enabled, lower_mesh_region
        if enabled(options):
            case = lower_mesh_region(case, options['segmentation_region'])
        else:
            from .region_segmentation import lower_region
            case = lower_region(case, options['segmentation_region'])
    if (options or {}).get('distance_method', 'nearest_vertex') == 'nearest_surface':
        from .surface import surface_distance
        return surface_distance(case, xyz)[0]
    if (options or {}).get('distance_method', 'nearest_vertex') != 'nearest_vertex':
        raise ValueError('Unknown distance_method')
    if '_lower_tree' not in case:
        case['_lower_tree'] = cKDTree(case['lower_xyz'])
    distance, ids = case['_lower_tree'].query(xyz, workers=-1 if len(xyz) >= 4096 else 1)
    direction = np.einsum('ij,ij->i', xyz - case['lower_xyz'][ids],
                          case['lower_normal'][ids])
    return np.where(direction >= 0, distance, -distance).astype(np.float32)


def candidate_indices(case, options=None):
    options = options or {}
    count = int(options.get('candidate_points', len(case['xyz'])))
    if not 1 <= count <= len(case['xyz']):
        raise ValueError('candidate_points must fit the upper mesh')
    key = (count, int(options.get('point_seed', 42)))
    candidates = case.setdefault('_uniform_candidates', {})
    if key not in candidates:
        candidates[key] = (np.arange(len(case['xyz'])) if count == len(case['xyz'])
                           else np.random.RandomState(key[1]).choice(len(case['xyz']), count, replace=False))
    return candidates[key]


def sample_upper_mesh(case, xyz, normal, count, fps_options=None, return_indices=False,
                      preselected_ids=None, precomputed_sdf=None):
    """Uniformly select mesh-vertex candidates, then distance-weighted FPS."""
    options = fps_options or {}
    candidate_count = int(options.get('candidate_points', len(case['xyz'])))
    if count < 1 or candidate_count < count or candidate_count > len(case['xyz']):
        raise ValueError('Require fps_points <= candidate_points <= upper mesh vertices')
    from .region_mesh import enabled, upper_mesh_candidates
    if preselected_ids is None and enabled(options):
        ids = upper_mesh_candidates(case, options, xyz=xyz, normal=normal)
    else:
        ids = candidate_indices(case, options) if preselected_ids is None else preselected_ids
    if preselected_ids is None:
        xyz, normal = xyz[ids], normal[ids]
    if len(xyz) != len(ids):
        raise ValueError('Candidate positions and original vertex IDs must match')
    xyz = np.asarray(xyz, dtype=np.float32)
    d = sample_sdf(case, xyz, options) if precomputed_sdf is None else precomputed_sdf
    if options.get('segmentation_region') and not enabled(options):
        from .region_segmentation import upper_region_mask
        mask = upper_region_mask(case, xyz, normal, ids, count, options['segmentation_region'])
        xyz, normal, ids, d = xyz[mask], normal[mask], ids[mask], d[mask]
    absolute = np.abs(d)
    scales = options.get('contact_scales', (0.15, 0.6, 1.5))
    importance = (options.get('weight_floor', 0.15)
                  + np.exp(-0.5 * (absolute / scales[1]) ** 2)
                  + 0.5 * (d < 0) * np.exp(-absolute))
    priority = (1 + options.get('fps_strength', 1.5) * importance).astype(np.float32)
    contact_fraction = float(options.get('contact_fps_fraction', 0.0))
    if not 0 <= contact_fraction <= 1:
        raise ValueError('contact_fps_fraction must be between 0 and 1')
    contact_count = min(count, int(round(count * contact_fraction)))
    coverage_count = count - contact_count
    contact_mask = None
    if contact_count:
        threshold = float(options.get('contact_fps_max_mm', 0.6))
        pool_factor = int(options.get('contact_fps_pool_factor', 4))
        if threshold <= 0 or pool_factor < 1:
            raise ValueError('Contact FPS threshold and pool factor must be positive')
        pool_size = min(len(absolute), contact_count * pool_factor)
        cutoff = max(threshold, float(np.partition(absolute, pool_size - 1)[pool_size - 1]))
        contact_mask = absolute <= cutoff
    nearest = np.full(len(xyz), np.inf, dtype=np.float32)
    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(len(xyz), dtype=bool)
    if coverage_count:
        current = int(np.argmax(np.einsum('ij,ij->i', xyz - xyz.mean(0), xyz - xyz.mean(0))))
    else:
        contact_center = xyz[contact_mask].mean(0)
        seed_distance = np.einsum('ij,ij->i', xyz - contact_center, xyz - contact_center)
        current = int(np.argmax(np.where(contact_mask, seed_distance, -np.inf)))
    for j in range(count):
        selected[j] = current
        used[current] = True
        delta = xyz - xyz[current]
        np.minimum(nearest, np.einsum('ij,ij->i', delta, delta), out=nearest)
        if j + 1 < count:
            scores = nearest * priority
            if contact_mask is not None and j + 1 >= coverage_count and (contact_mask & ~used).any():
                current = int(np.argmax(np.where(contact_mask & ~used, scores, -np.inf)))
            else:
                current = int(np.argmax(np.where(~used, scores, -np.inf)))
    selected_ids = ids[selected]
    features = np.column_stack((xyz[selected], normal[selected],
                                case['curvature'][selected_ids].reshape(-1), d[selected])).astype(np.float32)
    return (features, selected_ids) if return_indices else features


def features_at_pose(case, rotation, translation, count, options=None, return_indices=False):
    """Select candidates for this pose; preserve full-mesh pivot and vertex identity."""
    from .region_mesh import enabled, upper_mesh_candidates
    ids = (upper_mesh_candidates(case, options, rotation, translation) if enabled(options)
           else candidate_indices(case, options))
    if '_pivot' not in case:
        case['_pivot'] = case['xyz'].mean(0)
    pivot = case['_pivot']
    xyz = np.einsum('nj,ij->ni', case['xyz'][ids] - pivot, rotation) + pivot + translation
    normal = np.einsum('nj,ij->ni', case['normal'][ids], rotation)
    return sample_upper_mesh(case, xyz, normal, count, options, return_indices,
                             preselected_ids=ids)


def transform(points, normals, trans, angles_deg, pivot):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', angles_deg, degrees=True).as_matrix().astype(np.float32)
    return (points - pivot) @ rot.T + pivot + trans, normals @ rot.T


def load_case(path):
    with np.load(path) as f:
        required = ('xyz', 'normal', 'curvature', 'lower_xyz', 'lower_normal')
        if any(k not in f for k in required):
            raise ValueError('%s requires nearest-vertex format; rerun prepare from STL (keys: %s)' % (path, required))
        case = {k: f[k].astype(np.float32) for k in required}
        if 'lower_faces' in f:
            case['lower_faces'] = f['lower_faces'].astype(np.int64)
        if 'upper_faces' in f:
            case['upper_faces'] = f['upper_faces'].astype(np.int64)
        if 'lower_curvature' in f:
            case['lower_curvature'] = f['lower_curvature'].astype(np.float32)
    n = len(case['xyz'])
    assert case['xyz'].shape == (n, 3) and case['normal'].shape == (n, 3)
    assert case['curvature'].shape in ((n,), (n, 1))
    assert case['lower_xyz'].shape == case['lower_normal'].shape
    assert case['lower_xyz'].ndim == 2 and case['lower_xyz'].shape[1] == 3
    assert np.isfinite(case['lower_xyz']).all() and np.isfinite(case['lower_normal']).all()
    return case


class PoseDataset(Dataset):
    """One case per item; fresh synthetic pose pairs every epoch, no case leakage."""
    def __init__(self, paths, points=128, repeats=64, seed=42,
                 near_translation_mm=0.1, near_rotation_deg=0.2,
                 far_magnitude_min=0.35, far_magnitude_max=2.0, fps_options=None,
                 ranking_pose_bands=None):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError('No .npz cases found')
        self.points, self.repeats, self.seed, self.epoch = points, repeats, seed, 0
        self.near_translation_mm = near_translation_mm
        self.near_rotation_deg = near_rotation_deg
        self.far_magnitude_min = far_magnitude_min
        self.far_magnitude_max = far_magnitude_max
        self.fps_options = fps_options
        self.ranking_pose_bands = ranking_pose_bands
        if ranking_pose_bands is not None:
            if len(ranking_pose_bands) < 2 or any(
                band['min'] < 0 or band['max'] <= band['min'] or
                (i and band['min'] < ranking_pose_bands[i - 1]['max']) or
                band.get('translation_scale_mm', 1.0) <= 0 or
                band.get('rotation_scale_deg', 2.0) <= 0
                for i, band in enumerate(ranking_pose_bands)):
                raise ValueError('Ranking pose bands must be ordered and non-overlapping')

    def __len__(self):
        return len(self.paths) * self.repeats

    @lru_cache(maxsize=2)
    def case(self, index):
        # Keep only two case meshes and nearest-neighbor trees in memory.
        return load_case(self.paths[index])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def pose(self, case, trans, angles):
        xyz = case['xyz']
        normal = case['normal']
        pivot = case['xyz'].mean(0)
        xyz, normal = transform(xyz, normal, trans, angles, pivot)
        return torch.from_numpy(sample_upper_mesh(case, xyz, normal, self.points, self.fps_options))

    def __getitem__(self, index):
        item_seed = self.seed + self.epoch * 1000003 + index
        pose_rng = np.random.RandomState(item_seed + 1000000000)
        case = self.case(index // self.repeats)
        if self.ranking_pose_bands is not None:
            bands = self.ranking_pose_bands
            pairs = [(i, j) for i in range(len(bands)) for j in range(i + 1, len(bands))]
            good_band, bad_band = (bands[k] for k in pairs[(index % self.repeats + self.epoch) % len(pairs)])
            direction = pose_rng.normal(size=6)
            direction /= np.linalg.norm(direction) + 1e-8
            def draw(band):
                magnitude = pose_rng.uniform(band['min'], band['max'])
                trans = (direction[:3] * magnitude * np.sqrt(6) *
                         band.get('translation_scale_mm', 1.0)).astype(np.float32)
                angles = (direction[3:] * magnitude * np.sqrt(6) *
                          band.get('rotation_scale_deg', 2.0)).astype(np.float32)
                return self.pose(case, trans, angles)
            return draw(good_band), draw(bad_band)
        # Near-equivalent poses vs a farther pose. Magnitude determines pair only,
        # not an artificially calibrated continuous clinical quality score.
        good_t = pose_rng.uniform(-self.near_translation_mm, self.near_translation_mm, 3).astype(np.float32)
        good_r = pose_rng.uniform(-self.near_rotation_deg, self.near_rotation_deg, 3).astype(np.float32)
        magnitude = pose_rng.uniform(self.far_magnitude_min, self.far_magnitude_max)
        direction = pose_rng.normal(size=6)
        direction /= np.linalg.norm(direction) + 1e-8
        bad_t = (direction[:3] * magnitude * np.sqrt(6)).astype(np.float32)
        bad_r = (direction[3:] * magnitude * np.sqrt(6)).astype(np.float32)
        return self.pose(case, good_t, good_r), self.pose(case, bad_t, bad_r)


class CaseBlockSampler(Sampler):
    """Shuffle case order while keeping each case's samples adjacent for tree caching."""
    def __init__(self, length, block_size, seed=42):
        self.length, self.block_size, self.seed, self.epoch = length, block_size, seed, 0
        if length < 1 or block_size < 1:
            raise ValueError('Sampler length and block size must be positive')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        starts = list(range(0, self.length, self.block_size))
        rng.shuffle(starts)
        for start in starts:
            block = np.arange(start, min(start + self.block_size, self.length))
            rng.shuffle(block)
            yield from block.tolist()
