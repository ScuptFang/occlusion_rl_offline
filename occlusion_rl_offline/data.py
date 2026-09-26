from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from scipy.ndimage import map_coordinates


def sample_sdf(grid, origin, spacing, xyz):
    # grid[i,j,k] corresponds to origin + spacing * (i,j,k); negative inside lower jaw.
    ijk = ((xyz - origin) / spacing).T
    valid = ((ijk >= 0) & (ijk <= np.asarray(grid.shape)[:, None] - 1)).all(axis=0)
    if not valid.all():
        raise ValueError('Upper points lie outside SDF grid; enlarge the lower-jaw grid (no border clipping).')
    return map_coordinates(grid, ijk, order=1, mode='nearest').astype(np.float32)


def transform(points, normals, trans, angles_deg, pivot):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', angles_deg, degrees=True).as_matrix().astype(np.float32)
    return (points - pivot) @ rot.T + pivot + trans, normals @ rot.T


def load_case(path):
    with np.load(path) as f:
        required = ('xyz', 'normal', 'curvature', 'sdf_grid', 'origin', 'spacing')
        if any(k not in f for k in required):
            raise ValueError('%s missing keys: %s' % (path, required))
        case = {k: f[k].astype(np.float32) for k in required}
    n = len(case['xyz'])
    assert case['xyz'].shape == (n, 3) and case['normal'].shape == (n, 3)
    assert case['curvature'].shape in ((n,), (n, 1))
    assert case['origin'].shape == (3,) and case['spacing'].shape == (3,)
    assert (case['spacing'] > 0).all() and np.isfinite(case['sdf_grid']).all()
    return case


class PoseDataset(Dataset):
    """One case per item; fresh synthetic pose pairs every epoch, no case leakage."""
    def __init__(self, paths, points=1024, repeats=64, seed=42,
                 near_translation_mm=0.1, near_rotation_deg=0.2,
                 far_magnitude_min=0.35, far_magnitude_max=2.0):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError('No .npz cases found')
        self.points, self.repeats, self.seed, self.epoch = points, repeats, seed, 0
        self.near_translation_mm = near_translation_mm
        self.near_rotation_deg = near_rotation_deg
        self.far_magnitude_min = far_magnitude_min
        self.far_magnitude_max = far_magnitude_max

    def __len__(self):
        return len(self.paths) * self.repeats

    @lru_cache(maxsize=2)
    def case(self, index):
        # A full SDF grid can be tens of MB; keep only two cases in memory.
        return load_case(self.paths[index])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def pose(self, case, ids, trans, angles):
        xyz = case['xyz'][ids]
        normal = case['normal'][ids]
        pivot = case['xyz'].mean(0)
        xyz, normal = transform(xyz, normal, trans, angles, pivot)
        d = sample_sdf(case['sdf_grid'], case['origin'], case['spacing'], xyz)
        # xyz relative to fixed case pivot; translation remains visible through SDF.
        features = np.column_stack((xyz, normal, case['curvature'][ids].reshape(-1), d)).astype(np.float32)
        return torch.from_numpy(features)

    def __getitem__(self, index):
        item_seed = self.seed + self.epoch * 1000003 + index
        points_rng = np.random.RandomState(item_seed)
        pose_rng = np.random.RandomState(item_seed + 1000000000)
        case = self.case(index // self.repeats)
        ids = points_rng.choice(len(case['xyz']), self.points, replace=len(case['xyz']) < self.points)
        # Near-equivalent poses vs a farther pose. Magnitude determines pair only,
        # not an artificially calibrated continuous clinical quality score.
        good_t = pose_rng.uniform(-self.near_translation_mm, self.near_translation_mm, 3).astype(np.float32)
        good_r = pose_rng.uniform(-self.near_rotation_deg, self.near_rotation_deg, 3).astype(np.float32)
        magnitude = pose_rng.uniform(self.far_magnitude_min, self.far_magnitude_max)
        direction = pose_rng.normal(size=6)
        direction /= np.linalg.norm(direction) + 1e-8
        bad_t = (direction[:3] * magnitude * np.sqrt(6)).astype(np.float32)
        bad_r = (direction[3:] * magnitude * np.sqrt(6)).astype(np.float32)
        return self.pose(case, ids, good_t, good_r), self.pose(case, ids, bad_t, bad_r)


class CaseBlockSampler(Sampler):
    """Shuffle case order while keeping each case's samples adjacent for SDF caching."""
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
