"""Single-jaw Point Transformer segmentation of reference-pose <=5 mm regions.

Labels are used only by segmentation supervision/evaluation. Inference accepts
only current mesh coordinates, vertex normals, mesh curvature and jaw identity.
Vector attention follows Zhao et al., ICCV 2021 (2012.09164); this is a compact,
single-resolution implementation, not the paper's full hierarchical network.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uniform_fps(xyz, count):
    count = min(count, len(xyz))
    nearest = np.full(len(xyz), np.inf)
    current = int(np.argmax(((xyz - xyz.mean(0)) ** 2).sum(1)))
    chosen = np.empty(count, dtype=np.int64)
    used = np.zeros(len(xyz), dtype=bool)
    for i in range(count):
        chosen[i] = current
        used[current] = True
        nearest = np.minimum(nearest, ((xyz - xyz[current]) ** 2).sum(1))
        current = int(np.argmax(np.where(used, -np.inf, nearest)))
    return chosen


def sample_ids(xyz, points, candidates, seed):
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(xyz), min(candidates, len(xyz)), replace=False)
    return ids[uniform_fps(xyz[ids], points)]


class PointTransformerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.position = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width))
        self.attention = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Linear(width * 2, width))
        self.ff_norm = nn.LayerNorm(width)

    def forward(self, xyz, x, idx):
        batch = torch.arange(len(x), device=x.device)[:, None, None]
        neighbors = x[batch, idx]
        pos = self.position(xyz[:, :, None] - xyz[batch, idx])
        # Separate attention weight per feature channel, normalized over neighbors.
        weights = self.attention(self.q(x)[:, :, None] - self.key(neighbors) + pos).softmax(2)
        x = self.norm(x + self.output((weights * (self.value(neighbors) + pos)).sum(2)))
        return self.ff_norm(x + self.ff(x))


class RegionPointTransformer(nn.Module):
    def __init__(self, width=48, layers=3, k=16, coordinate_scale_mm=30.):
        super().__init__()
        self.k, self.coordinate_scale_mm = k, coordinate_scale_mm
        self.embed = nn.Sequential(nn.Linear(8, width), nn.LayerNorm(width), nn.ReLU())
        self.blocks = nn.ModuleList([PointTransformerBlock(width) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, features):
        xyz = (features[..., :3] - features[..., :3].mean(1, keepdim=True)) / self.coordinate_scale_mm
        idx = torch.cdist(xyz, xyz).topk(min(self.k, xyz.shape[1]), largest=False).indices
        x = self.embed(torch.cat((xyz, features[..., 3:]), -1))
        for block in self.blocks:
            x = block(xyz, x, idx)
        mean = x.mean(1, keepdim=True).expand_as(x)
        maximum = x.max(1, keepdim=True).values.expand_as(x)
        return self.head(torch.cat((x, mean, maximum), -1)).squeeze(-1)


def pack_features(xyz, normals, curvature, jaw):
    return np.column_stack((xyz, normals, curvature.reshape(-1), np.full(len(xyz), jaw))).astype(np.float32)


@lru_cache(maxsize=2)
def _load_predictor(path, expected_hash, device):
    if digest(path) != expected_hash:
        raise ValueError('Segmentation checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RegionPointTransformer(**checkpoint['architecture']).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint['sampling']


def predictor(options):
    from .config import resolve
    from .runtime import resolve_device
    path = resolve(options['checkpoint'])
    # Stat is part of the cache identity so replacing a file is never silently ignored.
    stat = path.stat()
    device = resolve_device(override=os.environ.get('OCCLUSION_DEVICE') or options.get('device', 'auto'))
    return _checked_predictor(str(path), options['sha256'], device,
                              stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _checked_predictor(path, expected_hash, device, mtime, size):
    _load_predictor.cache_clear()
    return _load_predictor(path, expected_hash, device)


@torch.no_grad()
def predict_probabilities(xyz, normal, curvature, jaw, options, sampled_ids=None):
    model, sampling = predictor(options)
    if sampled_ids is None:
        sampled_ids = sample_ids(xyz, sampling['points'], sampling['candidates'], sampling['seed'])
    f = pack_features(xyz[sampled_ids], normal[sampled_ids], curvature[sampled_ids], jaw)
    device = next(model.parameters()).device
    probs = model(torch.from_numpy(f)[None].to(device)).sigmoid()[0].cpu().numpy()
    return interpolate_probabilities(xyz, sampled_ids, probs, options)


def interpolate_probabilities(xyz, sampled_ids, probs, options):
    # Geometric interpolation uses current-pose points only; no labels or GT pose.
    neighbors = int(options.get('interpolation_neighbors', 3))
    if neighbors < 1:
        raise ValueError('interpolation_neighbors must be positive')
    distances, indices = cKDTree(xyz[sampled_ids]).query(xyz, k=min(neighbors, len(sampled_ids)))
    if distances.ndim == 1:
        return probs[indices]
    weights = 1 / np.maximum(distances, 1e-8) ** 2
    return ((probs[indices] * weights).sum(1) / weights.sum(1)).astype(np.float32)


def lower_region(case, options):
    key = (options['sha256'], float(options.get('threshold', .5)),
           int(options.get('minimum_lower_faces', 128)), int(options.get('interpolation_neighbors', 3)))
    cached = case.get('_predicted_lower_region')
    if cached is not None and cached[0] == key:
        return cached[1]
    if 'lower_curvature' not in case or 'lower_faces' not in case:
        raise ValueError('Region inference requires lower mesh curvature and triangles; run region prepare')
    probs = predict_probabilities(case['lower_xyz'], case['lower_normal'], case['lower_curvature'], 1, options)
    face_prob = probs[case['lower_faces']].mean(1)
    mask = face_prob >= key[1]
    fallback = int(mask.sum()) < key[2]
    if fallback:
        mask[np.argsort(face_prob)[-min(key[2], len(mask)):]] = True
    region = dict(lower_xyz=case['lower_xyz'], lower_normal=case['lower_normal'],
                  lower_faces=case['lower_faces'][mask])
    case['_predicted_lower_region'] = (key, region)
    case['_lower_region_audit'] = {'faces': int(mask.sum()), 'total_faces': len(mask), 'fallback': fallback}
    return region


def upper_sampled_ids(case, xyz, ids, options):
    key = (options['sha256'], tuple(ids.tolist()))
    cached = case.get('_upper_segmentation_ids')
    if cached is None or cached[0] != key:
        _, sampling = predictor(options)
        sampled = sample_ids(xyz, sampling['points'], sampling['candidates'], sampling['seed'])
        case['_upper_segmentation_ids'] = (key, sampled)
    else:
        sampled = cached[1]
    return sampled


def upper_region_mask(case, xyz, normal, ids, count, options):
    sampled = upper_sampled_ids(case, xyz, ids, options)
    probs = predict_probabilities(xyz, normal, case['curvature'][ids], 0, options, sampled)
    mask = probs >= options.get('threshold', .5)
    fallback = mask.sum() < count
    if fallback:
        mask[np.argsort(probs)[-count:]] = True
    case['_upper_region_audit'] = {'vertices': int(mask.sum()), 'candidates': len(mask), 'fallback': bool(fallback)}
    return mask
