import torch
from torch import nn
import torch.nn.functional as F


def sdf_weights(d, scales=(0.15, 0.6, 1.5), floor=0.15):
    # d in mm; retain penetration as important even beyond contact bandwidth.
    a = d.abs()
    return [floor + torch.exp(-0.5 * (a / s) ** 2) + 0.5 * (d < 0).float() * torch.exp(-a / 1.0) for s in scales]


def weighted_fps(xyz, d, n, strength=1.5, scales=(0.15, 0.6, 1.5), floor=0.15):
    # O(B*N*n); pure torch reference implementation. Sampling is discrete.
    b, count, _ = xyz.shape
    n = min(n, count)
    importance = sdf_weights(d, scales, floor)[1]
    chosen = torch.empty(b, n, dtype=torch.long, device=xyz.device)
    nearest = torch.full((b, count), float('inf'), device=xyz.device)
    seed = xyz.mean(1, keepdim=True)
    current = ((xyz - seed) ** 2).sum(-1).argmax(-1)
    batch = torch.arange(b, device=xyz.device)
    for j in range(n):
        chosen[:, j] = current
        center = xyz[batch, current].unsqueeze(1)
        nearest = torch.minimum(nearest, ((xyz - center) ** 2).sum(-1))
        current = (nearest * (1 + strength * importance)).argmax(-1)
    return chosen


def gather(x, idx):
    batch = torch.arange(x.shape[0], device=x.device)[:, None]
    return x[batch, idx]


class LocalAttention(nn.Module):
    def __init__(self, width, k=16):
        super().__init__()
        self.k = k
        self.q = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.val = nn.Linear(width, width)
        self.pos = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width))
        self.attn = nn.Linear(width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, xyz, feat):
        b, n, _ = xyz.shape
        idx = torch.cdist(xyz, xyz).topk(min(self.k, n), largest=False).indices
        batch = torch.arange(b, device=xyz.device)[:, None, None]
        neighbor = feat[batch, idx]
        delta = xyz[batch, idx] - xyz[:, :, None]
        pos = self.pos(delta)
        weight = torch.softmax(self.attn(torch.tanh(self.q(feat)[:, :, None] - self.key(neighbor) + pos)), 2)
        result = (weight * (self.val(neighbor) + pos)).sum(2)
        return self.norm(feat + result)


class OcclusionTeacher(nn.Module):
    """Input [B,N,8]: xyz(mm), normal, mean curvature, lower SDF(mm)."""
    def __init__(self, sample_count=128, width=64, k=16, encoder_layers=3, fps_strength=1.5,
                 contact_scales=(0.15, 0.6, 1.5), weight_floor=0.15,
                 contact_threshold=0.15, penetration_threshold=-0.1,
                 stat_clip_mm=3.0, candidate_points=None, point_seed=42,
                 contact_fps_fraction=0.0, contact_fps_max_mm=0.6,
                 contact_fps_pool_factor=4, sampling_mode='dynamic', distance_method='nearest_vertex',
                 segmentation_region=None):
        super().__init__()
        self.sample_count = sample_count
        self.fps_strength = fps_strength
        self.contact_scales = tuple(contact_scales)
        self.weight_floor = weight_floor
        self.contact_threshold = contact_threshold
        self.penetration_threshold = penetration_threshold
        self.stat_clip_mm = stat_clip_mm
        if encoder_layers < 1:
            raise ValueError('encoder_layers must be positive')
        self.embed = nn.Sequential(nn.Linear(8, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([LocalAttention(width, k) for _ in range(encoder_layers)])
        self.head = nn.Sequential(nn.Linear(width * 5 + 4, width), nn.ReLU(), nn.Linear(width, 1))

    def encode(self, points, pooling_override=None):
        xyz, d = points[:, :, :3], points[:, :, 7]
        if points.shape[1] != self.sample_count:
            raise ValueError('Expected %d candidate-FPS points, got %d' %
                             (self.sample_count, points.shape[1]))
        x = points
        # Center xyz to remove arbitrary scan-origin shortcut; keep distances in mm.
        x = torch.cat((xyz - xyz.mean(1, keepdim=True), x[:, :, 3:]), -1)
        f = self.embed(x)
        for block in self.blocks:
            f = block(xyz, f)
        pooled = [f.mean(1), f.max(1).values]
        for w in sdf_weights(d, self.contact_scales, self.weight_floor):
            if pooling_override == 'uniform':
                w = torch.ones_like(w)
            elif pooling_override == 'reversed':
                w = w.flip(1)
            elif pooling_override is not None:
                raise ValueError('Unknown pooling override: %s' % pooling_override)
            pooled.append((f * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6))
        stats = torch.stack(((d < self.penetration_threshold).float().mean(1),
                             (d.abs() < self.contact_threshold).float().mean(1),
                             d.clamp(-self.stat_clip_mm, self.stat_clip_mm).mean(1),
                             d.abs().clamp_max(self.stat_clip_mm).mean(1)), -1)
        return torch.cat(pooled + [stats], -1)

    def forward(self, points, pooling_override=None):
        return self.head(self.encode(points, pooling_override)).squeeze(-1)
