"""Held-out pose and inference-time sampling/pooling diagnostics for case 010."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import PoseDataset, load_case, sample_upper_mesh
from occlusion_rl_offline.rl import load_teacher


def score(model, features, pooling_override=None):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(features[i:i + 8]),
                                     pooling_override).numpy()
                               for i in range(0, len(features), 8)])


def summarize(scores, original, quality, bands):
    gap = quality[:, None] - quality[None, :]
    difference = scores[:, None] - scores[None, :]
    ordered = gap > 0.5
    metrics = {'spearman_quality': float(spearmanr(quality, scores).statistic),
               'wide_pair_accuracy': float((difference[ordered] > 0).mean()),
               'wide_pairs': int(ordered.sum()),
               'original_score': float(original),
               'original_above_perturbed': float((original > scores).mean())}
    for name in ('near', 'medium', 'full'):
        mask = bands == name
        metrics[name] = {'count': int(mask.sum()),
                         'original_above_fraction': float((original > scores[mask]).mean()),
                         'spearman_quality': float(spearmanr(quality[mask], scores[mask]).statistic),
                         'mean_score': float(scores[mask].mean())}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/case010_mesh128.json')
    parser.add_argument('--case', default='runs/case010_mesh128_20260930/case.npz')
    parser.add_argument('--ranking', default='runs/case010_mesh128_20260930/teacher/ranking/best.pt')
    parser.add_argument('--calibration', default='runs/case010_mesh128_20260930/teacher/calibration/best.pt')
    parser.add_argument('--out', default='runs/case010_mesh128_20260930/teacher_diagnostics.json')
    parser.add_argument('--per-band', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    cfg = load_config(args.config)
    case = load_case(resolve(args.case))
    xyz, normal = case['xyz'], case['normal']
    pivot = xyz.mean(0)
    n = cfg['teacher']['fps_points']
    spatial = dict(cfg['model'], fps_strength=0)
    bands_spec = (('near', 1.5, .5), ('medium', 4., 2.), ('full', 8., 4.))
    rng = np.random.RandomState(args.seed)
    selected = {'weighted': [], 'spatial': []}
    qualities, bands = [], []
    for name, max_angle, max_translation in bands_spec:
        for _ in range(args.per_band):
            rvec = rng.uniform(-max_angle, max_angle, 3)
            trans = rng.uniform(-max_translation, max_translation, 3)
            rotation = Rotation.from_rotvec(np.deg2rad(rvec))
            matrix = rotation.as_matrix().astype(np.float32)
            posed_xyz = (xyz - pivot) @ matrix.T + pivot + trans
            posed_normal = normal @ matrix.T
            for kind, options in (('weighted', cfg['model']), ('spatial', spatial)):
                selected[kind].append(sample_upper_mesh(case, posed_xyz, posed_normal, n, options))
            qualities.append(-np.rad2deg(rotation.magnitude()) / cfg['teacher']['quality_rotation_scale_deg']
                             - np.linalg.norm(trans) / cfg['teacher']['quality_translation_scale_mm'])
            bands.append(name)
        print(f'sampled {name}: {args.per_band} poses', flush=True)
    arrays = {key: np.stack(value) for key, value in selected.items()}
    originals = {kind: sample_upper_mesh(case, xyz, normal, n, options)[None]
                 for kind, options in (('weighted', cfg['model']), ('spatial', spatial))}
    quality, bands = np.asarray(qualities), np.asarray(bands)
    result = {'case': str(resolve(args.case)), 'seed': args.seed,
              'per_band': args.per_band, 'input_points': n,
              'weighted_contact_fraction': float((np.abs(arrays['weighted'][:, :, 7]) < .6).mean()),
              'spatial_contact_fraction': float((np.abs(arrays['spatial'][:, :, 7]) < .6).mean()),
              'weighted_penetration_fraction': float((arrays['weighted'][:, :, 7] < 0).mean()),
              'spatial_penetration_fraction': float((arrays['spatial'][:, :, 7] < 0).mean()),
              'stages': {}}
    ranking_data = PoseDataset([resolve(args.case)], n, args.per_band, args.seed + 10000,
                               near_translation_mm=cfg['teacher']['ranking_near_translation_mm'],
                               near_rotation_deg=cfg['teacher']['ranking_near_rotation_deg'],
                               far_magnitude_min=cfg['teacher']['ranking_far_magnitude_min'],
                               far_magnitude_max=cfg['teacher']['ranking_far_magnitude_max'],
                               fps_options=cfg['model'])
    good, bad = zip(*(ranking_data[i] for i in range(args.per_band)))
    good, bad = torch.stack(good).numpy(), torch.stack(bad).numpy()
    for stage, path in (('ranking', args.ranking), ('calibration', args.calibration)):
        model = load_teacher(resolve(path), 'cpu', n, **cfg['model'])
        stage_results = {}
        margin = score(model, good) - score(model, bad)
        stage_results['heldout_ranking_pairs'] = {
            'count': len(margin), 'accuracy': float((margin > 0).mean()),
            'mean_score_margin': float(margin.mean())}
        for name, kind, pooling in (('weighted', 'weighted', None),
                                    ('spatial_fps', 'spatial', None),
                                    ('uniform_pool', 'weighted', 'uniform'),
                                    ('reversed_pool', 'weighted', 'reversed'),
                                    ('spatial_and_uniform', 'spatial', 'uniform')):
            values = score(model, arrays[kind], pooling)
            original = score(model, originals[kind], pooling)[0]
            metrics = summarize(values, original, quality, bands)
            if stage == 'calibration':
                metrics['mae_to_synthetic_quality'] = float(np.abs(values - quality).mean())
            stage_results[name] = metrics
            print(stage, name, 'spearman', round(metrics['spearman_quality'], 3),
                  'pair', round(metrics['wide_pair_accuracy'], 3), flush=True)
        result['stages'][stage] = stage_results
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(out)


if __name__ == '__main__':
    main()
