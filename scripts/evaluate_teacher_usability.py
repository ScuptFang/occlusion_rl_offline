"""Held-out ranking, calibration, local, and corrective-step checks for one case."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from occlusion_rl_offline.calibrate_teacher import PoseScoreDataset
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import PoseDataset
from occlusion_rl_offline.rl import load_teacher


def scores(model, features):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.stack(features[i:i + 8])).numpy()
                               for i in range(0, len(features), 8)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/case010_candidate4096_fps64.json')
    parser.add_argument('--case', default='runs/case010_mesh128_20260930/case.npz')
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--pairs-per-type', type=int, default=16)
    parser.add_argument('--calibration-poses', type=int, default=128)
    parser.add_argument('--local-poses', type=int, default=64)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg['teacher']
    run = resolve(cfg['paths']['run_dir']) / 'teacher'
    path = resolve(args.case)
    options = cfg['model']
    bands = t['ranking_pose_bands']
    pair_types = [(i, j) for i in range(len(bands)) for j in range(i + 1, len(bands))]
    count = args.pairs_per_type * len(pair_types)
    pairs = PoseDataset([path], t['fps_points'], count, args.seed,
                        fps_options=options, ranking_pose_bands=bands)
    good, bad = zip(*(pairs[i] for i in range(count)))
    calibration = PoseScoreDataset(path, args.calibration_poses, t['fps_points'],
                                   args.seed + 1000, bands=t['calibration_pose_bands'],
                                   quality_rotation_scale_deg=t['quality_rotation_scale_deg'],
                                   quality_translation_scale_mm=t['quality_translation_scale_mm'],
                                   fps_options=options)
    calibration_x, calibration_y = zip(*(calibration[i] for i in range(args.calibration_poses)))
    target = np.asarray([float(y) for y in calibration_y])
    case = pairs.case(0)
    zero = pairs.pose(case, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    rng = np.random.RandomState(args.seed + 2000)
    local, local_quality = [], []
    for _ in range(args.local_poses):
        angle = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        trans = rng.uniform(-.5, .5, 3).astype(np.float32)
        local.append(pairs.pose(case, trans, angle))
        local_quality.append(-np.linalg.norm(angle) / 8 - np.linalg.norm(trans) / 4)
    current, toward, away = [], [], []
    near_current, near_toward, near_away = [], [], []
    for _ in range(args.steps):
        angle = rng.uniform(-8, 8, 3).astype(np.float32)
        trans = rng.uniform(-4, 4, 3).astype(np.float32)
        current.append(pairs.pose(case, trans, angle))
        toward.append(pairs.pose(case, .8 * trans, .8 * angle))
        away.append(pairs.pose(case, 1.2 * trans, 1.2 * angle))
        angle = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        trans = rng.uniform(-.5, .5, 3).astype(np.float32)
        near_current.append(pairs.pose(case, trans, angle))
        near_toward.append(pairs.pose(case, .8 * trans, .8 * angle))
        near_away.append(pairs.pose(case, 1.2 * trans, 1.2 * angle))
    output = {'case': str(path), 'seed': args.seed, 'pair_count': count,
              'calibration_poses': args.calibration_poses, 'local_poses': args.local_poses,
              'corrective_steps': args.steps, 'stages': {}}
    for name, checkpoint in (('ranking', run / 'ranking/best.pt'),
                             ('calibration', run / 'calibration/best.pt')):
        model = load_teacher(checkpoint, 'cpu', t['fps_points'], **options)
        good_score, bad_score = scores(model, good), scores(model, bad)
        hits = good_score > bad_score
        calibrated = scores(model, calibration_x)
        local_score = scores(model, local)
        current_score = scores(model, current)
        toward_score, away_score = scores(model, toward), scores(model, away)
        near_current_score = scores(model, near_current)
        near_toward_score, near_away_score = scores(model, near_toward), scores(model, near_away)
        zero_score = float(scores(model, [zero])[0])
        result = {'ranking_pair_accuracy': float(hits.mean()),
                  'ranking_margin_mean': float((good_score - bad_score).mean()),
                  'pair_accuracy_by_type': {},
                  'quality_spearman': float(spearmanr(target, calibrated).statistic),
                  'quality_mae': float(np.abs(target - calibrated).mean()),
                  'original_above_local_fraction': float((zero_score > local_score).mean()),
                  'local_quality_spearman': float(spearmanr(local_quality, local_score).statistic),
                  'toward_better_than_away_fraction': float((toward_score > away_score).mean()),
                  'toward_minus_away_mean': float((toward_score - away_score).mean()),
                  'toward_better_than_current_fraction': float((toward_score > current_score).mean()),
                  'near_toward_better_than_away_fraction': float((near_toward_score > near_away_score).mean()),
                  'near_toward_better_than_current_fraction': float((near_toward_score > near_current_score).mean())}
        for k, (i, j) in enumerate(pair_types):
            mask = np.arange(k, count, len(pair_types))
            result['pair_accuracy_by_type'][f'{bands[i]["name"]} > {bands[j]["name"]}'] = float(hits[mask].mean())
        output['stages'][name] = result
        print(name, json.dumps(result, ensure_ascii=False), flush=True)
    target_path = resolve(args.out) if args.out else run / 'usability.json'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(target_path)


if __name__ == '__main__':
    main()
