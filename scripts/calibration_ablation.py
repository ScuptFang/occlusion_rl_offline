"""Compare ranking-only and calibrated teachers on fixed held-out poses."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from occlusion_rl_offline.calibrate_teacher import PoseScoreDataset
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import PoseDataset
from occlusion_rl_offline.rl import load_teacher


def main():
    cfg = load_config()
    case = resolve(cfg['paths']['case'])
    teacher_cfg = cfg['teacher']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(20260927)
    continuous = PoseScoreDataset(
        case, 512, teacher_cfg['candidate_points'], 20260927,
        teacher_cfg['calibration_pose_bands'],
        teacher_cfg['quality_rotation_scale_deg'],
        teacher_cfg['quality_translation_scale_mm'])
    pairs = PoseDataset([case], teacher_cfg['candidate_points'], 512, 20260928,
                        near_translation_mm=teacher_cfg['ranking_near_translation_mm'],
                        near_rotation_deg=teacher_cfg['ranking_near_rotation_deg'],
                        far_magnitude_min=teacher_cfg['ranking_far_magnitude_min'],
                        far_magnitude_max=teacher_cfg['ranking_far_magnitude_max'])
    out = {}
    for name, checkpoint in (
        ('ranking_only', 'examples/checkpoints/teacher_ranking.pt'),
        ('calibrated', 'examples/checkpoints/teacher.pt')):
        model = load_teacher(resolve(checkpoint), device, teacher_cfg['fps_points'], **cfg['model'])
        predictions, targets, pair_hits, pair_margins = [], [], [], []
        with torch.no_grad():
            for x, y in DataLoader(continuous, batch_size=16):
                predictions.append(model(x.to(device)).cpu().numpy())
                targets.append(y.numpy())
            for good, bad in DataLoader(pairs, batch_size=16):
                difference = model(good.to(device)) - model(bad.to(device))
                delta = difference.cpu().numpy()
                pair_hits.append(delta > 0)
                pair_margins.append(delta)
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        hits = np.concatenate(pair_hits)
        margins = np.concatenate(pair_margins)
        out[name] = {
            'pose_count': len(prediction), 'pair_count': len(hits),
            'pose_quality_mae': float(np.mean(np.abs(prediction - target))),
            'pose_quality_correlation': float(np.corrcoef(prediction, target)[0, 1]),
            'near_far_pair_accuracy': float(hits.mean()),
            'median_good_bad_score_margin': float(np.median(margins)),
            'mean_teacher_score': float(prediction.mean()),
        }
    path = resolve('runs/calibration_ablation/teacher_comparison.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
