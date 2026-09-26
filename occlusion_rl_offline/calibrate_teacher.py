"""Calibrate the case teacher's logit to a continuous synthetic pose-quality score."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import PoseDataset
from .config import REPO_ROOT
from .model import OcclusionTeacher


class PoseScoreDataset(Dataset):
    def __init__(self, case, count, points, seed, bands=None,
                 quality_rotation_scale_deg=8.0, quality_translation_scale_mm=4.0):
        self.poses = PoseDataset([case], points, 1, seed)
        self.count, self.seed, self.epoch = count, seed, 0
        self.bands = bands or [
            {'probability': .2, 'rotation_deg': 1.5, 'translation_mm': .75},
            {'probability': .5, 'rotation_deg': 8.0, 'translation_mm': 4.0},
            {'probability': .3, 'rotation_deg': 20.0, 'translation_mm': 8.0}]
        self.probabilities = np.array([x['probability'] for x in self.bands], dtype=float)
        if not np.isclose(self.probabilities.sum(), 1):
            raise ValueError('Calibration pose-band probabilities must sum to 1')
        self.quality_rotation_scale_deg = quality_rotation_scale_deg
        self.quality_translation_scale_mm = quality_translation_scale_mm

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        item_seed = self.seed + self.epoch * 1000003 + index
        points_rng = np.random.RandomState(item_seed)
        pose_rng = np.random.RandomState(item_seed + 1000000000)
        case = self.poses.cases[0]
        ids = points_rng.choice(len(case['xyz']), self.poses.points, replace=len(case['xyz']) < self.poses.points)
        # Cover near, initial and reachable states, including poses an agent
        # might visit after moving away from the original orientation.
        band = self.bands[pose_rng.choice(len(self.bands), p=self.probabilities)]
        max_rotation = band['rotation_deg']
        max_translation = band['translation_mm']
        for _ in range(100):
            angles = pose_rng.uniform(-max_rotation, max_rotation, 3).astype(np.float32)
            trans = pose_rng.uniform(-max_translation, max_translation, 3).astype(np.float32)
            try:
                features = self.poses.pose(case, ids, trans, angles)
                angle = np.rad2deg(Rotation.from_euler('xyz', angles, degrees=True).magnitude())
                quality = (-angle / self.quality_rotation_scale_deg
                           - np.linalg.norm(trans) / self.quality_translation_scale_mm)
                return features, torch.tensor(quality, dtype=torch.float32)
            except ValueError:
                pass
        raise ValueError('Unable to sample an in-grid pose; reduce perturbation limits')


def evaluate(model, loader, device):
    model.eval()
    predicted, target = [], []
    with torch.no_grad():
        for x, y in loader:
            predicted.append(model(x.to(device)).cpu().numpy())
            target.append(y.numpy())
    predicted, target = np.concatenate(predicted), np.concatenate(target)
    return {'mae': float(np.abs(predicted - target).mean()),
            'correlation': float(np.corrcoef(predicted, target)[0, 1]),
            'mean_predicted': float(predicted.mean()), 'mean_target': float(target.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default=str(REPO_ROOT / 'examples/data/case.npz'))
    parser.add_argument('--source', default=str(REPO_ROOT / 'examples/checkpoints/teacher_ranking.pt'))
    parser.add_argument('--out', default=str(REPO_ROOT / 'runs/default/teacher/calibration'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-count', type=int, default=768)
    parser.add_argument('--val-count', type=int, default=256)
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--smooth-l1-beta', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--pose-bands', default=None, help='JSON list of sampling bands')
    parser.add_argument('--quality-rotation-scale-deg', type=float, default=8.0)
    parser.add_argument('--quality-translation-scale-mm', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-options', default='{}', help='JSON keyword arguments for OcclusionTeacher')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source = torch.load(args.source, map_location=device)
    model = OcclusionTeacher(sample_count=args.samples, **json.loads(args.model_options)).to(device)
    model.load_state_dict(source['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_options = {'bands': json.loads(args.pose_bands) if args.pose_bands else None,
                    'quality_rotation_scale_deg': args.quality_rotation_scale_deg,
                    'quality_translation_scale_mm': args.quality_translation_scale_mm}
    train = PoseScoreDataset(args.case, args.train_count, args.points, args.seed,
                             **data_options)
    validation = PoseScoreDataset(args.case, args.val_count, args.points,
                                  args.seed + 100000, **data_options)
    val_loader = DataLoader(validation, batch_size=args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        model.train()
        losses = []
        for x, y in DataLoader(train, batch_size=args.batch, shuffle=True):
            prediction = model(x.to(device))
            loss = nn.functional.smooth_l1_loss(prediction, y.to(device),
                                                beta=args.smooth_l1_beta)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, val_loader, device)
        result = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation': metrics}
        history.append(result)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(result), flush=True)
        if metrics['mae'] < best:
            best = metrics['mae']
            torch.save({'model': model.state_dict(), 'samples': args.samples,
                        'epoch': epoch + 1, 'case': str(args.case)}, out / 'best.pt')
            (out / 'validation.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
