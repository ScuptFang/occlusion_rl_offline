"""Fine-tune the ranking teacher for the one aligned case used by the RL task."""
from .runtime import resolve_device
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PoseDataset, CaseBlockSampler
from .config import REPO_ROOT
from .model import OcclusionTeacher
from .teacher_ranking import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default=str(REPO_ROOT / 'examples/data/case.npz'))
    parser.add_argument('--train-cases', default=None, help='JSON array of training case paths')
    parser.add_argument('--val-cases', default=None, help='JSON array of validation case paths')
    parser.add_argument('--source', default=str(REPO_ROOT / 'examples/checkpoints/teacher_initial.pt'))
    parser.add_argument('--from-scratch', action='store_true', help='Initialize teacher without pretrained weights')
    parser.add_argument('--out', default=str(REPO_ROOT / 'runs/default/teacher/ranking'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=256)
    parser.add_argument('--validation-repeats', type=int, default=128)
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--near-translation-mm', type=float, default=0.1)
    parser.add_argument('--near-rotation-deg', type=float, default=0.2)
    parser.add_argument('--far-magnitude-min', type=float, default=0.35)
    parser.add_argument('--far-magnitude-max', type=float, default=2.0)
    parser.add_argument('--ranking-pose-bands', default=None, help='Ordered JSON pose-magnitude bands')
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--anchor-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-options', default='{}', help='JSON keyword arguments for OcclusionTeacher')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(override=getattr(args,'device',None))
    model_options = json.loads(args.model_options)
    model = OcclusionTeacher(sample_count=args.samples, **model_options).to(device)
    if not args.from_scratch:
        source = torch.load(args.source, map_location=device)
        model.load_state_dict(source['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_options = {'near_translation_mm': args.near_translation_mm,
                    'near_rotation_deg': args.near_rotation_deg,
                    'far_magnitude_min': args.far_magnitude_min,
                    'far_magnitude_max': args.far_magnitude_max,
                    'ranking_pose_bands': json.loads(args.ranking_pose_bands) if args.ranking_pose_bands else None,
                    'fps_options': model_options}
    train_cases = json.loads(args.train_cases) if args.train_cases else [args.case]
    val_cases = json.loads(args.val_cases) if args.val_cases else [args.case]
    train = PoseDataset(train_cases, args.points, args.repeats, args.seed, **data_options)
    train_sampler = CaseBlockSampler(len(train), args.repeats, args.seed)
    validation = PoseDataset(val_cases, args.points, args.validation_repeats,
                             args.seed + 100000, **data_options)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float('-inf')
    best_loss = float('inf')
    history = []
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        tr = run(model, DataLoader(train, batch_size=args.batch, sampler=train_sampler),
                 device, optimizer, args.margin, args.anchor_weight)
        va = run(model, DataLoader(validation, batch_size=args.batch),
                 device, margin=args.margin, anchor_weight=args.anchor_weight)
        result = {'epoch': epoch + 1, 'train': tr, 'validation': va}
        history.append(result)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(result), flush=True)
        if va['pair_accuracy'] > best or (va['pair_accuracy'] == best and va['loss'] < best_loss):
            best, best_loss = va['pair_accuracy'], va['loss']
            torch.save({'model': model.state_dict(), 'samples': args.samples, 'model_options': model_options, 'epoch': epoch + 1,
                        'train_cases': train_cases, 'validation_cases': val_cases}, out / 'best.pt')


if __name__ == '__main__':
    main()
