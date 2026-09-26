"""Fine-tune the ranking teacher for the one aligned case used by the RL task."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PoseDataset
from .config import REPO_ROOT
from .model import OcclusionTeacher
from .teacher_ranking import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default=str(REPO_ROOT / 'examples/data/case.npz'))
    parser.add_argument('--source', default=str(REPO_ROOT / 'examples/checkpoints/teacher_initial.pt'))
    parser.add_argument('--out', default=str(REPO_ROOT / 'runs/default/teacher/ranking'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=256)
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--near-translation-mm', type=float, default=0.1)
    parser.add_argument('--near-rotation-deg', type=float, default=0.2)
    parser.add_argument('--far-magnitude-min', type=float, default=0.35)
    parser.add_argument('--far-magnitude-max', type=float, default=2.0)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--anchor-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-options', default='{}', help='JSON keyword arguments for OcclusionTeacher')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source = torch.load(args.source, map_location=device)
    model = OcclusionTeacher(sample_count=args.samples, **json.loads(args.model_options)).to(device)
    model.load_state_dict(source['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_options = {'near_translation_mm': args.near_translation_mm,
                    'near_rotation_deg': args.near_rotation_deg,
                    'far_magnitude_min': args.far_magnitude_min,
                    'far_magnitude_max': args.far_magnitude_max}
    train = PoseDataset([args.case], args.points, args.repeats, args.seed, **data_options)
    validation = PoseDataset([args.case], args.points, max(128, args.repeats),
                             args.seed + 100000, **data_options)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float('-inf')
    history = []
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        tr = run(model, DataLoader(train, batch_size=args.batch, shuffle=True),
                 device, optimizer, args.margin, args.anchor_weight)
        va = run(model, DataLoader(validation, batch_size=args.batch),
                 device, margin=args.margin, anchor_weight=args.anchor_weight)
        result = {'epoch': epoch + 1, 'train': tr, 'validation': va}
        history.append(result)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(result), flush=True)
        if va['pair_accuracy'] > best:
            best = va['pair_accuracy']
            torch.save({'model': model.state_dict(), 'samples': args.samples, 'epoch': epoch + 1,
                        'case': str(args.case)}, out / 'best.pt')


if __name__ == '__main__':
    main()
