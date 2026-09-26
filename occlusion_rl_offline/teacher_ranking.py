import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import OcclusionTeacher
from .data import PoseDataset


def run(model, loader, device, optimizer=None, margin=0.5, anchor_weight=0.1):
    model.train(optimizer is not None)
    total = hits = count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for good, bad in loader:
            good, bad = good.to(device), bad.to(device)
            a, b = model(good), model(bad)
            # Ranking + soft anchors; scores are logits, not calibrated probabilities.
            loss = F.softplus(margin - (a - b)).mean() + anchor_weight * (F.softplus(-a).mean() + F.softplus(b).mean())
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            size = len(good)
            total += loss.item() * size
            hits += (a > b).sum().item()
            count += size
    return {'loss': total / count, 'pair_accuracy': hits / count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='runs/teacher')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--repeats', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.2)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    paths = sorted(Path(args.data).glob('*.npz'))
    if len(paths) < 5:
        raise ValueError('Need at least five patient cases for a case-level train/validation split')
    rng = np.random.RandomState(args.seed)
    rng.shuffle(paths)
    nval = max(1, round(len(paths) * args.val_fraction))
    train_paths, val_paths = paths[nval:], paths[:nval]
    train = PoseDataset(train_paths, args.points, args.repeats, args.seed)
    val = PoseDataset(val_paths, args.points, max(8, args.repeats // 4), args.seed + 100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OcclusionTeacher(sample_count=args.samples).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'split.json').write_text(json.dumps({'train': [x.name for x in train_paths], 'val': [x.name for x in val_paths]}, indent=2))
    best = -1
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        tr = run(model, DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0), device, opt)
        va = run(model, DataLoader(val, batch_size=args.batch, num_workers=0), device)
        print(json.dumps({'epoch': epoch + 1, 'train': tr, 'val': va}), flush=True)
        if va['pair_accuracy'] > best:
            best = va['pair_accuracy']
            torch.save({'model': model.state_dict(), 'samples': args.samples, 'epoch': epoch + 1}, out / 'best.pt')


if __name__ == '__main__':
    main()
