"""Two-stage teacher with shared SE(3) labels, local ranking and delta regression."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from torch.nn import functional as F

from .cohort import read_manifest
from .config import load_config, resolve
from .data import load_case, features_at_pose
from .model import OcclusionTeacher
from .runtime import resolve_device, runtime_info


def pose_quality(rotation, translation, scales):
    return -np.rad2deg(Rotation.from_matrix(rotation).magnitude()) / scales[0] - np.linalg.norm(translation) / scales[1]


def pose_pairs(count, seed, options):
    """Both stages use identical labels and composed local actions, never Euler angles."""
    rng = np.random.RandomState(seed)
    bands = options['pose_bands']
    probabilities = np.array([b['probability'] for b in bands])
    if not np.isclose(probabilities.sum(), 1):
        raise ValueError('Pose probabilities must sum to one')
    records = []
    for index in range(count):
        band = bands[rng.choice(len(bands), p=probabilities)]
        def draw():
            r = Rotation.from_rotvec(np.deg2rad(rng.uniform(-1, 1, 3) * band['rotation_deg'])).as_matrix()
            t = rng.uniform(-1, 1, 3) * band['translation_mm']
            return r, t
        r, t = draw()
        local = index % 2 == 0
        if local:
            scale = options['local_steps'][index // 2 % len(options['local_steps'])]
            delta = rng.uniform(-1, 1, 6)
            if index % 4 == 0:
                axis = rng.randint(6)
                delta[np.arange(6) != axis] = 0
            r2 = Rotation.from_rotvec(np.deg2rad(delta[:3] * scale[0])).as_matrix() @ r
            t2 = t + delta[3:] * scale[1]
        else:
            r2, t2 = draw()
        y = [pose_quality(rr, tt, options['quality_scales']) for rr, tt in ((r, t), (r2, t2))]
        records.append((r, t, r2, t2, y, local, band['rotation_deg'] <= 1.5))
    return records


def build_bank(paths, cfg, count, seed, cache):
    options = cfg['teacher']['aligned_pair']
    identity = {'model': cfg['model'], 'points': cfg['teacher']['fps_points'],
                'options': options, 'count_per_case': count, 'seed': seed,
                'cases': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    if cache.exists():
        with np.load(cache) as f:
            if str(f['signature']) != signature:
                raise ValueError('Stale pose bank: use a new output directory')
            return {k: f[k] for k in f.files if k != 'signature'}
    result = {k: [] for k in ('x', 'y', 'local', 'near', 'case_id', 'rotation', 'translation')}
    for path in paths:
        case = load_case(path)
        records = pose_pairs(count, seed + int(path.parent.name) * 1009, options)
        for r, t, r2, t2, y, local, near in records:
            result['x'].append([features_at_pose(case, rr, tt, cfg['teacher']['fps_points'], cfg['model'])
                                for rr, tt in ((r, t), (r2, t2))])
            for key, value in [('y', y), ('local', local), ('near', near), ('case_id', path.parent.name),
                               ('rotation', [r, r2]), ('translation', [t, t2])]:
                result[key].append(value)
        print(json.dumps({'bank_case': path.parent.name, 'pairs': count, 'cache': str(cache)}), flush=True)
    from .data import pad_point_batches
    result['x'] = pad_point_batches([np.asarray(v) for v in result['x']])
    result = {k: np.asarray(v, dtype=np.float32 if k in ('x', 'y') else None) for k, v in result.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, signature=signature, **result)
    cache.with_suffix('.json').write_text(json.dumps(identity, indent=2))
    return result


def pair_loss(pred, target, options, stage):
    gap = target[:, 1] - target[:, 0]
    delta = pred[:, 1] - pred[:, 0]
    valid = gap.abs() > options['tie_epsilon']
    # Gap-scaled hinge becomes zero at equal quality; no forced fixed score gap.
    rank = F.relu(options['margin_fraction'] * gap.abs() - gap.sign() * delta)
    rank = rank[valid].mean() if valid.any() else delta.sum() * 0
    regression = F.smooth_l1_loss(pred, target, beta=options['regression_beta'])
    difference = F.smooth_l1_loss(delta, gap, beta=options['delta_beta'])
    weights = options[stage + '_weights']
    return weights['regression'] * regression + weights['ranking'] * rank + weights['delta'] * difference


@torch.no_grad()
def predict(model, bank, device, batch=64):
    model.eval()
    if model.all_mesh:
        batch = min(batch, 2)
    flat = torch.from_numpy(bank['x'].reshape(-1, *bank['x'].shape[2:]))
    return torch.cat([model(x.to(device)).cpu() for x in flat.split(batch)]).numpy().reshape(-1, 2)


def metrics(pred, bank, epsilon=1e-5):
    target = bank['y']
    gap, delta = target[:, 1] - target[:, 0], pred[:, 1] - pred[:, 0]
    valid = np.abs(gap) > epsilon
    def accuracy(mask):
        mask = mask & valid
        return float((gap[mask] * delta[mask] > 0).mean()) if mask.any() else None
    zero = np.abs(target) < 1e-8
    local_valid = bank['local'] & valid
    predicted_up = local_valid & (delta > 0)
    return {'mae': float(np.abs(pred - target).mean()),
            'zero_anchor_mae': float(np.abs(pred[zero]).mean()) if zero.any() else None,
            'near_mae': float(np.abs(pred[bank['near']] - target[bank['near']]).mean()),
            'local_improving_accuracy': accuracy(bank['local'] & (gap > 0)),
            'local_worsening_accuracy': accuracy(bank['local'] & (gap < 0)),
            'error_worse_given_score_up_fraction': (float((gap[predicted_up] < 0).mean())
                                                     if predicted_up.any() else None),
            'local_valid_pairs': int(local_valid.sum()),
            'spearman': float(spearmanr(pred.flatten(), target.flatten()).statistic),
            'delta_mae': float(np.abs(delta - gap).mean()),
            'local_delta_mae': float(np.abs(delta[bank['local']] - gap[bank['local']]).mean()),
            'pair_accuracy': accuracy(np.ones(len(gap), dtype=bool)),
            'local_accuracy': accuracy(bank['local']),
            'near_local_accuracy': accuracy(bank['near'] & bank['local']),
            'score_up_error_worse_fraction': float(((delta > 0) & (gap < -epsilon))[bank['local']].mean()),
            'near_local_pairs': int((bank['near'] & bank['local'] & valid).sum()),
            'pairs': len(target)}


def selection(m, options=None):
    w = (options or {}).get('selection_weights', [1., 1., .5])
    return w[0] * m['mae'] + w[1] * m['delta_mae'] + w[2] * (1 - m['local_accuracy'])


def train(cfg, manifest, out, stages=('ranking', 'calibration')):
    torch.set_num_threads(cfg['agent'].get('torch_threads', 4))
    seed = cfg['teacher']['seed']
    torch.manual_seed(seed)
    device = resolve_device(cfg)
    split = read_manifest(manifest)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'config.snapshot.json').write_text(json.dumps(cfg, indent=2))
    (out / 'runtime.json').write_text(json.dumps(runtime_info(device), indent=2))
    o = cfg['teacher']['aligned_pair']
    train_bank = build_bank(split['train'], cfg, o['train_pairs_per_case'], seed, out / 'banks/train.npz')
    val_bank = build_bank(split['val'], cfg, o['validation_pairs_per_case'], seed + 100000, out / 'banks/val.npz')
    model = OcclusionTeacher(sample_count=cfg['teacher']['fps_points'], **cfg['model']).to(device)
    x, y = torch.from_numpy(train_bank['x']), torch.from_numpy(train_bank['y'])
    for stage in stages:
        stage_out = out / stage; stage_out.mkdir(exist_ok=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['teacher'][stage + '_lr'],
                                      weight_decay=cfg['teacher'][stage + '_weight_decay'])
        best, history = float('inf'), []
        for epoch in range(cfg['teacher'][stage + '_epochs']):
            model.train(); losses = []
            order = np.random.RandomState(seed + epoch).permutation(len(x))
            for start in range(0, len(order), o['batch_size']):
                ids = order[start:start + o['batch_size']]
                pred = model(x[ids].flatten(0, 1).to(device)).reshape(-1, 2)
                loss = pair_loss(pred, y[ids].to(device), o, stage)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['teacher']['calibration_max_grad_norm'])
                optimizer.step(); losses.append(float(loss.detach()))
            m = metrics(predict(model, val_bank, device), val_bank, o['tie_epsilon'])
            row = {'stage': stage, 'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation': m}
            history.append(row)
            (stage_out / 'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
            if selection(m, o) < best:
                best = selection(m, o)
                torch.save({'model': model.state_dict(), 'samples': cfg['teacher']['fps_points'],
                            'model_options': cfg['model'], 'epoch': epoch + 1, 'stage': stage,
                            'train_cases': [str(p) for p in split['train']],
                            'validation_cases': [str(p) for p in split['val']],
                            'label': 'negative SO3 angle / rotation_scale - translation_norm / translation_scale',
                            'aligned_pair': o}, stage_out / 'best.pt')
                (stage_out / 'validation.json').write_text(json.dumps(m, indent=2))
        model.load_state_dict(torch.load(stage_out / 'best.pt', map_location=device, weights_only=False)['model'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    train(load_config(args.config), resolve(args.manifest), resolve(args.out))


if __name__ == '__main__':
    main()
