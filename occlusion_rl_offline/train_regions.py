"""Prepare reference labels, train and evaluate a pose-augmented region segmenter."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F

from .cohort import read_manifest
from .config import resolve
from .data import load_case
from .prepare import _mesh, _upper_features
from .region_segmentation import (RegionPointTransformer, digest, pack_features,
                                  predict_probabilities, sample_ids)
from .runtime import resolve_device, runtime_info
from .surface import surface_distance


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def reference_labels(upper, lower, threshold):
    def directed(source, target):
        query = dict(lower_xyz=np.asarray(target.vertices, dtype=np.float32),
                     lower_normal=np.asarray(target.vertex_normals, dtype=np.float32),
                     lower_faces=np.asarray(target.faces, dtype=np.int64))
        return np.abs(surface_distance(query, np.asarray(source.vertices, dtype=np.float32))[0])
    ud, ld = directed(upper, lower), directed(lower, upper)
    return (ud <= threshold).astype(np.float32), (ld <= threshold).astype(np.float32), ud, ld


def prepare(cfg):
    out = resolve(cfg['out'])
    source = resolve(cfg['source_manifest'])
    # Only the partition/IDs are needed; a fresh machine need not have old caches.
    manifest = json.loads(source.read_text())
    if manifest.get('version') != 1 or set(manifest.get('splits', {})) != {'train', 'val', 'test'}:
        raise ValueError('Expected version 1 train/val/test manifest')
    seen = set()
    for entries in manifest['splits'].values():
        if not entries:
            raise ValueError('All three splits must be nonempty')
        for entry in entries:
            identity = entry['id']
            if not identity.isdigit() or identity in seen:
                raise ValueError('Unique numeric case IDs required across all splits')
            seen.add(identity)
    audits = []
    for split, entries in manifest['splits'].items():
        for entry in entries:
            folder = resolve(cfg['data_root']) / entry['id']
            upper, lower = _mesh(folder / 'upper.stl'), _mesh(folder / 'lower.stl')
            ux, un, uc = _upper_features(upper)
            lx, ln, lc = _upper_features(lower)
            uy, ly, ud, ld = reference_labels(upper, lower, cfg['label_distance_mm'])
            dest = out / 'prepared' / 'cases' / entry['id']; dest.mkdir(parents=True, exist_ok=True)
            # Downstream teacher/agent case deliberately contains NO reference labels.
            np.savez_compressed(dest / 'case.npz', xyz=ux, normal=un, curvature=uc,
                                upper_faces=np.asarray(upper.faces, dtype=np.int64),
                                lower_xyz=lx, lower_normal=ln, lower_curvature=lc,
                                lower_faces=np.asarray(lower.faces, dtype=np.int64))
            np.savez_compressed(dest / 'region_labels.npz', upper=uy, lower=ly,
                                upper_distance=ud, lower_distance=ld)
            entry['case'] = os.path.relpath(dest / 'case.npz', out)
            audit = {'case': entry['id'], 'split': split, 'upper_vertices': len(ux),
                     'lower_vertices': len(lx), 'upper_positive_fraction': float(uy.mean()),
                     'lower_positive_fraction': float(ly.mean()),
                     'upper_stl_sha256': digest(folder / 'upper.stl'),
                     'lower_stl_sha256': digest(folder / 'lower.stl')}
            audits.append(audit); print(json.dumps(audit), flush=True)
    manifest['region_label_distance_mm'] = cfg['label_distance_mm']
    save_json(out / 'manifest.json', manifest)
    save_json(out / 'label_audit.json', audits)
    save_json(out / 'segmentation_config.json', cfg)


def load_jaws(paths):
    result = []
    for path in paths:
        with np.load(path) as f, np.load(path.parent / 'region_labels.npz') as labels:
            for jaw in (0, 1):
                prefix = 'lower_' if jaw else ''
                result.append(dict(xyz=f[prefix + 'xyz'], normal=f[prefix + 'normal'],
                                   curvature=f[prefix + 'curvature'],
                                   label=labels['lower' if jaw else 'upper'],
                                   jaw=jaw, case=path.parent.name))
    return result


def example(item, sampling, seed, rotation_deg, translation_mm):
    ids = sample_ids(item['xyz'], sampling['points'], sampling['candidates'], seed)
    rng = np.random.RandomState(seed + 100000)
    r = Rotation.from_rotvec(np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, 3))).as_matrix()
    t = rng.uniform(-translation_mm, translation_mm, 3)
    pivot = item['xyz'].mean(0)
    xyz = (item['xyz'][ids] - pivot) @ r.T + pivot + t
    normals = item['normal'][ids] @ r.T
    return pack_features(xyz, normals, item['curvature'][ids], item['jaw']), item['label'][ids], ids


def binary_metrics(prob, label, threshold=.5):
    p, y = prob >= threshold, label > .5
    tp, fp, fn, tn = [int(v.sum()) for v in (p & y, p & ~y, ~p & y, ~p & ~y)]
    return dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), accuracy=(tp + tn) / len(y),
                positive_fraction=float(y.mean()), predicted_fraction=float(p.mean()))


def validation_bank(items, cfg):
    result = []
    for item in items:
        for i in range(cfg['validation_poses_per_jaw']):
            x, y, _ = example(item, cfg['sampling'], 500000 + int(item['case']) * 101 + item['jaw'] * 10000 + i,
                              cfg['augmentation']['rotation_deg'], cfg['augmentation']['translation_mm'])
            result.append((x, y, item['jaw']))
    return result


@torch.no_grad()
def validate(model, bank, device):
    model.eval()
    rows = []
    for x, y, jaw in bank:
        p = model(torch.from_numpy(x)[None].to(device)).sigmoid()[0].cpu().numpy()
        rows.append(dict(jaw=jaw, **binary_metrics(p, y)))
    return {name: {k: float(np.mean([r[k] for r in rows if jaw is None or r['jaw'] == jaw]))
                   for k in rows[0] if k != 'jaw'} for name, jaw in [('macro', None), ('upper', 0), ('lower', 1)]}


def train(cfg):
    torch.set_num_threads(cfg['torch_threads'])
    torch.manual_seed(cfg['seed'])
    device = resolve_device({'runtime': cfg['runtime']})
    out = resolve(cfg['out']); split = read_manifest(out / 'manifest.json')
    train_items, val_items = load_jaws(split['train']), load_jaws(split['val'])
    bank = validation_bank(val_items, cfg)
    model = RegionPointTransformer(**cfg['architecture']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    positive = sum(float(i['label'].sum()) for i in train_items)
    negative = sum(len(i['label']) for i in train_items) - positive
    pos_weight = torch.tensor(min(4., max(.25, negative / positive)), device=device)
    history, best = [], -1.
    # Cache only sampled features, not full pose-dependent labels. IDs carry fixed labels.
    for epoch in range(cfg['epochs']):
        model.train(); losses = []
        examples = []
        for index, item in enumerate(train_items):
            for j in range(cfg['train_poses_per_jaw']):
                seed = cfg['seed'] + epoch * 100000 + index * 1000 + j
                x, y, _ = example(item, cfg['sampling'], seed, **cfg['augmentation'])
                examples.append((x, y))
        order = np.random.RandomState(cfg['seed'] + epoch).permutation(len(examples))
        for start in range(0, len(order), cfg['batch_size']):
            batch = [examples[i] for i in order[start:start + cfg['batch_size']]]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            p = logits.sigmoid()
            dice = 1 - ((2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean()
            loss = bce + cfg['dice_weight'] * dice
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        m = validate(model, bank, device)
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), validation=m)
        history.append(row); print(json.dumps(row), flush=True)
        save_json(out / 'segmentation/history.json', history)
        checkpoint = dict(model=model.state_dict(), architecture=cfg['architecture'], sampling=cfg['sampling'],
                          epoch=epoch + 1, config=cfg, validation=m,
                          manifest_sha256=digest(out / 'manifest.json'),
                          train_cases=[p.parent.name for p in split['train']],
                          validation_cases=[p.parent.name for p in split['val']],
                          runtime=runtime_info(device))
        if m['macro']['iou'] > best:
            best = m['macro']['iou']
            torch.save(checkpoint, out / 'segmentation/best.pt')
        torch.save(checkpoint, out / 'segmentation/latest.pt')


def evaluate(cfg):
    torch.set_num_threads(cfg['torch_threads'])
    out = resolve(cfg['out']); split = read_manifest(out / 'manifest.json')
    checkpoint = out / 'segmentation/best.pt'
    options = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), device='cpu', **cfg['region'])
    rows = []
    # Full original vertex set, predictions interpolated from sparse FPS observations.
    for item in load_jaws(split['test']):
        pivot = item['xyz'].mean(0)
        original_prediction = None
        for band, angle, translation in [('zero', 0., 0.), ('near', 1.5, .5), ('medium', 8., 4.), ('wide', 20., 8.)]:
            repeats = 1 if band == 'zero' else cfg['test_poses_per_band']
            for j in range(repeats):
                seed = 926001 + int(item['case']) * 1009 + item['jaw'] * 100 + j
                rng = np.random.RandomState(seed)
                r = Rotation.from_rotvec(np.deg2rad(rng.uniform(-angle, angle, 3))).as_matrix()
                t = rng.uniform(-translation, translation, 3)
                xyz = (item['xyz'] - pivot) @ r.T + pivot + t
                normal = item['normal'] @ r.T
                prob = predict_probabilities(xyz, normal, item['curvature'], item['jaw'], options)
                if original_prediction is None:
                    original_prediction = prob >= options['threshold']
                    target = out / 'segmentation/predictions'; target.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target / (item['case'] + '_' + str(item['jaw']) + '.npz'),
                                        probability=prob, label=item['label'], xyz=item['xyz'])
                row = dict(case=item['case'], jaw=item['jaw'], band=band, seed=seed,
                           prediction_flip_fraction=float(((prob >= options['threshold']) != original_prediction).mean()),
                           **binary_metrics(prob, item['label'], options['threshold']))
                rows.append(row)
        print(json.dumps({'segmentation_test_case': item['case'], 'jaw': item['jaw'],
                          'mean_iou': float(np.mean([r['iou'] for r in rows if r['case'] == item['case'] and r['jaw'] == item['jaw']]))}), flush=True)
    summary = {band: {jaw: {key: float(np.mean([r[key] for r in rows if r['band'] == band and r['jaw'] == j]))
                            for key in ('iou', 'dice', 'precision', 'recall', 'prediction_flip_fraction')}
                      for jaw, j in [('upper', 0), ('lower', 1)]}
               for band in ('zero', 'near', 'medium', 'wide')}
    save_json(out / 'segmentation/test.json', dict(checkpoint_sha256=digest(checkpoint), rows=rows, summary=summary))
    print(json.dumps(summary), flush=True)


def configure(cfg):
    out = resolve(cfg['out'])
    checkpoint = out / 'segmentation/best.pt'
    config = json.loads(resolve(cfg['base_training_config']).read_text())
    root = cfg['out']
    config['paths'].update(run_dir=root, packaged_teacher=root + '/teacher/calibration/best.pt',
                           packaged_agent=root + '/agent/agent.pt', case=root + '/prepared/cases/010/case.npz')
    config['model']['segmentation_region'] = dict(checkpoint=root + '/segmentation/best.pt', sha256=digest(checkpoint),
                                                 device=resolve_device({'runtime': cfg['runtime']}), **cfg['region'])
    config['inference']['out_dir'] = root + '/inference'
    config['runtime'] = cfg['runtime'].copy()
    save_json(out / 'teacher_agent_config.json', config)
    rl = copy.deepcopy(config)
    base_rl = json.loads(resolve('configs/filtered8_diffusion_rl.json').read_text())
    rl['agent']['trainer'] = 'diffusion_rl'
    rl['agent']['diffusion_rl'] = base_rl['agent']['diffusion_rl']
    rl['paths'].update(run_dir=root + '/rl', imitation_agent=root + '/agent/agent.pt',
                       packaged_agent=root + '/rl/agent/agent.pt')
    rl['inference']['out_dir'] = root + '/rl/inference'
    save_json(out / 'diffusion_rl_config.json', rl)


def infer(cfg, case_path, output):
    """Segment an observed pose, without any labels or a reference pose."""
    torch.set_num_threads(cfg['torch_threads'])
    checkpoint = resolve(cfg['out']) / 'segmentation/best.pt'
    options = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
                   device=resolve_device({'runtime': cfg['runtime']}), **cfg['region'])
    case = load_case(case_path)
    predictions = {}
    for name, jaw, prefix in [('upper', 0, ''), ('lower', 1, 'lower_')]:
        probability = predict_probabilities(case[prefix + 'xyz'], case[prefix + 'normal'],
                                            case[prefix + 'curvature'], jaw, options)
        predictions[name + '_probability'] = probability
        predictions[name + '_mask'] = probability >= options['threshold']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **predictions)
    print(json.dumps(dict(output=str(output), checkpoint_sha256=options['sha256'],
                          upper_points=int(predictions['upper_mask'].sum()),
                          lower_points=int(predictions['lower_mask'].sum()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/region_point_transformer.json')
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'configure', 'infer'])
    parser.add_argument('--case', help='Prepared observed mesh pair; required for infer')
    parser.add_argument('--out', help='Prediction NPZ; required for infer')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text())
    if args.stage == 'infer':
        if not args.case or not args.out:
            parser.error('infer requires --case and --out')
        infer(cfg, resolve(args.case), resolve(args.out))
    else:
        globals()[args.stage](cfg)


if __name__ == '__main__':
    main()
