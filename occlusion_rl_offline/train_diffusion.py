"""Offline expert action-chunk generation and conditional diffusion training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .config import load_config, resolve
from .cohort import read_manifest
from .diffusion import DiffusionAgent
from .evaluation import evaluate_continuous
from .imitation import expert_actions
from .rl import PoseTask, action_steps, load_teacher, step_context
from .runtime import resolve_device, runtime_info


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expert_chunk(rotations, translations, a, step, length):
    """Only label generation sees target-relative poses; future points are absent."""
    r, t = rotations.copy(), translations.copy()
    chunk = []
    for offset in range(length):
        rs, ts = action_steps(a, step + offset)
        action = expert_actions(r, t, rs, ts)
        chunk.append(action)
        r = Rotation.from_rotvec(np.deg2rad(action[:, :3] * rs)).as_matrix() @ r
        t = t + action[:, 3:] * ts
    return np.stack(chunk, axis=1)


def collect(cfg, manifest, out):
    a, d = cfg['agent'], cfg['agent']['diffusion']
    paths = read_manifest(manifest)['train']
    # Fail closed on stale caches, including changes to mesh data or split.
    metadata = {'manifest_sha256': sha256(manifest), 'model': cfg['model'],
                'agent': a, 'train_cases': {str(p): sha256(p) for p in paths}}
    cache, info = out / 'expert_chunks.npz', out / 'expert_chunks.json'
    if cache.exists() or info.exists():
        if not (cache.exists() and info.exists()) or json.loads(info.read_text()) != metadata:
            raise ValueError('Expert cache configuration/data mismatch; use a new output directory')
        with np.load(cache) as f:
            return {k: torch.from_numpy(f[k]) for k in ('observations', 'contexts', 'actions')}
    observations, contexts, actions = [], [], []
    for ci, path in enumerate(paths):
        task = PoseTask(path, a['fps_points'], a['seed'] + ci * 100003, 8, 4, 'cpu',
                        a['action_rotation_step_deg'], a['action_translation_step_mm'], cfg['model'])
        for first in range(0, d['episodes_per_case'], d['collection_batch']):
            count = min(d['collection_batch'], d['episodes_per_case'] - first)
            r, t, points = task.reset(count, a['training_pose_bands'])
            for step in range(a['horizon']):
                target = expert_chunk(r, t, a, step, d['prediction_horizon'])
                observations.append(points.copy())
                contexts.append(np.tile([step / a['horizon'], 1 - step / a['horizon']], (count, 1)))
                actions.append(target)
                task.rotation_step_deg, task.translation_step_mm = action_steps(a, step)
                random_action = task.rng.uniform(-1, 1, (count, 6))
                random_mask = task.rng.rand(count, 1) < d['random_action_probability']
                executed = np.where(random_mask, random_action, target[:, 0])
                r, t, points, _ = task.move(r, t, executed)
        print('collected', path.parent.name, d['episodes_per_case'] * a['horizon'], flush=True)
    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in
              [('observations', observations), ('contexts', contexts), ('actions', actions)]}
    np.savez(cache, **arrays)
    info.write_text(json.dumps(metadata, indent=2)+'\n')
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


def run(cfg, manifest, teacher_path, out, resume=None):
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)
    a, d = cfg['agent'], cfg['agent']['diffusion']
    torch.set_num_threads(a.get('torch_threads', 4))
    torch.manual_seed(a['seed'])
    teacher = load_teacher(teacher_path, device, a['fps_points'], **cfg['model'])
    agent = DiffusionAgent(a, cfg['model']).to(device)
    agent.encoder.load_state_dict(teacher.state_dict())
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in agent.named_parameters() if n.startswith('encoder.') and p.requires_grad],
         'lr': d['encoder_learning_rate']},
        {'params': [p for n, p in agent.named_parameters() if not n.startswith('encoder.') and p.requires_grad],
         'lr': d['learning_rate']}], weight_decay=d['weight_decay'])
    ema = copy.deepcopy(agent).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    dataset = collect(cfg, manifest, out)
    train_count = len(dataset['actions'])
    teacher_hash, manifest_hash = sha256(teacher_path), sha256(manifest)
    history, start, best, loss_sum, loss_count = [], 0, float('inf'), 0., 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if (checkpoint['agent_options'] != a or checkpoint['model_options'] != cfg['model'] or
                checkpoint['teacher_sha256'] != teacher_hash or checkpoint['manifest_sha256'] != manifest_hash):
            raise ValueError('Diffusion resume configuration/teacher/data mismatch')
        agent.load_state_dict(checkpoint['raw_model'])
        ema.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, history = checkpoint['iteration'], checkpoint['best_error'], checkpoint['history']
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(), device)
        if not (out/'agent.pt').exists():
            import shutil
            shutil.copyfile(Path(resume).parent/'agent.pt', out/'agent.pt')
            previous_evaluation = Path(resume).parent/'evaluation.json'
            if previous_evaluation.exists():
                shutil.copyfile(previous_evaluation, out/'evaluation.json')
    (out/'config.snapshot.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(runtime_info(device), indent=2)+'\n')
    paths = read_manifest(manifest)
    for iteration in range(start, d['training_steps']):
        ids = torch.randint(train_count, (d['batch_size'],))
        batch = {k: v[ids].to(device) for k, v in dataset.items()}
        loss = agent.noise_loss(batch['observations'], batch['contexts'], batch['actions'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diffusion noise loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), d['max_grad_norm'])
        optimizer.step()
        with torch.no_grad():
            decay = min(d['ema_decay'], (1 + iteration) / (10 + iteration))
            for ep, p in zip(ema.parameters(), agent.parameters()):
                ep.lerp_(p, 1 - decay)
        loss_sum += float(loss.detach()); loss_count += 1
        if (iteration + 1) % d['validate_every'] == 0 or iteration + 1 == d['training_steps']:
            result = evaluate_continuous(ema, teacher, paths['val'], cfg, a['validation_episodes'],
                                         a.get('validation_seed', 426101), (a['horizon'],), details=False)
            error = result['macro'][str(a['horizon'])]['error']
            improved = error < best
            best = min(best, error)
            row = {'iteration': iteration + 1, 'noise_mse': loss_sum / loss_count,
                   'sample_presentations': (iteration + 1) * d['batch_size'], 'validation': result}
            history.append(row)
            loss_sum, loss_count = 0., 0
            checkpoint = {'model': ema.state_dict(), 'raw_model': agent.state_dict(),
                          'model_options': cfg['model'], 'agent_options': a,
                          'teacher_sha256': teacher_hash, 'manifest_sha256': manifest_hash,
                          'teacher_checkpoint': str(teacher_path), 'iteration': iteration + 1,
                          'best_error': best, 'history': history, 'optimizer': optimizer.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state(device) if device.startswith('cuda') else None}
            torch.save(checkpoint, out/'latest.pt')
            if improved:
                torch.save(checkpoint, out/'agent.pt')
                (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
            (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(json.dumps({'iteration': iteration + 1, 'noise_mse': row['noise_mse'],
                              'validation': result['macro']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/diffusion_policy.json')
    p.add_argument('--manifest', required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--resume')
    p.add_argument('--device')
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg['runtime']['device'] = resolve_device(cfg, args.device)
    run(cfg, resolve(args.manifest), resolve(args.teacher), resolve(args.out),
        resolve(args.resume) if args.resume else None)


if __name__ == '__main__':
    main()
