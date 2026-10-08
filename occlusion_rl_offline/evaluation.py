"""Paired rotvec start banks and full-budget continuous recovery evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .runtime import resolve_device
from .config import load_config, resolve
from .cohort import read_manifest
from .rl import Agent, PoseTask, load_teacher, teacher_score, action_steps, step_context, continuous_error, validate_teacher_pair

BANDS = [('zero', 0., 0.), ('near', 1.5, .5), ('medium', 4., 2.), ('wide', 8., 4.)]


def load_agent(path, cfg, device='cpu', teacher_path=None):
    a = cfg['agent']
    agent = Agent(a['fps_points'], a['policy_width'], a['initial_log_std'],
                  time_context=a.get('time_context', False), **cfg['model']).to(device)
    checkpoint = torch.load(path, map_location=device)
    if teacher_path is not None:
        validate_teacher_pair(checkpoint, teacher_path)
    if checkpoint.get('model_options') is not None and checkpoint['model_options'] != cfg['model']:
        raise ValueError('Agent model/sampling configuration does not match checkpoint')
    agent.load_state_dict(checkpoint['model'])
    agent.eval()
    return agent


def start_bank(case_id, band_index, episodes, seed):
    rng = np.random.RandomState(seed + int(case_id) * 1009 + band_index * 100003)
    _, r, t = BANDS[band_index]
    # Interleaved draws make prefixes stable when episode count is changed.
    draws = rng.uniform(-1, 1, (episodes, 6))
    return Rotation.from_rotvec(np.deg2rad(draws[:, :3] * r)).as_matrix(), draws[:, 3:] * t


def evaluate_continuous(agent, teacher, paths, cfg, episodes=32, seed=426101,
                        horizons=(10, 20, 40), mode='agent', details=True):
    a = cfg['agent']
    device = str(next(agent.parameters()).device)
    scales = a.get('error_scales', [8., 4.])
    rows = []
    for path in paths:
        case_id = Path(path).parent.name
        task = PoseTask(path, a['fps_points'], seed, 8, 4, device,
                        a['action_rotation_step_deg'], a['action_translation_step_mm'], cfg['model'])
        for bi, (band_name, _, _) in enumerate(BANDS):
            n = 1 if bi == 0 else episodes
            r, t = start_bank(case_id, bi, n, seed)
            initial_r, initial_t = r.copy(), t.copy()
            task.begin_episode()
            points = task.points_at(r, t)
            initial_ids = task.last_ids.copy()
            scores = teacher_score(teacher, points, device)
            rng = np.random.RandomState(seed + int(case_id) * 1709 + bi * 3001)
            trace, snapshots = [], {}
            initial_angle, initial_distance = task.error(r, t)
            initial_score = scores.copy()
            for step in range(max(horizons) + 1):
                angle, distance = task.error(r, t)
                error = continuous_error(angle, distance, scales)
                trace.append({'step': step, 'rotation_deg': angle.tolist(),
                              'translation_mm': distance.tolist(), 'teacher_score': scores.tolist(),
                              'contact_fraction': (np.abs(points[:,:,7]) < .6).mean(1).tolist(),
                              'median_abs_distance_mm': np.median(np.abs(points[:,:,7]),axis=1).tolist()})
                if details:
                    trace[-1].update(rotations=r.tolist(), translations=t.tolist(),
                                     selected_vertex_ids=task.last_ids.tolist())
                if step in horizons:
                    gain = scores - initial_score
                    snapshots[str(step)] = {
                        'rotation_deg': float(angle.mean()), 'translation_mm': float(distance.mean()),
                        'rotation_median_deg': float(np.median(angle)),
                        'translation_median_mm': float(np.median(distance)),
                        'rotation_p90_deg': float(np.quantile(angle, .9)),
                        'translation_p90_mm': float(np.quantile(distance, .9)),
                        'error': float(error.mean()), 'teacher_gain': float(gain.mean()),
                        'rotation_reduction_deg': float((initial_angle-angle).mean()),
                        'translation_reduction_mm': float((initial_distance-distance).mean()),
                        'rotation_improved_fraction': float((angle < initial_angle).mean()),
                        'translation_improved_fraction': float((distance < initial_distance).mean()),
                        'score_up_error_worse_fraction': float(((gain > 0) & (error > continuous_error(initial_angle, initial_distance, scales))).mean())}
                if step == max(horizons):
                    break  # Exhausted the externally specified budget, not a policy decision.
                task.rotation_step_deg, task.translation_step_mm = action_steps(a, step)
                obs = task.observation(points)
                with torch.no_grad():
                    if mode == 'agent':
                        ctx = step_context(agent, n, step, max(horizons), device)
                        action = torch.tanh(agent(obs, ctx)[0]).cpu().numpy()
                    elif mode == 'random':
                        action = rng.uniform(-1, 1, (n, 6))
                    elif mode == 'constant':
                        action = np.tile([.15, 0, .2, .15, 0, -.15], (n, 1))
                    else:
                        raise ValueError(mode)
                if details:
                    trace[-1]['action'] = action.tolist()
                r, t, points, scores, _, _ = task.move_scored(r, t, points, scores, action, teacher, False)
            row = {'case': case_id, 'band': band_name, 'episodes': n,
                   'initial_rotation_deg': float(initial_angle.mean()),
                   'initial_translation_mm': float(initial_distance.mean()),
                   'initial_error': float(continuous_error(initial_angle, initial_distance, scales).mean()),
                   'checkpoints': snapshots}
            if details:
                row.update(initial_rotations=initial_r.tolist(), initial_translations=initial_t.tolist(),
                           initial_vertex_ids=initial_ids.tolist(), trace=trace)
            rows.append(row)
    macro = {str(h): {k: float(np.mean([r['checkpoints'][str(h)][k] for r in rows if r['band'] != 'zero']))
                      for k in rows[1]['checkpoints'][str(h)]} for h in horizons}
    return {'seed': seed, 'mode': mode, 'episodes_per_nonzero_band': episodes,
            'horizons': list(horizons), 'error_scales': scales,
            'note': 'Full-budget actions; zero band separate from equally weighted case/band macro. Intermediate checkpoints are prefixes of the longest rollout.',
            'initial_macro_error': float(np.mean([r['initial_error'] for r in rows if r['band'] != 'zero'])),
            'macro': macro, 'cases': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    p.add_argument('--teacher', required=True)
    p.add_argument('--agent', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--seed', type=int, default=426101)
    p.add_argument('--horizons', nargs='+', type=int, default=[10, 20, 40])
    p.add_argument('--mode', choices=['agent','random','constant'], default='agent')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if args.episodes < 1 or min(args.horizons) < 1:
        p.error('Positive episode counts and horizons required')
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    device = resolve_device(cfg,args.device)
    teacher = load_teacher(resolve(args.teacher), device, cfg['teacher']['fps_points'], **cfg['model'])
    agent = load_agent(resolve(args.agent), cfg, device=device, teacher_path=resolve(args.teacher))
    paths = read_manifest(resolve(args.manifest))[args.split]
    result = evaluate_continuous(agent, teacher, paths, cfg, args.episodes, args.seed, args.horizons, args.mode)
    result.update(config=cfg, split=args.split, teacher=args.teacher, agent=args.agent,
                  checkpoint_sha256={label: hashlib.sha256(resolve(file).read_bytes()).hexdigest()
                                     for label, file in [('teacher',args.teacher),('agent',args.agent)]})
    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['macro']), flush=True)

if __name__ == '__main__':
    main()
