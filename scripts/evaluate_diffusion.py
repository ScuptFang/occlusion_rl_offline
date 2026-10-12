"""Paired per-step diffusion/PPO trajectories, with diffusion sampling replicates.

Checkpoints must already be selected on validation. All large traces stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.evaluation import evaluate_continuous, load_agent
from occlusion_rl_offline.rl import load_teacher, action_steps
from occlusion_rl_offline.runtime import resolve_device, runtime_info


def summarize_steps(result):
    rows = [r for r in result['cases'] if r['band'] != 'zero']
    return [{'step': step, **{metric: float(np.mean([np.mean(r['trace'][step][metric]) for r in rows]))
                              for metric in ('rotation_deg', 'translation_mm', 'teacher_score')}}
            for step in range(len(rows[0]['trace']))]


def audit_trajectory(result, options):
    """Replay the recorded physical updates and quantify action variation."""
    executed = score_decreasing = 0
    changes = []
    for row in result['cases']:
        trace = row['trace']
        assert len(trace) == max(result['horizons']) + 1
        physical_actions = []
        for step, (before, after) in enumerate(zip(trace, trace[1:])):
            action = np.asarray(before['action'])
            assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
            assert all(before['valid_action']) and all(before['accepted_action'])
            if 'planned_actions' in before:
                np.testing.assert_array_equal(action, np.asarray(before['planned_actions'])[:, 0])
            rs, ts = action_steps(options, step)
            rotation = Rotation.from_rotvec(np.deg2rad(action[:, :3] * rs)).as_matrix() @ np.asarray(before['rotations'])
            translation = np.asarray(before['translations']) + action[:, 3:] * ts
            np.testing.assert_allclose(rotation, after['rotations'], atol=1e-6)
            np.testing.assert_allclose(translation, after['translations'], atol=1e-6)
            physical_actions.append(action * np.array([rs] * 3 + [ts] * 3))
            executed += len(action)
            score_decreasing += int((np.asarray(after['teacher_score']) < before['teacher_score']).sum())
        if len(physical_actions) > 1:
            changes.append(np.diff(physical_actions, axis=0).reshape(-1, 6))
    delta = np.concatenate(changes) if changes else None
    return {'executed_actions': executed, 'score_decreasing_actions': score_decreasing,
            'mean_action_change_rotation_deg': float(np.linalg.norm(delta[:, :3], axis=-1).mean()) if delta is not None else None,
            'mean_action_change_translation_mm': float(np.linalg.norm(delta[:, 3:], axis=-1).mean()) if delta is not None else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/diffusion_policy.json')
    p.add_argument('--manifest', required=True)
    p.add_argument('--teacher')
    p.add_argument('--agent')
    p.add_argument('--baseline-config', default='configs/reagent_curriculum_large.json')
    p.add_argument('--baseline', default='runs/reagent_curriculum_large/agent/agent.pt')
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--seed', type=int, default=426201)
    p.add_argument('--sampling-seeds', type=int, nargs='+', default=[91001, 91002, 91003])
    p.add_argument('--device')
    p.add_argument('--out', default='runs/diffusion_policy/test')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('Positive episode count required')
    cfg, base = load_config(args.config), load_config(args.baseline_config)
    if cfg['model'] != base['model'] or any(cfg['agent'][k] != base['agent'][k]
            for k in ('fps_points', 'horizon', 'error_scales', 'action_rotation_step_deg', 'action_translation_step_mm')):
        p.error('Baseline must use identical geometry, density, action scales and horizon')
    if cfg['agent'].get('action_schedule') != base['agent'].get('action_schedule'):
        p.error('Baseline must use the same action schedule')
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    teacher_path = resolve(args.teacher or cfg['paths']['packaged_teacher'])
    checkpoint = resolve(args.agent or cfg['paths']['packaged_agent'])
    device = resolve_device(cfg, args.device)
    torch.set_num_threads(cfg['agent'].get('torch_threads', 4))
    paths = read_manifest(resolve(args.manifest))['test']
    teacher = load_teacher(teacher_path, device, cfg['agent']['fps_points'], **cfg['model'])
    result = {'seed': args.seed, 'episodes_per_band': args.episodes, 'runtime': runtime_info(device),
              'config': cfg, 'baseline_config': base,
              'teacher_sha256': hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(resolve(args.manifest).read_bytes()).hexdigest(),
              'note': 'One locked diffusion checkpoint; sampling seeds are replicates, never selected by test error.',
              'stages': {}}
    policies = [('ppo', base, resolve(args.baseline), None)]
    policies += [('diffusion_%d' % seed, cfg, checkpoint, seed) for seed in args.sampling_seeds]
    for name, config, path, sampling_seed in policies:
        agent = load_agent(path, config, device, teacher_path)
        evaluated = evaluate_continuous(agent, teacher, paths, config, args.episodes, args.seed,
                                        (config['agent']['horizon'],), details=True, sampling_seed=sampling_seed)
        trace_file = out/(name+'_trajectories.json')
        trace_file.write_text(json.dumps(evaluated, separators=(',', ':'))+'\n')
        summary = {k: v for k, v in evaluated.items() if k != 'cases'}
        summary['per_step'] = summarize_steps(evaluated)
        summary['trajectory_audit'] = audit_trajectory(evaluated, config['agent'])
        summary['cases'] = [{k: v for k, v in row.items() if k not in
                             ('trace', 'initial_vertex_ids', 'initial_rotations', 'initial_translations')}
                            for row in evaluated['cases']]
        summary['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary['selected_iteration'] = torch.load(path, map_location='cpu', weights_only=True).get('iteration')
        summary['trajectories'] = trace_file.name
        result['stages'][name] = summary
        print(name, json.dumps(summary['macro']), flush=True)
    diffusion = [r['macro'][str(cfg['agent']['horizon'])] for k, r in result['stages'].items() if k.startswith('diffusion_')]
    result['diffusion_sampling_mean'] = {k: float(np.mean([r[k] for r in diffusion])) for k in diffusion[0]}
    result['diffusion_sampling_std'] = {k: float(np.std([r[k] for r in diffusion])) for k in diffusion[0]}
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
