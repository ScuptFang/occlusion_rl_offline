"""Replay held-out 010/011 agent trajectories in the multi-case browser view."""
from occlusion_rl_offline.runtime import resolve_device
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.rl import Agent, PoseTask, load_teacher, teacher_score, action_steps, step_context
from occlusion_rl_offline.evaluation import start_bank, BANDS, load_agent
from generate import surface_points


def frame(task, rotations, translations, features, scores, index, action=None, accepted=None):
    angle, distance = task.error(rotations, translations)
    points = features[index]
    return {'rotation': np.round(rotations[index], 7).reshape(-1).tolist(),
            'translation': np.round(translations[index], 5).tolist(),
            'rotation_error': round(float(angle[index]), 4),
            'translation_error': round(float(distance[index]), 4),
            'teacher_logit': round(float(scores[index]), 5),
            'action': None if action is None else np.round(action[index], 5).tolist(),
            'accepted_action': None if accepted is None else bool(accepted[index]),
            'selected_points': np.round(points[:, :3], 3).tolist(),
            'selected_indices': task.last_ids[index].tolist(),
            'selected_sdf': np.round(points[:, 7], 4).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/data11_agent_multi_pose.json')
    parser.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    parser.add_argument('--teacher', default='runs/data11_candidate4096_fps64_more/teacher/calibration/best.pt')
    parser.add_argument('--agent', default='runs/data11_candidate4096_fps64_more/agent_multi_pose/agent.pt')
    parser.add_argument('--data-root', default='../data')
    parser.add_argument('--out', default='demo/multi_pose_data.js')
    parser.add_argument('--examples-per-band', type=int, default=32)
    parser.add_argument('--surface-points', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.examples_per_band < 1 or args.examples_per_band > 32:
        parser.error('examples-per-band must be between 1 and 32')
    cfg = load_config(args.config)
    a = cfg['agent']
    device = resolve_device(cfg,args.device)
    teacher = load_teacher(resolve(args.teacher), device, a['fps_points'], **cfg['model'])
    agent = load_agent(resolve(args.agent), cfg, device, teacher_path=resolve(args.teacher))
    paths = read_manifest(resolve(args.manifest))['test']
    data_root = resolve(args.data_root)
    geometries, trajectories = {}, []
    for case_index, path in enumerate(paths):
        case_id = path.parent.name
        jaw_dir = data_root / case_id
        rng = np.random.default_rng(args.seed + case_index)
        geometries[case_id] = {'upper': surface_points(jaw_dir / 'upper.stl', args.surface_points, rng),
                               'lower': surface_points(jaw_dir / 'lower.stl', args.surface_points, rng)}
        for band_index, band in enumerate(a.get('demo_pose_bands', a['training_pose_bands'])):
            task = PoseTask(path, a['fps_points'], a['seed'], band['rotation_deg'],
                            band['translation_mm'], device, a['action_rotation_step_deg'],
                            a['action_translation_step_mm'], cfg['model'])
            task.rng = np.random.RandomState(args.seed + case_index * 1009 + band_index * 100003)
            if a.get('trainer') == 'continuous':
                matched = next(i for i, (_, r, t) in enumerate(BANDS)
                               if r == band['rotation_deg'] and t == band['translation_mm'])
                all_rotations, all_translations = start_bank(case_id, matched, 32, args.seed)
                all_features = task.points_at(all_rotations, all_translations)
            else:
                all_rotations, all_translations, all_features = task.reset(32)
            count = 1 if band['rotation_deg'] == band['translation_mm'] == 0 else args.examples_per_band
            rotations, translations, features = (array[:count].copy() for array in
                                                 (all_rotations, all_translations, all_features))
            if task.episode_ids is not None:
                task.episode_ids = task.episode_ids[:count]
            scores = teacher_score(teacher, features, device)
            snapshots = [[frame(task, rotations, translations, features, scores, i)]
                         for i in range(count)]
            for step in range(a['horizon']):
                task.rotation_step_deg, task.translation_step_mm = action_steps(a, step)
                with torch.no_grad():
                    action = torch.tanh(agent(task.observation(features),
                                              step_context(agent,count,step,a['horizon'],device))[0]).cpu().numpy()
                rotations, translations, features, scores, _, accepted = task.move_scored(
                    rotations, translations, features, scores, action, teacher,
                    a.get('score_gated_actions', False))
                for i in range(count):
                    snapshots[i].append(frame(task, rotations, translations, features,
                                              scores, i, action, accepted))
            for i, steps in enumerate(snapshots):
                name = f'{case_id} · ±{band["rotation_deg"]:g}° / ±{band["translation_mm"]:g} mm · 样本 {i + 1}'
                trajectories.append({'name': name, 'case_id': case_id,
                                     'band': {'rotation_deg': band['rotation_deg'],
                                              'translation_mm': band['translation_mm']},
                                     'test_episode_index': i, 'steps': steps})
                print(f'{name}: {steps[0]["rotation_error"]:.2f}° / '
                      f'{steps[0]["translation_error"]:.2f} mm → '
                      f'{steps[-1]["rotation_error"]:.2f}° / '
                      f'{steps[-1]["translation_error"]:.2f} mm', flush=True)
            if 'pivot' not in geometries[case_id]:
                ids = task.case['_uniform_candidates'][(cfg['model']['candidate_points'],
                                                        cfg['model']['point_seed'])]
                geometries[case_id].update({
                    'pivot': np.round(task.pivot, 6).tolist(),
                    'candidate_upper': np.round(task.xyz[ids], 3).tolist(),
                    'upper_mesh_vertices': len(task.xyz)})
    payload = {'title': '多病例、多随机位姿 Agent 轨迹',
               'subtitle': '010 / 011 探索性测试；执行完整动作预算，按连续误差观察恢复效果。',
               'case_id': '010 / 011', 'seed': args.seed,
               'score_gated_actions': a.get('score_gated_actions', False),
               'config_path': args.config, 'teacher_checkpoint': args.teacher,
               'agent_checkpoint': args.agent, 'sampling_mode': cfg['model'].get('sampling_mode','dynamic'),
               'action_schedule': a.get('action_schedule',[]),
               'cases': geometries, 'trajectories': trajectories,
               'action_rotation_step_deg': a['action_rotation_step_deg'],
               'action_translation_step_mm': a['action_translation_step_mm'],
               'success_threshold': {'rotation_deg': a['success_rotation_deg'],
                                     'translation_mm': a['success_translation_mm']}}
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('window.DEMO_DATA = ' + json.dumps(payload, ensure_ascii=False,
                                                     separators=(',', ':')) + ';\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()
