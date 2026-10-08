"""Evaluate deterministic pose recovery by case and initial perturbation band."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.rl import Agent, PoseTask, load_teacher, teacher_score


def summarize(initial_angle, initial_distance, final_angle, final_distance,
              initial_score, final_score, rotation_limit, translation_limit):
    success = (final_angle <= rotation_limit) & (final_distance <= translation_limit)
    gain = final_score - initial_score
    return {
        'episodes': len(success),
        'success_count': int(success.sum()),
        'success_rate': float(success.mean()),
        'initial_rotation_mean_deg': float(initial_angle.mean()),
        'final_rotation_mean_deg': float(final_angle.mean()),
        'final_rotation_p90_deg': float(np.quantile(final_angle, .9)),
        'initial_translation_mean_mm': float(initial_distance.mean()),
        'final_translation_mean_mm': float(final_distance.mean()),
        'final_translation_p90_mm': float(np.quantile(final_distance, .9)),
        'initial_teacher_mean': float(initial_score.mean()),
        'final_teacher_mean': float(final_score.mean()),
        'score_improved_fraction': float((gain > 0).mean()),
        'score_up_rotation_worse_fraction': float(((gain > 0) & (final_angle > initial_angle)).mean()),
        'score_up_translation_worse_fraction': float(((gain > 0) & (final_distance > initial_distance)).mean()),
    }


def evaluate_band(case_path, case_index, band_index, band, config, teacher, agent,
                  episodes, rotation_limit, translation_limit, seed):
    a = config['agent']
    task = PoseTask(case_path, a['fps_points'], seed + case_index * 1009 + band_index * 100003,
                    band['rotation_deg'], band['translation_mm'], 'cpu',
                    a['action_rotation_step_deg'], a['action_translation_step_mm'],
                    config['model'])
    rotations, translations, points = task.reset(episodes)
    initial_angle, initial_distance = task.error(rotations, translations)
    initial_score = teacher_score(teacher, points, 'cpu')
    score = initial_score.copy()
    for _ in range(a['horizon']):
        with torch.no_grad():
            action = torch.tanh(agent(task.observation(points))[0]).cpu().numpy()
        rotations, translations, points, score, _, _ = task.move_scored(
            rotations, translations, points, score, action, teacher,
            a.get('score_gated_actions', False))
    final_angle, final_distance = task.error(rotations, translations)
    final_score = teacher_score(teacher, points, 'cpu')
    summary = summarize(initial_angle, initial_distance, final_angle, final_distance,
                        initial_score, final_score, rotation_limit, translation_limit)
    summary['case_id'] = Path(case_path).parent.name
    summary['band'] = band['name']
    summary['episodes_detail'] = [
        {'initial_rotation_deg': float(ia), 'initial_translation_mm': float(it),
         'final_rotation_deg': float(fa), 'final_translation_mm': float(ft),
         'teacher_gain': float(fs - ins),
         'success': bool(fa <= rotation_limit and ft <= translation_limit)}
        for ia, it, fa, ft, ins, fs in zip(initial_angle, initial_distance,
                                           final_angle, final_distance, initial_score, final_score)]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--episodes-per-case', type=int, default=64)
    parser.add_argument('--bands', nargs='+', choices=('near', 'medium', 'full'),
                        default=['near', 'medium', 'full'])
    parser.add_argument('--seed', type=int, default=242042)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.episodes_per_case < 1:
        parser.error('--episodes-per-case must be positive')
    config = load_config(resolve(args.config))
    paths = read_manifest(resolve(args.manifest))[args.split]
    torch.set_num_threads(4)
    teacher = load_teacher(resolve(args.teacher), 'cpu', config['teacher']['fps_points'],
                           **config['model'])
    agent = Agent(config['agent']['fps_points'], config['agent']['policy_width'],
                  config['agent']['initial_log_std'], **config['model'])
    agent.load_state_dict(torch.load(resolve(args.agent), map_location='cpu')['model'])
    agent.eval()
    bands = [{'name': 'near', 'rotation_deg': 1.5, 'translation_mm': .5},
             {'name': 'medium', 'rotation_deg': 4., 'translation_mm': 2.},
             {'name': 'full', 'rotation_deg': 8., 'translation_mm': 4.}]
    bands = [band for band in bands if band['name'] in args.bands]
    results = [evaluate_band(path, case_index, band_index, band, config, teacher, agent,
                             args.episodes_per_case, config['agent']['success_rotation_deg'],
                             config['agent']['success_translation_mm'], args.seed)
               for band_index, band in enumerate(bands)
               for case_index, path in enumerate(paths)]
    output = {'split': args.split, 'episodes_per_case': args.episodes_per_case,
              'rotation_limit_deg': config['agent']['success_rotation_deg'],
              'translation_limit_mm': config['agent']['success_translation_mm'],
              'bands': bands, 'results': results}
    target = resolve(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2))
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'episodes_detail'}))


if __name__ == '__main__':
    main()
