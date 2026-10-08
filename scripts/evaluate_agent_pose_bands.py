"""Evaluate one trained agent on held-out cases across random start-pose bands."""
import argparse
import json
from pathlib import Path

import torch

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.rl import Agent, PoseTask, evaluate, load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/data11_agent_multi_pose.json')
    parser.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    parser.add_argument('--teacher', default='runs/data11_candidate4096_fps64_more/teacher/calibration/best.pt')
    parser.add_argument('--agent', default='runs/data11_candidate4096_fps64_more/agent_multi_pose/agent.pt')
    parser.add_argument('--out', default='runs/data11_candidate4096_fps64_more/agent_multi_pose/pose_band_test.json')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    cfg = load_config(args.config)
    a = cfg['agent']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher = load_teacher(resolve(args.teacher), device, a['fps_points'], **cfg['model'])
    agent = Agent(a['fps_points'], a['policy_width'], a['initial_log_std'],
                  **cfg['model']).to(device)
    agent.load_state_dict(torch.load(resolve(args.agent), map_location=device)['model'])
    agent.eval()
    cases = read_manifest(resolve(args.manifest))['test']
    result = {'seed': args.seed, 'episodes_per_case_band': args.episodes,
              'score_gated_actions': a.get('score_gated_actions', False), 'cases': {}}
    for index, path in enumerate(cases):
        case_id = path.parent.name
        result['cases'][case_id] = {}
        for band_index, band in enumerate(a['training_pose_bands']):
            task = PoseTask(path, a['fps_points'], a['seed'], band['rotation_deg'],
                            band['translation_mm'], device, a['action_rotation_step_deg'],
                            a['action_translation_step_mm'], cfg['model'])
            metrics = evaluate(agent, task, teacher, args.episodes, a['horizon'],
                               args.seed + index * 1009 + band_index * 100003,
                               a['success_rotation_deg'], a['success_translation_mm'],
                               a.get('score_gated_actions', False))
            name = f'{band["rotation_deg"]}deg_{band["translation_mm"]}mm'
            result['cases'][case_id][name] = metrics
            print(case_id, name, json.dumps(metrics), flush=True)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
