"""Compare locked IL/PPO checkpoints on one shared held-out start bank.

No model selection happens here. agent.pt must already be selected on validation.
Outputs small aggregate JSON; no meshes, weights or demo trajectories.
"""
import argparse
import hashlib
import json

import torch

from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.evaluation import evaluate_continuous, load_agent
from occlusion_rl_offline.rl import load_teacher
from occlusion_rl_offline.runtime import resolve_device, runtime_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reagent_curriculum.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--run-dir', default='runs/reagent_curriculum/agent')
    parser.add_argument('--baseline', required=True, help='Teacher-only validation-selected checkpoint')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seed', type=int, default=426201)
    parser.add_argument('--device')
    parser.add_argument('--out', default='runs/reagent_curriculum/comparison.json')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    cfg = load_config(args.config)
    device = resolve_device(cfg, args.device)
    torch.set_num_threads(cfg['agent'].get('torch_threads', 4))
    teacher_path = resolve(args.teacher)
    teacher = load_teacher(teacher_path, device, cfg['agent']['fps_points'], **cfg['model'])
    paths = read_manifest(resolve(args.manifest))['test']
    root = resolve(args.run_dir)
    checkpoints = {'imitation': root / 'imitation.pt', 'curriculum_best': root / 'agent.pt',
                   'curriculum_final': root / 'latest.pt', 'teacher_only': resolve(args.baseline)}
    # Resolve every file before beginning; a missing baseline is an error.
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in checkpoints.items()}
    result = {'seed': args.seed, 'episodes_per_band': args.episodes, 'split': 'test',
              'runtime': runtime_info(device), 'config': cfg,
              'teacher_sha256': hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(resolve(args.manifest).read_bytes()).hexdigest(),
              'checkpoint_sha256': hashes, 'stages': {}}
    for name, path in checkpoints.items():
        agent = load_agent(path, cfg, device, teacher_path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        stage = evaluate_continuous(agent, teacher, paths, cfg, args.episodes, args.seed,
                                    horizons=(cfg['agent']['horizon'],), details=False)
        stage['selected_iteration'] = checkpoint.get('iteration')
        stage['checkpoint'] = str(path.relative_to(resolve('.'))) if path.is_relative_to(resolve('.')) else str(path)
        result['stages'][name] = stage
        print(name, json.dumps(stage['macro']), flush=True)
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
