"""Compare two independently configured agents on identical held-out start poses."""
import argparse
import hashlib
import json
import time

import torch

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.evaluation import evaluate_continuous, load_agent
from occlusion_rl_offline.rl import load_teacher
from occlusion_rl_offline.runtime import resolve_device, runtime_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/reagent_curriculum_large.json')
    parser.add_argument('--baseline-config', default='configs/reagent_curriculum.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--teacher')
    parser.add_argument('--agent')
    parser.add_argument('--baseline-teacher', required=True)
    parser.add_argument('--baseline-agent', required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seed', type=int, default=426201)
    parser.add_argument('--device')
    parser.add_argument('--out', default='runs/reagent_curriculum_large/comparison.json')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    cfg = load_config(args.config)
    baseline = load_config(args.baseline_config)
    if cfg['agent']['horizon'] != baseline['agent']['horizon']:
        parser.error('Both agents must use the same evaluation horizon')
    if cfg['agent'].get('error_scales', [8., 4.]) != baseline['agent'].get('error_scales', [8., 4.]):
        parser.error('Both agents must use the same error normalization')
    paths = read_manifest(resolve(args.manifest))['test']
    result = {'seed': args.seed, 'episodes_per_band': args.episodes, 'split': 'test',
              'manifest_sha256': hashlib.sha256(resolve(args.manifest).read_bytes()).hexdigest(),
              'note': 'Paired start poses; each agent uses its own matching teacher and FPS density. '
                      'Teacher score gains across different teachers are not directly comparable.',
              'stages': {}}
    runs = [('baseline', baseline, args.baseline_teacher, args.baseline_agent),
            ('large', cfg, args.teacher or cfg['paths']['packaged_teacher'],
             args.agent or cfg['paths']['packaged_agent'])]
    for name, config, teacher_path, agent_path in runs:
        teacher_path, agent_path = resolve(teacher_path), resolve(agent_path)
        device = resolve_device(config, args.device)
        torch.set_num_threads(config['agent'].get('torch_threads', 4))
        teacher = load_teacher(teacher_path, device, config['agent']['fps_points'], **config['model'])
        agent = load_agent(agent_path, config, device, teacher_path)
        checkpoint = torch.load(agent_path, map_location='cpu', weights_only=True)
        start = time.perf_counter()
        stage = evaluate_continuous(agent, teacher, paths, config, args.episodes, args.seed,
                                    horizons=(config['agent']['horizon'],), details=False)
        stage.update(config=config, runtime=runtime_info(device),
                     selected_iteration=checkpoint.get('iteration'),
                     elapsed_seconds=time.perf_counter()-start,
                     teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                     agent_sha256=hashlib.sha256(agent_path.read_bytes()).hexdigest(),
                     total_parameters=sum(p.numel() for p in agent.parameters()),
                     trainable_parameters=sum(p.numel() for p in agent.parameters() if p.requires_grad),
                     policy_parameters=sum(p.numel() for n, p in agent.named_parameters()
                                           if not n.startswith('encoder.')))
        result['stages'][name] = stage
        print(name, json.dumps(stage['macro']), flush=True)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
