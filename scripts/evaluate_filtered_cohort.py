"""Evaluate a retrained teacher and policy after cohort exclusions.

Checkpoints are selected on validation only. Historical results are descriptive,
not a controlled ablation of the data exclusion when geometry also changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from occlusion_rl_offline.aligned_teacher import build_bank, metrics, predict
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.evaluation import evaluate_continuous, load_agent
from occlusion_rl_offline.rl import load_teacher
from evaluate_diffusion import audit_trajectory, summarize_steps


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--stage', choices=['teacher', 'agent'], required=True)
    parser.add_argument('--agent', help='Optional policy checkpoint override')
    parser.add_argument('--out', help='Optional evaluation directory')
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg['agent'].get('torch_threads', 4))
    manifest = resolve(args.manifest)
    paths = read_manifest(manifest)['test']
    root = resolve(cfg['paths']['run_dir'])
    out = resolve(args.out) if args.out else root / 'test' / args.stage
    out.mkdir(parents=True, exist_ok=True)
    result = {'manifest_sha256': digest(manifest), 'config': cfg, 'results': {}}
    if args.stage == 'teacher':
        checkpoints = {stage: root / 'teacher' / stage / 'best.pt'
                       for stage in ['ranking', 'calibration']}
        result['checkpoints'] = {k: {'path': str(v), 'sha256': digest(v)}
                                 for k, v in checkpoints.items()}
        (out / 'locked_checkpoints.json').write_text(json.dumps(result['checkpoints'], indent=2))
        bank = build_bank(paths, cfg, cfg['teacher']['aligned_pair']['test_pairs_per_case'],
                          242042, out / 'test_bank.npz')
        for stage, path in checkpoints.items():
            model = load_teacher(path, 'cpu', cfg['teacher']['fps_points'], **cfg['model'])
            pred = predict(model, bank, 'cpu')
            m = metrics(pred, bank)
            m['by_case'] = {str(c): metrics(pred[bank['case_id'] == c],
                              {k: v[bank['case_id'] == c] for k, v in bank.items()})
                            for c in np.unique(bank['case_id'])}
            m['selected_epoch'] = torch.load(path, map_location='cpu', weights_only=False)['epoch']
            result['results'][stage] = m
            np.save(out / (stage + '_predictions.npy'), pred)
            print(stage, json.dumps(m), flush=True)
    else:
        teacher_path = resolve(cfg['paths']['packaged_teacher'])
        agent_path = resolve(args.agent or cfg['paths']['packaged_agent'])
        result.update(teacher_sha256=digest(teacher_path), agent_sha256=digest(agent_path),
                      agent_path=str(agent_path), selected_iteration=torch.load(agent_path, map_location='cpu', weights_only=True).get('iteration'))
        (out / 'locked_checkpoints.json').write_text(json.dumps(result, indent=2))
        teacher = load_teacher(teacher_path, 'cpu', cfg['agent']['fps_points'], **cfg['model'])
        agent = load_agent(agent_path, cfg, 'cpu', teacher_path)
        seeds = [91001, 91002, 91003] if cfg['agent']['trainer'] in ('diffusion', 'diffusion_rl') else [None]
        for seed in seeds:
            evaluated = evaluate_continuous(agent, teacher, paths, cfg, 32, 426201,
                                            (cfg['agent']['horizon'],), details=True, sampling_seed=seed)
            name = 'policy_' + str(seed)
            (out / (name + '_trajectories.json')).write_text(json.dumps(evaluated, separators=(',', ':')))
            summary = {k: v for k, v in evaluated.items() if k != 'cases'}
            summary['per_step'] = summarize_steps(evaluated)
            summary['trajectory_audit'] = audit_trajectory(evaluated, cfg['agent'])
            summary['cases'] = [{k: v for k, v in r.items() if k not in
                                ('trace', 'initial_vertex_ids', 'initial_rotations', 'initial_translations')}
                               for r in evaluated['cases']]
            result['results'][name] = summary
            print(name, json.dumps(summary['macro']), flush=True)
        rows = [r['macro'][str(cfg['agent']['horizon'])] for r in result['results'].values()]
        result['sampling_mean'] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        result['sampling_std'] = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
