"""Run the region segmentation -> teacher -> imitation -> diffusion RL experiment.

Every subprocess logs to the ignored run directory. Select stages to continue
an experiment; use a new config.out for a new run. Existing trained stages are
never silently overwritten by this runner.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from occlusion_rl_offline.config import REPO_ROOT, resolve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/region_point_transformer.json')
    p.add_argument('--stages', nargs='+', default=['prepare', 'segment', 'configure', 'teacher', 'imitation', 'rl', 'evaluate'],
                   choices=['prepare', 'segment', 'configure', 'teacher', 'imitation', 'rl', 'evaluate'])
    args = p.parse_args()
    cfg = json.loads(resolve(args.config).read_text())
    env = os.environ.copy()
    # Small geometry dot products otherwise oversubscribe the CPU next to Torch.
    for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
        env[key] = str(cfg.get('math_threads', 1))
    out = resolve(cfg['out']); logs = out / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    manifest = str(out / 'manifest.json')
    training = str(out / 'teacher_agent_config.json')
    rl = str(out / 'diffusion_rl_config.json')
    py = sys.executable
    region = [py, '-m', 'occlusion_rl_offline.train_regions', '--config', str(resolve(args.config))]
    commands = {
        'prepare': [region + ['prepare']], 'segment': [region + ['train']],
        'configure': [region + ['configure']],
        'teacher': [[py, '-m', 'occlusion_rl_offline.cli', '--config', training, 'train-teacher', '--manifest', manifest]],
        'imitation': [[py, '-m', 'occlusion_rl_offline.cli', '--config', training, 'train-agent', '--manifest', manifest]],
        'rl': [[py, '-m', 'occlusion_rl_offline.cli', '--config', rl, 'train-agent', '--manifest', manifest]],
        'evaluate': [region + ['evaluate'],
            [py, 'scripts/evaluate_filtered_cohort.py', '--config', training, '--manifest', manifest, '--stage', 'teacher'],
            [py, 'scripts/evaluate_filtered_cohort.py', '--config', training, '--manifest', manifest, '--stage', 'agent'],
            [py, 'scripts/evaluate_filtered_cohort.py', '--config', rl, '--manifest', manifest, '--stage', 'agent'],
            [py, 'scripts/evaluate_filtered_cohort.py', '--config', rl, '--manifest', manifest, '--stage', 'agent',
             '--agent', str(out / 'rl/agent/imitation.pt'), '--out', str(out / 'rl/test/imitation')]]}
    sentinels = dict(prepare='manifest.json', segment='segmentation/best.pt', configure='teacher_agent_config.json',
                     teacher='teacher/ranking/best.pt', imitation='agent/latest.pt', rl='rl/agent/latest.pt')
    def execute(stage, i, command):
        log = logs / ('%s_%d.log' % (stage, i))
        print(json.dumps(dict(stage=stage, command=command, log=str(log))), flush=True)
        with log.open('w') as f:
            subprocess.run(command, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
        print(json.dumps(dict(stage=stage, task=i, completed=True)), flush=True)

    for stage in args.stages:
        if stage in sentinels and (out / sentinels[stage]).exists():
            raise FileExistsError('Stage already exists: %s; omit it from --stages or use a new config.out' % stage)
        if stage == 'evaluate':
            # All weights are frozen; tasks write to distinct output directories.
            with ThreadPoolExecutor(max_workers=cfg.get('evaluation_workers', 3)) as pool:
                futures = [pool.submit(execute, stage, i, command) for i, command in enumerate(commands[stage])]
                for future in futures:
                    future.result()
        else:
            for i, command in enumerate(commands[stage]):
                execute(stage, i, command)


if __name__ == '__main__':
    main()
