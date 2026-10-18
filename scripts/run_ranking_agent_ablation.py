"""Train and test a fresh diffusion agent using the region ranking teacher.

The calibration stage is omitted; all sampling, splits and budgets are inherited
from the completed region experiment. Artifacts stay under the ignored run dir.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from occlusion_rl_offline.config import REPO_ROOT, load_config, resolve
from occlusion_rl_offline.region_segmentation import digest
from report_region_experiment import policy_summary, read


def report(out, source):
    paths = dict(imitation='test/agent/comparison.json',
                 stochastic_imitation='rl/test/imitation/comparison.json',
                 rl='rl/test/agent/comparison.json')
    summary = {'ranking_teacher': {k: policy_summary(out / v) for k, v in paths.items()},
               'calibrated_teacher': {k: policy_summary(source / v) for k, v in paths.items()}}
    for group, root in [('ranking_teacher', out), ('calibrated_teacher', source)]:
        for policy, path in paths.items():
            evaluations = read(root / path)['results'].values()
            rows = [c for result in evaluations for c in result['cases'] if c['band'] != 'zero']
            summary[group][policy]['by_case'] = {
                case: {k: float(np.mean([r['checkpoints']['10'][k] for r in rows if r['case'] == case]))
                       for k in ('rotation_deg', 'translation_mm', 'error')}
                for case in sorted({r['case'] for r in rows})}
    # Compare every test start, including the zero-pose drift probes.
    for seed in (91001, 91002, 91003):
        baseline = read(source / ('rl/test/agent/policy_%d_trajectories.json' % seed))
        for path in paths.values():
            current = read(out / Path(path).parent / ('policy_%d_trajectories.json' % seed))
            assert len(current['cases']) == len(baseline['cases'])
            for x, y in zip(current['cases'], baseline['cases']):
                assert (x['case'], x['band']) == (y['case'], y['band'])
                for key in ('initial_rotations', 'initial_translations'):
                    np.testing.assert_array_equal(x[key], y[key])
    cfg = read(out / 'teacher_agent_config.json')
    source_cfg = read(source / 'teacher_agent_config.json')
    for key in ('agent', 'model', 'teacher'):
        assert cfg[key] == source_cfg[key], 'Training configuration changed: ' + key
    # The supervised samples must be identical; only encoder initialization differs.
    with np.load(out / 'agent/expert_chunks.npz') as fresh, np.load(source / 'agent/expert_chunks.npz') as old:
        for key in ('observations', 'contexts', 'actions'):
            np.testing.assert_array_equal(fresh[key], old[key])
    teacher_hash = digest(resolve(cfg['paths']['packaged_teacher']))
    initial = torch.load(out / 'rl/agent/imitation.pt', map_location='cpu', weights_only=True)
    selected = torch.load(out / 'rl/agent/agent.pt', map_location='cpu', weights_only=True)
    assert initial['teacher_sha256'] == selected['teacher_sha256'] == teacher_hash
    frozen = [k for k in initial['model'] if k.startswith(('reference.', 'actor.encoder.'))]
    assert all(torch.equal(initial['model'][k], selected['model'][k]) for k in frozen)
    region = cfg['model']['segmentation_region']
    assert digest(resolve(region['checkpoint'])) == region['sha256']
    history = read(out / 'rl/agent/history.json')
    imitation_history = read(out / 'agent/history.json')
    summary['audit'] = dict(paired_starts_verified=True, teacher_sha256=teacher_hash,
                            identical_expert_samples=True,
                            frozen_tensors_unchanged=len(frozen),
                            rl_iterations=len(history),
                            optimizer_updates=sum(r['optimizer_updates'] for r in history),
                            selected_iteration=selected['iteration'],
                            score_decreasing_training_actions=sum(r.get('score_decreasing_actions', 0) for r in history))
    summary['training'] = dict(imitation_updates=imitation_history[-1]['iteration'],
        imitation_first_mse=imitation_history[0]['noise_mse'],
        imitation_last_mse=imitation_history[-1]['noise_mse'],
        imitation_validation=read(out / 'agent/evaluation.json')['macro'],
        stochastic_imitation_validation=read(out / 'rl/agent/imitation_evaluation.json')['macro'],
        rl_validation=read(out / 'rl/agent/evaluation.json')['macro'],
        best_including_imitation_iteration=torch.load(out / 'rl/agent/best_including_imitation.pt',
                                                       map_location='cpu', weights_only=True)['iteration'])
    summary['teacher_test'] = read(source / 'test/teacher/comparison.json')['results']
    lines = ['# Ranking teacher agent 对照实验', '',
             '未矫正指跳过第二阶段 calibration；ranking 本身仍含姿态质量回归及分数差监督。', '',
             '| Teacher | Agent | 角度 ° | 平移 mm | 综合误差 E |',
             '|---|---|---:|---:|---:|']
    for group in ('calibrated_teacher', 'ranking_teacher'):
        for name, value in summary[group].items():
            m = value['mean']
            lines.append('| %s | %s | %.4f | %.4f | %.5f |' %
                         (group, name, m['rotation_deg'], m['translation_mm'], m['error']))
    ranking = summary['ranking_teacher']['rl']['mean']
    calibrated = summary['calibrated_teacher']['rl']['mean']
    imitation = summary['ranking_teacher']['stochastic_imitation']['mean']
    summary['rl_relative_change'] = {name: {k: ranking[k] / baseline[k] - 1
                                           for k in ('rotation_deg', 'translation_mm', 'error')}
                                    for name, baseline in [('vs_calibrated_rl', calibrated),
                                                           ('vs_matching_imitation', imitation)]}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines += ['', '未校准 RL 相对校准 RL：角度 %+.2f%%，平移 %+.2f%%，E %+.2f%%；相对自身随机推理模仿基线 E %+.2f%%。' % (
        100 * summary['rl_relative_change']['vs_calibrated_rl']['rotation_deg'],
        100 * summary['rl_relative_change']['vs_calibrated_rl']['translation_mm'],
        100 * summary['rl_relative_change']['vs_calibrated_rl']['error'],
        100 * summary['rl_relative_change']['vs_matching_imitation']['error'])]
    lines += ['', 'E = 0.5 × (角度/8° + 平移/4 mm)。主表为非零扰动，3 个推理采样种子均值；零位姿漂移和分档结果见 summary.json。',
              '训练病例 003、002、005、004、007；验证 006；测试 010、011。所有 checkpoint 按验证集选取。',
              '每个种子使用相同的 194 条轨迹，固定 10 步；不拒绝降分动作，不按精度提前停止。',
              '两组分别从各自 teacher 编码器初始化并重新模仿训练，所以比较的是完整训练栈；不是只替换奖励的因果消融。',
              '只有一个训练种子，三个推理种子不代表独立重复训练；历史测试病例也不属于全新外部验证。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, root in [('calibrated', source), ('ranking', out)]:
        h = read(root / 'agent/history.json')
        axes[0].plot([r['iteration'] for r in h],
                     [r['validation']['macro']['10']['error'] for r in h], label=label)
        initial_error = read(root / 'rl/agent/imitation_evaluation.json')['macro']['10']['error']
        h = [r for r in read(root / 'rl/agent/history.json') if 'validation' in r]
        axes[1].plot([0] + [r['iteration'] for r in h],
                     [initial_error] + [r['validation']['macro']['10']['error'] for r in h], label=label)
    for ax, title in zip(axes, ['Imitation validation error', 'Diffusion RL validation error']):
        ax.set_title(title); ax.set_xlabel('Updates / iterations'); ax.set_ylabel('E (lower is better)')
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/region_ranking_agent.json')
    p.add_argument('--stages', nargs='+', choices=['configure', 'imitation', 'rl', 'evaluate', 'report'],
                   default=['configure', 'imitation', 'rl', 'evaluate', 'report'])
    args = p.parse_args()
    experiment = read(resolve(args.config))
    source, out = resolve(experiment['source_run']), resolve(experiment['out'])
    if source.resolve() == out.resolve():
        raise ValueError('Use a separate output directory for this ablation')
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'; logs.mkdir(exist_ok=True)
    teacher = source / 'teacher/ranking/best.pt'
    manifest = source / 'manifest.json'
    configs = [out / 'teacher_agent_config.json', out / 'diffusion_rl_config.json']
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
        env[key] = str(experiment.get('math_threads', 1))

    def execute(name, command):
        print(json.dumps(dict(stage=name, command=command)), flush=True)
        with (logs / (name + '.log')).open('w') as f:
            subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(name + ' completed', flush=True)

    for stage in args.stages:
        if stage != 'configure':
            locked = read(out / 'experiment.json')
            if (locked['teacher_sha256'] != digest(teacher) or
                    locked['manifest_sha256'] != digest(manifest) or
                    resolve(locked['source_run']) != source):
                raise ValueError('Source teacher or manifest changed since configuration')
        if stage == 'configure':
            if any(c.exists() for c in configs):
                raise FileExistsError('Configuration already exists; omit configure to continue')
            for i, original in enumerate(('teacher_agent_config.json', 'diffusion_rl_config.json')):
                cfg = copy.deepcopy(read(source / original))
                run = out if i == 0 else out / 'rl'
                cfg['paths'].update(run_dir=str(run), packaged_teacher=str(teacher),
                                    packaged_agent=str(run / 'agent/agent.pt'))
                if i:
                    cfg['paths']['imitation_agent'] = str(out / 'agent/agent.pt')
                cfg['inference']['out_dir'] = str(run / 'inference')
                configs[i].write_text(json.dumps(cfg, indent=2) + '\n')
                load_config(configs[i])
            (out / 'experiment.json').write_text(json.dumps(dict(experiment,
                teacher_sha256=digest(teacher), manifest_sha256=digest(manifest)), indent=2) + '\n')
        elif stage in ('imitation', 'rl'):
            i = int(stage == 'rl')
            run = out if i == 0 else out / 'rl'
            if (run / 'agent/latest.pt').exists():
                raise FileExistsError('Training already exists: ' + str(run))
            execute(stage, [sys.executable, '-m', 'occlusion_rl_offline.cli', '--config', str(configs[i]),
                            'train-agent', '--manifest', str(manifest), '--teacher', str(teacher)])
        elif stage == 'evaluate':
            common = [sys.executable, 'scripts/evaluate_filtered_cohort.py', '--manifest', str(manifest), '--stage', 'agent']
            tasks = [('imitation_test', common + ['--config', str(configs[0])]),
                     ('rl_test', common + ['--config', str(configs[1])]),
                     ('stochastic_imitation_test', common + ['--config', str(configs[1]),
                        '--agent', str(out / 'rl/agent/imitation.pt'), '--out', str(out / 'rl/test/imitation')])]
            with ThreadPoolExecutor(max_workers=experiment.get('evaluation_workers', 3)) as pool:
                futures = [pool.submit(execute, *task) for task in tasks]
                for future in futures:
                    future.result()
        else:
            report(out, source)


if __name__ == '__main__':
    main()
