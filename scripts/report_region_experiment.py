"""Summarize locked region experiment results and compare existing regression runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from occlusion_rl_offline.config import resolve
from occlusion_rl_offline.region_segmentation import digest


def read(path):
    return json.loads(Path(path).read_text())


def policy_summary(path):
    d = read(path)
    bands = {}
    for band in ('zero', 'near', 'medium', 'wide'):
        rows = [c['checkpoints']['10'] for result in d['results'].values()
                for c in result['cases'] if c['band'] == band]
        bands[band] = {k: float(np.mean([r[k] for r in rows]))
                       for k in ('rotation_deg', 'translation_mm', 'error')}
    direction_correct = valid_count = positive_count = positive_wrong = 0
    scales = d['config']['agent'].get('error_scales', [8., 4.])
    for name in d['results']:
        trace_path = Path(path).parent / (name + '_trajectories.json')
        if not trace_path.exists():
            continue
        for case in read(trace_path)['cases']:
            for before, after in zip(case['trace'], case['trace'][1:]):
                eb = .5 * (np.asarray(before['rotation_deg']) / scales[0] + np.asarray(before['translation_mm']) / scales[1])
                ea = .5 * (np.asarray(after['rotation_deg']) / scales[0] + np.asarray(after['translation_mm']) / scales[1])
                gain = np.asarray(after['teacher_score']) - np.asarray(before['teacher_score'])
                valid = (np.abs(eb - ea) > 1e-8) & (np.abs(gain) > 1e-8)
                direction_correct += int(((gain * (eb - ea) > 0) & valid).sum())
                valid_count += int(valid.sum())
                positive = valid & (gain > 0)
                positive_count += int(positive.sum())
                positive_wrong += int((positive & (ea > eb)).sum())
    return dict(mean=d['sampling_mean'], std=d['sampling_std'], bands=bands,
                on_policy_step_ranking_accuracy=direction_correct / valid_count if valid_count else None,
                on_policy_error_worse_given_score_up=positive_wrong / positive_count if positive_count else None,
                selected_iteration=d.get('selected_iteration'),
                actions=sum(r['trajectory_audit']['executed_actions'] for r in d['results'].values()),
                negative_actions=sum(r['trajectory_audit']['score_decreasing_actions'] for r in d['results'].values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/region_point_transformer')
    a = p.parse_args(); out = resolve(a.out)
    summary = dict(segmentation=read(out / 'segmentation/test.json')['summary'],
                   segmentation_checkpoint_validation=read(out / 'segmentation/history.json'),
                   teacher=read(out / 'test/teacher/comparison.json')['results'],
                   policies={name: policy_summary(out / path) for name, path in [
                       ('imitation', 'test/agent/comparison.json'),
                       ('stochastic_imitation', 'rl/test/imitation/comparison.json'),
                       ('rl', 'rl/test/agent/comparison.json')]})
    hist = summary.pop('segmentation_checkpoint_validation')
    best = max(hist, key=lambda r: r['validation']['macro']['iou'])
    summary['segmentation_training'] = dict(selected_epoch=best['epoch'], validation=best['validation'],
                                           first_loss=hist[0]['loss'], final_loss=hist[-1]['loss'])
    old_teacher = resolve('runs/filtered8_surface_diffusion/test/teacher/comparison.json')
    summary['previous_teacher'] = read(old_teacher)['results'] if old_teacher.exists() else None
    for key, path in [('previous_imitation', 'runs/filtered8_surface_diffusion/test/agent/comparison.json'),
                      ('previous_rl', 'runs/filtered8_diffusion_rl/test/agent/comparison.json')]:
        summary[key] = policy_summary(resolve(path)) if resolve(path).exists() else None
    # Verify that differences are not caused by altered test starts.
    historical_verified = True
    for seed in (91001, 91002, 91003):
        fresh = read(out / ('rl/test/agent/policy_%d_trajectories.json' % seed))
        baseline = read(out / ('rl/test/imitation/policy_%d_trajectories.json' % seed))
        historical = resolve('runs/filtered8_diffusion_rl/test/agent/policy_%d_trajectories.json' % seed)
        old = read(historical) if historical.exists() else baseline
        historical_verified &= historical.exists()
        for x, y, z in zip(fresh['cases'], old['cases'], baseline['cases']):
            assert (x['case'], x['band']) == (y['case'], y['band']) == (z['case'], z['band'])
            for key in ('initial_rotations', 'initial_translations'):
                np.testing.assert_array_equal(x[key], y[key])
                np.testing.assert_array_equal(x[key], z[key])
    summary['paired_starts_verified'] = True
    summary['historical_starts_verified'] = historical_verified
    config = read(out / 'teacher_agent_config.json')
    region = config['model']['segmentation_region']
    assert digest(resolve(region['checkpoint'])) == region['sha256']
    teacher_hash = digest(resolve(config['paths']['packaged_teacher']))
    initial = torch.load(out / 'rl/agent/imitation.pt', map_location='cpu', weights_only=True)
    selected = torch.load(out / 'rl/agent/agent.pt', map_location='cpu', weights_only=True)
    assert initial['teacher_sha256'] == selected['teacher_sha256'] == teacher_hash
    frozen = [key for key in initial['model'] if key.startswith(('reference.', 'actor.encoder.'))]
    assert all(torch.equal(initial['model'][key], selected['model'][key]) for key in frozen)
    changed = [key for key in initial['model'] if key.startswith('actor.') and
               not torch.equal(initial['model'][key], selected['model'][key])]
    assert changed, 'Selected RL actor did not change from imitation'
    rl_history = read(out / 'rl/agent/history.json')
    summary['training_audit'] = dict(segmentation_sha256=region['sha256'], teacher_sha256=teacher_hash,
                                     frozen_tensors_unchanged=len(frozen), changed_actor_tensors=len(changed),
                                     rl_iterations=len(rl_history), selected_rl_iteration=selected['iteration'],
                                     rl_score_decreasing_actions=sum(r.get('score_decreasing_actions', 0) for r in rl_history))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['## 实际结果', '', '分割验证最优 epoch %d，IoU %.4f。训练 loss %.4f → %.4f。' %
             (best['epoch'], best['validation']['macro']['iou'], hist[0]['loss'], hist[-1]['loss']), '',
             '| 分割测试位姿 | 上颌 IoU | 下颌 IoU | 上颌 Dice | 下颌 Dice |',
             '|---|---:|---:|---:|---:|']
    for band, d in summary['segmentation'].items():
        lines.append('| %s | %.4f | %.4f | %.4f | %.4f |' %
                     (band, d['upper']['iou'], d['lower']['iou'], d['upper']['dice'], d['lower']['dice']))
    lines += ['', '| Teacher | MAE | 局部排序准确率 | 近区局部排序准确率 |', '|---|---:|---:|---:|']
    for name, m in [('原流程 calibration', (summary['previous_teacher'] or {}).get('calibration')),
                    ('区域流程 ranking', summary['teacher']['ranking']),
                    ('区域流程 calibration', summary['teacher']['calibration'])]:
        if m is None:
            continue
        lines.append('| %s | %.4f | %.2f%% | %.2f%% |' % (name, m['mae'], 100*m['local_accuracy'], 100*m['near_local_accuracy']))
    lines += ['', '| Agent（3 个采样种子均值） | 角度 ° | 平移 mm | 综合误差 E |', '|---|---:|---:|---:|']
    for name, d in [('原流程 diffusion 模仿', summary['previous_imitation']),
                    ('原流程 diffusion RL', summary['previous_rl']),
                    ('区域流程 diffusion 模仿', summary['policies']['imitation']),
                    ('区域流程随机推理模仿基线', summary['policies']['stochastic_imitation']),
                    ('区域流程 diffusion RL', summary['policies']['rl'])]:
        if d is None:
            continue
        m = d['mean']
        lines.append('| %s | %.4f | %.4f | %.5f |' % (name, m['rotation_deg'], m['translation_mm'], m['error']))
    lines += ['', '综合误差 E = 0.5 × (角度/8° + 平移/4 mm)，主表不含零位姿；各档、零位姿漂移和标准差见 `summary.json`。相同起点已逐项验证，RL 与随机推理模仿基线使用相同采样种子。分割阈值和所有 checkpoint 均不按测试结果选择。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot([r['epoch'] for r in hist], [r['loss'] for r in hist]); axes[0, 0].set_title('Segmentation training loss')
    axes[0, 1].plot([r['epoch'] for r in hist], [r['validation']['macro']['iou'] for r in hist]); axes[0, 1].set_title('Segmentation validation IoU')
    for stage in ('ranking', 'calibration'):
        h = read(out / ('teacher/%s/history.json' % stage))
        axes[1, 0].plot([r['epoch'] for r in h], [r['validation']['local_accuracy'] for r in h], label=stage)
    axes[1, 0].legend(); axes[1, 0].set_title('Teacher validation local ranking')
    h = read(out / 'agent/history.json')
    axes[1, 1].plot([r['iteration'] for r in h], [r['validation']['macro']['10']['error'] for r in h])
    axes[1, 1].set_title('Imitation validation pose error')
    for ax in axes.flat: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)
    # Spatial sanity check: red false positives, blue false negatives, green TP.
    fig = plt.figure(figsize=(11, 8))
    for index, (case, jaw) in enumerate([('010', 0), ('010', 1), ('011', 0), ('011', 1)]):
        with np.load(out / ('segmentation/predictions/%s_%d.npz' % (case, jaw))) as f:
            xyz, p, y = f['xyz'], f['probability'] >= .5, f['label'] > .5
        ids = np.random.RandomState(42).choice(len(xyz), min(6000, len(xyz)), replace=False)
        colors = np.full(len(xyz), '#bbbbbb', dtype='<U7')
        colors[p & y] = '#369c56'; colors[p & ~y] = '#dc4a43'; colors[~p & y] = '#357cc5'
        ax = fig.add_subplot(2, 2, index + 1, projection='3d')
        ax.scatter(*xyz[ids].T, c=colors[ids], s=1.5, depthshade=False)
        ax.set_title(case + (' upper' if jaw == 0 else ' lower'))
        ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1e-6))
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
    fig.suptitle('Reference pose: green TP / red FP / blue FN / gray TN')
    fig.tight_layout(); fig.savefig(out / 'segmentation_preview.png', dpi=150); plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
