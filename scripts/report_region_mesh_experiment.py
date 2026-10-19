"""Compare the original-mesh 10240/256 experiment with the 4096/128 baseline."""
import argparse
import json

import numpy as np
import torch

from occlusion_rl_offline.config import resolve
from occlusion_rl_offline.region_segmentation import digest
from report_region_experiment import read, policy_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/region_mesh10240_fps256')
    p.add_argument('--baseline',default='runs/region_ranking_agent')
    args=p.parse_args();out=resolve(args.out);baseline=resolve(args.baseline)
    paths=dict(imitation='test/agent/comparison.json',stochastic_imitation='rl/test/imitation/comparison.json',
               rl='rl/test/agent/comparison.json')
    summary=dict(mesh_audit=read(out/'mesh_audit.json'),teacher=read(out/'test/teacher/comparison.json')['results'],
                 policies={k:policy_summary(out/v) for k,v in paths.items()},
                 baseline={k:policy_summary(baseline/v) for k,v in paths.items()})
    if (out/'sampling_stability.json').exists():
        summary['sampling_stability']=read(out/'sampling_stability.json')
    for name,root in [('policies',out),('baseline',baseline)]:
        for policy,path in paths.items():
            rows=[c for r in read(root/path)['results'].values() for c in r['cases'] if c['band']!='zero']
            summary[name][policy]['by_case']={case:{k:float(np.mean([r['checkpoints']['10'][k] for r in rows if r['case']==case]))
                for k in ('rotation_deg','translation_mm','error')} for case in sorted({r['case'] for r in rows})}
    for seed in (91001,91002,91003):
        original=read(baseline/('rl/test/agent/policy_%d_trajectories.json'%seed))['cases']
        for relative in paths.values():
            current=read((out/relative).parent/('policy_%d_trajectories.json'%seed))['cases']
            assert len(current)==len(original)
            for a,b in zip(current,original):
                assert (a['case'],a['band'])==(b['case'],b['band'])
                for key in ('initial_rotations','initial_translations'):
                    np.testing.assert_array_equal(a[key],b[key])
    cfg=read(out/'teacher_agent_config.json')
    with np.load(out/'agent/expert_chunks.npz') as new, np.load(baseline/'agent/expert_chunks.npz') as old:
        for key in ('contexts','actions'):
            np.testing.assert_array_equal(new[key],old[key])
    initial=torch.load(out/'rl/agent/imitation.pt',map_location='cpu',weights_only=True)
    selected=torch.load(out/'rl/agent/agent.pt',map_location='cpu',weights_only=True)
    frozen=[k for k in initial['model'] if k.startswith(('reference.','actor.encoder.'))]
    assert all(torch.equal(initial['model'][k],selected['model'][k]) for k in frozen)
    assert initial['teacher_sha256']==selected['teacher_sha256']==digest(resolve(cfg['paths']['packaged_teacher']))
    segment=cfg['model']['segmentation_region'];assert digest(resolve(segment['checkpoint']))==segment['sha256']
    h=read(out/'rl/agent/history.json')
    summary['audit']=dict(paired_starts_verified=True,identical_expert_actions_and_contexts=True,
        frozen_tensors=len(frozen),selected_rl_iteration=selected['iteration'],
        training_actions=h[-1]['interactions'],score_decreasing_training_actions=sum(r['score_decreasing_actions'] for r in h),
        segmentation_sha256=segment['sha256'],teacher_sha256=initial['teacher_sha256'])
    summary['training']={}
    summary['training']['rl']=dict(
        initial_validation=read(out/'rl/agent/imitation_evaluation.json')['macro'],
        selected_validation=read(out/'rl/agent/evaluation.json')['macro'],
        best_including_imitation_iteration=torch.load(out/'rl/agent/best_including_imitation.pt',
            map_location='cpu',weights_only=True)['iteration'])
    for stage,path in [('ranking','teacher/ranking'),('calibration','teacher/calibration'),('imitation','agent')]:
        h=read(out/path/'history.json');loss='noise_mse' if stage=='imitation' else 'train_loss'
        checkpoint=torch.load(out/path/('agent.pt' if stage=='imitation' else 'best.pt'),map_location='cpu',weights_only=True)
        summary['training'][stage]=dict(first_loss=h[0][loss],last_loss=h[-1][loss],
            selected=checkpoint['iteration' if stage=='imitation' else 'epoch'])
    new=summary['policies']['rl']['mean']
    summary['relative_change']={name:{k:new[k]/row[k]-1 for k in ('rotation_deg','translation_mm','error')}
        for name,row in [('vs_baseline',summary['baseline']['rl']['mean']),('vs_imitation',summary['policies']['stochastic_imitation']['mean'])]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 原网格区域 10240 / 256 实验结果','','| 方案 | 策略 | 旋转 ° | 平移 mm | E |','|---|---|---:|---:|---:|']
    for group in ('baseline','policies'):
        for policy,r in summary[group].items():
            m=r['mean'];lines.append('| %s | %s | %.4f | %.4f | %.5f |'%(group,policy,m['rotation_deg'],m['translation_mm'],m['error']))
    lines+=['','baseline 为此前 4096 候选 / 128 点的未校准 teacher；policies 为先恢复原网格区域，再 10240 候选 / 256 点。',
        '两组均为单个训练种子，测试 010/011，3 个推理种子，相同起点、完整 10 步。E=0.5×(角度/8°+平移/4mm)，主表不含零位姿。',
        '同时改变区域恢复规则和采样数量，不能把差异单独归因于点数。训练/验证/测试划分保持不变。',
        '所有病例的源三角面顺序已经与原始 STL 核对。完整顶点/面片映射在 prepared/cases/*/*_region.npz。',
        '分档、病例差异、零位姿漂移、逐步奖励方向及训练记录见 summary.json。','']
    (out/'report.md').write_text('\n'.join(lines));print('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for stage in ('ranking','calibration'):
        h=read(out/('teacher/'+stage+'/history.json'))
        axes[0].plot([r['epoch'] for r in h],[r['validation']['local_accuracy'] for r in h],label=stage)
    for label,root in [('10240/256',out),('4096/128',baseline)]:
        h=read(root/'agent/history.json')
        axes[1].plot([r['iteration'] for r in h],[r['validation']['macro']['10']['error'] for r in h],label=label)
        h=[r for r in read(root/'rl/agent/history.json') if 'validation' in r]
        initial_error=read(root/'rl/agent/imitation_evaluation.json')['macro']['10']['error']
        axes[2].plot([0]+[r['iteration'] for r in h],[initial_error]+[r['validation']['macro']['10']['error'] for r in h],label=label)
    for ax,title in zip(axes,['Teacher validation local accuracy','Imitation validation error','RL validation error']):
        ax.set_title(title);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
