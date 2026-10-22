"""Independent semantic/mesh/teacher/diffusion pipeline; paths relative to repo root."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ..config import resolve, load_config
from ..train_regions import save_json
from ..cohort import read_manifest
from ..runtime import resolve_device

STAGES=['prepare','segment','segment-test','regions','configure','teacher','imitate','rl','test','end-to-end','visualize']


def configuration(path, smoke=False):
    cfg=json.loads(resolve(path).read_text())
    cfg.setdefault('horizon',16)
    if isinstance(cfg['horizon'],bool) or not isinstance(cfg['horizon'],int) or cfg['horizon']<1:
        raise ValueError('horizon must be a positive integer')
    if len(cfg['pose_bands']) != 5 or not np.isclose(sum(b['probability'] for b in cfg['pose_bands']),1):
        raise ValueError('Exactly five pose bands with total probability one required')
    if not 4 <= cfg['simplification']['max_vertices'] < 4096:raise ValueError('Simplified mesh must have fewer than 4096 vertices')
    manifest=json.loads(resolve(cfg['source_manifest']).read_text())
    if [x['id'] for x in manifest['splits']['test']] != ['011'] or any(x['id']=='011' for key in ('train','val') for x in manifest['splits'][key]):
        raise ValueError('011 must be held out exclusively for testing')
    from .rewards import validate
    validate(cfg['rl']['trajectory_curriculum'])
    if smoke:
        cfg['out']+='_smoke_h%d' % cfg['horizon']
        cfg['segmentation'].update(epochs=48,poses_per_jaw=2,validation_rotations=1,test_rotations=2)
        cfg['segmentation']['sampling']['max_points']=512
        cfg['segmentation']['architecture'].update(width=24,layers=2,k=12)
        cfg['teacher'].update(ranking_epochs=4,pairs_per_case=16)
        cfg['imitation'].update(training_steps=32,episodes_per_case=2)
        cfg['rl'].update(iterations=6,parallel_envs=2)
        cfg['smoke']=True
    return cfg


def configure(cfg):
    root=cfg['out'];out=resolve(root)
    base=json.loads(resolve(cfg['base_training_config']).read_text())
    base['model'].pop('segmentation_region',None)
    base['model'].update(candidate_source='all_mesh',candidate_points=cfg['simplification']['max_vertices'])
    base['teacher'].update(fps_points=cfg['simplification']['max_vertices'],ranking_epochs=cfg['teacher']['ranking_epochs'],calibration_epochs=0)
    base['teacher']['aligned_pair']['batch_size']=2
    base['agent']['diffusion']['batch_size']=4
    base['agent']['diffusion']['collection_batch']=2
    pairs=base['teacher']['aligned_pair'];pairs['pose_bands']=cfg['pose_bands']
    for k in ('train_pairs_per_case','validation_pairs_per_case','test_pairs_per_case'):pairs[k]=cfg['teacher']['pairs_per_case']
    base['agent'].update(fps_points=cfg['simplification']['max_vertices'],training_pose_bands=cfg['pose_bands'],torch_threads=cfg['torch_threads'],horizon=cfg.get('horizon',16))
    base['agent']['diffusion'].update(**cfg['imitation'])
    base['runtime']=cfg['runtime']
    base['paths'].update(run_dir=root,packaged_teacher=root+'/teacher/ranking/best.pt',
                         packaged_agent=root+'/imitation/agent.pt',case=root+'/regions/cases/010/case.npz')
    base['inference'].update(out_dir=root+'/inference',steps=cfg.get('horizon',16))
    if cfg.get('smoke'):
        base['agent'].update(validation_episodes=1,test_episodes=1)
        base['agent']['diffusion'].update(validate_every=8,batch_size=8,collection_batch=2,hidden_width=64,hidden_layers=2,
                                        inference_steps=4,prediction_horizon=2)
    save_json(out/'imitation_config.json',base)
    rl=copy.deepcopy(base);rl['agent']['trainer']='diffusion_rl'
    rl['agent']['diffusion_rl']=json.loads(resolve(cfg['base_rl_config']).read_text())['agent']['diffusion_rl']
    rl['agent']['diffusion_rl'].update(cfg['rl'])
    if cfg.get('smoke'):rl['agent']['diffusion_rl'].update(tail_steps=2,validate_every=1,minibatch=6)
    rl['agent']['diffusion_rl']['minibatch']=2
    rl['paths'].update(run_dir=root+'/rl',imitation_agent=root+'/imitation/agent.pt',packaged_agent=root+'/rl/agent.pt')
    save_json(out/'rl_config.json',rl)
    # Parse with shared validators before any expensive stage.
    load_config(out/'imitation_config.json');load_config(out/'rl_config.json')


def test(cfg):
    from .. import aligned_teacher
    from ..evaluation import evaluate_continuous,load_agent
    from ..rl import load_teacher
    out=resolve(cfg['out']);c=load_config(out/'rl_config.json');split=read_manifest(out/'regions/manifest.json');device=resolve_device(c)
    bank=aligned_teacher.build_bank(split['test'],c,c['teacher']['aligned_pair']['test_pairs_per_case'],94101,out/'teacher/banks/test.npz')
    result={}
    for stage in ('ranking',):
        teacher=load_teacher(out/('teacher/'+stage+'/best.pt'),device,c['teacher']['fps_points'],**c['model'])
        result[stage]=aligned_teacher.metrics(aligned_teacher.predict(teacher,bank,device),bank)
    save_json(out/'teacher/test.json',result)
    teacher=load_teacher(resolve(c['paths']['packaged_teacher']),device,c['teacher']['fps_points'],**c['model'])
    result={}
    for name,checkpoint in [('imitation',out/'rl/imitation.pt'),('rl',out/'rl/agent.pt')]:
        agent=load_agent(checkpoint,c,device,resolve(c['paths']['packaged_teacher']))
        result[name]=evaluate_continuous(agent,teacher,split['test'],c,c['agent']['test_episodes'],94102,
                                        (c['agent']['horizon'],),details=True)
    save_json(out/'agent_test.json',result)


def run_stage(stage,cfg):
    out=resolve(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['torch_threads'])
    snapshot=out/'pipeline_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:raise ValueError('Output configuration differs; use new out directory')
    save_json(snapshot,cfg)
    marker=out/('stage_'+stage+'.json')
    if marker.exists():raise ValueError(stage+' already completed; use a new run or select remaining stages')
    print('STAGE',stage,flush=True)
    if stage=='prepare':
        from ..train_regions import prepare
        prepare(cfg)
    elif stage in ('segment','segment-test'):
        from .segmentation import train,evaluate
        (train if stage=='segment' else evaluate)(cfg)
    elif stage=='regions':
        from .regions import prepare
        prepare(cfg)
    elif stage=='configure':configure(cfg)
    elif stage in ('teacher','imitate','rl'):
        c=load_config(out/('rl_config.json' if stage=='rl' else 'imitation_config.json'))
        manifest=out/'regions/manifest.json';teacher=resolve(c['paths']['packaged_teacher'])
        if stage=='teacher':
            from ..aligned_teacher import train
            train(c,manifest,out/'teacher',stages=('ranking',))
        elif stage=='imitate':
            from ..train_diffusion import run
            run(c,manifest,teacher,out/'imitation')
        else:
            from ..train_diffusion_rl import run
            run(c,manifest,teacher,out/'imitation/agent.pt',out/'rl')
    elif stage=='test':test(cfg)
    elif stage=='end-to-end':
        from .end_to_end import evaluate
        evaluate(cfg)
    elif stage=='visualize':
        from .viewer import export
        export(cfg)
        if (out/'rl_config.json').exists() and all((out/('teacher/'+name+'/best.pt')).exists() for name in ('ranking',)):
            from .recovery_viewer import export as export_recovery
            kind='both' if all((out/('rl/'+name)).exists() for name in ('imitation.pt','agent.pt')) else 'teacher'
            export_recovery(cfg,kind=kind)
    save_json(marker,dict(completed=True,stage=stage,smoke=bool(cfg.get('smoke'))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/semantic_pipeline.json');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES)
    p.add_argument('--device',help='auto, cpu, cuda or cuda:N')
    args=p.parse_args();cfg=configuration(args.config,args.smoke)
    if args.device:cfg['runtime']['device']=args.device
    for stage in args.stages:run_stage(stage,cfg)

if __name__=='__main__':main()
