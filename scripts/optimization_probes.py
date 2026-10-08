"""Sampling sensitivity and teacher-only search diagnostics (not PPO actions)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.config import load_config,resolve
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.data import load_case,features_at_pose,sample_sdf,candidate_indices
from occlusion_rl_offline.rl import PoseTask,load_teacher,teacher_score,continuous_error
from occlusion_rl_offline.evaluation import start_bank


def sampling(paths,cfg,teacher):
    rows=[]
    for path in paths:
        case=load_case(path);pivot=case['xyz'].mean(0);rng=np.random.RandomState(671+int(path.parent.name))
        for anchor in range(3):
            pose=rng.uniform(-1,1,6)*np.array([0 if anchor==0 else (1.5 if anchor==1 else 8)]*3+[0 if anchor==0 else (.5 if anchor==1 else 4)]*3)
            for variant in ('quota','no_quota','fixed_initial_ids'):
                options=dict(cfg['model'])
                if variant=='no_quota':options['contact_fps_fraction']=0.
                base,ids=features_at_pose(case,Rotation.from_rotvec(np.deg2rad(pose[:3])).as_matrix(),pose[3:],64,options,True)
                zero=teacher_score(teacher,base[None],'cpu')[0]
                for axis in range(6):
                    for sign in (-1,1):
                        p=pose.copy();p[axis]+=sign*(.05 if axis<3 else .01)
                        r=Rotation.from_rotvec(np.deg2rad(p[:3])).as_matrix()
                        if variant=='fixed_initial_ids':
                            x=(case['xyz'][ids]-pivot)@r.T+pivot+p[3:];n=case['normal'][ids]@r.T
                            f=np.column_stack((x,n,case['curvature'][ids],sample_sdf(case,x))).astype(np.float32);selected=ids
                        else:f,selected=features_at_pose(case,r,p[3:],64,options,True)
                        score=teacher_score(teacher,f[None],'cpu')[0]
                        rows.append({'case':path.parent.name,'anchor':anchor,'variant':variant,
                                     'replaced':1-len(np.intersect1d(ids,selected))/64,
                                     'score_delta_abs':float(abs(score-zero)),
                                     'contact_fraction':float((np.abs(f[:,7])<.6).mean())})
    summary={}
    for variant in ('quota','no_quota','fixed_initial_ids'):
        rr=[r for r in rows if r['variant']==variant]
        summary[variant]={'replaced_mean':float(np.mean([r['replaced'] for r in rr])),
                          'delta_p95':float(np.quantile([r['score_delta_abs'] for r in rr],.95)),
                          'contact_mean':float(np.mean([r['contact_fraction'] for r in rr]))}
    return {'summary':summary,'rows':rows,'note':'Sampler changes are sensitivity probes with fixed weights, not retrained comparisons. Fixed IDs originate from the actual perturbed starting pose.'}


def search(paths,cfg,teacher,episodes):
    rows=[];a=cfg['agent']
    for path in paths:
        task=PoseTask(path,64,1,8,4,'cpu',fps_options=cfg['model'])
        for bi in (1,2,3):
            task.begin_episode()
            r,t=start_bank(path.parent.name,bi,episodes,426101);x=task.points_at(r,t)
            scores=teacher_score(teacher,x,'cpu');ai,di=task.error(r,t);si=scores.copy()
            for step in range(10):
                scale=1 if step<5 else .25
                actions=np.concatenate([np.zeros((1,6)),np.eye(6),-np.eye(6)])*scale
                task.rotation_step_deg=1.;task.translation_step_mm=.2
                rr=np.repeat(r,13,axis=0);tt=np.repeat(t,13,axis=0)
                if task.episode_ids is not None:task.episode_ids=np.repeat(task.episode_ids,13,axis=0)
                nr,nt,nx,_=task.move(rr,tt,np.tile(actions,(episodes,1)))
                ns=teacher_score(teacher,nx,'cpu').reshape(episodes,13)
                pick=np.arange(episodes)*13+ns.argmax(1)
                if task.episode_ids is not None:task.episode_ids=task.episode_ids[pick]
                r,t,x,scores=nr[pick],nt[pick],nx[pick],ns.max(1)
            af,df=task.error(r,t)
            rows.append({'case':path.parent.name,'band':bi,'initial_rotation':float(ai.mean()),'final_rotation':float(af.mean()),
                         'initial_translation':float(di.mean()),'final_translation':float(df.mean()),
                         'initial_error':float(continuous_error(ai,di).mean()),'final_error':float(continuous_error(af,df).mean()),
                         'teacher_gain':float((scores-si).mean())})
            print('search',rows[-1],flush=True)
    return {'rows':rows,'teacher_evaluations_per_trajectory':131,
            'note':'Diagnostic teacher-search baseline may choose zero; PPO and deployed agent still execute every proposed action without rejection.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/optimization_20261001.json')
    p.add_argument('--teacher',required=True);p.add_argument('--mode',choices=['sampling','search'],required=True)
    p.add_argument('--episodes',type=int,default=4);p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);cfg=load_config(args.config)
    teacher=load_teacher(resolve(args.teacher),'cpu',64,**cfg['model'])
    paths=read_manifest(resolve('runs/data11_candidate4096_fps64_more/prepared/manifest.json'))['val']
    result=sampling(paths,cfg,teacher) if args.mode=='sampling' else search(paths,cfg,teacher,args.episodes)
    out=resolve(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result.get('summary',{})),flush=True)
if __name__=='__main__':main()
