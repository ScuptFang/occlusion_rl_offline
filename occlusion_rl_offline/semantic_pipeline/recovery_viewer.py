"""Audited offline teacher poses and paired policy rollouts for the semantic pipeline."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..config import resolve,load_config
from ..cohort import read_manifest
from ..data import load_case,features_at_pose
from ..evaluation import load_agent
from ..inference import _matrix
from ..region_segmentation import digest
from ..rl import PoseTask,load_teacher,teacher_score,action_steps,step_context,policy_action
from ..train_regions import save_json


def display_mesh(case, original):
    def preview(x):
        return np.round(x[np.linspace(0,len(x)-1,min(8000,len(x)),dtype=int)],4).tolist()
    return dict(upper=case['xyz'].tolist(),lower=case['lower_xyz'].tolist(),
                upper_faces=case['upper_faces'].tolist(),lower_faces=case['lower_faces'].tolist(),
                upper_context=preview(original['xyz']),lower_context=preview(original['lower_xyz']))


def pose_frame(case,r,t,points,ids,scores,action=None):
    pivot=case['xyz'].mean(0)
    return dict(matrix=_matrix(r,t,pivot).tolist(),rotation_deg=float(np.rad2deg(Rotation.from_matrix(r).magnitude())),
        translation_mm=float(np.linalg.norm(t)),rotvec_deg=np.rad2deg(Rotation.from_matrix(r).as_rotvec()).tolist(),
        translation=t.tolist(),scores={key:float(value) for key,value in scores.items()},
        selected_points=points[:,:3].tolist(),selected_ids=ids.tolist(),sdf=points[:,7].tolist(),action=action)


def draw_starts(case_id,band_index,band,count,seed=96101):
    rng=np.random.RandomState(seed+int(case_id)*1009+band_index*100003)
    values=rng.uniform(-1,1,(count,6))
    return Rotation.from_rotvec(np.deg2rad(values[:,:3]*band['rotation_deg'])).as_matrix(),values[:,3:]*band['translation_mm']


def teacher_series(case,cfg,models,device,points):
    series=[]
    axes=['Rx','Ry','Rz','Tx','Ty','Tz']
    bounds=[max(b['rotation_deg'] for b in cfg['pose_bands'])]*3+[max(b['translation_mm'] for b in cfg['pose_bands'])]*3
    poses=[]
    for axis,name in enumerate(axes):
        vectors=np.zeros((points,6));vectors[:,axis]=np.linspace(-bounds[axis],bounds[axis],points)
        poses.append((name+' 单轴扫描',vectors))
    for bi,band in enumerate(cfg['pose_bands']):
        r,t=draw_starts(case['_id'],bi,band,1 if bi==0 else 8)
        poses.append((band['name']+' 五档预设',np.column_stack((np.rad2deg(Rotation.from_matrix(r).as_rotvec()),t))))
    for name,vectors in poses:
        frames=[]
        for pose in vectors:
            r=Rotation.from_rotvec(np.deg2rad(pose[:3])).as_matrix();t=pose[3:]
            x,ids=features_at_pose(case,r,t,cfg['_training']['teacher']['fps_points'],cfg['_training']['model'],True)
            scores={key:teacher_score(model,x[None],device)[0] for key,model in models.items()}
            frame=pose_frame(case,r,t,x,ids,scores)
            scales=cfg['_training']['teacher']['aligned_pair']['quality_scales']
            frame['quality']=-frame['rotation_deg']/scales[0]-frame['translation_mm']/scales[1]
            frames.append(frame)
        series.append(dict(name=name,frames=frames))
    return series


def policy_rollouts(case_id,path,cfg,teacher,models,device,episodes):
    a=cfg['agent'];rows=[]
    for bi,band in enumerate(a['training_pose_bands']):
        n=1 if bi==0 else episodes
        initial_r,initial_t=draw_starts(case_id,bi,band,n)
        for policy_name,policy in models.items():
            task=PoseTask(path,a['fps_points'],96101,8,4,device,a['action_rotation_step_deg'],a['action_translation_step_mm'],cfg['model'])
            r,t=initial_r.copy(),initial_t.copy();x=task.points_at(r,t);score=teacher_score(teacher,x,device)
            generator=torch.Generator(device=device).manual_seed(97101+int(case_id)*1009+bi*100003)
            frames=[[] for _ in range(n)]
            previous_action=None
            for step in range(a['horizon']+1):
                for i in range(n):
                    f=pose_frame(task.case,r[i],t[i],x[i],task.last_ids[i],{'teacher':score[i]},
                                 previous_action[i].tolist() if previous_action is not None else None)
                    if previous_action is not None:
                        rs,ts=action_steps(a,step-1)
                        f.update(action_delta_deg=(previous_action[i,:3]*rs).tolist(),action_delta_mm=(previous_action[i,3:]*ts).tolist())
                    frames[i].append(f)
                if step==a['horizon']:break
                task.rotation_step_deg,task.translation_step_mm=action_steps(a,step)
                with torch.no_grad():
                    action,plan=policy_action(policy,task.observation(x),step_context(policy,n,step,a['horizon'],device),generator)
                previous_action=action.cpu().numpy()
                nr,nt,nx,ns,valid,accepted=task.move_scored(r,t,x,score,previous_action,teacher,False)
                if not valid.all() or not accepted.all():raise ValueError('Viewer rollout contains invalid/rejected action')
                # Independent action composition check, not just recorded UI matrices.
                expected=Rotation.from_rotvec(np.deg2rad(previous_action[:,:3]*task.rotation_step_deg)).as_matrix()@r
                np.testing.assert_allclose(nr,expected,atol=1e-7)
                np.testing.assert_allclose(nt,t+previous_action[:,3:]*task.translation_step_mm,atol=1e-7)
                r,t,x,score=nr,nt,nx,ns
            for i,trace in enumerate(frames):
                rows.append(dict(case=case_id,geometry=case_id,mode='固定区域',policy=policy_name,band=band['name'],episode=i,
                    rotation_limit=band['rotation_deg'],translation_limit=band['translation_mm'],frames=trace))
    return rows


def end_to_end_traces(root,cfg,teacher,device,raw_paths,policy):
    """Replay saved full-STL trajectories; verify scores against current checkpoint."""
    path=root/'end_to_end/summary.json'
    if not path.exists():return [],{},[]
    summary=json.loads(path.read_text());rows=[];geometry={};failures=[]
    for record in summary['rows']:
        if not record['success']:
            failures.append(dict(case=record['case'],band=record['band'],error=record['error']));continue
        folder=root/'end_to_end'/record['case']/record['band']
        result=json.loads((folder/'trajectory.json').read_text())
        if digest(resolve(result['teacher_checkpoint'])) != digest(resolve(cfg['paths']['packaged_teacher'])):
            raise ValueError('End-to-end teacher differs from current teacher')
        case=load_case(folder/'regions/case.npz');original=load_case(raw_paths[record['case']])
        # Exact endpoint IDs provide the reference mesh for audit, not network input.
        with np.load(folder/'regions/upper_mapping.npz') as f:ids=f['simplified_vertex_ids']
        source=original['xyz'][ids].astype(float);observed=case['xyz'].astype(float)
        u,_,vh=np.linalg.svd((source-source.mean(0)).T@(observed-observed.mean(0)))
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(vh.T@u.T)
        r=vh.T@correction@u.T;t=observed.mean(0)-r@source.mean(0)
        initial=np.eye(4);initial[:3,:3]=r;initial[:3,3]=t
        np.testing.assert_allclose(source@r.T+t,observed,atol=2e-5)
        transformed_original=dict(original,xyz=(original['xyz']@r.T+t).astype(np.float32))
        key=record['case']+'_'+record['band']+'_observed'
        geometry[key]=display_mesh(case,transformed_original)
        geometry[key]['ghost_upper']=source.tolist()
        frames=[]
        generator=torch.Generator(device=device).manual_seed(result['sampling_seed'])
        for item in result['trajectory']:
            matrix=np.asarray(item['matrix']);rr=matrix[:3,:3];pivot=case['xyz'].mean(0)
            tt=matrix[:3,3]-pivot+rr@pivot
            x,selected=features_at_pose(case,rr,tt,cfg['agent']['fps_points'],cfg['model'],True)
            score=float(teacher_score(teacher,x[None],device)[0])
            if not np.isclose(score,item['teacher_logit'],atol=1e-5,rtol=0):raise ValueError('Stale end-to-end teacher scores')
            frame=pose_frame(case,rr,tt,x,selected,{'teacher':score},item.get('action'))
            residual=matrix@initial;reference_pivot=original['xyz'].mean(0)
            frame['rotation_deg']=float(np.rad2deg(Rotation.from_matrix(residual[:3,:3]).magnitude()))
            frame['translation_mm']=float(np.linalg.norm(residual[:3,:3]@reference_pivot+residual[:3,3]-reference_pivot))
            if item.get('action') is not None:
                rs,ts=action_steps(cfg['agent'],item['step']-1)
                action=np.asarray(item['action']);frame.update(action_delta_deg=(action[:3]*rs).tolist(),action_delta_mm=(action[3:]*ts).tolist())
            if item['step'] < result['steps']:
                with torch.no_grad():
                    predicted,_=policy_action(policy,torch.from_numpy(x[None]).to(device),
                        step_context(policy,1,item['step'],result['steps'],device),generator)
                np.testing.assert_allclose(predicted[0].cpu().numpy(),result['trajectory'][item['step']+1]['action'],atol=1e-6,rtol=0,
                                           err_msg='Saved full-STL actions differ from current agent')
            frames.append(frame)
        np.testing.assert_allclose([frames[-1]['rotation_deg'],frames[-1]['translation_mm']],
                                   [record['rotation_deg'],record['translation_mm']],atol=2e-5)
        rows.append(dict(case=record['case'],geometry=key,mode='完整 STL',policy='RL',band=record['band'],episode=0,frames=frames))
    return rows,geometry,failures


def export(cfg,kind='both',episodes=3,sweep_points=21):
    if not 1<=episodes<=32 or sweep_points<3 or sweep_points%2!=1:
        raise ValueError('episodes must be 1..32; sweep-points must be odd and >=3')
    root=resolve(cfg['out']);directory=root/'viewer';directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg['torch_threads'])
    c=load_config(root/'rl_config.json')
    from ..runtime import resolve_device
    device=resolve_device(c)
    paths=read_manifest(root/'regions/manifest.json');raw=read_manifest(root/'manifest.json')
    raw_paths={p.parent.name:p for p in raw['test']}
    geometry={};cases={}
    for path in paths['test']:
        case=load_case(path);case['_id']=path.parent.name;cases[path.parent.name]=case
        geometry[path.parent.name]=display_mesh(case,load_case(raw_paths[path.parent.name]))
    if kind in ('teacher','both'):
        models={name:load_teacher(root/('teacher/'+name+'/best.pt'),device,c['teacher']['fps_points'],**c['model']) for name in ('ranking',)}
        tc=dict(cfg,_training=c)
        payload=dict(kind='teacher',smoke=bool(cfg.get('smoke')),cases=geometry,
            data={key:teacher_series(case,tc,models,device,sweep_points) for key,case in cases.items()},
            checkpoints={name:digest(root/('teacher/'+name+'/best.pt')) for name in models},fps_points=c['teacher']['fps_points'])
        write_page(directory,'teacher',payload)
    if kind in ('agent','both'):
        teacher_path=resolve(c['paths']['packaged_teacher']);teacher=load_teacher(teacher_path,device,c['teacher']['fps_points'],**c['model'])
        models={name:load_agent(path,c,device,teacher_path) for name,path in [('模仿基线',root/'rl/imitation.pt'),('RL',root/'rl/agent.pt')]}
        traces=[]
        for path in paths['test']:traces+=policy_rollouts(path.parent.name,path,c,teacher,models,device,episodes)
        full,extra,failures=end_to_end_traces(root,c,teacher,device,raw_paths,models['RL']);traces+=full
        audit=dict(trajectories=len(traces),actions=sum(len(t['frames'])-1 for t in traces),
            decreasing_actions=sum(b['scores']['teacher']<a['scores']['teacher'] for t in traces for a,b in zip(t['frames'],t['frames'][1:])),
            end_to_end_failures=failures,seed=96101,sampling_seed=97101,paired_starts_verified=True,
            end_to_end_actions_and_scores_replayed=True)
        for key in cases:
            first=[t for t in traces if t['case']==key and t['mode']=='固定区域' and t['policy']=='模仿基线']
            second=[t for t in traces if t['case']==key and t['mode']=='固定区域' and t['policy']=='RL']
            for a,b in zip(first,second):np.testing.assert_array_equal(a['frames'][0]['matrix'],b['frames'][0]['matrix'])
        payload=dict(kind='agent',smoke=bool(cfg.get('smoke')),cases={**geometry,**extra},trajectories=traces,audit=audit,
            teacher_stage='ranking',fps_points=c['agent']['fps_points'],horizon=c['agent']['horizon'],
            selected_iteration=torch.load(root/'rl/agent.pt',map_location='cpu',weights_only=True)['iteration'],
            reward_history=json.loads((root/'rl/history.json').read_text()),
            checkpoints={'teacher':digest(teacher_path),'imitation':digest(root/'rl/imitation.pt'),'rl':digest(root/'rl/agent.pt')})
        write_page(directory,'agent',payload);save_json(directory/'agent_audit.json',audit)
    assets=Path(__file__).parent/'assets'
    for name in ('recovery.js','recovery.css'):shutil.copyfile(assets/name,directory/name)
    # Add links to an existing semantic page without rerunning segmentation.
    index=directory/'index.html'
    if index.exists():
        html=index.read_text()
        if 'id="pipeline-links"' not in html:
            html=html.replace('<main>','<main><nav id="pipeline-links"><a href="teacher.html">Teacher 评分</a> · <a href="agent.html">Agent 轨迹</a></nav>',1)
            index.write_text(html)
    print('Visualizations:',directory,flush=True)


def write_page(directory,kind,payload):
    (directory/(kind+'_data.js')).write_text('window.VIEW_DATA='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';\n')
    template=(Path(__file__).parent/'assets/recovery.html').read_text()
    (directory/(kind+'.html')).write_text(template.replace('__KIND__',kind))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/semantic_pipeline')
    p.add_argument('--kind',choices=['teacher','agent','both'],default='both');p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--sweep-points',type=int,default=21);args=p.parse_args()
    cfg=json.loads((resolve(args.run)/'pipeline_config.json').read_text())
    export(cfg,args.kind,args.episodes,args.sweep_points)

if __name__=='__main__':main()
