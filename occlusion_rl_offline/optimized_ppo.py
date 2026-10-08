"""Full-budget PPO with teacher-only rewards and continuous validation selection."""
import argparse
import os
import hashlib
import json
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from .runtime import resolve_device, runtime_info
from .cohort import read_manifest
from .config import load_config, resolve
from .evaluation import evaluate_continuous
from .rl import Agent, PoseTask, load_teacher, teacher_score, action_steps, step_context, validate_teacher_pair


def advantages_and_returns(rewards, values, bootstrap, gamma, gae_lambda):
    gae=torch.zeros_like(bootstrap); advantages=[]; next_value=bootstrap
    for reward,value in zip(reversed(rewards),reversed(values)):
        gae=reward+gamma*next_value-value+gamma*gae_lambda*gae
        advantages.append(gae);next_value=value
    advantages=torch.stack(advantages[::-1])
    return advantages, advantages+torch.stack(values)


def run(cfg, manifest, teacher_path, out, resume=None):
    device=resolve_device(cfg)
    a=cfg['agent']; torch.set_num_threads(a.get('torch_threads',4))
    torch.manual_seed(a['seed']);np.random.seed(a['seed'])
    if a.get('score_gated_actions',False):raise ValueError('Continuous PPO cannot reject score-decreasing actions')
    paths=read_manifest(manifest);out.mkdir(parents=True,exist_ok=True)
    (out/'config.snapshot.json').write_text(json.dumps(cfg,indent=2))
    (out/'runtime.json').write_text(json.dumps(runtime_info(device),indent=2))
    teacher=load_teacher(teacher_path,device,a['fps_points'],**cfg['model'])
    teacher_hash=hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    agent=Agent(a['fps_points'],a['policy_width'],a['initial_log_std'],
                time_context=a.get('time_context',True),**cfg['model']).to(device)
    agent.encoder.load_state_dict(teacher.state_dict())
    # Small initial policy means reduce arbitrary drift; sampled actions remain
    # stochastic and every step is executed, including near the original pose.
    nn.init.orthogonal_(agent.actor.weight,gain=a.get('actor_init_gain',.01))
    nn.init.zeros_(agent.actor.bias)
    encoder=[p for n,p in agent.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    policy=[p for n,p in agent.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    optimizer=torch.optim.Adam([{'params':policy,'lr':a['learning_rate']},
                                {'params':encoder,'lr':a.get('encoder_learning_rate',1e-5)}])
    start=0;best=float('inf');history=[]
    if resume:
        ckpt=torch.load(resume,map_location=device)
        validate_teacher_pair(ckpt, teacher_path)
        if ckpt['model_options']!=cfg['model']:raise ValueError('Resume sampler mismatch')
        for key in ('seed','horizon','parallel_envs','action_schedule','time_context','finite_horizon','reward_scale'):
            if ckpt['agent_options'].get(key)!=a.get(key):raise ValueError('Resume training option mismatch: '+key)
        if str(teacher_path)!=ckpt['teacher_checkpoint']:raise ValueError('Resume teacher mismatch')
        agent.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        start=ckpt['iteration'];best=ckpt['best_error'];torch.set_rng_state(ckpt['rng_state'].cpu())
        if device.startswith('cuda') and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(ckpt['cuda_rng_state'].cpu(),device)
        history=ckpt['history']
        if not (out/'agent.pt').exists():
            # A resumed run may never improve its earlier best; preserve that
            # actual best checkpoint, not the latest checkpoint used to resume.
            previous_best=resolve(resume).parent/'agent.pt'
            if not previous_best.exists():raise ValueError('Resume requires the previous agent.pt beside latest.pt')
            import shutil
            shutil.copyfile(previous_best,out/'agent.pt')
            previous_eval=resolve(resume).parent/'evaluation.json'
            if previous_eval.exists():shutil.copyfile(previous_eval,out/'evaluation.json')
    cache=OrderedDict()
    def task_for(path):
        if path not in cache:
            cache[path]=PoseTask(path,a['fps_points'],a['seed'],a['start_rotation_deg'],
                                 a['start_translation_mm'],device,a['action_rotation_step_deg'],
                                 a['action_translation_step_mm'],cfg['model'])
        cache.move_to_end(path)
        if len(cache)>2:cache.popitem(last=False)
        return cache[path]
    def validation():
        agent.eval()
        result=evaluate_continuous(agent,teacher,paths['val'],cfg,a['validation_episodes'],
                                    a.get('validation_seed',426101),(a['horizon'],),details=False)
        agent.train();return result
    if not resume:
        initial=validation();(out/'untrained.json').write_text(json.dumps(initial,indent=2))
        print('untrained',json.dumps(initial['macro']),flush=True)
    stale=0
    for iteration in range(start,a['iterations']):
        frozen=iteration<a.get('freeze_encoder_iterations',0)
        for parameter in encoder:parameter.requires_grad_(not frozen)
        task=task_for(paths['train'][iteration%len(paths['train'])])
        task.rng=np.random.RandomState(a['seed']+iteration*1009)
        rotations,translations,points=task.reset(a['parallel_envs'],a['training_pose_bands'])
        scores=teacher_score(teacher,points,device);rollout=[]
        for step in range(a['horizon']):
            task.rotation_step_deg,task.translation_step_mm=action_steps(a,step)
            obs=task.observation(points);ctx=step_context(agent,len(points),step,a['horizon'],device)
            with torch.no_grad():
                encoded=agent.encoder.encode(obs) if frozen else None
                dist,value=(agent.distribution_encoded(encoded,ctx) if frozen else agent.distribution(obs,ctx))
                raw=dist.sample();log_prob=dist.log_prob(raw).sum(-1)
            nr,nt,np_,ns,_,_=task.move_scored(rotations,translations,points,scores,
                                            torch.tanh(raw).cpu().numpy(),teacher,False)
            reward=torch.tensor((ns-scores)*a['reward_scale'],dtype=torch.float32,device=device)
            rollout.append((obs,ctx,raw,log_prob,value,reward,encoded))
            rotations,translations,points,scores=nr,nt,np_,ns
        if a.get('finite_horizon',True):bootstrap=torch.zeros(a['parallel_envs'],device=device)
        else:
            with torch.no_grad():
                bootstrap=agent(task.observation(points),step_context(agent,len(points),a['horizon'],a['horizon'],device))[1]
        adv,returns=advantages_and_returns([r[5] for r in rollout],[r[4] for r in rollout],bootstrap,a['gamma'],a['gae_lambda'])
        obs=torch.cat([r[0] for r in rollout]);ctx=torch.cat([r[1] for r in rollout]) if agent.time_context else None
        encodings=torch.cat([r[6] for r in rollout]) if frozen else None
        actions=torch.cat([r[2] for r in rollout]);oldlp=torch.cat([r[3] for r in rollout])
        old_values=torch.cat([r[4] for r in rollout]);returns=returns.flatten();adv=adv.flatten()
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        stats=[]
        for epoch in range(a['ppo_epochs']):
            for ids in torch.randperm(len(actions),device=device).split(a['minibatch']):
                context=ctx[ids] if ctx is not None else None
                dist,value=(agent.distribution_encoded(encodings[ids],context) if frozen
                            else agent.distribution(obs[ids],context))
                logratio=dist.log_prob(actions[ids]).sum(-1)-oldlp[ids];ratio=logratio.exp()
                policy_loss=-torch.minimum(ratio*adv[ids],ratio.clamp(1-a['ppo_clip'],1+a['ppo_clip'])*adv[ids]).mean()
                value_loss=(value-returns[ids]).square().mean();entropy=dist.entropy().sum(-1).mean()
                loss=policy_loss+a['value_coef']*value_loss-a['entropy_coef']*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(agent.parameters(),a['max_grad_norm']);optimizer.step()
                stats.append([float(policy_loss.detach()),float(value_loss.detach()),float(entropy.detach()),
                              float(((ratio-1)-logratio).mean().detach()),float(((ratio-1).abs()>a['ppo_clip']).float().mean().detach())])
            if np.mean(stats,axis=0)[3]>a.get('target_kl',.03):break  # Stop optimizer passes only, never rollout actions.
        with torch.no_grad():agent.log_std.clamp_(a.get('min_log_std',-3),a.get('max_log_std',.5))
        if (iteration+1)%a['log_every']==0 or iteration+1==a['iterations']:
            result=validation();metric=result['macro'][str(a['horizon'])]['error']
            value_var=float(returns.var(unbiased=False));ev=1-float((returns-old_values).var(unbiased=False))/max(value_var,1e-8)
            row={'iteration':iteration+1,'interactions':(iteration+1)*a['parallel_envs']*a['horizon'],
                 'reward_mean':float(torch.stack([r[5] for r in rollout]).mean()),
                 'ppo':dict(zip(['policy_loss','value_loss','raw_entropy','approx_kl','clip_fraction'],map(float,np.mean(stats,axis=0)))),
                 'explained_variance':ev,'action_mean':torch.tanh(actions).mean(0).tolist(),
                 'action_saturated_fraction':float((torch.tanh(actions).abs()>.95).float().mean()),
                 'log_std':agent.log_std.detach().tolist(),'encoder_frozen':frozen,'validation':result}
            history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({'iteration':iteration+1,'interactions':row['interactions'],'validation':result['macro'],'ppo':row['ppo']}),flush=True)
            improved=metric<best
            if improved:best=metric;stale=0
            else:stale+=1
            ckpt={'model':agent.state_dict(),'optimizer':optimizer.state_dict(),'iteration':iteration+1,
                  'model_options':cfg['model'],'agent_options':a,'best_error':best,'history':history,
                  'rng_state':torch.get_rng_state(),
                  'cuda_rng_state':torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,'teacher_checkpoint':str(teacher_path),
                  'teacher_sha256':teacher_hash}
            torch.save(ckpt,out/'latest.pt')
            if improved:
                torch.save(ckpt,out/'agent.pt');(out/'evaluation.json').write_text(json.dumps(result,indent=2))
            if stale>=a.get('early_stopping_checks',100000):
                print('Training validation plateau: all completed rollouts used full budget.',flush=True);break
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--manifest',default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    p.add_argument('--teacher',required=True);p.add_argument('--out',required=True);p.add_argument('--resume')
    p.add_argument('--device',default=None)
    args=p.parse_args();cfg=load_config(args.config);cfg.setdefault('runtime',{})['device']=resolve_device(cfg,args.device)
    os.environ['OCCLUSION_DEVICE']=cfg['runtime']['device']
    run(cfg,resolve(args.manifest),resolve(args.teacher),resolve(args.out),resolve(args.resume) if args.resume else None)

if __name__=='__main__':main()
