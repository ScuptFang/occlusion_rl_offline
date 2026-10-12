"""Teacher-reward RL fine-tuning of an imitation-pretrained diffusion policy."""
import argparse
import hashlib
import json
from collections import OrderedDict

import numpy as np
import torch

from .cohort import read_manifest
from .config import load_config, resolve
from .diffusion_rl import DiffusionRLAgent, clipped_objective
from .evaluation import evaluate_continuous
from .optimized_ppo import advantages_and_returns
from .rl import PoseTask, action_steps, load_teacher, step_context, teacher_score
from .runtime import resolve_device, runtime_info


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg, manifest, teacher_path, source, out, resume=None):
    a = cfg['agent']; o = a['diffusion_rl']; device = resolve_device(cfg)
    for k in ['iterations', 'parallel_envs', 'ppo_epochs', 'minibatch', 'validate_every']:
        if isinstance(o[k], bool) or not isinstance(o[k], int) or o[k] < 1:
            raise ValueError('Positive integer required: ' + k)
    for k in ['learning_rate', 'critic_learning_rate', 'max_grad_norm', 'target_kl', 'reward_scale']:
        if not np.isfinite(o[k]) or o[k] <= 0:
            raise ValueError('Positive finite value required: ' + k)
    if not 0 < o['clip'] < 1 or not 0 <= o['gamma'] <= 1 or not 0 <= o['gae_lambda'] <= 1:
        raise ValueError('Invalid PPO/GAE options')
    if any(not np.isfinite(o[k]) or o[k] < 0 for k in ['reference_kl_coef', 'value_coef']):
        raise ValueError('Invalid loss weights')
    if a.get('score_gated_actions', False):
        raise ValueError('RL cannot reject score-decreasing actions')
    torch.set_num_threads(a.get('torch_threads', 4)); torch.manual_seed(a['seed'])
    paths = read_manifest(manifest)
    teacher = load_teacher(teacher_path, device, a['fps_points'], **cfg['model'])
    pretrained = torch.load(source, map_location=device, weights_only=True)
    if pretrained.get('model_options') != cfg['model']:
        raise ValueError('Imitation geometry mismatch')
    if pretrained.get('teacher_sha256') != sha(teacher_path) or pretrained.get('manifest_sha256') != sha(manifest):
        raise ValueError('Imitation teacher/cohort mismatch')
    if pretrained['agent_options']['trainer'] != 'diffusion':
        raise ValueError('A supervised diffusion checkpoint is required')
    for key in ['diffusion', 'fps_points', 'horizon', 'action_rotation_step_deg', 'action_translation_step_mm', 'time_context']:
        if pretrained['agent_options'].get(key) != a.get(key):
            raise ValueError('Imitation action/diffusion mismatch: ' + key)
    agent = DiffusionRLAgent(a, cfg['model']).to(device)
    agent.initialize_imitation(pretrained['model'])
    optimizer = torch.optim.AdamW([
        {'params': [p for p in agent.actor.parameters() if p.requires_grad], 'lr': o['learning_rate']},
        {'params': agent.critic.parameters(), 'lr': o['critic_learning_rate']}], weight_decay=0)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'teacher_sha256': sha(teacher_path), 'manifest_sha256': sha(manifest), 'imitation_sha256': sha(source)}
    start = 0; best = float('inf'); overall_best = float('inf'); history = []
    def validate():
        agent.eval()
        result = evaluate_continuous(agent, teacher, paths['val'], cfg, a['validation_episodes'],
                                     a['validation_seed'], (a['horizon'],), details=False)
        agent.train()
        return result
    def checkpoint(iteration):
        return {'model': agent.state_dict(), 'agent_options': a, 'model_options': cfg['model'],
                'teacher_checkpoint': str(teacher_path), 'imitation_checkpoint': str(source),
                **identity, 'iteration': iteration, 'best_error': best, 'overall_best_error': overall_best,
                'history': history, 'optimizer': optimizer.state_dict(), 'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(device) if device.startswith('cuda') else None}
    if resume:
        c = torch.load(resume, map_location=device, weights_only=True)
        if any(c[k] != v for k, v in identity.items()) or c['agent_options'] != a or c['model_options'] != cfg['model']:
            raise ValueError('Resume identity/configuration mismatch')
        if resolve(resume).resolve().parent != out.resolve():
            raise ValueError('Resume in the original output directory to preserve selected checkpoints')
        agent.load_state_dict(c['model']); optimizer.load_state_dict(c['optimizer'])
        start, best, overall_best, history = c['iteration'], c['best_error'], c['overall_best_error'], c['history']
        torch.set_rng_state(c['rng_state'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state(c['cuda_rng_state'].cpu(), device)
    else:
        if (out / 'latest.pt').exists() or (out / 'imitation.pt').exists():
            raise ValueError('Existing run: resume explicitly or use a new output directory')
        initial = validate(); overall_best = initial['macro'][str(a['horizon'])]['error']
        (out / 'imitation_evaluation.json').write_text(json.dumps(initial, indent=2))
        torch.save(checkpoint(0), out / 'imitation.pt')
        torch.save(checkpoint(0), out / 'best_including_imitation.pt')
        print('imitation_stochastic', json.dumps(initial['macro']), flush=True)
    (out / 'config.snapshot.json').write_text(json.dumps(cfg, indent=2))
    (out / 'runtime.json').write_text(json.dumps(runtime_info(device), indent=2))
    cache = OrderedDict()
    for iteration in range(start, o['iterations']):
        path = paths['train'][iteration % len(paths['train'])]
        if path not in cache:
            cache[path] = PoseTask(path, a['fps_points'], a['seed'], a['start_rotation_deg'],
                                  a['start_translation_mm'], device, a['action_rotation_step_deg'],
                                  a['action_translation_step_mm'], cfg['model'])
        cache.move_to_end(path)
        if len(cache) > 2: cache.popitem(last=False)
        task = cache[path]; task.rng = np.random.RandomState(a['seed'] + iteration * 1009)
        r, t, points = task.reset(o['parallel_envs'], a['training_pose_bands'])
        scores = teacher_score(teacher, points, device); rollout = []; negative = 0
        for step in range(a['horizon']):
            obs = task.observation(points); context = step_context(agent, len(points), step, a['horizon'], device)
            plan, before, after, lp, value = agent.sample_chain(obs, context)
            task.rotation_step_deg, task.translation_step_mm = action_steps(a, step)
            nr, nt, np_, ns, valid, accepted = task.move_scored(r, t, points, scores,
                                       plan[:, 0].cpu().numpy(), teacher, False)
            if not np.all(valid) or not np.all(accepted):
                raise RuntimeError('Unexpected invalid/rejected diffusion action')
            reward = torch.as_tensor((ns - scores) * o['reward_scale'], device=device, dtype=torch.float32)
            negative += int((ns < scores).sum())
            rollout.append((obs, context, before, after, lp, value, reward))
            r, t, points, scores = nr, nt, np_, ns
        adv, returns = advantages_and_returns([r[6] for r in rollout], [r[5] for r in rollout],
                           torch.zeros(o['parallel_envs'], device=device), o['gamma'], o['gae_lambda'])
        obs, context, before, after, oldlp = [torch.cat([r[k] for r in rollout]) for k in range(5)]
        adv = adv.flatten(); adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        returns = returns.flatten(); stats = []
        for epoch in range(o['ppo_epochs']):
            stop = False
            for ids in torch.randperm(len(obs), device=device).split(o['minibatch']):
                lp, value, refkl = agent.evaluate_chain(obs[ids], context[ids], before[ids], after[ids])
                policy_loss, kl, clip_fraction = clipped_objective(lp, oldlp[ids], adv[ids], o['clip'])
                if not torch.isfinite(kl): raise FloatingPointError('Nonfinite denoising KL')
                if float(kl.detach()) > o['target_kl']:
                    stop = True; break
                value_loss = (value - returns[ids]).square().mean()
                loss = policy_loss + o['value_coef'] * value_loss + o['reference_kl_coef'] * refkl.mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diffusion PPO loss')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), o['max_grad_norm']); optimizer.step()
                stats.append([float(x.detach()) for x in [policy_loss, value_loss, kl, clip_fraction, refkl.mean()]])
            if stop: break
        row = {'iteration': iteration + 1, 'case': path.parent.name,
               'interactions': (iteration + 1) * o['parallel_envs'] * a['horizon'],
               'reward_mean': float(torch.stack([r[6] for r in rollout]).mean()),
               'score_decreasing_actions': negative, 'optimizer_updates': len(stats),
               'ppo': dict(zip(['policy_loss', 'value_loss', 'approx_kl', 'clip_fraction', 'reference_kl'],
                              np.mean(stats, axis=0).tolist() if stats else [None] * 5))}
        if (iteration + 1) % o['validate_every'] == 0 or iteration + 1 == o['iterations']:
            result = validate(); metric = result['macro'][str(a['horizon'])]['error']; row['validation'] = result
            improved = metric < best; overall_improved = metric < overall_best
            best = min(best, metric); overall_best = min(overall_best, metric)
            history.append(row)
            c = checkpoint(iteration + 1)
            if improved:
                torch.save(c, out / 'agent.pt')
                (out / 'evaluation.json').write_text(json.dumps(result, indent=2))
            if overall_improved: torch.save(c, out / 'best_including_imitation.pt')
            torch.save(c, out / 'latest.pt')
            print(json.dumps({'iteration': iteration + 1, 'validation': result['macro'], 'ppo': row['ppo']}), flush=True)
        else:
            history.append(row)
            print(json.dumps({k: row[k] for k in ['iteration','case','reward_mean','optimizer_updates']}), flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--teacher', required=True); p.add_argument('--out', required=True)
    p.add_argument('--resume')
    args = p.parse_args(); cfg = load_config(args.config)
    run(cfg, resolve(args.manifest), resolve(args.teacher), resolve(cfg['paths']['imitation_agent']),
        resolve(args.out), resolve(args.resume) if args.resume else None)


if __name__ == '__main__': main()
