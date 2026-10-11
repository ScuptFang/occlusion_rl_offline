"""ReAgent-inspired continuous expert, behavior cloning and reward curriculum.

Ground-truth rotations/translations are training targets, never observations.
The teacher remains frozen and the environment never gates an action by score.
"""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from .rl import action_steps, continuous_error, step_context


def validate_curriculum(options):
    for key in ('iterations', 'epochs', 'batch_size'):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('imitation.%s must be a positive integer' % key)
    for key in ('gt_weight_start', 'gt_weight_end', 'expert_rollout_probability'):
        if not 0 <= options[key] <= 1:
            raise ValueError('imitation.%s must be in [0,1]' % key)
    if options['gt_weight_start'] < options['gt_weight_end']:
        raise ValueError('Ground-truth weight must not increase')
    if not 0 <= options['decay_start'] < options['decay_end'] <= 1:
        raise ValueError('Require 0 <= decay_start < decay_end <= 1')
    if not 0 <= options['bc_weight_end'] <= options['bc_weight_start']:
        raise ValueError('BC weights must be nonnegative and nonincreasing')
    for key in ('learning_rate', 'encoder_learning_rate', 'gt_reward_scale'):
        if not np.isfinite(options[key]) or options[key] <= 0:
            raise ValueError('imitation.%s must be finite and positive' % key)
    if not 0 < options['expert_action_limit'] < 1:
        raise ValueError('expert_action_limit must be in (0,1)')


def curriculum_weights(options, iteration, iterations):
    # Fixed for an entire rollout; evaluate progress against the original budget.
    progress = iteration / max(iterations - 1, 1)
    fraction = np.clip((progress - options['decay_start']) /
                       (options['decay_end'] - options['decay_start']), 0., 1.)
    gt = options['gt_weight_start'] + fraction * (options['gt_weight_end'] - options['gt_weight_start'])
    bc = options['bc_weight_start'] + fraction * (options['bc_weight_end'] - options['bc_weight_start'])
    return float(gt), float(bc)


def expert_actions(rotations, translations, rotation_step_deg, translation_step_mm, limit=.999):
    """A bounded, non-overshooting inverse for PoseTask's disentangled update.

    Scale a whole rotation vector, not individual axes: this keeps its axis
    and monotonically reduces SO(3) distance under left composition.
    """
    if min(rotation_step_deg, translation_step_mm) <= 0 or not 0 < limit < 1:
        raise ValueError('Positive steps and action limit in (0,1) required')
    delta = np.rad2deg(Rotation.from_matrix(rotations).inv().as_rotvec()) / rotation_step_deg
    delta /= np.maximum(1., np.max(np.abs(delta), axis=1, keepdims=True) / limit)
    translation = np.clip(-np.asarray(translations) / translation_step_mm, -limit, limit)
    return np.concatenate((delta, translation), axis=1).astype(np.float32)


def mixed_reward(task, rotations, translations, next_rotations, next_translations,
                 scores, next_scores, agent_options, options, gt_weight):
    before = continuous_error(*task.error(rotations, translations), agent_options['error_scales'])
    after = continuous_error(*task.error(next_rotations, next_translations), agent_options['error_scales'])
    gt = options['gt_reward_scale'] * (before - after)
    teacher = agent_options['reward_scale'] * (next_scores - scores)
    return gt_weight * gt + (1 - gt_weight) * teacher, gt, teacher


def cloning_loss(mean, targets):
    # Supervise the bounded action used by the environment, not raw Gaussian logits.
    return (torch.tanh(mean) - targets).square().mean()


def pretrain(agent, paths, task_for, cfg, device, out):
    """Online expert labels on a mix of expert- and learner-visited states.

    Independent episodes execute full horizons. Only the train split is used.
    The per-state expert label is recomputed even on learner trajectories.
    """
    a, options = cfg['agent'], cfg['agent']['imitation']
    encoder = [p for n, p in agent.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    policy = [p for n, p in agent.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    optimizer = torch.optim.Adam([{'params': policy, 'lr': options['learning_rate']},
                                  {'params': encoder, 'lr': options['encoder_learning_rate']}])
    history = []
    agent.train()
    for iteration in range(options['iterations']):
        task = task_for(paths[iteration % len(paths)])
        task.rng = np.random.RandomState(a['seed'] + 7000001 + iteration * 1009)
        r, t, points = task.reset(a['parallel_envs'], a['training_pose_bands'])
        observations, contexts, targets = [], [], []
        for step in range(a['horizon']):
            task.rotation_step_deg, task.translation_step_mm = action_steps(a, step)
            target = expert_actions(r, t, task.rotation_step_deg, task.translation_step_mm,
                                    options['expert_action_limit'])
            obs = task.observation(points)
            ctx = step_context(agent, len(points), step, a['horizon'], device)
            observations.append(obs)
            contexts.append(ctx)
            targets.append(torch.as_tensor(target, device=device))
            with torch.no_grad():
                learner = torch.tanh(agent(obs, ctx)[0]).cpu().numpy()
            use_expert = task.rng.rand(len(points), 1) < options['expert_rollout_probability']
            r, t, points, _ = task.move(r, t, np.where(use_expert, target, learner))
        observations = torch.cat(observations)
        contexts = torch.cat(contexts) if agent.time_context else None
        targets = torch.cat(targets)
        losses = []
        for _ in range(options['epochs']):
            for ids in torch.randperm(len(targets), device=device).split(options['batch_size']):
                mean, _ = agent(observations[ids], contexts[ids] if contexts is not None else None)
                loss = cloning_loss(mean, targets[ids])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite imitation loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), a['max_grad_norm'])
                optimizer.step()
                losses.append(float(loss.detach()))
        row = {'iteration': iteration + 1, 'action_mse': float(np.mean(losses)),
               'labels': len(targets)}
        history.append(row)
        if (iteration + 1) % a['log_every'] == 0 or iteration + 1 == options['iterations']:
            print('imitation', json.dumps(row), flush=True)
            (out / 'imitation_history.json').write_text(json.dumps(history, indent=2))
    return history
