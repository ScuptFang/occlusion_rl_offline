"""Single-case PPO pose recovery rewarded only by teacher-score change."""
from .runtime import resolve_device
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.distributions import Normal

from .data import load_case, sample_upper_mesh, candidate_indices, sample_sdf
from .config import REPO_ROOT
from .model import OcclusionTeacher


class PoseTask:
    def __init__(self, case, points, seed, max_rotation, max_translation, device,
                 rotation_step_deg=2.0, translation_step_mm=0.4, fps_options=None):
        self.case = load_case(case)
        self.rng = np.random.RandomState(seed)
        self.xyz = self.case['xyz']
        self.normal = self.case['normal']
        self.points = points
        self.fps_options = fps_options
        self.pivot = self.case['xyz'].mean(0)
        self.max_rotation, self.max_translation = max_rotation, max_translation
        self.rotation_step_deg, self.translation_step_mm = rotation_step_deg, translation_step_mm
        self.device = device
        self.episode_ids = None

    def begin_episode(self):
        self.episode_ids = None

    def points_at(self, rotations, translations):
        fixed = (self.fps_options or {}).get('sampling_mode') == 'episode_fixed'
        if fixed and self.episode_ids is not None:
            if len(self.episode_ids) != len(rotations):
                raise ValueError('Call begin_episode before changing fixed-ID batch size')
            xyz = (np.einsum('bnj,bij->bni', self.xyz[self.episode_ids] - self.pivot, rotations)
                   + self.pivot + translations[:, None]).astype(np.float32)
            normal = np.einsum('bnj,bij->bni', self.normal[self.episode_ids], rotations)
            distances = sample_sdf(self.case, xyz.reshape(-1, 3), self.fps_options).reshape(len(rotations), -1)
            self.last_ids = self.episode_ids
            return np.concatenate((xyz, normal, self.case['curvature'][self.episode_ids].reshape(len(rotations), -1, 1),
                                   distances[:, :, None]), axis=-1).astype(np.float32)
        ids = candidate_indices(self.case, self.fps_options)
        xyz = (np.einsum('nj,bij->bni', self.xyz[ids] - self.pivot, rotations)
               + self.pivot + translations[:, None]).astype(np.float32)
        normal = np.einsum('nj,bij->bni', self.normal[ids], rotations)
        distances = sample_sdf(self.case, xyz.reshape(-1, 3), self.fps_options).reshape(len(rotations), -1)
        selected = [sample_upper_mesh(self.case, x, n, self.points, self.fps_options,
                                     return_indices=True, preselected_ids=ids, precomputed_sdf=d)
                    for x, n, d in zip(xyz, normal, distances)]
        features = [x[0] for x in selected]
        self.last_ids = np.asarray([x[1] for x in selected])
        if fixed:
            self.episode_ids = self.last_ids
        return np.asarray(features, dtype=np.float32)

    def reset(self, count, pose_bands=None):
        self.begin_episode()
        while True:
            if pose_bands:
                probabilities = np.asarray([band['probability'] for band in pose_bands], dtype=float)
                choices = self.rng.choice(len(pose_bands), size=count, p=probabilities)
                rotation_limits = np.asarray([pose_bands[i]['rotation_deg'] for i in choices])[:, None]
                translation_limits = np.asarray([pose_bands[i]['translation_mm'] for i in choices])[:, None]
                rv = self.rng.uniform(-1, 1, (count, 3)) * rotation_limits
                translations = self.rng.uniform(-1, 1, (count, 3)) * translation_limits
            else:
                rv = self.rng.uniform(-self.max_rotation, self.max_rotation, (count, 3))
                translations = self.rng.uniform(-self.max_translation, self.max_translation, (count, 3))
            rotations = Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            try:
                points = self.points_at(rotations, translations)
                return rotations, translations, points
            except ValueError:
                # Retry only if a pose or feature is invalid.
                continue

    def observation(self, points):
        # Match the teacher's [xyz, normal, curvature, lower-SDF] input exactly.
        return torch.from_numpy(points).to(self.device)

    def error(self, rotations, translations):
        angle = np.rad2deg(Rotation.from_matrix(rotations).magnitude())
        distance = np.linalg.norm(translations, axis=1)
        return angle, distance

    def move(self, rotations, translations, action):
        # Action is bounded to one incremental rotation and translation.
        action = np.clip(action, -1, 1)
        delta = Rotation.from_rotvec(np.deg2rad(action[:, :3] * self.rotation_step_deg)).as_matrix()
        next_rotations = delta @ rotations
        next_translations = translations + action[:, 3:] * self.translation_step_mm
        try:
            points = self.points_at(next_rotations, next_translations)
            return next_rotations, next_translations, points, np.ones(len(action), dtype=bool)
        except ValueError:
            # Keep valid states for the rest of the fixed-length rollout.
            valid = np.ones(len(action), dtype=bool)
            for i in range(len(action)):
                try:
                    self.points_at(next_rotations[i:i + 1], next_translations[i:i + 1])
                except ValueError:
                    valid[i] = False
                    next_rotations[i] = rotations[i]
                    next_translations[i] = translations[i]
            points = self.points_at(next_rotations, next_translations)
            return next_rotations, next_translations, points, valid

    def move_scored(self, rotations, translations, points, scores, action, teacher,
                    score_gate=False):
        next_rotations, next_translations, next_points, valid = self.move(
            rotations, translations, action)
        next_scores = teacher_score(teacher, next_points, self.device)
        if score_gate:
            accepted = valid & (next_scores > scores)
            next_rotations = np.where(accepted[:, None, None], next_rotations, rotations)
            next_translations = np.where(accepted[:, None], next_translations, translations)
            next_points = np.where(accepted[:, None, None], next_points, points)
            next_scores = np.where(accepted, next_scores, scores)
        else:
            accepted = valid
        return next_rotations, next_translations, next_points, next_scores, valid, accepted


class Agent(nn.Module):
    def __init__(self, samples=128, policy_width=128, initial_log_std=-0.5,
                 time_context=False, **model_options):
        super().__init__()
        # The shared input sampler performs candidate FPS; the encoder applies
        # local attention and multi-scale distance-weighted pooling.
        self.encoder = OcclusionTeacher(sample_count=samples, **model_options)
        for parameter in self.encoder.head.parameters():
            parameter.requires_grad_(False)
        width = model_options.get('width', 64)
        self.time_context = time_context
        self.body = nn.Sequential(nn.Linear(width * 5 + 4 + (2 if time_context else 0), policy_width), nn.Tanh())
        self.actor = nn.Linear(policy_width, 6)
        self.critic = nn.Linear(policy_width, 1)
        self.log_std = nn.Parameter(torch.full((6,), initial_log_std))

    def forward(self, obs, context=None):
        encoded = self.encoder.encode(obs)
        return self.forward_encoded(encoded, context)

    def forward_encoded(self, encoded, context=None):
        if self.time_context:
            if context is None:
                raise ValueError('Time-conditioned agent requires explicit step context')
            encoded = torch.cat((encoded, context), -1)
        h = self.body(encoded)
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs, context=None):
        mean, value = self(obs, context)
        return Normal(mean, self.log_std.exp().expand_as(mean)), value

    def distribution_encoded(self, encoded, context=None):
        mean, value = self.forward_encoded(encoded, context)
        return Normal(mean, self.log_std.exp().expand_as(mean)), value


def policy_action(agent, observations, context, generator=None):
    """Return the executed action and optional newly predicted action chunk."""
    if hasattr(agent, 'predict_action_chunk'):
        plan = agent.predict_action_chunk(observations, context, generator=generator)
        return plan[:, 0], plan
    return torch.tanh(agent(observations, context)[0]), None


def step_context(agent, count, step, horizon, device):
    if not agent.time_context:
        return None
    return torch.tensor([step / max(horizon, 1), (horizon - step) / max(horizon, 1)],
                        dtype=torch.float32, device=device).expand(count, -1)


def action_steps(options, step):
    """Absolute step schedule; never stops or rejects an action."""
    for stage in options.get('action_schedule', []):
        if step < stage['until_step']:
            return stage['rotation_deg'], stage['translation_mm']
    stages = options.get('action_schedule', [])
    if stages:
        return stages[-1]['rotation_deg'], stages[-1]['translation_mm']
    return options['action_rotation_step_deg'], options['action_translation_step_mm']


def continuous_error(angle, distance, scales=(8., 4.)):
    return .5 * (np.asarray(angle) / scales[0] + np.asarray(distance) / scales[1])


def load_teacher(path, device, samples, **model_options):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('model_options') is not None and checkpoint['model_options'] != model_options:
        raise ValueError('Teacher model/sampling configuration does not match checkpoint')
    if checkpoint['samples'] != samples:
        raise ValueError('Teacher checkpoint FPS count %d does not match requested %d; retrain the teacher or use its original configuration' %
                         (checkpoint['samples'], samples))
    teacher = OcclusionTeacher(sample_count=samples,
                               **model_options).to(device)
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def validate_teacher_pair(checkpoint, teacher_path):
    if checkpoint.get('teacher_sha256'):
        import hashlib
        if hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest() != checkpoint['teacher_sha256']:
            raise ValueError('Scoring teacher differs from the frozen teacher used to train this agent')


@torch.no_grad()
def teacher_score(teacher, points, device):
    return teacher(torch.from_numpy(points).to(device)).cpu().numpy()


def evaluate(agent, task, teacher, episodes, horizon, seed,
             success_rotation_deg=1.0, success_translation_mm=0.5,
             score_gate=False):
    task.rng = np.random.RandomState(seed)
    rotations, translations, points = task.reset(episodes)
    initial_angle, initial_distance = task.error(rotations, translations)
    score = teacher_score(teacher, points, task.device)
    initial_score = score.copy()
    for _ in range(horizon):
        with torch.no_grad():
            action = torch.tanh(agent(task.observation(points))[0]).cpu().numpy()
        rotations, translations, points, score, _, _ = task.move_scored(
            rotations, translations, points, score, action, teacher, score_gate)
    angle, distance = task.error(rotations, translations)
    return {'episodes': episodes, 'initial_rotation_deg': float(initial_angle.mean()),
            'final_rotation_deg': float(angle.mean()), 'initial_translation_mm': float(initial_distance.mean()),
            'final_translation_mm': float(distance.mean()), 'initial_teacher_logit': float(initial_score.mean()),
            'final_teacher_logit': float(score.mean()),
            'initial_success_rate': float(((initial_angle <= success_rotation_deg) &
                                           (initial_distance <= success_translation_mm)).mean()),
            'rotation_improved_fraction': float((angle < initial_angle).mean()),
            'translation_improved_fraction': float((distance < initial_distance).mean()),
            'success_rate': float(((angle <= success_rotation_deg) &
                                   (distance <= success_translation_mm)).mean()),
            'success_rotation_deg': success_rotation_deg,
            'success_translation_mm': success_translation_mm}


def evaluate_cases(agent, case_paths, make_task, teacher, episodes, horizon, seed,
                   success_rotation_deg, success_translation_mm, score_gate=False):
    results = [evaluate(agent, make_task(path, index), teacher, episodes, horizon,
                        seed + index * 1009, success_rotation_deg, success_translation_mm,
                        score_gate)
               for index, path in enumerate(case_paths)]
    averaged = {key: float(np.mean([result[key] for result in results]))
                for key in ('initial_rotation_deg', 'final_rotation_deg',
                            'initial_translation_mm', 'final_translation_mm',
                            'initial_teacher_logit', 'final_teacher_logit',
                            'initial_success_rate', 'rotation_improved_fraction',
                            'translation_improved_fraction', 'success_rate')}
    return {'episodes': episodes * len(case_paths), 'cases': len(case_paths), **averaged,
            'success_rotation_deg': success_rotation_deg,
            'success_translation_mm': success_translation_mm}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(override=getattr(args,'device',None))
    model_options = json.loads(args.model_options)
    train_cases = json.loads(args.train_cases) if args.train_cases else [args.case]
    eval_paths = json.loads(args.eval_cases) if args.eval_cases else [args.case]
    if not train_cases or not eval_paths:
        raise ValueError('Train and evaluation case lists must be nonempty')
    if not args.eval_only and args.iterations < len(train_cases):
        raise ValueError('agent.iterations must be at least the number of training cases')
    pose_bands = json.loads(args.training_pose_bands) if args.training_pose_bands else None
    if pose_bands:
        probabilities = np.asarray([band['probability'] for band in pose_bands], dtype=float)
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
            raise ValueError('Training pose-band probabilities must be nonnegative and sum to 1')

    def make_task(path, index):
        return PoseTask(path, args.points, args.seed, args.max_rotation, args.max_translation,
                        device, args.rotation_step_deg, args.translation_step_mm,
                        model_options)

    def evaluate_split(seed):
        return evaluate_cases(agent, eval_paths, make_task, teacher,
                              args.eval_episodes, args.horizon, seed,
                              args.success_rotation_deg, args.success_translation_mm,
                              args.score_gated_actions)
    teacher = load_teacher(args.teacher, device, args.teacher_samples, **model_options)
    agent = Agent(args.teacher_samples, args.policy_width, args.initial_log_std,
                  **model_options).to(device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device)['model'])
    else:
        # Start from the teacher's learned point representation; the policy
        # encoder is then updated by PPO, while the scoring teacher stays frozen.
        agent.encoder.load_state_dict(teacher.state_dict())
    if args.eval_only:
        result = evaluate_split(args.seed + args.eval_seed_offset)
        print(json.dumps({'stage': 'evaluation', **result}), flush=True)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'test_evaluation.json').write_text(json.dumps(result, indent=2))
        return
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_split(args.seed + args.validation_seed_offset)
    print(json.dumps({'stage': 'untrained', **baseline}), flush=True)
    best_score_gain = float('-inf')
    history = []
    for iteration in range(args.iterations):
        task = make_task(train_cases[iteration % len(train_cases)], iteration % len(train_cases))
        task.rng = np.random.RandomState(args.seed + iteration * 1009)
        rotations, translations, points = task.reset(args.envs, pose_bands)
        scores = teacher_score(teacher, points, device)
        rollout = []
        for step in range(args.horizon):
            obs = task.observation(points)
            with torch.no_grad():
                dist, value = agent.distribution(obs)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action).sum(-1)
            next_rot, next_trans, next_points, next_scores, _, _ = task.move_scored(
                rotations, translations, points, scores, torch.tanh(raw_action).cpu().numpy(),
                teacher, args.score_gated_actions)
            # The only reward signal is teacher-score change. A score decrease
            # gives negative reward; invalid SDF moves are rolled back (zero).
            reward = args.reward_scale * (next_scores - scores)
            rollout.append((obs, raw_action, log_prob, value, torch.as_tensor(reward, dtype=torch.float32, device=device)))
            rotations, translations, points, scores = next_rot, next_trans, next_points, next_scores
        with torch.no_grad():
            next_value = agent(task.observation(points))[1]
        advantages = []
        gae = torch.zeros(args.envs, device=device)
        for obs, action, log_prob, value, reward in reversed(rollout):
            gae = reward + args.gamma * next_value - value + args.gamma * args.gae_lambda * gae
            advantages.append(gae)
            next_value = value
        advantages = torch.stack(advantages[::-1]).flatten()
        returns = advantages + torch.stack([x[3] for x in rollout]).flatten()
        observations = torch.cat([x[0] for x in rollout])
        actions = torch.cat([x[1] for x in rollout])
        old_log_prob = torch.cat([x[2] for x in rollout])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(args.ppo_epochs):
            for ids in torch.randperm(len(actions), device=device).split(args.minibatch):
                dist, value = agent.distribution(observations[ids])
                ratio = (dist.log_prob(actions[ids]).sum(-1) - old_log_prob[ids]).exp()
                surrogate = torch.minimum(ratio * advantages[ids],
                                          ratio.clamp(1 - args.ppo_clip, 1 + args.ppo_clip) * advantages[ids])
                loss = -surrogate.mean() + args.value_coef * (value - returns[ids]).square().mean() - args.entropy_coef * dist.entropy().sum(-1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        if (iteration + 1) % args.log_every == 0 or iteration + 1 == args.iterations:
            result = evaluate_split(args.seed + args.validation_seed_offset)
            record = {'stage': 'train', 'iteration': iteration + 1,
                      'reward': float(torch.stack([x[4] for x in rollout]).mean()), **result}
            history.append(record)
            (out / 'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(record), flush=True)
            checkpoint = {'model': agent.state_dict(), 'config': vars(args), 'iteration': iteration + 1}
            torch.save(checkpoint, out / 'latest.pt')
            score_gain = result['final_teacher_logit'] - result['initial_teacher_logit']
            if score_gain > best_score_gain:
                best_score_gain = score_gain
                torch.save(checkpoint, out / 'agent.pt')
                (out / 'evaluation.json').write_text(json.dumps({'untrained': baseline, 'best': result,
                                                                 'iteration': iteration + 1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default=str(REPO_ROOT / 'examples/data/case.npz'))
    parser.add_argument('--train-cases', default=None, help='JSON array of training case paths')
    parser.add_argument('--eval-cases', default=None, help='JSON array of validation or test case paths')
    parser.add_argument('--teacher', default=str(REPO_ROOT / 'examples/checkpoints/teacher.pt'))
    parser.add_argument('--out', default=str(REPO_ROOT / 'runs/default/agent'))
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--points', type=int, default=128)
    parser.add_argument('--teacher-samples', type=int, default=128)
    parser.add_argument('--score-gated-actions', action='store_true')
    parser.add_argument('--training-pose-bands', default=None)
    parser.add_argument('--max-rotation', type=float, default=8.0)
    parser.add_argument('--max-translation', type=float, default=4.0)
    parser.add_argument('--rotation-step-deg', type=float, default=2.0)
    parser.add_argument('--translation-step-mm', type=float, default=0.4)
    parser.add_argument('--model-options', default='{}')
    parser.add_argument('--policy-width', type=int, default=128)
    parser.add_argument('--initial-log-std', type=float, default=-0.5)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--envs', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ppo-clip', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--eval-episodes', type=int, default=64)
    parser.add_argument('--eval-seed-offset', type=int, default=100000)
    parser.add_argument('--validation-seed-offset', type=int, default=100000)
    parser.add_argument('--success-rotation-deg', type=float, default=1.0)
    parser.add_argument('--success-translation-mm', type=float, default=0.5)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.iterations < 1 or args.envs < 1 or args.horizon < 1 or args.points < 1:
        parser.error('iterations, envs, horizon and points must be positive')
    if args.max_rotation <= 0 or args.max_translation <= 0 or args.log_every < 1:
        parser.error('max-rotation, max-translation and log-every must be positive')
    train(args)


if __name__ == '__main__':
    main()
