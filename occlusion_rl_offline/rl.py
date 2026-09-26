"""Single-case PPO pose recovery rewarded only by teacher-score change."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.distributions import Normal

from .data import load_case, sample_sdf
from .config import REPO_ROOT
from .model import OcclusionTeacher


class PoseTask:
    def __init__(self, case, points, seed, max_rotation, max_translation, device,
                 rotation_step_deg=2.0, translation_step_mm=0.4, point_seed=None):
        self.case = load_case(case)
        self.rng = np.random.RandomState(seed)
        point_rng = np.random.RandomState(seed if point_seed is None else point_seed)
        self.ids = point_rng.choice(len(self.case['xyz']), points, replace=len(self.case['xyz']) < points)
        self.xyz = self.case['xyz'][self.ids]
        self.normal = self.case['normal'][self.ids]
        self.curvature = self.case['curvature'][self.ids].reshape(-1)
        self.pivot = self.case['xyz'].mean(0)
        self.max_rotation, self.max_translation = max_rotation, max_translation
        self.rotation_step_deg, self.translation_step_mm = rotation_step_deg, translation_step_mm
        self.device = device

    def points_at(self, rotations, translations):
        xyz = np.einsum('nj,bij->bni', self.xyz - self.pivot, rotations) + self.pivot + translations[:, None]
        normals = np.einsum('nj,bij->bni', self.normal, rotations)
        features = []
        for x, normal in zip(xyz, normals):
            d = sample_sdf(self.case['sdf_grid'], self.case['origin'], self.case['spacing'], x)
            features.append(np.column_stack((x, normal, self.curvature, d)))
        return np.asarray(features, dtype=np.float32)

    def reset(self, count):
        while True:
            rv = self.rng.uniform(-self.max_rotation, self.max_rotation, (count, 3))
            translations = self.rng.uniform(-self.max_translation, self.max_translation, (count, 3))
            rotations = Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            try:
                points = self.points_at(rotations, translations)
                return rotations, translations, points
            except ValueError:
                # The teacher must never see clipped SDF values.
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


class Agent(nn.Module):
    def __init__(self, samples=32, policy_width=128, initial_log_std=-0.5, **model_options):
        super().__init__()
        # The teacher encoder performs SDF-weighted FPS, local attention,
        # global pooling and multi-scale SDF-weighted pooling on the same 8 channels.
        self.encoder = OcclusionTeacher(sample_count=samples, **model_options)
        for parameter in self.encoder.head.parameters():
            parameter.requires_grad_(False)
        width = model_options.get('width', 64)
        self.body = nn.Sequential(nn.Linear(width * 5 + 4, policy_width), nn.Tanh())
        self.actor = nn.Linear(policy_width, 6)
        self.critic = nn.Linear(policy_width, 1)
        self.log_std = nn.Parameter(torch.full((6,), initial_log_std))

    def forward(self, obs):
        h = self.body(self.encoder.encode(obs))
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs):
        mean, value = self(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean)), value


def load_teacher(path, device, samples, **model_options):
    checkpoint = torch.load(path, map_location=device)
    teacher = OcclusionTeacher(sample_count=min(samples, checkpoint['samples']),
                               **model_options).to(device)
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def teacher_score(teacher, points, device):
    return teacher(torch.from_numpy(points).to(device)).cpu().numpy()


def evaluate(agent, task, teacher, episodes, horizon, seed,
             success_rotation_deg=1.0, success_translation_mm=0.5):
    task.rng = np.random.RandomState(seed)
    rotations, translations, points = task.reset(episodes)
    initial_angle, initial_distance = task.error(rotations, translations)
    initial_score = teacher_score(teacher, points, task.device)
    for _ in range(horizon):
        with torch.no_grad():
            action = torch.tanh(agent(task.observation(points))[0]).cpu().numpy()
        rotations, translations, points, _ = task.move(rotations, translations, action)
    angle, distance = task.error(rotations, translations)
    score = teacher_score(teacher, points, task.device)
    return {'episodes': episodes, 'initial_rotation_deg': float(initial_angle.mean()),
            'final_rotation_deg': float(angle.mean()), 'initial_translation_mm': float(initial_distance.mean()),
            'final_translation_mm': float(distance.mean()), 'initial_teacher_logit': float(initial_score.mean()),
            'final_teacher_logit': float(score.mean()),
            'success_rate': float(((angle < success_rotation_deg) &
                                   (distance < success_translation_mm)).mean()),
            'success_rotation_deg': success_rotation_deg,
            'success_translation_mm': success_translation_mm}


def evaluate_cases(agent, case_paths, make_task, teacher, episodes, horizon, seed,
                   success_rotation_deg, success_translation_mm):
    results = [evaluate(agent, make_task(path, index), teacher, episodes, horizon,
                        seed + index * 1009, success_rotation_deg, success_translation_mm)
               for index, path in enumerate(case_paths)]
    averaged = {key: float(np.mean([result[key] for result in results]))
                for key in ('initial_rotation_deg', 'final_rotation_deg',
                            'initial_translation_mm', 'final_translation_mm',
                            'initial_teacher_logit', 'final_teacher_logit', 'success_rate')}
    return {'episodes': episodes * len(case_paths), 'cases': len(case_paths), **averaged,
            'success_rotation_deg': success_rotation_deg,
            'success_translation_mm': success_translation_mm}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_options = json.loads(args.model_options)
    train_cases = json.loads(args.train_cases) if args.train_cases else [args.case]
    eval_paths = json.loads(args.eval_cases) if args.eval_cases else [args.case]
    if not train_cases or not eval_paths:
        raise ValueError('Train and evaluation case lists must be nonempty')
    if not args.eval_only and args.iterations < len(train_cases):
        raise ValueError('agent.iterations must be at least the number of training cases')

    def make_task(path, index):
        return PoseTask(path, args.points, args.seed, args.max_rotation, args.max_translation,
                        device, args.rotation_step_deg, args.translation_step_mm,
                        args.point_seed + index * 1009)

    def evaluate_split(seed):
        return evaluate_cases(agent, eval_paths, make_task, teacher,
                              args.eval_episodes, args.horizon, seed,
                              args.success_rotation_deg, args.success_translation_mm)
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
        rotations, translations, points = task.reset(args.envs)
        scores = teacher_score(teacher, points, device)
        rollout = []
        for step in range(args.horizon):
            obs = task.observation(points)
            with torch.no_grad():
                dist, value = agent.distribution(obs)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action).sum(-1)
            next_rot, next_trans, next_points, _ = task.move(rotations, translations, torch.tanh(raw_action).cpu().numpy())
            next_scores = teacher_score(teacher, next_points, device)
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
    parser.add_argument('--points', type=int, default=256)
    parser.add_argument('--teacher-samples', type=int, default=64)
    parser.add_argument('--max-rotation', type=float, default=8.0)
    parser.add_argument('--max-translation', type=float, default=4.0)
    parser.add_argument('--rotation-step-deg', type=float, default=2.0)
    parser.add_argument('--translation-step-mm', type=float, default=0.4)
    parser.add_argument('--point-seed', type=int, default=42)
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
    args = parser.parse_args()
    if args.iterations < 1 or args.envs < 1 or args.horizon < 1 or args.points < 1:
        parser.error('iterations, envs, horizon and points must be positive')
    if args.max_rotation <= 0 or args.max_translation <= 0 or args.log_every < 1:
        parser.error('max-rotation, max-translation and log-every must be positive')
    train(args)


if __name__ == '__main__':
    main()
