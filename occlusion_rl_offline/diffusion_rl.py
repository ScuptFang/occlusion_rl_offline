"""DPPO-inspired last-denoising-step PPO, after diffusion imitation learning.

The frozen prefix is deterministic DDIM. Tail kernels have strictly positive
Gaussian variance. Likelihoods use RAW latent samples, before action clipping.
"""
import copy
import math

import torch
from torch import nn
from torch.distributions import Normal

from .diffusion import DiffusionAgent


class DiffusionRLAgent(nn.Module):
    def __init__(self, a, model_options):
        super().__init__()
        self.actor = DiffusionAgent(a, model_options)
        self.reference = copy.deepcopy(self.actor)
        self.options = dict(a['diffusion_rl'])
        o = self.options
        if not 1 <= o['tail_steps'] <= a['diffusion']['inference_steps']:
            raise ValueError('tail_steps must fit inference_steps')
        if not 0 < o['eta'] <= 1 or not math.isfinite(o['eta']):
            raise ValueError('eta must be in (0,1]')
        if not 0 < o['min_std'] < 1 or not math.isfinite(o['min_std']):
            raise ValueError('min_std must be in (0,1)')
        for p in self.reference.parameters():
            p.requires_grad_(False)
        for p in self.actor.encoder.parameters():
            p.requires_grad_(False)
        width = a['diffusion']['hidden_width']
        self.critic = nn.Sequential(nn.Linear(width, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.time_context = True
        self.timesteps = torch.linspace(a['diffusion']['noise_steps'] - 1, 0,
                                        a['diffusion']['inference_steps']).round().long().tolist()
        self.tail_start = len(self.timesteps) - o['tail_steps']

    def initialize_imitation(self, state):
        self.actor.load_state_dict(state)
        self.reference.load_state_dict(state)

    def kernel(self, policy, x, index, condition, stochastic=True):
        t = self.timesteps[index]
        alpha = policy.alpha_bar[t]
        ts = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        eps = policy.denoise(x, ts, condition)
        clean = ((x - (1 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1, 1)
        if index == len(self.timesteps) - 1:
            return clean, torch.as_tensor(self.options['min_std'], device=x.device)
        previous = policy.alpha_bar[self.timesteps[index + 1]]
        variance = (1 - previous) / (1 - alpha) * (1 - alpha / previous)
        std = (self.options['eta'] * variance.clamp_min(0).sqrt()).clamp_min(self.options['min_std'])
        if not stochastic:
            std = torch.zeros_like(std)
        eps = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        mean = previous.sqrt() * clean + (1 - previous - std.square()).clamp_min(0).sqrt() * eps
        return mean, std

    @torch.no_grad()
    def sample_chain(self, observations, context, generator=None):
        condition = self.actor.encode(observations, context)
        prior = self.reference.encode(observations, context)
        shape = (len(observations), self.actor.prediction_horizon, 6)
        x = torch.randn(shape, device=observations.device, generator=generator)
        before, after, logprobs = [], [], []
        for i in range(len(self.timesteps)):
            if i < self.tail_start:
                x, _ = self.kernel(self.reference, x, i, prior, stochastic=False)
            else:
                mean, std = self.kernel(self.actor, x, i, condition)
                nx = mean + std * torch.randn(shape, device=x.device, generator=generator)
                before.append(x); after.append(nx)
                logprobs.append(Normal(mean, std).log_prob(nx).flatten(1).sum(-1))
                x = nx
        return (x.clamp(-1, 1), torch.stack(before, 1), torch.stack(after, 1),
                torch.stack(logprobs, 1), self.critic(condition).squeeze(-1))

    @torch.no_grad()
    def predict_action_chunk(self, observations, context, generator=None):
        return self.sample_chain(observations, context, generator)[0]

    def evaluate_chain(self, observations, context, before, after):
        condition = self.actor.encode(observations, context)
        with torch.no_grad():
            prior = self.reference.encode(observations, context)
        logprobs, kl = [], []
        for k in range(self.options['tail_steps']):
            i = self.tail_start + k
            mean, std = self.kernel(self.actor, before[:, k], i, condition)
            logprobs.append(Normal(mean, std).log_prob(after[:, k]).flatten(1).sum(-1))
            with torch.no_grad():
                reference_mean, _ = self.kernel(self.reference, before[:, k], i, prior)
            kl.append(((mean - reference_mean).square() / (2 * std.square())).flatten(1).sum(-1))
        return torch.stack(logprobs, 1), self.critic(condition.detach()).squeeze(-1), torch.stack(kl, 1)


def clipped_objective(logprobs, old_logprobs, advantages, clip):
    logratio = logprobs - old_logprobs
    ratio = logratio.exp()
    loss = -torch.minimum(ratio * advantages[:, None],
                          ratio.clamp(1 - clip, 1 + clip) * advantages[:, None]).mean()
    kl = ((ratio - 1) - logratio).mean()
    return loss, kl, ((ratio - 1).abs() > clip).float().mean()
