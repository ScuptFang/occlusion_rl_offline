"""Conditional action-chunk diffusion with one-action receding-horizon execution.

Small point-cloud/MLP adaptation of Diffusion Policy, not DPPO or an exact
reproduction of the original temporal U-Net. No target pose enters this policy.
"""
import math

import torch
from torch import nn

from .model import OcclusionTeacher


def validate_options(a):
    d = a['diffusion']
    for key in ('horizon', 'validation_episodes', 'test_episodes', 'torch_threads'):
        if isinstance(a[key], bool) or not isinstance(a[key], int) or a[key] < 1:
            raise ValueError('agent.%s must be a positive integer' % key)
    if isinstance(d['sampling_seed'], bool) or not isinstance(d['sampling_seed'], int) or d['sampling_seed'] < 0:
        raise ValueError('diffusion.sampling_seed must be a nonnegative integer')
    for key in ('action_rotation_step_deg', 'action_translation_step_mm'):
        if not math.isfinite(a[key]) or a[key] <= 0:
            raise ValueError('agent.%s must be finite and positive' % key)
    for key in ('prediction_horizon', 'noise_steps', 'inference_steps', 'hidden_width',
                'hidden_layers', 'time_embed_dim', 'training_steps', 'batch_size',
                'episodes_per_case', 'collection_batch', 'validate_every'):
        if isinstance(d[key], bool) or not isinstance(d[key], int) or d[key] < 1:
            raise ValueError('diffusion.%s must be a positive integer' % key)
    if not 2 <= d['inference_steps'] <= d['noise_steps']:
        raise ValueError('Require 2 <= inference_steps <= noise_steps')
    if d['time_embed_dim'] < 4 or d['time_embed_dim'] % 2:
        raise ValueError('time_embed_dim must be even and >= 4')
    for key in ('learning_rate', 'encoder_learning_rate', 'max_grad_norm'):
        if not math.isfinite(d[key]) or d[key] <= 0:
            raise ValueError('diffusion.%s must be finite and positive' % key)
    if not 0 <= d['ema_decay'] < 1 or not 0 <= d['random_action_probability'] <= 1:
        raise ValueError('Invalid EMA decay or collection random-action probability')
    if d['weight_decay'] < 0 or not math.isfinite(d['weight_decay']):
        raise ValueError('Invalid diffusion weight_decay')
    if a.get('score_gated_actions', False):
        raise ValueError('Diffusion must execute score-decreasing actions')
    if not a.get('time_context', False) or a['horizon'] < 1:
        raise ValueError('Diffusion requires time context and a positive horizon')


def cosine_alpha_bar(steps):
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    curve = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    beta = (1 - curve[1:] / curve[:-1]).clamp(max=.999)
    return (1 - beta).cumprod(0).float()


class ConditionalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, width * 2)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x, condition):
        scale, shift = self.film(condition).chunk(2, -1)
        return x + self.net(self.norm(x) * (1 + scale) + shift)


class DiffusionAgent(nn.Module):
    def __init__(self, a, model_options):
        super().__init__()
        validate_options(a)
        self.options = dict(a['diffusion'])
        d = self.options
        self.time_context = True
        self.prediction_horizon = d['prediction_horizon']
        self.encoder = OcclusionTeacher(sample_count=a['fps_points'], **model_options)
        for p in self.encoder.head.parameters():
            p.requires_grad_(False)
        width = d['hidden_width']
        self.condition = nn.Sequential(nn.LayerNorm(model_options.get('width', 64) * 5 + 6),
                                       nn.Linear(model_options.get('width', 64) * 5 + 6, width), nn.SiLU())
        self.time_projection = nn.Sequential(nn.Linear(d['time_embed_dim'], width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Linear(self.prediction_horizon * 6, width)
        self.blocks = nn.ModuleList([ConditionalBlock(width) for _ in range(d['hidden_layers'])])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, self.prediction_horizon * 6))
        self.register_buffer('alpha_bar', cosine_alpha_bar(d['noise_steps']))
        frequencies = torch.exp(-math.log(10000) * torch.arange(d['time_embed_dim'] // 2) /
                                (d['time_embed_dim'] // 2 - 1))
        self.register_buffer('frequencies', frequencies)

    def encode(self, observations, context):
        if context is None:
            raise ValueError('Diffusion requires explicit step context')
        return self.condition(torch.cat((self.encoder.encode(observations), context), -1))

    def denoise(self, noisy, timesteps, condition):
        phase = timesteps[:, None].float() * self.frequencies[None]
        embedding = self.time_projection(torch.cat((phase.sin(), phase.cos()), -1))
        h = self.input(noisy.flatten(1))
        for block in self.blocks:
            h = block(h, condition + embedding)
        return self.output(h).reshape_as(noisy)

    def noise_loss(self, observations, context, actions, noise=None, timesteps=None):
        if tuple(actions.shape[1:]) != (self.prediction_horizon, 6):
            raise ValueError('Expected a [batch, prediction_horizon, 6] action target')
        if noise is None:
            noise = torch.randn_like(actions)
        if timesteps is None:
            timesteps = torch.randint(len(self.alpha_bar), (len(actions),), device=actions.device)
        alpha = self.alpha_bar[timesteps][:, None, None]
        noisy = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
        predicted = self.denoise(noisy, timesteps, self.encode(observations, context))
        return (predicted - noise).square().mean()

    @torch.no_grad()
    def predict_action_chunk(self, observations, context, generator=None, initial_noise=None):
        """DDIM eta=0; randomness enters only through the initial Gaussian noise.

        Recompute epsilon after clipping x0 so the DDIM direction is consistent
        with bounded physical actions, including at the terminal t=0 update.
        """
        condition = self.encode(observations, context)
        shape = (len(observations), self.prediction_horizon, 6)
        x = (torch.randn(shape, device=observations.device, generator=generator)
             if initial_noise is None else initial_noise.clone())
        if tuple(x.shape) != shape:
            raise ValueError('Initial noise has incorrect shape')
        timesteps = torch.linspace(len(self.alpha_bar) - 1, 0,
                                   self.options['inference_steps']).round().long().tolist()
        for i, t in enumerate(timesteps):
            ts = torch.full((len(x),), t, dtype=torch.long, device=x.device)
            alpha = self.alpha_bar[t]
            epsilon = self.denoise(x, ts, condition)
            clean = ((x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
            if i == len(timesteps) - 1:
                x = clean
            else:
                previous = self.alpha_bar[timesteps[i + 1]]
                epsilon = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                x = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        return x
