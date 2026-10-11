"""Configuration loading and paths relative to the repository root."""
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'configs/default.json'


def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text())
    config.setdefault('cohort', {'upper_filename': 'upper.stl', 'lower_filename': 'lower.stl',
                                 'split_seed': 42, 'validation_fraction': 0.1,
                                 'test_fraction': 0.1})
    for section in ('paths', 'prepare', 'model', 'teacher', 'agent', 'inference'):
        if section not in config:
            raise ValueError('Missing config section: %s' % section)
    layers = config['model'].get('encoder_layers', 3)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError('model.encoder_layers must be a positive integer')
    for stage in ('teacher', 'agent'):
        values = config[stage]
        if not isinstance(values['fps_points'], int) or values['fps_points'] < 1:
            raise ValueError('%s.fps_points must be a positive integer' % stage)
    if config['teacher']['fps_points'] != config['agent']['fps_points']:
        raise ValueError('Teacher and agent point densities must match for this experiment')
    candidates = config['model'].get('candidate_points', 4096)
    if config['model'].get('sampling_mode','dynamic') not in ('dynamic','episode_fixed'):
        raise ValueError('model.sampling_mode must be dynamic or episode_fixed')
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < config['teacher']['fps_points']:
        raise ValueError('model.candidate_points must be an integer >= fps_points')
    fraction = config['model'].get('contact_fps_fraction', 0.0)
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 <= fraction <= 1:
        raise ValueError('model.contact_fps_fraction must be between 0 and 1')
    pool_factor = config['model'].get('contact_fps_pool_factor', 4)
    if fraction and (config['model'].get('contact_fps_max_mm', 0.6) <= 0 or
                     isinstance(pool_factor, bool) or not isinstance(pool_factor, int) or pool_factor < 1):
        raise ValueError('Contact FPS threshold and pool factor must be positive')
    a = config['agent']
    if a.get('trainer') in ('continuous', 'imitation_curriculum'):
        if a.get('score_gated_actions', False):
            raise ValueError('Continuous PPO must execute score-decreasing actions')
        if a.get('reward_scale', 1) <= 0 or not 0 < a.get('gamma', 1) <= 1:
            raise ValueError('Positive reward scale and gamma in (0,1] required')
        if len(a.get('error_scales', [8., 4.])) != 2 or min(a.get('error_scales', [8., 4.])) <= 0:
            raise ValueError('Two positive continuous error scales required')
        previous = 0
        for stage in a.get('action_schedule', []):
            if stage['until_step'] <= previous or min(stage['rotation_deg'], stage['translation_mm']) <= 0:
                raise ValueError('Action schedule requires increasing steps and positive scales')
            previous = stage['until_step']
        for key in ('horizon', 'parallel_envs', 'iterations', 'ppo_epochs', 'minibatch', 'log_every'):
            if not isinstance(a[key], int) or a[key] < 1:
                raise ValueError('agent.%s must be a positive integer' % key)
    if a.get('trainer') == 'imitation_curriculum':
        from .imitation import validate_curriculum
        validate_curriculum(a['imitation'])
        if a['iterations'] < 2:
            raise ValueError('Curriculum needs at least two PPO iterations')
    return config


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
