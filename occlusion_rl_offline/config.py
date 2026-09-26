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
    for stage in ('teacher', 'agent'):
        values = config[stage]
        if not 0 < values['fps_points'] <= values['candidate_points']:
            raise ValueError('%s.fps_points must be in (0, candidate_points]' % stage)
    if config['teacher']['candidate_points'] != config['agent']['candidate_points'] or \
            config['teacher']['fps_points'] != config['agent']['fps_points']:
        raise ValueError('Teacher and agent point densities must match for this experiment')
    return config


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
