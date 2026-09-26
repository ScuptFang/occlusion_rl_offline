"""Case-directory discovery and leakage-free cohort manifests."""
import json
from pathlib import Path

import numpy as np

from .data import load_case
from .prepare import prepare_case


def _case_dirs(root, upper_name, lower_name):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Case directory does not exist: %s' % root)
    found = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        upper, lower = folder / upper_name, folder / lower_name
        if upper.is_file() and lower.is_file():
            found.append((folder.name, upper, lower))
        elif upper.is_file() or lower.is_file():
            raise ValueError('Incomplete upper/lower STL pair in %s' % folder)
    if not found:
        raise ValueError('No case folders with %s and %s under %s' % (upper_name, lower_name, root))
    return found


def split_cases(cases, seed=42, validation_fraction=0.1, test_fraction=0.1):
    """Split whole cases, never poses or points from the same case."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError('Validation and test fractions must be between 0 and 1')
    if validation_fraction + test_fraction >= 1 or len(cases) < 3:
        raise ValueError('At least three cases and a positive train fraction are required')
    shuffled = list(cases)
    np.random.RandomState(seed).shuffle(shuffled)
    nval = max(1, round(len(cases) * validation_fraction))
    ntest = max(1, round(len(cases) * test_fraction))
    if nval + ntest >= len(cases):
        raise ValueError('Fractions leave no training cases')
    return {'train': shuffled[nval + ntest:], 'val': shuffled[:nval],
            'test': shuffled[nval:nval + ntest]}


def write_manifest(cases, path, seed=42, validation_fraction=0.1, test_fraction=0.1):
    """Write portable paths relative to the manifest file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [case_id for case_id, _ in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case IDs are not allowed')
    resolved_paths = [Path(case_path).resolve() for _, case_path in cases]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError('The same case file cannot be indexed under multiple IDs')
    for _, case_path in cases:
        load_case(case_path)
    split = split_cases(cases, seed, validation_fraction, test_fraction)
    import os
    manifest = {'version': 1, 'seed': seed,
                'validation_fraction': validation_fraction, 'test_fraction': test_fraction,
                'splits': {key: [{'id': case_id,
                                   'case': os.path.relpath(case_path, path.parent)}
                                  for case_id, case_path in values]
                           for key, values in split.items()}}
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('version') != 1 or set(data.get('splits', {})) != {'train', 'val', 'test'}:
        raise ValueError('Expected a version 1 manifest with train, val and test splits')
    seen = set()
    seen_paths = set()
    result = {}
    for split, entries in data['splits'].items():
        if not entries:
            raise ValueError('Empty %s split' % split)
        paths = []
        for item in entries:
            case_id = item['id']
            if case_id in seen:
                raise ValueError('Case ID occurs in more than one split: %s' % case_id)
            seen.add(case_id)
            case = (path.parent / item['case']).resolve()
            if not case.is_file():
                raise FileNotFoundError(case)
            if case in seen_paths:
                raise ValueError('Case file occurs in more than one split: %s' % case)
            seen_paths.add(case)
            paths.append(case)
        result[split] = paths
    return result


def prepare_batch(root, out, cfg, upper_name='upper.stl', lower_name='lower.stl',
                  seed=42, validation_fraction=0.1, test_fraction=0.1, skip_existing=False):
    root, out = Path(root), Path(out)
    case_dirs = _case_dirs(root, upper_name, lower_name)
    cases = []
    for case_id, upper, lower in case_dirs:
        target = out / 'cases' / case_id / 'case.npz'
        if skip_existing and target.is_file():
            load_case(target)
        else:
            prepare_case(upper, lower, target, cfg)
        cases.append((case_id, target.resolve()))
    manifest_path = out / 'manifest.json'
    manifest = write_manifest(cases, manifest_path, seed, validation_fraction, test_fraction)
    return manifest_path.resolve(), {key: len(value) for key, value in manifest['splits'].items()}


def index_prepared(root, out, seed=42, validation_fraction=0.1, test_fraction=0.1):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Prepared case directory does not exist: %s' % root)
    cases = [(folder.name, (folder / 'case.npz').resolve())
             for folder in sorted(root.iterdir()) if folder.is_dir() and (folder / 'case.npz').is_file()]
    if not cases:
        raise ValueError('Expected case_id/case.npz folders under %s' % root)
    manifest = write_manifest(cases, out, seed, validation_fraction, test_fraction)
    return Path(out).resolve(), {key: len(value) for key, value in manifest['splits'].items()}
