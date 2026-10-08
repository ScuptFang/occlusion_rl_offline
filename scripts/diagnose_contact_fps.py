"""Compare distance-weighted FPS and the contact-quota variant on all cases."""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import load_case, sample_upper_mesh


def summarize(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-config', default='configs/data11_agent_multi_pose.json')
    parser.add_argument('--new-config', default='configs/data11_contact_fps.json')
    parser.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    parser.add_argument('--poses-per-case', type=int, default=5)
    parser.add_argument('--seed', type=int, default=99000)
    parser.add_argument('--out', default='runs/data11_contact_fps/sampling_diagnostics.json')
    args = parser.parse_args()
    old, new = load_config(args.old_config), load_config(args.new_config)
    count = new['teacher']['fps_points']
    if count != old['teacher']['fps_points'] or old['model']['candidate_points'] != new['model']['candidate_points']:
        raise ValueError('Compare with the same FPS and candidate counts')
    split = read_manifest(resolve(args.manifest))
    cases = sorted((path for paths in split.values() for path in paths), key=lambda path: path.parent.name)
    result = {'seed': args.seed, 'poses_per_case': args.poses_per_case,
              'old_config': args.old_config, 'new_config': args.new_config, 'cases': {}}
    all_rows = {'old': [], 'new': []}
    for path in cases:
        case = load_case(path)
        xyz, normal = case['xyz'], case['normal']
        pivot = xyz.mean(0)
        rng = np.random.RandomState(args.seed + int(path.parent.name))
        rows = {'old': [], 'new': []}
        for _ in range(args.poses_per_case):
            angles = rng.uniform(-4, 4, 3)
            trans = rng.uniform(-2, 2, 3)
            rotation = Rotation.from_rotvec(np.deg2rad(angles)).as_matrix().astype(np.float32)
            posed = (xyz - pivot) @ rotation.T + pivot + trans
            posed_normal = normal @ rotation.T
            for label, options in (('old', old['model']), ('new', new['model'])):
                features = sample_upper_mesh(case, posed, posed_normal, count, options)
                candidate_indices = case['_uniform_candidates'][(options['candidate_points'],
                                                                 options['point_seed'])]
                coverage = cKDTree(features[:, :3]).query(posed[candidate_indices])[0]
                distance = np.abs(features[:, 7])
                record = {'within_0_2mm': float((distance < .2).mean()),
                          'within_0_6mm': float((distance < .6).mean()),
                          'median_distance_mm': float(np.median(distance)),
                          'coverage_p95_mm': float(np.quantile(coverage, .95))}
                rows[label].append(record)
                all_rows[label].append(record)
        result['cases'][path.parent.name] = {label: summarize(items) for label, items in rows.items()}
        print(path.parent.name, result['cases'][path.parent.name], flush=True)
    result['overall'] = {label: summarize(items) for label, items in all_rows.items()}
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('overall', result['overall'], flush=True)
    print(out)


if __name__ == '__main__':
    main()
