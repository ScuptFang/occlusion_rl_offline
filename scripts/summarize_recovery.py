"""Paired final-pose metrics, including displacement of original mesh vertices."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.data import load_case, candidate_indices
from occlusion_rl_offline.rl import action_steps


def final_pose(row, config):
    last = row['trace'][-1]
    if 'rotations' in last:
        return np.asarray(last['rotations']), np.asarray(last['translations'])
    r, t = np.asarray(row['initial_rotations']), np.asarray(row['initial_translations'])
    for step in row['trace'][:-1]:
        a = np.asarray(step['action'], dtype=np.float32)
        rs, ts = action_steps(config['agent'], step['step'])
        r = Rotation.from_rotvec(np.deg2rad(a[:, :3] * rs)).as_matrix() @ r
        t = t + a[:, 3:] * ts
    np.testing.assert_allclose(np.rad2deg(Rotation.from_matrix(r).magnitude()), last['rotation_deg'], atol=3e-5)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), last['translation_mm'], atol=3e-5)
    return r, t


def geometry_error(case, rotations, translations, options):
    xyz = case['xyz'].astype(np.float64)
    centered = xyz - case['xyz'].mean(0)
    delta = rotations - np.eye(3)
    second = centered.T @ centered / len(centered)
    shifted_mean = np.einsum('bij,j->bi', delta, centered.mean(0))
    rms = np.sqrt(np.maximum(0, np.einsum('bij,jk,bik->b', delta, second, delta)
                             + 2 * np.sum(shifted_mean * translations, axis=1)
                             + np.sum(translations ** 2, axis=1)))
    candidates = centered[candidate_indices(case, options)]
    displacements = np.einsum('bij,nj->bni', delta, candidates) + translations[:, None]
    return rms, np.quantile(np.linalg.norm(displacements, axis=2), .95, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', action='append', required=True, help='Label=JSON')
    p.add_argument('--cases', default='runs/data11_candidate4096_fps64_more/prepared/cases')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    results, reference, cache = [], {}, {}
    for spec in args.input:
        label, file = spec.split('=', 1)
        data = json.loads(Path(file).read_text()); rows = []
        for row in data['cases']:
            key = (row['case'], row['band'])
            current = np.concatenate((np.asarray(row['initial_rotations']).reshape(-1, 9), row['initial_translations']), axis=1)
            if key in reference:
                np.testing.assert_allclose(current, reference[key], atol=1e-7)
            else:
                reference[key] = current
            if row['case'] not in cache:
                cache[row['case']] = load_case(Path(args.cases) / row['case'] / 'case.npz')
            case = cache[row['case']]
            r, t = final_pose(row, data['config'])
            rms, p95 = geometry_error(case, r, t, data['config']['model'])
            irms, ip95 = geometry_error(case, np.asarray(row['initial_rotations']),
                                       np.asarray(row['initial_translations']), data['config']['model'])
            last = row['trace'][-1]
            summary = {'case': row['case'], 'band': row['band'], 'episodes': row['episodes'],
                       'vertex_rms_mm': float(rms.mean()), 'initial_vertex_rms_mm': float(irms.mean()),
                       'candidate_displacement_p95_mm': float(p95.mean()),
                       'initial_candidate_displacement_p95_mm': float(ip95.mean())}
            for name in ('rotation_deg', 'translation_mm'):
                for stat, fn in [('mean', np.mean), ('median', np.median), ('p90', lambda v: np.quantile(v, .9))]:
                    summary[name + '_' + stat] = float(fn(last[name]))
            rows.append(summary)
        fields = [k for k in rows[0] if k not in ('case', 'band', 'episodes')]
        results.append({'label': label, 'input': file, 'macro': {
            k: float(np.mean([r[k] for r in rows if r['band'] != 'zero'])) for k in fields}, 'cases': rows})
    Path(args.out).write_text(json.dumps({'note': 'Equal case/band macro; quantiles averaged across groups, not pooled. RMS is exact over all original vertices; displacement p95 uses 4096 uniform candidate vertices, not area-weighted surface samples.',
                                       'results': results}, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
