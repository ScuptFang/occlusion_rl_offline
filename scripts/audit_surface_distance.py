"""Compare nearest vertex and triangle-surface distances on prepared cases."""
import argparse
import json
import time

import numpy as np

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import resolve
from occlusion_rl_offline.data import load_case, sample_sdf
from occlusion_rl_offline.surface import surface_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='runs/teacher_refinement/prepared/manifest.json')
    p.add_argument('--out', default='runs/teacher_refinement/geometry_audit.json')
    args = p.parse_args()
    result = []
    for split, paths in read_manifest(resolve(args.manifest)).items():
        for path in paths:
            case = load_case(path)
            ids = np.random.RandomState(42).choice(len(case['xyz']), min(512, len(case['xyz'])), False)
            points = case['xyz'][ids]
            vertex = sample_sdf(case, points)
            start = time.perf_counter()
            surface, confidence = surface_distance(case, points)
            if not np.isfinite(surface).all() or not np.all(np.abs(surface) <= np.abs(vertex) + 1e-5):
                raise AssertionError('Invalid closest surface distance')
            row = {'case': path.parent.name, 'split': split, 'vertices': len(case['lower_xyz']),
                   'faces': len(case['lower_faces']), 'finite': True,
                   'mean_vertex_abs_mm': float(np.abs(vertex).mean()),
                   'mean_surface_abs_mm': float(np.abs(surface).mean()),
                   'mean_absolute_distance_reduction_mm': float((np.abs(vertex)-np.abs(surface)).mean()),
                   'sign_disagreement_fraction': float(((vertex < 0) != (surface < 0)).mean()),
                   'surface_sign_confident_fraction': float(confidence.mean()),
                   'degenerate_faces_excluded': case['_surface_degenerate_faces'],
                   'seconds': time.perf_counter() - start}
            result.append(row)
            print(json.dumps(row), flush=True)
    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
