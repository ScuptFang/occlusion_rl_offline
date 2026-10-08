"""Export actual teacher scores for independent pose sweeps (no agent)."""
from occlusion_rl_offline.runtime import resolve_device
import argparse
import hashlib
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.rl import PoseTask, load_teacher, teacher_score
from occlusion_rl_offline.data import candidate_indices
from generate import surface_points
from generate_multi_pose import frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/optimization_selected.json')
    p.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    p.add_argument('--teacher', default=None)
    p.add_argument('--cases', nargs='+', default=None)
    p.add_argument('--out', default='demo/teacher_scores_data.js')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    device = resolve_device(cfg,args.device)
    path = resolve(args.teacher or cfg['paths']['packaged_teacher'])
    teacher = load_teacher(path, device, cfg['teacher']['fps_points'], **cfg['model'])
    payload = dict(cases={}, trajectories=[], teacher_checkpoint=str(path),
                   teacher_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   config_path=args.config, seed=20261008)
    manifest = read_manifest(resolve(args.manifest))
    for split, paths in manifest.items():
        for case in paths:
            case_id = case.parent.name
            if args.cases and case_id not in args.cases:
                continue
            task = PoseTask(case, cfg['teacher']['fps_points'], 42, 8, 4, device, fps_options=cfg['model'])
            rng = np.random.default_rng(20261008 + int(case_id))
            jaw = resolve('../data') / case_id
            ids = candidate_indices(task.case, cfg['model'])
            geometry = dict(pivot=task.pivot.tolist(), candidate_upper=task.xyz[ids].round(3).tolist(),
                            upper_mesh_vertices=len(task.xyz), split=split,
                            upper=surface_points(jaw/'upper.stl', 3000, rng),
                            lower=surface_points(jaw/'lower.stl', 3000, rng))
            task.begin_episode()
            original = task.points_at(np.eye(3)[None], np.zeros((1, 3)))
            geometry['reference_score'] = float(teacher_score(teacher, original, device)[0])
            payload['cases'][case_id] = geometry
            for axis in range(7):
                count = 41 if axis < 6 else 33
                rv, t = np.zeros((count, 3)), np.zeros((count, 3))
                if axis < 3:
                    values = np.linspace(-10, 10, count)
                    rv[:, axis] = values
                    name, unit = '绕' + 'XYZ'[axis] + '轴旋转', '°'
                elif axis < 6:
                    values = np.linspace(-4, 4, count)
                    t[:, axis-3] = values
                    name, unit = '沿' + 'XYZ'[axis-3] + '轴平移', 'mm'
                else:
                    values = np.arange(count)
                    rv[1:] = rng.uniform(-8, 8, (count-1, 3))
                    t[1:] = rng.uniform(-4, 4, (count-1, 3))
                    name, unit = '随机组合姿态', '号'
                r = Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
                steps = []
                for start in range(0, count, 16):
                    task.begin_episode()
                    rr, tt = r[start:start+16], t[start:start+16]
                    features = task.points_at(rr, tt)
                    scores = teacher_score(teacher, features, device)
                    for i in range(len(rr)):
                        record = frame(task, rr, tt, features, scores, i)
                        record.update(value=float(values[start+i]), rotvec=rv[start+i].tolist())
                        steps.append(record)
                payload['trajectories'].append(dict(case_id=case_id, name=name, unit=unit,
                                                      steps=steps, reference_index=20 if axis < 6 else 0))
            print(f'{case_id} ({split}): 279 poses, original score {geometry["reference_score"]:.4f}', flush=True)
    if not payload['cases']:
        raise ValueError('No matching cases')
    resolve(args.out).write_text('window.DEMO_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';\n')


if __name__ == '__main__':
    main()
