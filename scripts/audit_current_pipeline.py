"""Read-only data audit, local teacher scans, and zero-pose agent rollouts."""
import argparse
import hashlib
import json

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import load_case, sample_sdf, sample_upper_mesh
from occlusion_rl_offline.rl import Agent, PoseTask, load_teacher


def quantiles(values):
    return dict(zip(('min', 'p50', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .5, .95, 1]))))


def data_audit(path, data_dir, catalog):
    case = load_case(path)
    number = path.parent.name
    row = {'meshes': {}}
    for jaw, key in (('upper', 'xyz'), ('lower', 'lower_xyz')):
        source = data_dir / number / (jaw + '.stl')
        mesh = trimesh.load(str(source), force='mesh', process=True)
        edge_counts = np.bincount(mesh.edges_unique_inverse)
        boundary_ids = np.unique(mesh.edges_unique[edge_counts == 1])
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        row['meshes'][jaw] = {
            'sha256_matches_catalog': hashlib.sha256(source.read_bytes()).hexdigest()
                == catalog['files'][jaw]['sha256'],
            'prepared_vertices_match': bool(np.array_equal(vertices, case[key])),
            'vertices': len(vertices), 'faces': len(mesh.faces),
            'extent_mm': np.ptp(vertices, axis=0).tolist(),
            'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'boundary_vertex_fraction': len(boundary_ids) / len(vertices),
            'nonmanifold_edges': int((edge_counts > 2).sum()),
            'degenerate_faces': int((mesh.area_faces < 1e-12).sum()),
        }
        if jaw == 'upper':
            row['meshes'][jaw]['curvature_boundary_p50'] = (
                float(np.median(case['curvature'][boundary_ids])) if len(boundary_ids) else None)
    lower = case['lower_xyz']
    rng = np.random.RandomState(812)
    probe = rng.choice(len(lower), min(4096, len(lower)), replace=False)
    spacing = cKDTree(lower).query(lower[probe], k=2)[0][:, 1]
    upper_ids = rng.choice(len(case['xyz']), min(4096, len(case['xyz'])), replace=False)
    d = sample_sdf(case, case['xyz'][upper_ids])
    row.update(lower_vertex_spacing_mm=quantiles(spacing),
               original_abs_distance_mm=quantiles(np.abs(d)),
               original_negative_distance_fraction=float((d < 0).mean()),
               original_contact_0_6mm_fraction=float((np.abs(d) < .6).mean()),
               upper_curvature=quantiles(case['curvature']),
               centroid_upper_minus_lower_mm=(case['xyz'].mean(0) - lower.mean(0)).tolist())
    return row


@torch.no_grad()
def predict(model, features):
    return model(torch.from_numpy(np.asarray(features, dtype=np.float32))).numpy()


def local_audit(path, cfg, teachers, agent):
    a = cfg['agent']
    task = PoseTask(path, a['fps_points'], a['seed'], a['start_rotation_deg'],
                    a['start_translation_mm'], 'cpu', a['action_rotation_step_deg'],
                    a['action_translation_step_mm'], cfg['model'])
    c = task.case
    base, ids = sample_upper_mesh(c, task.xyz, task.normal, task.points,
                                   cfg['model'], return_indices=True)
    records, dynamic, fixed = [], [], []
    for axis in range(6):
        values = [-2, -1, -.5, -.1, -.05, 0, .05, .1, .5, 1, 2] if axis < 3 else [
            -1, -.5, -.25, -.05, -.01, 0, .01, .05, .25, .5, 1]
        for offset in values:
            rv, t = np.zeros(3), np.zeros(3)
            (rv if axis < 3 else t)[axis % 3] = offset
            rot = Rotation.from_rotvec(np.deg2rad(rv)).as_matrix()
            xyz = (task.xyz - task.pivot) @ rot.T + task.pivot + t
            normal = task.normal @ rot.T
            f, selected = sample_upper_mesh(c, xyz, normal, task.points,
                                            cfg['model'], return_indices=True)
            held = np.column_stack((xyz[ids], normal[ids], c['curvature'][ids].reshape(-1),
                                     sample_sdf(c, xyz[ids]))).astype(np.float32)
            dynamic.append(f)
            fixed.append(held)
            records.append({'axis': ['rx', 'ry', 'rz', 'tx', 'ty', 'tz'][axis],
                            'offset': offset,
                            'selected_set_replaced_fraction': 1 - len(np.intersect1d(ids, selected)) / len(ids)})
    for name, teacher in teachers.items():
        ds, fs = predict(teacher, dynamic), predict(teacher, fixed)
        for row, dynamic_score, fixed_score in zip(records, ds, fs):
            row[name + '_dynamic_score'] = float(dynamic_score)
            row[name + '_fixed_ids_score'] = float(fixed_score)
    rotations, translations = np.eye(3)[None], np.zeros((1, 3))
    points = base[None]
    teacher = teachers['calibration']
    score = predict(teacher, points)
    rollout = [{'step': 0, 'score': float(score[0]), 'rotation_deg': 0., 'translation_mm': 0.}]
    for step in range(a['horizon']):
        with torch.no_grad():
            action = torch.tanh(agent(task.observation(points))[0]).numpy()
        rotations, translations, points, score, _, _ = task.move_scored(
            rotations, translations, points, score, action, teacher, score_gate=False)
        angle, distance = task.error(rotations, translations)
        rollout.append({'step': step + 1, 'score': float(score[0]),
                        'rotation_deg': float(angle[0]), 'translation_mm': float(distance[0]),
                        'action': action[0].tolist()})
    return {'axis_scans': records, 'zero_pose_rollout': rollout,
            'fixed_ids_note': 'Diagnostic only: indices fixed at original pose; normal training resamples.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/data11_contact_fps.json')
    p.add_argument('--manifest', default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    p.add_argument('--data-dir', default='../data')
    p.add_argument('--local-cases', nargs='+', default=['001', '006', '010', '011'])
    p.add_argument('--out', default='runs/data11_contact_fps/pipeline_audit.json')
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = load_config(args.config)
    manifest = read_manifest(resolve(args.manifest))
    catalog = {r['number']: r for r in json.loads(resolve('docs/data_catalog.json').read_text())['pairs']}
    run = resolve(cfg['paths']['run_dir'])
    checkpoints = {s: run / 'teacher' / s / 'best.pt' for s in ('ranking', 'calibration')}
    teachers = {s: load_teacher(f, 'cpu', cfg['teacher']['fps_points'], **cfg['model'])
                for s, f in checkpoints.items()}
    a = cfg['agent']
    agent_file = run / 'agent/agent.pt'
    agent = Agent(a['fps_points'], a['policy_width'], a['initial_log_std'], **cfg['model'])
    agent.load_state_dict(torch.load(agent_file, map_location='cpu')['model'])
    agent.eval()
    result = {'config': cfg, 'manifest': args.manifest,
              'checkpoint_sha256': {s: hashlib.sha256(f.read_bytes()).hexdigest()
                                     for s, f in dict(checkpoints, agent=agent_file).items()},
              'data': {}, 'local': {}}
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for split, paths in manifest.items():
        for path in paths:
            case_id = path.parent.name
            row = data_audit(path, resolve(args.data_dir), catalog[case_id])
            result['data'][case_id] = dict(row, split=split, source=catalog[case_id]['source'])
            if case_id in args.local_cases:
                result['local'][case_id] = local_audit(path, cfg, teachers, agent)
            out.write_text(json.dumps(result, indent=2) + '\n')
            print(case_id, split, 'audited', flush=True)
    print(out)


if __name__ == '__main__':
    main()
