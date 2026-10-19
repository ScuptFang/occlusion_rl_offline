"""Export complete held-out recovery traces; do not select successful episodes."""
import argparse
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.data import load_case, candidate_indices, sample_sdf
from occlusion_rl_offline.evaluation import BANDS
from occlusion_rl_offline.rl import load_teacher, teacher_score, action_steps
from occlusion_rl_offline.region_mesh import enabled, upper_mesh_candidates


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='runs/region_ranking_agent/diffusion_rl_config.json')
    p.add_argument('--manifest', default='runs/region_point_transformer/manifest.json')
    p.add_argument('--evaluation', default='runs/region_ranking_agent/rl/test/agent')
    p.add_argument('--sampling-seed', type=int, default=91001)
    p.add_argument('--out', default='demo/ranking_recovery_data.js')
    p.add_argument('--examples-per-band', type=int, default=32,
                   help='Show the first N episodes in each band, never select by outcome')
    args = p.parse_args()
    if not 1 <= args.examples_per_band <= 32:
        p.error('examples-per-band must be between 1 and 32')
    cfg = load_config(args.config); a = cfg['agent']; torch.set_num_threads(4)
    evaluation = resolve(args.evaluation)
    # A completed seed can be replayed while other frozen-policy seeds are still
    # evaluating; both files contain the same locked checkpoint/config identity.
    identity = evaluation / 'comparison.json'
    if not identity.exists():
        identity = evaluation / 'locked_checkpoints.json'
    comparison = json.loads(identity.read_text())
    manifest = resolve(args.manifest)
    assert comparison['manifest_sha256'] == digest(manifest)
    assert comparison['config']['model'] == cfg['model']
    assert comparison['config']['agent'] == a
    for key in ('teacher', 'agent'):
        assert comparison[key + '_sha256'] == digest(resolve(cfg['paths']['packaged_' + key]))
    traces = json.loads((evaluation / ('policy_%d_trajectories.json' % args.sampling_seed)).read_text())
    paths = read_manifest(manifest)
    test = {path.parent.name: path for path in paths['test']}
    assert not set(test) & {path.parent.name for path in paths['train'] + paths['val']}
    assert {row['case'] for row in traces['cases']} == set(test)
    teacher = load_teacher(resolve(cfg['paths']['packaged_teacher']), 'cpu', a['fps_points'], **cfg['model'])
    payload = dict(title='未校准 Teacher · 未参与训练病例恢复',
        subtitle='010 / 011 未参与训练，曾用于测试。完整展示随机起点，包括恢复不佳的轨迹。',
        teacher_label='Ranking 原始分数 · 未经过 calibration',
        config_path=args.config, score_gated_actions=False, cases={}, trajectories=[],
        seed=traces['seed'], sampling_seed=traces['sampling_seed'], fps_points=a['fps_points'],
        action_rotation_step_deg=a['action_rotation_step_deg'],
        action_translation_step_mm=a['action_translation_step_mm'],
        checkpoint_sha256={key: comparison[key + '_sha256'] for key in ('teacher', 'agent')})
    if enabled(cfg['model']):
        payload.update(title='原网格咬合区域 · %d 候选 / %d 点恢复' % (cfg['model']['candidate_points'], a['fps_points']),
            subtitle='010 / 011 未参与训练。展示各档前 %d 个随机起点及正确位姿；没有按恢复效果筛选。' % args.examples_per_band,
            mesh_before_sampling=True, candidate_points=cfg['model']['candidate_points'])
    max_score_difference = 0.; action_count = negative_count = 0
    for case_id, path in test.items():
        case = load_case(path); pivot = case['xyz'].mean(0)
        rng = np.random.RandomState(421 + int(case_id))
        def display_vertices(key):
            vertices = case[key]
            ids = rng.choice(len(vertices), min(10000, len(vertices)), replace=False)
            return np.round(vertices[ids], 3).tolist()
        payload['cases'][case_id] = dict(pivot=pivot.tolist(), upper=display_vertices('xyz'),
            lower=display_vertices('lower_xyz'), upper_mesh_vertices=len(case['xyz']),
            candidate_upper=[] if enabled(cfg['model']) else np.round(case['xyz'][candidate_indices(case, cfg['model'])], 3).tolist())
        for row in [r for r in traces['cases'] if r['case'] == case_id]:
            count = min(row['episodes'], args.examples_per_band)
            row = dict(row, episodes=count,
                       trace=[{k: v[:count] if isinstance(v, list) else v for k, v in f.items()}
                              for f in row['trace']])
            np.testing.assert_array_equal(pivot, row['pivot'])
            band = next(b for b in BANDS if b[0] == row['band'])
            episodes = [[] for _ in range(row['episodes'])]
            for index, frame in enumerate(row['trace']):
                ids = np.asarray(frame['selected_vertex_ids'])
                r, t = np.asarray(frame['rotations']), np.asarray(frame['translations'])
                assert ids.shape == (row['episodes'], a['fps_points'])
                xyz = (np.einsum('bnj,bij->bni', case['xyz'][ids] - pivot, r) + pivot + t[:, None]).astype(np.float32)
                normal = np.einsum('bnj,bij->bni', case['normal'][ids], r)
                sdf = sample_sdf(case, xyz.reshape(-1, 3), cfg['model']).reshape(ids.shape)
                features = np.concatenate((xyz, normal, case['curvature'][ids].reshape(*ids.shape, 1), sdf[..., None]), -1).astype(np.float32)
                scores = teacher_score(teacher, features, 'cpu')
                delta = float(np.max(np.abs(scores - frame['teacher_score'])))
                max_score_difference = max(max_score_difference, delta)
                np.testing.assert_allclose(scores, frame['teacher_score'], atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(np.rad2deg(Rotation.from_matrix(r).magnitude()), frame['rotation_deg'], atol=1e-6)
                np.testing.assert_allclose(np.linalg.norm(t, axis=1), frame['translation_mm'], atol=1e-6)
                previous = row['trace'][index - 1] if index else None
                if previous:
                    action = np.asarray(previous['action']); rs, ts = action_steps(a, index - 1)
                    assert all(previous['accepted_action']) and all(previous['valid_action'])
                    np.testing.assert_allclose(Rotation.from_rotvec(np.deg2rad(action[:, :3] * rs)).as_matrix() @ np.asarray(previous['rotations']), r, atol=1e-7)
                    np.testing.assert_allclose(np.asarray(previous['translations']) + action[:, 3:] * ts, t, atol=1e-7)
                    action_count += len(action)
                    negative_count += int((np.asarray(frame['teacher_score']) < previous['teacher_score']).sum())
                for i in range(row['episodes']):
                    region_detail = {}
                    if enabled(cfg['model']):
                        candidates = upper_mesh_candidates(case, cfg['model'], r[i], t[i])
                        assert np.isin(ids[i], candidates).all(), 'Recorded FPS vertices escaped the recovered region candidates'
                        preview = (case['xyz'][candidates[:512]] - pivot) @ r[i].T + pivot + t[i]
                        region_detail = dict(candidate_count=len(candidates), candidate_preview_points=np.round(preview,3).tolist(),
                            region_vertices=case['_upper_region_audit']['vertices'], region_faces=case['_upper_region_audit']['faces'])
                    episodes[i].append(dict(rotation=r[i].reshape(-1).tolist(), translation=t[i].tolist(),
                        rotation_error=frame['rotation_deg'][i], translation_error=frame['translation_mm'][i],
                        teacher_logit=frame['teacher_score'][i], selected_indices=ids[i].tolist(),
                        selected_points=np.round(xyz[i], 3).tolist(), selected_sdf=np.round(sdf[i], 4).tolist(),
                        action=previous['action'][i] if previous else None,
                        accepted_action=True if previous else None,
                        rotation_step_deg=rs if previous else None, translation_step_mm=ts if previous else None, **region_detail))
            for i, steps in enumerate(episodes):
                payload['trajectories'].append(dict(case_id=case_id, band_name=band[0],
                    band=dict(rotation_deg=band[1], translation_mm=band[2]),
                    test_episode_index=i, steps=steps))
            print(case_id, band[0], len(episodes), 'trajectories verified', flush=True)
    payload['audit'] = dict(trajectory_count=len(payload['trajectories']), actions=action_count,
        score_decreasing_actions=negative_count, max_score_reconstruction_difference=max_score_difference,
        source_trace_sha256=digest(evaluation / ('policy_%d_trajectories.json' % args.sampling_seed)))
    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('window.DEMO_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n')
    out.with_suffix('.audit.json').write_text(json.dumps(payload['audit'], indent=2) + '\n')
    print(json.dumps(payload['audit']), flush=True)


if __name__ == '__main__':
    main()
