"""Generate the browser replay from a trained agent on one case."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from occlusion_rl_offline.rl import Agent, PoseTask, load_teacher, teacher_score


def surface_points(path, count, rng):
    mesh = trimesh.load_mesh(path, process=False)
    areas = mesh.area_faces
    ids = rng.choice(len(mesh.faces), count, p=areas / areas.sum())
    triangles = mesh.vertices[mesh.faces[ids]]
    u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    return np.round((1-u)*triangles[:, 0] + u*(1-v)*triangles[:, 1] + u*v*triangles[:, 2], 3).tolist()


def record(task, teacher, config, rotation, translation, points, score, action=None, accepted=None):
    angle, distance = task.error(rotation, translation)
    return {'rotation': np.round(rotation[0], 7).reshape(-1).tolist(),
            'translation': np.round(translation[0], 5).tolist(),
            'rotation_error': round(float(angle[0]), 4),
            'translation_error': round(float(distance[0]), 4),
            'teacher_logit': round(float(score[0]), 5),
            'action': None if action is None else np.round(action[0], 5).tolist(),
            'accepted_action': accepted,
            'selected_points': np.round(points[0, :, :3], 3).tolist(),
            'selected_indices': list(range(len(points[0]))),
            'selected_sdf': np.round(points[0, :, 7], 4).tolist()}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=root/'runs/case010_mesh128_20260930/case.npz')
    parser.add_argument('--upper', type=Path, default=root.parent/'data/010/upper.stl')
    parser.add_argument('--lower', type=Path, default=root.parent/'data/010/lower.stl')
    parser.add_argument('--config', type=Path, default=root/'configs/case010_mesh128.json')
    parser.add_argument('--teacher', type=Path, default=root/'runs/case010_mesh128_20260930/teacher/calibration/best.pt')
    parser.add_argument('--agent', type=Path, default=root/'runs/case010_mesh128_20260930/agent/agent.pt')
    parser.add_argument('--out', type=Path, default=root/'demo/data.js')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--surface-points', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    ac = cfg['agent']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = PoseTask(args.case, ac['fps_points'], ac['seed'], ac['start_rotation_deg'],
                    ac['start_translation_mm'], device, ac['action_rotation_step_deg'],
                    ac['action_translation_step_mm'], cfg['model'])
    task.rng = np.random.RandomState(args.seed)
    teacher = load_teacher(args.teacher, device, cfg['teacher']['fps_points'], **cfg['model'])
    agent = Agent(ac['fps_points'], ac['policy_width'], ac['initial_log_std'], **cfg['model']).to(device)
    agent.load_state_dict(torch.load(args.agent, map_location=device)['model'])
    agent.eval()
    trajectories = []
    for i in range(args.cases):
        rotation, translation, points = task.reset(1)
        score = teacher_score(teacher, points, device)
        steps = [record(task, teacher, cfg, rotation, translation, points, score)]
        for _ in range(cfg['inference']['steps']):
            with torch.no_grad():
                action = torch.tanh(agent(task.observation(points))[0]).cpu().numpy()
            rotation, translation, points, score, _, accepted = task.move_scored(
                rotation, translation, points, score, action, teacher,
                ac.get('score_gated_actions', False))
            steps.append(record(task, teacher, cfg, rotation, translation, points, score,
                                action, bool(accepted[0])))
        trajectories.append({'name': f'{args.upper.parent.name} · 随机扰动 {i+1:02d}', 'steps': steps})
        print(f'{trajectories[-1]["name"]}: {steps[0]["rotation_error"]:.2f}° / '
              f'{steps[0]["translation_error"]:.2f} mm → '
              f'{steps[-1]["rotation_error"]:.2f}° / '
              f'{steps[-1]["translation_error"]:.2f} mm; '
              f'accepted {sum(s["accepted_action"] is True for s in steps)}/{len(steps)-1}', flush=True)
    rng = np.random.default_rng(args.seed)
    data = {'seed': args.seed, 'case_id': args.upper.parent.name,
            'score_gated_actions': ac.get('score_gated_actions', False),
            'pivot': np.round(task.pivot, 6).tolist(),
            'candidate_upper': [], 'upper_mesh_vertices': len(task.xyz),
            'upper': surface_points(args.upper, args.surface_points, rng),
            'lower': surface_points(args.lower, args.surface_points, rng),
            'trajectories': trajectories,
            'action_rotation_step_deg': ac['action_rotation_step_deg'],
            'action_translation_step_mm': ac['action_translation_step_mm'],
            'success_threshold': {'rotation_deg': ac['success_rotation_deg'],
                                  'translation_mm': ac['success_translation_mm']}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('window.DEMO_DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
