"""Teacher scoring and deterministic agent rollout on an observed upper pose."""
from .runtime import resolve_device
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from .rl import PoseTask, load_teacher, teacher_score, action_steps, step_context
from .evaluation import load_agent


def _task(case_path, cfg, device):
    agent = cfg['agent']
    return PoseTask(case_path, agent['fps_points'], agent['seed'],
                    agent['start_rotation_deg'], agent['start_translation_mm'],
                    device, agent['action_rotation_step_deg'],
                    agent['action_translation_step_mm'], cfg['model'])


def score_case(case_path, teacher_path, cfg):
    device = resolve_device(cfg)
    task = _task(case_path, cfg, device)
    teacher = load_teacher(teacher_path, device, cfg['teacher']['fps_points'],
                           **cfg['model'])
    points = task.points_at(np.eye(3)[None], np.zeros((1, 3)))
    return {'teacher_logit': float(teacher_score(teacher, points, device)[0]),
            'upper_mesh_vertices': len(task.xyz),
            'weighted_fps_points': teacher.sample_count}


def _matrix(rotation, translation, pivot):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot + translation - rotation @ pivot
    return matrix


def infer_agent(case_path, teacher_path, agent_path, out_dir, cfg, upper_stl=None, steps=None):
    device = resolve_device(cfg)
    task = _task(case_path, cfg, device)
    teacher = load_teacher(teacher_path, device, cfg['teacher']['fps_points'],
                           **cfg['model'])
    policy = load_agent(agent_path, cfg, device, teacher_path)
    rotation = np.eye(3)[None]
    translation = np.zeros((1, 3))
    points = task.points_at(rotation, translation)
    score = teacher_score(teacher, points, device)
    history = [{'step': 0, 'teacher_logit': float(score[0]),
                'matrix': _matrix(rotation[0], translation[0], task.pivot).tolist()}]
    horizon = cfg['inference']['steps'] if steps is None else steps
    for step in range(horizon):
        task.rotation_step_deg, task.translation_step_mm = action_steps(cfg['agent'], step)
        with torch.no_grad():
            context = step_context(policy, len(points), step, horizon, device)
            action = torch.tanh(policy(task.observation(points), context)[0]).cpu().numpy()
        rotation, translation, points, score, valid, accepted = task.move_scored(
            rotation, translation, points, score, action, teacher,
            cfg['agent'].get('score_gated_actions', False))
        history.append({'step': step + 1, 'teacher_logit': float(score[0]),
                        'action': action[0].tolist(), 'valid_sdf': bool(valid[0]),
                        'accepted_action': bool(accepted[0]),
                        'matrix': _matrix(rotation[0], translation[0], task.pivot).tolist()})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = _matrix(rotation[0], translation[0], task.pivot)
    result = {'case': str(case_path), 'teacher_checkpoint': str(teacher_path),
              'agent_checkpoint': str(agent_path), 'steps': horizon,
              'initial_teacher_logit': history[0]['teacher_logit'],
              'final_teacher_logit': history[-1]['teacher_logit'],
              'input_to_recovered_matrix': final.tolist(), 'trajectory': history,
              'note': 'No original-pose ground truth is used during inference.'}
    if upper_stl is not None:
        mesh = trimesh.load(str(upper_stl), force='mesh', process=False)
        mesh.apply_transform(final)
        mesh.export(str(out / 'upper_recovered.stl'))
        result['upper_recovered_stl'] = str(out / 'upper_recovered.stl')
    (out / 'trajectory.json').write_text(json.dumps(result, indent=2))
    return result
