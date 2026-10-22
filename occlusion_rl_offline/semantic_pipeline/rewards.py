"""Training-only expert trajectory alignment, smoothly replaced by teacher delta."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..imitation import expert_actions


def validate(options):
    if not 0 <= options['decay_start'] < options['decay_end'] <= 1:
        raise ValueError('Require 0 <= decay_start < decay_end <= 1')
    if not 0 <= options['expert_weight_end'] <= options['expert_weight_start'] <= 1:
        raise ValueError('Expert weight must decrease within [0,1]')
    if any(not np.isfinite(options[k]) or options[k] <= 0 for k in ('expert_scale', 'teacher_scale')):
        raise ValueError('Reward scales must be finite and positive')


def weights(options, iteration, iterations):
    validate(options)
    progress = iteration / max(iterations - 1, 1)
    x = np.clip((progress-options['decay_start'])/(options['decay_end']-options['decay_start']), 0, 1)
    x = x*x*(3-2*x)
    w = options['expert_weight_start'] + x*(options['expert_weight_end']-options['expert_weight_start'])
    return float(w), float(1-w)


def trajectory_reward(r, t, nr, nt, score, next_score, rotation_step, translation_step,
                      options, iteration, iterations):
    """Replan bounded expert's next waypoint at each learner-visited state.

    Reward is improvement in normalized distance to that waypoint, not a BC loss.
    The waypoint is fixed for the before/after comparison; no target enters obs.
    """
    action = expert_actions(r, t, rotation_step, translation_step)
    er = Rotation.from_rotvec(np.deg2rad(action[:, :3]*rotation_step)).as_matrix() @ r
    et = t + action[:, 3:]*translation_step
    def error(rr, tt):
        angle = np.rad2deg(Rotation.from_matrix(rr @ er.transpose(0,2,1)).magnitude())
        return .5*(angle/(np.sqrt(3)*rotation_step) + np.linalg.norm(tt-et, axis=1)/(np.sqrt(3)*translation_step))
    expert = options['expert_scale']*(error(r,t)-error(nr,nt))
    teacher = options['teacher_scale']*(np.asarray(next_score)-score)
    ew, tw = weights(options, iteration, iterations)
    return ew*expert+tw*teacher, expert, teacher, ew, tw
