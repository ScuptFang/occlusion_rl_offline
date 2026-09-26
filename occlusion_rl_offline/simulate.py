"""Create a reproducible observed upper pose while keeping the lower SDF fixed."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .data import load_case, sample_sdf


def perturb_case(case_path, out_case, rotvec_deg, translation_mm, upper_stl=None,
                 out_stl=None):
    case = load_case(case_path)
    rotvec = np.asarray(rotvec_deg, dtype=np.float64)
    translation = np.asarray(translation_mm, dtype=np.float64)
    if rotvec.shape != (3,) or translation.shape != (3,):
        raise ValueError('rotvec_deg and translation_mm must contain three numbers')
    rotation = Rotation.from_rotvec(np.deg2rad(rotvec)).as_matrix()
    pivot = case['xyz'].mean(0).astype(np.float64)
    xyz = ((case['xyz'] - pivot) @ rotation.T + pivot + translation).astype(np.float32)
    normal = (case['normal'] @ rotation.T).astype(np.float32)
    # Check grid coverage before writing a case that the teacher cannot score.
    sample_sdf(case['sdf_grid'], case['origin'], case['spacing'], xyz)
    out_case = Path(out_case)
    out_case.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_case, xyz=xyz, normal=normal, curvature=case['curvature'],
                        sdf_grid=case['sdf_grid'], origin=case['origin'], spacing=case['spacing'])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot + translation - rotation @ pivot
    if upper_stl is not None:
        if out_stl is None:
            out_stl = out_case.parent / 'upper_perturbed.stl'
        mesh = trimesh.load(str(upper_stl), force='mesh', process=False)
        mesh.apply_transform(matrix)
        mesh.export(str(out_stl))
    metadata = {'observed_case': str(out_case), 'input_to_perturbed_matrix': matrix.tolist(),
                'rotation_vector_deg': rotvec.tolist(), 'translation_mm': translation.tolist(),
                'upper_perturbed_stl': None if out_stl is None else str(out_stl)}
    (out_case.parent / (out_case.stem + '.perturbation.json')).write_text(json.dumps(metadata, indent=2))
    return metadata
