import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.region_mesh import recover_mesh, upper_mesh_candidates, mesh_probabilities
from occlusion_rl_offline.region_segmentation import RegionPointTransformer, digest
from occlusion_rl_offline.prepare import prepare_case
from occlusion_rl_offline.data import features_at_pose, load_case
from occlusion_rl_offline.rl import PoseTask
from occlusion_rl_offline.simulate import perturb_case


class RegionMeshTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_topology_and_candidates_after_region(self):
        xyz = np.array([[0,0,0], [1,0,0], [0,1,0], [4,0,0], [5,0,0], [4,1,0]], np.float32)
        case = dict(xyz=xyz, normal=np.tile([0,0,1], (6,1)), curvature=np.arange(6),
                    upper_faces=np.array([[0,1,2], [3,4,5]]))
        probability = np.array([.9,.9,.9,.1,.1,.1])
        region = dict(mesh_before_sampling=True, threshold=.5, face_rule='all', minimum_upper_faces=1)
        mesh = recover_mesh(case, 0, probability, region)
        np.testing.assert_array_equal(mesh['vertex_ids'][mesh['faces']], case['upper_faces'][mesh['face_ids']])
        np.testing.assert_array_equal(mesh['xyz'], xyz[mesh['vertex_ids']])
        np.testing.assert_array_equal(mesh['curvature'], case['curvature'][mesh['vertex_ids']])
        opts = dict(segmentation_region=region, candidate_points=3, point_seed=42)
        with patch('occlusion_rl_offline.region_mesh.mesh_probabilities', return_value=probability):
            ids = upper_mesh_candidates(case, opts, np.eye(3), np.zeros(3))
            self.assertEqual(set(ids), {0,1,2})
            with self.assertRaisesRegex(ValueError, 'no duplication'):
                upper_mesh_candidates(case, dict(opts, candidate_points=4), np.eye(3), np.zeros(3))

    def test_interpolation_cache_does_not_freeze_pose_predictions(self):
        class NormalModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1))
            def forward(self, x):
                return 4*x[..., 3] + self.anchor
        xyz = np.random.RandomState(1).normal(size=(40,3)).astype(np.float32)
        case = dict(xyz=xyz, normal=np.tile([1.,0.,0.], (40,1)), curvature=np.zeros(40))
        with patch('occlusion_rl_offline.region_mesh.predictor', return_value=(NormalModel(), dict(points=12,candidates=24,seed=42))):
            a=mesh_probabilities(case,0,dict(sha256='test'),np.eye(3),np.zeros(3))
            r=Rotation.from_euler('z',180,degrees=True).as_matrix()
            b=mesh_probabilities(case,0,dict(sha256='test'),r,np.ones(3))
            self.assertTrue((a>.9).all());self.assertTrue((b<.1).all())
            self.assertEqual(len(case['_mesh_interpolation']),1)

    def test_prepare_perturb_and_teacher_agent_sampling_match(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mesh=trimesh.creation.icosphere(subdivisions=2)
            mesh.export(root/'upper.stl');mesh.apply_translation([0,0,3]);mesh.export(root/'lower.stl')
            path=root/'case.npz'
            case,_=prepare_case(root/'upper.stl',root/'lower.stl',path,dict(model=dict(distance_method='nearest_surface')))
            self.assertIn('upper_faces',case)
            perturb_case(path,root/'observed.npz',[2,3,4],[.2,.3,.4])
            np.testing.assert_array_equal(load_case(root/'observed.npz')['upper_faces'],case['upper_faces'])
            model=RegionPointTransformer(width=8,layers=1,k=4)
            with torch.no_grad(): model.head[-1].weight.zero_(); model.head[-1].bias.fill_(4)
            checkpoint=root/'segment.pt'
            torch.save(dict(model=model.state_dict(),architecture=dict(width=8,layers=1,k=4),
                            sampling=dict(points=16,candidates=32,seed=42)),checkpoint)
            opts=dict(candidate_points=64,point_seed=42,distance_method='nearest_surface',segmentation_region=dict(
                checkpoint=str(checkpoint),sha256=digest(checkpoint),mesh_before_sampling=True,face_rule='all',
                minimum_upper_faces=1,minimum_lower_faces=1))
            r=Rotation.from_rotvec([.1,.2,.3]).as_matrix();t=np.array([.2,-.1,.3])
            features,ids=features_at_pose(case,r,t,16,opts,True)
            task=PoseTask(path,16,42,8,4,'cpu',fps_options=opts)
            np.testing.assert_array_equal(task.points_at(r[None],t[None])[0],features)
            np.testing.assert_array_equal(task.last_ids[0],ids)
            self.assertEqual(case['_upper_region_audit']['candidates'],64)
            self.assertEqual(len(set(ids)),16)
            np.testing.assert_allclose(features[:,:3],(case['xyz'][ids]-task.pivot)@r.T+task.pivot+t,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
