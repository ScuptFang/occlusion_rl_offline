import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.aligned_teacher import pose_quality, pose_pairs, pair_loss
from occlusion_rl_offline.config import load_config
from occlusion_rl_offline.data import sample_sdf, features_at_pose, load_case
from occlusion_rl_offline.rl import PoseTask
from occlusion_rl_offline.surface import surface_distance
from occlusion_rl_offline.simulate import perturb_case


class TeacherRefinementTests(unittest.TestCase):
    def mesh(self):
        xyz = np.array([[0,0,0],[2,0,0],[0,2,0],[2,2,0]], dtype=np.float32)
        normals = np.tile([0,0,1.], (4,1)).astype(np.float32)
        return dict(lower_xyz=xyz, lower_normal=normals, lower_faces=np.array([[0,1,2],[1,3,2]]),
                    xyz=xyz+np.array([0,0,.2],dtype=np.float32), normal=normals.copy(),
                    curvature=np.arange(4,dtype=np.float32)/4)

    def test_surface_distance_and_interpolated_vertex_sign(self):
        case=self.mesh(); points=np.array([[.4,.4,.1],[.4,.4,-.1],[3,0,.1]])
        d,conf=surface_distance(case,points)
        np.testing.assert_allclose(d,[.1,-.1,np.sqrt(1.01)],rtol=1e-6)
        self.assertGreater(sample_sdf(case,points)[0],d[0])
        self.assertFalse(conf.any())  # All vertices touch the open boundary.
        # Deliberately tilted vertex normals prove sign uses vertex interpolation,
        # not the +Z triangle normal.
        case['lower_normal'][:]=[1,0,0]
        d,_=surface_distance(case,np.array([[3,0,-.1]]))
        self.assertGreater(d[0],0)
        del case['lower_faces']
        with self.assertRaisesRegex(ValueError,'rerun prepare'):
            sample_sdf(case,points,{'distance_method':'nearest_surface'})

    def test_degenerate_faces_are_excluded(self):
        case=self.mesh()
        case['lower_faces']=np.vstack(([0,0,1],case['lower_faces']))
        d,_=surface_distance(case,np.array([[.4,.4,.1]]))
        np.testing.assert_allclose(d,[.1])
        self.assertEqual(case['_surface_degenerate_faces'],1)
        case=self.mesh();case['lower_faces'][0,0]=100
        with self.assertRaisesRegex(ValueError,'indices'):
            surface_distance(case,np.array([[0,0,1.]]))

    def test_surface_distance_independent_of_tessellation(self):
        coarse=self.mesh();refined=self.mesh()
        refined['lower_xyz']=np.vstack((refined['lower_xyz'],[1,1,0]))
        refined['lower_normal']=np.vstack((refined['lower_normal'],[0,0,1]))
        refined['lower_faces']=np.array([[0,1,4],[1,3,4],[3,2,4],[2,0,4]])
        points=np.array([[1,1,.1],[.3,.7,-.2]])
        np.testing.assert_allclose(surface_distance(coarse,points)[0],
                                   surface_distance(refined,points)[0],atol=1e-7)
        self.assertNotAlmostEqual(sample_sdf(coarse,points)[0],sample_sdf(refined,points)[0])

    def test_pose_and_delta_loss(self):
        options=load_config('configs/teacher_aligned_surface.json')['teacher']['aligned_pair']
        r=Rotation.from_rotvec(np.deg2rad([3,4,0])).as_matrix()
        self.assertAlmostEqual(pose_quality(r,[0,0,2],[8,4]),-1.125)
        pairs=pose_pairs(32,42,options)
        for r,t,r2,t2,y,local,near in pairs:
            np.testing.assert_allclose(y,[pose_quality(r,t,[8,4]),pose_quality(r2,t2,[8,4])])
            np.testing.assert_allclose(r2@r2.T,np.eye(3),atol=1e-10)
        target=torch.tensor([[-1.,-.95],[-.1,-.1]])
        self.assertEqual(float(pair_loss(target,target,options,'calibration')),0.)
        wrong=target.flip(1).clone().requires_grad_()
        loss=pair_loss(wrong,target,options,'calibration');self.assertGreater(float(loss.detach()),0.)
        loss.backward();self.assertTrue(torch.isfinite(wrong.grad).all())
        # A very small quality gap does not demand a fixed 0.5 score margin.
        close=torch.tensor([[-1.,-.99999]])
        self.assertEqual(float(pair_loss(close,close,options,'ranking')),0.)

    def test_training_inference_geometry_and_perturbation(self):
        case=self.mesh();options={'distance_method':'nearest_surface','candidate_points':4}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'case.npz';np.savez(path,**case)
            task=PoseTask(path,4,42,8,4,'cpu',fps_options=options)
            r=Rotation.from_rotvec([[.01,.02,0]]).as_matrix();t=np.array([[.1,0,.1]])
            expected=features_at_pose(load_case(path),r[0],t[0],4,options)
            np.testing.assert_allclose(task.points_at(r,t)[0],expected,atol=1e-7)
            out=Path(d)/'perturbed.npz';perturb_case(path,out,[1,2,3],[.1,.2,.3])
            np.testing.assert_array_equal(load_case(out)['lower_faces'],case['lower_faces'])


if __name__=='__main__':unittest.main()
