"""Mathematical and behavior regressions for continuous teacher-only PPO."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.data import sample_upper_mesh
from occlusion_rl_offline.inference import _matrix
from occlusion_rl_offline.optimized_ppo import advantages_and_returns
from occlusion_rl_offline.rl import PoseTask, Agent, action_steps, step_context


class ContinuousTests(unittest.TestCase):
    def test_finite_returns_and_telescoping_reward(self):
        scores=torch.tensor([[-2.,-3.],[-1.,-4.],[-1.5,-2.]])
        rewards=list(scores[1:]-scores[:-1]);zero=[torch.zeros(2),torch.zeros(2)]
        adv,returns=advantages_and_returns(rewards,zero,torch.zeros(2),1.,1.)
        torch.testing.assert_close(returns[0],scores[-1]-scores[0])
        torch.testing.assert_close(returns[-1],rewards[-1])
        self.assertLess(float(rewards[0][1]),0)  # Negative reward is retained.

    def test_candidates_transform_and_export(self):
        rng=np.random.RandomState(8);xyz=rng.normal(size=(256,3)).astype(np.float32)
        normal=xyz/np.linalg.norm(xyz,axis=1,keepdims=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'case.npz'
            np.savez(p,xyz=xyz,normal=normal,curvature=np.arange(256,dtype=np.float32)/256,
                     lower_xyz=xyz-2,lower_normal=normal)
            options={'candidate_points':128,'point_seed':42,'contact_fps_fraction':.5}
            task=PoseTask(p,16,42,8,4,'cpu',fps_options=options)
            r=Rotation.from_rotvec([[.1,.2,-.1],[-.2,.1,.3]]).as_matrix();t=rng.normal(size=(2,3))
            actual=task.points_at(r,t)
            full=np.einsum('nj,bij->bni',xyz-task.pivot,r)+task.pivot+t[:,None]
            ns=np.einsum('nj,bij->bni',normal,r)
            expected=np.array([sample_upper_mesh(task.case,x,n,16,options) for x,n in zip(full,ns)])
            np.testing.assert_array_equal(actual,expected)
            m=_matrix(r[0],t[0],task.pivot)
            transformed=np.column_stack((xyz,np.ones(len(xyz))))@m.T
            np.testing.assert_allclose(transformed[:,:3],full[0],atol=1e-6)
            recovered=transformed@np.linalg.inv(m).T
            np.testing.assert_allclose(recovered[:,:3],xyz,atol=1e-6)
            fixed=PoseTask(p,16,42,8,4,'cpu',fps_options=dict(options,sampling_mode='episode_fixed'))
            start=fixed.points_at(r,t);ids=fixed.episode_ids.copy()
            np.testing.assert_array_equal(start,actual)
            moved=fixed.points_at(r,t+.1)
            np.testing.assert_array_equal(ids,fixed.episode_ids)
            np.testing.assert_allclose(moved[:,:,:3],start[:,:,:3]+.1,atol=1e-6)
            np.testing.assert_array_equal(moved[:,:,6],start[:,:,6])
            fixed.begin_episode();self.assertIsNone(fixed.episode_ids)

    def test_schedule_never_becomes_stop(self):
        opt={'action_schedule':[{'until_step':10,'rotation_deg':2,'translation_mm':.4},
                                {'until_step':20,'rotation_deg':.5,'translation_mm':.1}]}
        self.assertEqual(action_steps(opt,0),(2,.4))
        self.assertEqual(action_steps(opt,100),(.5,.1))
        a=Agent(16,32,-1.,time_context=True,width=8,k=4,encoder_layers=1)
        x=torch.randn(2,16,8)
        with self.assertRaises(ValueError):a(x)
        for step in (0,10,39):
            action,value=a(x,step_context(a,2,step,40,'cpu'))
            self.assertEqual(tuple(action.shape),(2,6))
            self.assertTrue(torch.isfinite(action).all())

    def test_frozen_encoding_preserves_policy_gradients(self):
        torch.manual_seed(1)
        a=Agent(16,32,-1.,time_context=True,width=8,k=4,encoder_layers=1)
        for p in a.encoder.parameters():p.requires_grad_(False)
        x=torch.randn(3,16,8);ctx=step_context(a,3,2,10,'cpu')
        mean,value=a(x,ctx);loss=mean.square().mean()+value.square().mean();loss.backward()
        gradients={n:p.grad.clone() for n,p in a.named_parameters() if p.grad is not None}
        a.zero_grad()
        with torch.no_grad():encoded=a.encoder.encode(x)
        m,v=a.forward_encoded(encoded,ctx);(m.square().mean()+v.square().mean()).backward()
        for n,p in a.named_parameters():
            if n in gradients:torch.testing.assert_close(p.grad,gradients[n])

if __name__=='__main__':unittest.main()
