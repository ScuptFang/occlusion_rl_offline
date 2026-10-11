"""Geometry, reward and checkpoint regressions for imitation curriculum."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.imitation import (expert_actions, curriculum_weights,
    mixed_reward, validate_curriculum, cloning_loss)
from occlusion_rl_offline.rl import PoseTask

CONFIG = Path(__file__).resolve().parents[1] / 'configs/reagent_curriculum.json'


class ImitationTests(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(CONFIG.read_text())['agent']
        self.il = self.a['imitation']

    def test_inverse_uses_actual_environment_composition(self):
        rng = np.random.RandomState(121)
        xyz = rng.normal(size=(80, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'case.npz'
            np.savez(path, xyz=xyz, normal=xyz / np.linalg.norm(xyz,axis=1,keepdims=True),
                     curvature=np.zeros(80,dtype=np.float32),lower_xyz=xyz-2,
                     lower_normal=np.tile([0.,0.,1.],(80,1)))
            task = PoseTask(path, 8, 1, 8, 4, 'cpu', fps_options={'candidate_points':64})
            # Include a large arbitrary-axis rotation and exact identity.
            r = Rotation.from_rotvec([[.3,-.2,.4],[.001,.002,-.001],[0,0,0]]).as_matrix()
            t = np.array([[.5,-.3,2],[.01,-.02,.01],[0,0,0]])
            before = task.error(r,t)
            for _ in range(20):
                action = expert_actions(r,t,task.rotation_step_deg,task.translation_step_mm)
                self.assertLessEqual(float(abs(action).max()), 1.)
                r,t,_,valid = task.move(r,t,action)
                after=task.error(r,t)
                self.assertTrue(valid.all())
                self.assertTrue(np.all(after[0] <= before[0]+1e-6))
                self.assertTrue(np.all(after[1] <= before[1]+1e-7))
                before=after
            self.assertLess(max(before[0]),1e-5)
            self.assertLess(max(before[1]),1e-6)

    def test_curriculum_endpoints_and_monotonicity(self):
        weights=np.array([curriculum_weights(self.il,i,168) for i in range(168)])
        np.testing.assert_allclose(weights[0],[1.,.5])
        np.testing.assert_allclose(weights[-1],[0.,0.])
        self.assertTrue((np.diff(weights,axis=0)<=1e-12).all())
        self.assertTrue((weights[134:]==0).all())
        bad=copy.deepcopy(self.il);bad['decay_start']=bad['decay_end']
        with self.assertRaises(ValueError):validate_curriculum(bad)
        bad=copy.deepcopy(self.il);bad['gt_weight_end']=1.1
        with self.assertRaises(ValueError):validate_curriculum(bad)

    def test_reward_sign_scale_and_telescoping(self):
        task=object.__new__(PoseTask)
        r=np.repeat(np.eye(3)[None],2,axis=0)
        t=np.array([[1.,0,0],[2.,0,0]])
        nt=np.array([[.5,0,0],[3.,0,0]])
        scores=np.array([0.,0.]);next_scores=np.array([-1.,1.])
        gt,_,_=mixed_reward(task,r,t,r,nt,scores,next_scores,self.a,self.il,1.)
        self.assertGreater(gt[0],0);self.assertLess(gt[1],0)
        teacher,_,_=mixed_reward(task,r,t,r,nt,scores,next_scores,self.a,self.il,0.)
        np.testing.assert_allclose(teacher,[-2.,2.])
        combined,_,_=mixed_reward(task,r,t,r,nt,scores,next_scores,self.a,self.il,.3)
        np.testing.assert_allclose(combined,.3*gt+.7*teacher)
        reverse,_,_=mixed_reward(task,r,nt,r,t,next_scores,scores,self.a,self.il,.3)
        np.testing.assert_allclose(combined+reverse,0.)

    def test_full_pipeline_resume_and_inference_without_pose_labels(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from occlusion_rl_offline.optimized_ppo import run
        from occlusion_rl_offline.model import OcclusionTeacher
        from occlusion_rl_offline.evaluation import load_agent
        cfg=json.loads(CONFIG.read_text())
        cfg['runtime']={'device':'cpu'}
        cfg['model']={'width':8,'k':4,'encoder_layers':1,'candidate_points':64}
        cfg['teacher']['fps_points']=8
        cfg['agent'].update(fps_points=8,policy_width=16,parallel_envs=2,horizon=3,
            iterations=3,ppo_epochs=1,minibatch=6,validation_episodes=1,log_every=1,
            freeze_encoder_iterations=1,torch_threads=1)
        cfg['agent']['imitation'].update(iterations=2,epochs=1,batch_size=6)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            root=Path(folder);rng=np.random.RandomState(55)
            splits={}
            for split,case_id in zip(('train','val','test'),('001','002','003')):
                directory=root/case_id;directory.mkdir()
                xyz=rng.normal(size=(80,3)).astype(np.float32)
                np.savez(directory/'case.npz',xyz=xyz,
                    normal=xyz/np.linalg.norm(xyz,axis=1,keepdims=True),
                    curvature=np.zeros(80,dtype=np.float32),lower_xyz=xyz-2,
                    lower_normal=np.tile([0.,0.,1.],(80,1)))
                splits[split]=[{'id':case_id,'case':case_id+'/case.npz'}]
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'splits':splits}))
            teacher_path=root/'teacher.pt'
            model=OcclusionTeacher(sample_count=8,**cfg['model'])
            torch.save({'model':model.state_dict(),'samples':8,'model_options':cfg['model']},teacher_path)
            run(cfg,manifest,teacher_path,root/'full')
            real_save=torch.save
            class Interrupted(Exception):pass
            def interrupt_after_checkpoint(obj,path,*args,**kwargs):
                real_save(obj,path,*args,**kwargs)
                if Path(path).name=='agent.pt' and obj['iteration']==1:
                    raise Interrupted()
            with patch('torch.save',side_effect=interrupt_after_checkpoint):
                with self.assertRaises(Interrupted):run(cfg,manifest,teacher_path,root/'resumed')
            run(cfg,manifest,teacher_path,root/'resumed',root/'resumed/latest.pt')
            full=torch.load(root/'full/latest.pt',weights_only=True)
            resumed=torch.load(root/'resumed/latest.pt',weights_only=True)
            for key,value in full['model'].items():
                torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(full['history'],resumed['history'])
            self.assertEqual(full['history'][-1]['curriculum']['gt_weight'],0.)
            self.assertEqual(full['history'][-1]['curriculum']['bc_weight'],0.)
            self.assertEqual(full['history'][-1]['interactions'],18)
            agent=load_agent(root/'resumed/agent.pt',cfg,teacher_path=teacher_path)
            with torch.no_grad():
                mean,_=agent(torch.randn(2,8,8),torch.tensor([[0.,1.],[0.,1.]]))
            self.assertTrue(torch.isfinite(mean).all())
            changed=copy.deepcopy(cfg);changed['agent']['iterations']=4
            with self.assertRaisesRegex(ValueError,'curriculum/options mismatch'):
                run(changed,manifest,teacher_path,root/'invalid',root/'resumed/latest.pt')

    def test_cloning_has_finite_gradient_at_bounded_targets(self):
        mean=torch.zeros(3,6,requires_grad=True)
        loss=cloning_loss(mean,torch.ones(3,6)*.999)
        loss.backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue((mean.grad<0).all())

if __name__=='__main__':unittest.main()
