import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal

from test_diffusion import small_config
from occlusion_rl_offline.config import load_config
from occlusion_rl_offline.diffusion import DiffusionAgent
from occlusion_rl_offline.diffusion_rl import DiffusionRLAgent, clipped_objective
from occlusion_rl_offline.evaluation import load_agent, evaluate_continuous
from occlusion_rl_offline.model import OcclusionTeacher
from occlusion_rl_offline.train_diffusion_rl import run


class DiffusionRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(2)
        self.cfg = small_config(); self.cfg['agent']['trainer'] = 'diffusion_rl'
        o = copy.deepcopy(load_config('configs/filtered8_diffusion_rl.json')['agent']['diffusion_rl'])
        o.update(tail_steps=2, iterations=2, parallel_envs=2, validate_every=1, minibatch=3)
        self.cfg['agent']['diffusion_rl'] = o

    def test_raw_chain_likelihood_and_frozen_reference(self):
        c = self.cfg; m = DiffusionRLAgent(c['agent'], c['model'])
        obs = torch.randn(4, 8, 8); ctx = torch.zeros(4, 2)
        plan, before, after, old, _ = m.sample_chain(obs, ctx)
        lp, _, kl = m.evaluate_chain(obs, ctx, before, after)
        torch.testing.assert_close(lp, old)
        torch.testing.assert_close(kl, torch.zeros_like(kl))
        torch.testing.assert_close(plan, after[:, -1].clamp(-1, 1))
        self.assertTrue((plan.abs() <= 1).all())
        cond = m.actor.encode(obs, ctx)
        mu, std = m.kernel(m.actor, before[:, 0], m.tail_start, cond)
        torch.testing.assert_close(lp[:, 0], Normal(mu, std).log_prob(after[:, 0]).flatten(1).sum(-1))
        advantages = torch.tensor([1., -1., .5, -.5])
        loss, approx_kl, _ = clipped_objective(lp, old, advantages, .1)
        self.assertAlmostEqual(float(approx_kl.detach()), 0.)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.actor.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in m.reference.parameters()))
        self.assertTrue(all(p.grad is None for p in m.actor.encoder.parameters()))
        # Density is defined on unbounded latent draws, not clipped final actions.
        raw = after.clone(); raw[:, -1] = 3
        rawlp, _, _ = m.evaluate_chain(obs, ctx, before, raw)
        clippedlp, _, _ = m.evaluate_chain(obs, ctx, before, raw.clamp(-1, 1))
        self.assertFalse(torch.allclose(rawlp[:, -1], clippedlp[:, -1]))

    def test_full_training_resume_and_inference(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as d, redirect_stdout(StringIO()):
            root = Path(d); rng = np.random.RandomState(3); splits = {}
            for split, case in zip(['train','val','test'], ['001','002','003']):
                p = root / case; p.mkdir(); xyz = rng.normal(size=(80,3)).astype('float32')
                np.savez(p/'case.npz', xyz=xyz, normal=xyz/np.linalg.norm(xyz,axis=1,keepdims=True),
                         curvature=np.zeros(80,np.float32), lower_xyz=xyz-2,
                         lower_normal=np.tile([0.,0.,1.],(80,1)))
                splits[split] = [{'id':case,'case':case+'/case.npz'}]
            manifest=root/'manifest.json'; manifest.write_text(json.dumps({'version':1,'splits':splits}))
            teacher=OcclusionTeacher(sample_count=8,**c['model']); tp=root/'teacher.pt'
            torch.save({'model':teacher.state_dict(),'samples':8,'model_options':c['model']},tp)
            pretrained=DiffusionAgent(c['agent'],c['model']); source=root/'source.pt'
            oldopts=copy.deepcopy(c['agent']); oldopts['trainer']='diffusion'; oldopts.pop('diffusion_rl')
            torch.save({'model':pretrained.state_dict(),'model_options':c['model'],'agent_options':oldopts,
                        'teacher_sha256':hashlib.sha256(tp.read_bytes()).hexdigest(),
                        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},source)
            run(c,manifest,tp,source,root/'full')
            original=torch.save
            class Interrupted(Exception):pass
            def save(obj,path,*args,**kw):
                original(obj,path,*args,**kw)
                if Path(path).name=='latest.pt' and obj['iteration']==1:raise Interrupted()
            with patch('torch.save',side_effect=save):
                with self.assertRaises(Interrupted):run(c,manifest,tp,source,root/'resume')
            run(c,manifest,tp,source,root/'resume',root/'resume/latest.pt')
            full=torch.load(root/'full/latest.pt',weights_only=True)
            resumed=torch.load(root/'resume/latest.pt',weights_only=True)
            self.assertEqual(full['history'],resumed['history'])
            for k,v in full['model'].items():torch.testing.assert_close(v,resumed['model'][k],atol=0,rtol=0)
            self.assertTrue(any(not torch.equal(v, full['model']['actor.'+k]) for k,v in pretrained.state_dict().items()))
            for k,v in pretrained.state_dict().items():torch.testing.assert_close(v,full['model']['reference.'+k],atol=0,rtol=0)
            agent=load_agent(root/'full/agent.pt',c,teacher_path=tp)
            result=evaluate_continuous(agent,teacher,[root/'003/case.npz'],c,episodes=1,horizons=(3,))
            for row in result['cases']:
                self.assertEqual(len(row['trace']),4)
                for step in row['trace'][:-1]:
                    self.assertTrue(all(step['accepted_action']))
                    np.testing.assert_array_equal(step['action'],np.asarray(step['planned_actions'])[:,0])
            from occlusion_rl_offline.inference import infer_agent
            inferred=infer_agent(root/'003/case.npz',tp,root/'full/agent.pt',root/'infer',c,steps=3)
            self.assertEqual(len(inferred['trajectory']),4)
            bad=copy.deepcopy(c);bad['agent']['diffusion']['inference_steps']+=1
            with self.assertRaises(ValueError):load_agent(root/'full/agent.pt',bad)


if __name__=='__main__':unittest.main()
