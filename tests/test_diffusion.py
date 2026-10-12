"""DDIM math, action chunks, deterministic inference and resumable training."""
import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.config import load_config
from occlusion_rl_offline.diffusion import DiffusionAgent, cosine_alpha_bar
from occlusion_rl_offline.evaluation import load_agent, evaluate_continuous
from occlusion_rl_offline.model import OcclusionTeacher
from occlusion_rl_offline.rl import PoseTask, step_context
from occlusion_rl_offline.train_diffusion import expert_chunk, run


def small_config():
    c = load_config('configs/diffusion_policy.json')
    c['runtime'] = {'device': 'cpu'}
    c['model'] = {'width': 8, 'k': 4, 'encoder_layers': 1, 'candidate_points': 64}
    c['teacher']['fps_points'] = 8
    c['agent'].update(fps_points=8, horizon=3, validation_episodes=1, torch_threads=1)
    c['agent']['diffusion'].update(hidden_width=16, hidden_layers=1, time_embed_dim=8,
        noise_steps=10, inference_steps=4, training_steps=2, batch_size=4,
        episodes_per_case=2, collection_batch=2, validate_every=1, prediction_horizon=3)
    return c


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.cfg = small_config()

    def test_ddim_oracle_and_gradients(self):
        c = self.cfg
        m = DiffusionAgent(c['agent'], c['model'])
        ab = cosine_alpha_bar(100)
        self.assertTrue(((ab > 0) & (ab < 1)).all())
        self.assertTrue((ab[1:] < ab[:-1]).all())
        x, ctx = torch.randn(2, 8, 8), torch.tensor([[0., 1.], [0., 1.]])
        target = torch.rand(2, 3, 6) * 1.8 - .9
        def oracle(noisy, timesteps, condition):
            alpha = m.alpha_bar[timesteps][:, None, None]
            return (noisy - alpha.sqrt() * target) / (1 - alpha).sqrt()
        with patch.object(m, 'denoise', side_effect=oracle):
            recovered = m.predict_action_chunk(x, ctx)
        torch.testing.assert_close(recovered, target, rtol=0, atol=2e-6)
        loss = m.noise_loss(x, ctx, target)
        loss.backward()
        for p in (m.encoder.embed[0].weight, m.input.weight, m.output[-1].weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
        noise = torch.randn_like(target)
        torch.testing.assert_close(m.predict_action_chunk(x, ctx, initial_noise=noise),
                                   m.predict_action_chunk(x, ctx, initial_noise=noise), rtol=0, atol=0)

    def test_chunks_follow_environment_and_continue_past_budget(self):
        a = self.cfg['agent']
        r = Rotation.from_rotvec([[.03, -.04, .02], [0, 0, 0]]).as_matrix()
        t = np.array([[.4, -.2, .1], [0, 0, 0]])
        actions = expert_chunk(r, t, a, a['horizon'] - 1, 5)
        self.assertEqual(actions.shape, (2, 5, 6))
        previous = (Rotation.from_matrix(r).magnitude(), np.linalg.norm(t, axis=-1))
        for action in actions.transpose(1, 0, 2):
            r = Rotation.from_rotvec(np.deg2rad(action[:, :3] * 2)).as_matrix() @ r
            t += action[:, 3:] * .4
            current = (Rotation.from_matrix(r).magnitude(), np.linalg.norm(t, axis=-1))
            self.assertTrue(np.all(current[0] <= previous[0] + 1e-7))
            self.assertTrue(np.all(current[1] <= previous[1] + 1e-7))
            previous = current
        self.assertLess(float(previous[0].max()), 1e-6)
        self.assertLess(float(previous[1].max()), 1e-6)
        np.testing.assert_array_equal(actions[1], 0)

    def test_training_resume_trace_and_no_target_inference(self):
        c = self.cfg
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            root = Path(folder).resolve()
            splits = {}
            rng = np.random.RandomState(23)
            for split, case in zip(('train', 'val', 'test'), ('001', '002', '003')):
                directory = root / case; directory.mkdir()
                xyz = rng.normal(size=(80, 3)).astype(np.float32)
                np.savez(directory/'case.npz', xyz=xyz, normal=xyz/np.linalg.norm(xyz, axis=1, keepdims=True),
                         curvature=np.zeros(80, np.float32), lower_xyz=xyz-2,
                         lower_normal=np.tile([0., 0., 1.], (80, 1)))
                splits[split] = [{'id': case, 'case': case+'/case.npz'}]
            manifest = root/'manifest.json'; manifest.write_text(json.dumps({'version': 1, 'splits': splits}))
            teacher = OcclusionTeacher(sample_count=8, **c['model'])
            teacher_path = root/'teacher.pt'
            torch.save({'model': teacher.state_dict(), 'samples': 8, 'model_options': c['model']}, teacher_path)
            run(c, manifest, teacher_path, root/'full')
            original_save = torch.save
            class Interrupted(Exception): pass
            def interrupt(obj, path, *args, **kwargs):
                original_save(obj, path, *args, **kwargs)
                if Path(path).name == 'agent.pt' and obj['iteration'] == 1:
                    raise Interrupted()
            with patch('torch.save', side_effect=interrupt):
                with self.assertRaises(Interrupted):
                    run(c, manifest, teacher_path, root/'resumed')
            run(c, manifest, teacher_path, root/'resumed', root/'resumed/latest.pt')
            full = torch.load(root/'full/latest.pt', weights_only=True)
            resumed = torch.load(root/'resumed/latest.pt', weights_only=True)
            self.assertEqual(full['history'], resumed['history'])
            for state in ('model', 'raw_model'):
                for key, value in full[state].items():
                    torch.testing.assert_close(value, resumed[state][key], rtol=0, atol=0)
            metadata = json.loads((root/'full/expert_chunks.json').read_text())
            self.assertEqual(list(metadata['train_cases']), [str(root/'001/case.npz')])
            agent = load_agent(root/'full/agent.pt', c, teacher_path=teacher_path)
            result = evaluate_continuous(agent, teacher, [root/'003/case.npz'], c,
                                         episodes=1, seed=52, horizons=(3,), details=True)
            repeat = evaluate_continuous(agent, teacher, [root/'003/case.npz'], c,
                                         episodes=1, seed=52, horizons=(3,), details=True)
            self.assertEqual(result, repeat)
            for row in result['cases']:
                self.assertEqual(len(row['trace']), 4)
                for step in row['trace'][:-1]:
                    np.testing.assert_array_equal(step['action'], np.array(step['planned_actions'])[:, 0])
            from occlusion_rl_offline.inference import infer_agent
            inference = infer_agent(root/'003/case.npz', teacher_path, root/'full/agent.pt', root/'infer', c, steps=3)
            self.assertEqual(len(inference['trajectory']), 4)
            self.assertTrue(all(row['accepted_action'] for row in inference['trajectory'][1:]))
            changed = copy.deepcopy(c); changed['agent']['diffusion']['training_steps'] += 1
            with self.assertRaises(ValueError):
                run(changed, manifest, teacher_path, root/'resumed', root/'resumed/latest.pt')
            changed = copy.deepcopy(c); changed['agent']['diffusion']['noise_steps'] += 1
            with self.assertRaisesRegex(ValueError, 'noise_steps'):
                load_agent(root/'full/agent.pt', changed)
            changed = copy.deepcopy(c); changed['agent']['diffusion']['inference_steps'] += 1
            restored = load_agent(root/'full/agent.pt', changed)
            self.assertEqual(restored.options['inference_steps'], 5)


if __name__ == '__main__':
    unittest.main()
