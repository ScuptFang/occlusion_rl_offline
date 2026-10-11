"""Expanded inputs, PPO gradients and checkpoint compatibility."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from occlusion_rl_offline.config import load_config
from occlusion_rl_offline.evaluation import load_agent
from occlusion_rl_offline.rl import Agent, step_context


class CapacityTests(unittest.TestCase):
    def test_large_network_update_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        cfg = load_config('configs/reagent_curriculum_large.json')
        a = cfg['agent']
        agent = Agent(a['fps_points'], a['policy_width'], a['initial_log_std'],
                      time_context=a['time_context'], **cfg['model'])
        obs = torch.randn(2, a['fps_points'], 8)
        context = step_context(agent, 2, 0, 10, 'cpu')
        # A real actor/value update must propagate through the larger input encoder.
        mean, value = agent(obs, context)
        loss = (mean - .2).square().mean() + (value - .5).square().mean()
        optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad])
        loss.backward()
        for module in (agent.encoder.embed[0], agent.body[0], agent.actor, agent.critic):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)
        optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'agent.pt'
            torch.save({'model': agent.state_dict(), 'model_options': cfg['model'],
                        'agent_options': a}, path)
            restored = load_agent(path, cfg)
            with torch.no_grad():
                expected = agent(obs, context)
                actual = restored(obs, context)
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            # Tensor shapes alone cannot detect an incorrect FPS count.
            for key, wrong in [('fps_points', 64), ('policy_width', 128), ('time_context', False)]:
                changed = copy.deepcopy(cfg)
                changed['agent'][key] = wrong
                with self.assertRaisesRegex(ValueError, key):
                    load_agent(path, changed)


if __name__ == '__main__':
    unittest.main()
