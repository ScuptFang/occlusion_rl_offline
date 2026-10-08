import os
import unittest
from unittest.mock import patch
from occlusion_rl_offline.runtime import resolve_device


class RuntimeTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch('torch.cuda.is_available', return_value=False)
    def test_no_silent_cuda_fallback(self, available):
        self.assertEqual(resolve_device(), 'cpu')
        self.assertEqual(resolve_device({'runtime': {'device': 'cpu'}}), 'cpu')
        with self.assertRaises(RuntimeError):
            resolve_device(override='cuda')
        with self.assertRaises(ValueError):
            resolve_device(override='mps')

    @patch.dict(os.environ, {'OCCLUSION_DEVICE': 'cpu'})
    def test_explicit_override_then_environment_then_config(self):
        self.assertEqual(resolve_device({'runtime': {'device': 'cuda'}}), 'cpu')
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device(override='cuda:0')

    @patch.dict(os.environ, {}, clear=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda_indices(self, available, current, count):
        self.assertEqual(resolve_device(), 'cuda:0')
        self.assertEqual(resolve_device(override='cuda:1'), 'cuda:1')
        with self.assertRaises(ValueError):
            resolve_device(override='cuda:2')
