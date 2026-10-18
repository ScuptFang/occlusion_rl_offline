"""Shared device selection for training, evaluation and inference."""
import os
import torch


def resolve_device(config=None, override=None):
    requested = override or os.environ.get('OCCLUSION_DEVICE') or (config or {}).get('runtime', {}).get('device', 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver, or use --device cpu.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError('CUDA device index is out of range')
        device = torch.device('cuda', index)
    return str(device)


def runtime_info(device):
    return {'device': str(device), 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None}
