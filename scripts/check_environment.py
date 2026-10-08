"""Check the selected device with a real encoder forward/backward pass."""
import argparse
import json
import torch
from occlusion_rl_offline.runtime import resolve_device, runtime_info
from occlusion_rl_offline.model import OcclusionTeacher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--device', default='auto')
a = p.parse_args()
d = resolve_device(override=a.device)
m = OcclusionTeacher(sample_count=64, encoder_layers=3).to(d)
x = torch.randn(2, 64, 8, device=d)
y = m(x)
y.square().mean().backward()
assert torch.isfinite(y).all()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print(json.dumps(dict(runtime_info(d), forward_backward='passed'), indent=2))
