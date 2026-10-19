"""Measure original-vertex identity stability under small validation poses."""
import argparse
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.data import load_case, features_at_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/region_mesh10240_fps256')
    args=p.parse_args();root=resolve(args.out);cfg=load_config(root/'teacher_agent_config.json')
    case=load_case(read_manifest(root/'manifest.json')['val'][0]);torch.set_num_threads(1)
    rows=[]
    for rv,t in [([0,0,0],[0,0,0]),([.05,0,0],[0,0,0]),([0,0,0],[.01,0,0])]:
        _,ids=features_at_pose(case,Rotation.from_rotvec(np.deg2rad(rv)).as_matrix(),np.asarray(t),
                               cfg['teacher']['fps_points'],cfg['model'],True)
        current=[case['_upper_region_mesh']['vertex_ids'].copy(),case['_upper_candidate_ids'].copy(),ids.copy()]
        if not rows:initial=current
        row=dict(rotation=rv,translation=t)
        for name,a,b in zip(('region','candidates','fps'),initial,current):
            row[name+'_jaccard']=len(np.intersect1d(a,b))/len(np.union1d(a,b))
        rows.append(row)
    (root/'sampling_stability.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
