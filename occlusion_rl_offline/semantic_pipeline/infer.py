"""Run the complete pipeline on an observed STL pair, without target poses/labels."""
import argparse
import json
from ..config import resolve,load_config
from ..prepare import _mesh,_upper_features
from ..inference import infer_agent
from .segmentation import load_model
from .regions import prepare_observed
import numpy as np
import torch


def mesh_pair(upper_path,lower_path):
    case={}
    for prefix,name,path in [('', 'upper',upper_path),('lower_', 'lower',lower_path)]:
        mesh=_mesh(path);xyz,normal,curvature=_upper_features(mesh)
        case.update({prefix+'xyz':xyz,prefix+'normal':normal,prefix+'curvature':curvature,name+'_faces':np.asarray(mesh.faces,dtype=np.int64)})
    return case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/semantic_pipeline');p.add_argument('--upper',required=True)
    p.add_argument('--lower',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=resolve(args.run);cfg=json.loads((root/'pipeline_config.json').read_text())
    torch.set_num_threads(cfg['torch_threads'])
    target=resolve(args.out)
    if (target/'trajectory.json').exists():raise ValueError('Use a fresh inference output directory')
    prepared,_=prepare_observed(mesh_pair(resolve(args.upper),resolve(args.lower)),load_model(cfg),cfg,target/'regions')
    c=load_config(root/'rl_config.json')
    result=infer_agent(prepared,resolve(c['paths']['packaged_teacher']),resolve(c['paths']['packaged_agent']),target,c,
                       upper_stl=resolve(args.upper))
    print(json.dumps({'output':str(target),'steps':result['steps'],'final_score':result['final_teacher_logit']}))

if __name__=='__main__':main()
