"""Locked-checkpoint comparison on identical held-out poses for both teacher stages."""
import argparse
import hashlib
import json

import numpy as np
import torch

from occlusion_rl_offline.aligned_teacher import build_bank, predict, metrics
from occlusion_rl_offline.cohort import read_manifest
from occlusion_rl_offline.config import load_config, resolve
from occlusion_rl_offline.rl import load_teacher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='runs/teacher_refinement/prepared/manifest.json')
    p.add_argument('--out',default='runs/teacher_refinement/evaluation')
    p.add_argument('--seed',type=int,default=242042)
    args=p.parse_args();torch.set_num_threads(4)
    out=resolve(args.out);out.mkdir(parents=True,exist_ok=True)
    paths=read_manifest(resolve(args.manifest))['test']
    configurations={'baseline':load_config('configs/reagent_curriculum_large.json'),
                    'aligned_vertex':load_config('configs/teacher_aligned_vertex.json'),
                    'aligned_surface':load_config('configs/teacher_aligned_surface.json')}
    checkpoints={}
    for name,cfg in configurations.items():
        for stage in ('ranking','calibration'):
            path=resolve(cfg['paths']['run_dir'])/'teacher'/stage/'best.pt'
            checkpoints[name+'/'+stage]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # All six checkpoints are locked before test generation/scoring.
    (out/'locked_checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
    epsilon=configurations['aligned_vertex']['teacher']['aligned_pair']['tie_epsilon']
    result={'seed':args.seed,'split':'test','tie_epsilon':epsilon,'checkpoints':checkpoints,'results':{}}
    vertex=build_bank(paths,configurations['aligned_vertex'],
                      configurations['aligned_vertex']['teacher']['aligned_pair']['test_pairs_per_case'],
                      args.seed,out/'vertex_test.npz')
    surface=build_bank(paths,configurations['aligned_surface'],
                       configurations['aligned_surface']['teacher']['aligned_pair']['test_pairs_per_case'],
                       args.seed,out/'surface_test.npz')
    for key in ('y','rotation','translation','local','near','case_id'):
        np.testing.assert_array_equal(vertex[key],surface[key])
    for name,cfg in configurations.items():
        bank=surface if name=='aligned_surface' else vertex
        for stage in ('ranking','calibration'):
            key=name+'/'+stage
            model=load_teacher(checkpoints[key]['path'],'cpu',cfg['teacher']['fps_points'],**cfg['model'])
            pred=predict(model,bank,'cpu')
            m=metrics(pred,bank,epsilon)
            m['by_case']={}
            for case_id in np.unique(bank['case_id']):
                mask=bank['case_id']==case_id
                subset={k:v[mask] for k,v in bank.items()}
                m['by_case'][str(case_id)]=metrics(pred[mask],subset,epsilon)
            m['selected_epoch']=torch.load(checkpoints[key]['path'],map_location='cpu',weights_only=False)['epoch']
            result['results'][key]=m
            np.save(out/(key.replace('/','_')+'_predictions.npy'),pred)
            print(key,json.dumps(m),flush=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
