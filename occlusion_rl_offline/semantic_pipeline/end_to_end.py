"""Held-out raw-mesh perturbation BEFORE segmentation; failures stay in denominator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ..config import resolve,load_config
from ..cohort import read_manifest
from ..data import load_case
from ..inference import infer_agent,_matrix
from ..train_regions import save_json
from .segmentation import load_model
from .regions import prepare_observed


def evaluate(cfg):
    out=resolve(cfg['out']);model=load_model(cfg);c=load_config(out/'rl_config.json');rows=[]
    bands=cfg['pose_bands'] if not cfg.get('smoke') else [cfg['pose_bands'][0],cfg['pose_bands'][2]]
    target=out/'end_to_end'
    for path in read_manifest(out/'manifest.json')['test']:
        original=load_case(path);pivot=original['xyz'].mean(0)
        for index,band in enumerate(bands):
            rng=np.random.RandomState(95101+int(path.parent.name)*1009+index)
            r=Rotation.from_rotvec(np.deg2rad(rng.uniform(-1,1,3)*band['rotation_deg'])).as_matrix()
            t=rng.uniform(-1,1,3)*band['translation_mm'];initial=_matrix(r,t,pivot)
            observed={k:v.copy() for k,v in original.items() if isinstance(v,np.ndarray)}
            observed['xyz']=((original['xyz']-pivot)@r.T+pivot+t).astype(np.float32)
            observed['normal']=(original['normal']@r.T).astype(np.float32)
            destination=target/path.parent.name/band['name']
            row=dict(case=path.parent.name,band=band['name'],initial_rotation_deg=float(np.rad2deg(Rotation.from_matrix(r).magnitude())),
                     initial_translation_mm=float(np.linalg.norm(t)))
            try:
                prepared,audit=prepare_observed(observed,model,cfg,destination/'regions')
                result=infer_agent(prepared,resolve(c['paths']['packaged_teacher']),resolve(c['paths']['packaged_agent']),
                                   destination,c,steps=c['agent']['horizon'])
                residual=np.asarray(result['input_to_recovered_matrix'])@initial
                displacement=residual[:3,:3]@pivot+residual[:3,3]-pivot
                row.update(success=True,rotation_deg=float(np.rad2deg(Rotation.from_matrix(residual[:3,:3]).magnitude())),
                           translation_mm=float(np.linalg.norm(displacement)),steps=result['steps'],meshes=audit)
            except ValueError as error:
                row.update(success=False,error=str(error))
            rows.append(row);print(json.dumps(row),flush=True)
    good=[r for r in rows if r['success']]
    summary=dict(attempted=len(rows),completed=len(good),failed=len(rows)-len(good),
                 conditional_mean={k:float(np.mean([r[k] for r in good])) if good else None for k in ('rotation_deg','translation_mm')},
                 note='Conditional errors exclude failed inputs; always report failure count. Smoke is not a quality benchmark.',rows=rows)
    save_json(target/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/semantic_pipeline')
    args=p.parse_args();cfg=json.loads((resolve(args.run)/'pipeline_config.json').read_text());torch.set_num_threads(cfg['torch_threads'])
    evaluate(cfg)

if __name__=='__main__':main()
