"""Original predicted mesh regions -> 10240 candidates -> 256 points experiment."""
import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.config import REPO_ROOT, resolve, load_config
from occlusion_rl_offline.prepare import prepare_case
from occlusion_rl_offline.data import features_at_pose
from occlusion_rl_offline.region_segmentation import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def setup(experiment, out):
    if (out / 'teacher_agent_config.json').exists():
        raise FileExistsError('Use a new out directory or omit setup')
    for i, source in enumerate(('source_config', 'source_rl_config')):
        cfg = json.loads(resolve(experiment[source]).read_text())
        run = out if i == 0 else out / 'rl'
        cfg['model']['candidate_points'] = experiment['candidate_points']
        cfg['model']['segmentation_region'].update(mesh_before_sampling=True,
            face_rule=experiment['face_rule'], minimum_upper_faces=experiment['minimum_upper_faces'])
        for stage in ('teacher', 'agent'):
            cfg[stage]['fps_points'] = experiment['fps_points']
        cfg['paths'].update(run_dir=str(run), packaged_teacher=str(out / 'teacher/ranking/best.pt'),
            packaged_agent=str(run / 'agent/agent.pt'), case=str(out / 'prepared/cases/010/case.npz'))
        if i:
            cfg['paths']['imitation_agent'] = str(out / 'agent/agent.pt')
        cfg['inference']['out_dir'] = str(run / 'inference')
        path = out / ('diffusion_rl_config.json' if i else 'teacher_agent_config.json')
        write(path, cfg); load_config(path)
    write(out / 'experiment.json', experiment)


def prepare(experiment, out):
    if (out / 'manifest.json').exists():
        raise FileExistsError('Prepared experiment already exists')
    torch.set_num_threads(4)
    cfg = load_config(out / 'teacher_agent_config.json')
    manifest = json.loads(resolve(experiment['source_manifest']).read_text())
    audits = []
    for split, rows in manifest['splits'].items():
        for row in rows:
            folder = resolve(experiment['data_root']) / row['id']
            path = out / 'prepared/cases' / row['id'] / 'case.npz'
            case, audit = prepare_case(folder / 'upper.stl', folder / 'lower.stl', path,
                                      dict(model=dict(distance_method='nearest_surface')))
            for jaw, key, faces in [('upper', 'xyz', 'upper_faces'), ('lower', 'lower_xyz', 'lower_faces')]:
                original = trimesh.load_mesh(folder / (jaw + '.stl'), process=False)
                # Prove that all triangles retain their original STL face order.
                np.testing.assert_array_equal(case[key][case[faces]], original.triangles.astype(np.float32))
                audit[jaw + '_stl_sha256'] = digest(folder / (jaw + '.stl'))
            audit.update(case=row['id'], split=split, original_triangle_order_verified=True, poses=[])
            rng = np.random.RandomState(93101 + int(row['id']))
            for band, angle, distance in [('zero',0,0), ('near',1.5,.5), ('medium',4,2), ('wide',8,4)]:
                r = Rotation.from_rotvec(np.deg2rad(rng.uniform(-angle,angle,3))).as_matrix()
                t = rng.uniform(-distance,distance,3)
                features, ids = features_at_pose(case,r,t,cfg['teacher']['fps_points'],cfg['model'],True)
                mesh = case['_upper_region_mesh']
                np.testing.assert_array_equal(mesh['vertex_ids'][mesh['faces']],case['upper_faces'][mesh['face_ids']])
                assert np.isin(ids, mesh['vertex_ids']).all()
                assert len(np.unique(ids)) == cfg['teacher']['fps_points']
                audit['poses'].append(dict(band=band, upper=case['_upper_region_audit'],lower=case['_lower_region_audit'],
                                           contact_fraction=float((np.abs(features[:,7])<.6).mean())))
                if band == 'zero':
                    for jaw in ('upper','lower'):
                        region = case['_' + jaw + '_region_mesh']
                        np.savez_compressed(path.parent / (jaw + '_region.npz'), **region)
                        trimesh.Trimesh(vertices=region['xyz'],faces=region['faces'],process=False).export(
                            path.parent / (jaw + '_region.stl'))
            row['case'] = str(path.relative_to(out))
            audits.append(audit)
            print(json.dumps(dict(case=row['id'],poses=audit['poses'])),flush=True)
    write(out / 'manifest.json',manifest);write(out / 'mesh_audit.json',audits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/region_mesh10240_fps256.json')
    p.add_argument('--stages',nargs='+',default=['setup','prepare','teacher','imitation','rl','evaluate'],
                   choices=['setup','prepare','teacher','imitation','rl','evaluate'])
    args=p.parse_args(); experiment=json.loads(resolve(args.config).read_text());out=resolve(experiment['out'])
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS'):
        env[key]=str(experiment['math_threads'])
    config=out/'teacher_agent_config.json';rl=out/'diffusion_rl_config.json';manifest=out/'manifest.json'
    def run(name,command):
        print(json.dumps(dict(stage=name,command=command)),flush=True)
        with (logs/(name+'.log')).open('w') as f:
            subprocess.run(command,cwd=REPO_ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(name+' completed',flush=True)
    for stage in args.stages:
        if stage=='setup':setup(experiment,out)
        elif stage=='prepare':prepare(experiment,out)
        elif stage in ('teacher','imitation','rl'):
            sentinel={'teacher':'teacher/ranking/best.pt','imitation':'agent/latest.pt','rl':'rl/agent/latest.pt'}[stage]
            if (out/sentinel).exists():raise FileExistsError('Stage exists: '+stage)
            command=[sys.executable,'-m','occlusion_rl_offline.cli','--config',str(rl if stage=='rl' else config),
                     'train-teacher' if stage=='teacher' else 'train-agent','--manifest',str(manifest)]
            if stage!='teacher':command+=['--teacher',str(out/'teacher/ranking/best.pt')]
            run(stage,command)
        else:
            common=[sys.executable,'scripts/evaluate_filtered_cohort.py','--manifest',str(manifest)]
            tasks=[('teacher_test',common+['--config',str(config),'--stage','teacher']),
                   ('imitation_test',common+['--config',str(config),'--stage','agent']),
                   ('rl_test',common+['--config',str(rl),'--stage','agent']),
                   ('stochastic_imitation_test',common+['--config',str(rl),'--stage','agent',
                    '--agent',str(out/'rl/agent/imitation.pt'),'--out',str(out/'rl/test/imitation')])]
            with ThreadPoolExecutor(max_workers=experiment['evaluation_workers']) as pool:
                futures=[pool.submit(run,*task) for task in tasks]
                for f in futures:f.result()


if __name__=='__main__':main()
