"""Segment observed meshes once, recover original faces, simplify, then perturb."""
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from ..config import resolve
from ..cohort import read_manifest
from ..data import load_case
from ..surface import surface_distance
from ..region_segmentation import digest
from ..train_regions import save_json
from .segmentation import load_model, predict
from .geometry import extract_region, simplify_region


def prepare_observed(case, model, cfg, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    arrays={}; audits=[]
    for jaw,prefix,name in [(0,'','upper'),(1,'lower_','lower')]:
        item={key:case[prefix+key] for key in ('xyz','normal','curvature')};item['jaw']=jaw
        probability,mapping=predict(model,item,cfg['segmentation']['sampling'])
        region=extract_region(case,prefix,probability,cfg['segmentation']['threshold'])
        mesh=simplify_region(region,cfg['simplification'])
        np.savez_compressed(output/(name+'_mapping.npz'),probability=probability,
            voxel_vertex_ids=mapping['ids'],vertex_to_voxel=mapping['inverse'],
            region_vertex_ids=region['vertex_ids'],region_face_ids=region['face_ids'],
            simplified_vertex_ids=mesh['vertex_ids'],simplified_faces=mesh['faces'])
        for key in ('xyz','normal','curvature'): arrays[prefix+key]=mesh[key]
        arrays[name+'_faces']=mesh['faces']
        trimesh.Trimesh(vertices=mesh['xyz'],faces=mesh['faces'],process=False).export(output/(name+'_simplified.stl'))
        # Exact full-resolution colored original mesh, not the display subsample.
        color=np.tile([95,110,130,255],(len(probability),1));color[probability>=cfg['segmentation']['threshold']]=[40,210,180,255]
        trimesh.Trimesh(vertices=item['xyz'],faces=case[name+'_faces'],vertex_colors=color,process=False).export(output/(name+'_semantic.ply'))
        probe_ids=np.linspace(0,len(region['xyz'])-1,min(2048,len(region['xyz'])),dtype=int)
        distance=np.abs(surface_distance(dict(lower_xyz=mesh['xyz'],lower_normal=mesh['normal'],lower_faces=mesh['faces']),region['xyz'][probe_ids])[0])
        before_mesh=trimesh.Trimesh(region['xyz'],region['faces'],process=False)
        after_mesh=trimesh.Trimesh(mesh['xyz'],mesh['faces'],process=False)
        audits.append(dict(euler_before=int(before_mesh.euler_number),euler_after=int(after_mesh.euler_number),
            sampled_surface_distance_mean_mm=float(distance.mean()),sampled_surface_distance_p95_mm=float(np.quantile(distance,.95)),
            sampled_surface_distance_max_mm=float(distance.max()),jaw=name,original_vertices=len(item['xyz']),voxels=len(mapping['ids']),
            voxel_size_mm=mapping['size_mm'],region_vertices=len(region['xyz']),region_faces=len(region['faces']),
            simplified_vertices=len(mesh['xyz']),simplified_faces=len(mesh['faces']),rounds=mesh['simplification_rounds']))
    # No labels/probabilities/target transforms enter the teacher/agent case.
    np.savez_compressed(output/'case.npz',**arrays)
    save_json(output/'mesh_audit.json',audits)
    return output/'case.npz',audits


def prepare(cfg):
    out=resolve(cfg['out']);model=load_model(cfg)
    manifest=json.loads((out/'manifest.json').read_text()); rows=[]
    for split,entries in manifest['splits'].items():
        for entry in entries:
            source=out/entry['case']; target=out/'regions/cases'/entry['id']
            case,audit=prepare_observed(load_case(source),model,cfg,target)
            entry['case']=os.path.relpath(case,out/'regions')
            rows.append(dict(case=entry['id'],split=split,meshes=audit,source_sha256=digest(source)))
            print(json.dumps(rows[-1]),flush=True)
    manifest['segmentation_sha256']=digest(out/'segmentation/best.pt')
    manifest['simplification']=cfg['simplification']; manifest['sampling']=cfg['segmentation']['sampling']
    save_json(out/'regions/manifest.json',manifest);save_json(out/'regions/audit.json',rows)
