"""Variable-size voxel PointTransformer training, full SO(3) augmentation and inverse maps."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F
from ..cohort import read_manifest
from ..config import resolve
from ..region_segmentation import RegionPointTransformer, pack_features, digest
from ..runtime import resolve_device
from ..train_regions import load_jaws, binary_metrics, save_json
from .geometry import voxel_sample


def pose_example(item, sampling, seed, rotated=True, translation_mm=10.):
    rng = np.random.RandomState(seed)
    r = Rotation.random(random_state=rng).as_matrix() if rotated else np.eye(3)
    t = rng.uniform(-translation_mm, translation_mm, 3) if rotated else np.zeros(3)
    xyz = (item['xyz']-item['xyz'].mean(0)) @ r.T + item['xyz'].mean(0) + t
    normal = item['normal'] @ r.T
    mapping = voxel_sample(xyz, **sampling)
    ids = mapping['ids']
    x = pack_features(xyz[ids], normal[ids], item['curvature'][ids], item['jaw'])
    return x, item['label'][ids] if 'label' in item else None, mapping, xyz, normal


def load_model(cfg):
    device = resolve_device(cfg)
    path = resolve(cfg['out'])/'segmentation/best.pt'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['sampling'] != cfg['segmentation']['sampling']:
        raise ValueError('Voxel sampling differs from segmentation training')
    model = RegionPointTransformer(**checkpoint['architecture']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    return model


@torch.no_grad()
def predict(model, item, sampling):
    mapping = voxel_sample(item['xyz'], **sampling)
    ids = mapping['ids']
    features = pack_features(item['xyz'][ids], item['normal'][ids], item['curvature'][ids], item['jaw'])
    p = model(torch.from_numpy(features)[None].to(next(model.parameters()).device)).sigmoid()[0].cpu().numpy()
    return p[mapping['inverse']], mapping


@torch.no_grad()
def evaluate_items(model, items, cfg, repeats):
    options = cfg['segmentation']; rows=[]; model.eval()
    for item in items:
        reference = None
        for i in range(repeats+1):
            x, _, mapping, _, _ = pose_example(item, options['sampling'], 840001+int(item['case'])*100+i, i>0)
            p = model(torch.from_numpy(x)[None].to(next(model.parameters()).device)).sigmoid()[0].cpu().numpy()[mapping['inverse']]
            if reference is None: reference=p >= options['threshold']
            rows.append(dict(case=item['case'], jaw=item['jaw'], pose=i, voxels=len(mapping['ids']),
                effective_voxel_mm=mapping['size_mm'], flip=float(((p>=options['threshold'])!=reference).mean()),
                **binary_metrics(p,item['label'],options['threshold'])))
    return dict(rows=rows, mean={k:float(np.mean([r[k] for r in rows])) for k in ('iou','dice','precision','recall','flip')})


def train(cfg):
    options=cfg['segmentation']; out=resolve(cfg['out']); device=resolve_device(cfg)
    torch.set_num_threads(cfg['torch_threads']);torch.manual_seed(cfg['seed'])
    split=read_manifest(out/'manifest.json')
    items=load_jaws(split['train']); validation=load_jaws(split['val'])
    model=RegionPointTransformer(**options['architecture']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=1e-4)
    positive=sum(i['label'].sum() for i in items); total=sum(len(i['label']) for i in items)
    if not 0 < positive < total: raise ValueError('Segmentation needs positive and negative training vertices')
    weight=torch.tensor(float(np.clip((total-positive)/positive,.25,4)),device=device)
    best=-1; history=[]
    target=out/'segmentation';target.mkdir(parents=True,exist_ok=True)
    if (target/'latest.pt').exists():raise ValueError('Use a new output directory for segmentation training')
    for epoch in range(options['epochs']):
        model.train();losses=[]
        order=np.random.RandomState(cfg['seed']+epoch).permutation(len(items)*options['poses_per_jaw'])
        for index in order:
            item=items[index%len(items)]
            x,y,_,_,_=pose_example(item,options['sampling'],cfg['seed']+epoch*100000+int(index),rotated=True)
            logits=model(torch.from_numpy(x)[None].to(device));y=torch.from_numpy(y)[None].to(device)
            p=logits.sigmoid()
            dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=weight)+options['dice_weight']*dice
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            losses.append(float(loss.detach()))
        result=evaluate_items(model,validation,cfg,options['validation_rotations'])
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=result['mean']);history.append(row)
        checkpoint=dict(model=model.state_dict(),architecture=options['architecture'],sampling=options['sampling'],
            epoch=epoch+1,validation=result,manifest_sha256=digest(out/'manifest.json'),
            train_cases=[p.parent.name for p in split['train']],config=cfg)
        torch.save(checkpoint,target/'latest.pt')
        if result['mean']['iou']>best:
            best=result['mean']['iou'];torch.save(checkpoint,target/'best.pt')
        save_json(target/'history.json',history);print(json.dumps(row),flush=True)


def evaluate(cfg):
    model=load_model(cfg);out=resolve(cfg['out'])
    result=evaluate_items(model,load_jaws(read_manifest(out/'manifest.json')['test']),cfg,cfg['segmentation']['test_rotations'])
    result['checkpoint_sha256']=digest(out/'segmentation/best.pt')
    save_json(out/'segmentation/test.json',result);print(json.dumps(result['mean']),flush=True)
