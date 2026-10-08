"""Rotvec teacher training with regression, local ranking and sampling consistency."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from torch.nn import functional as F

from .runtime import resolve_device, runtime_info
from .cohort import read_manifest
from .config import load_config, resolve
from .data import load_case, features_at_pose, sample_sdf
from .model import OcclusionTeacher


def quality(pose, scales):
    # pose[:3] is a rotation vector in degrees, hence its norm is the geodesic
    # angle within the (<180 deg) generated range.
    return -np.linalg.norm(pose[:3]) / scales[0] - np.linalg.norm(pose[3:]) / scales[1]


def build_bank(paths, cfg, count, seed, augment=True):
    options = cfg['model']; t = cfg['teacher']; joint = cfg['optimization']['teacher']
    x, y, case_ids, kinds = [], [], [], []
    for path in paths:
        case = load_case(path); case_id = path.parent.name
        rng = np.random.RandomState(seed + int(case_id) * 1009)
        for index in range(count):
            category = rng.choice(4, p=joint['pose_probabilities'])
            limits = joint.get('pose_limits', [(0.,0.), (1.5,.5), (4.,2.), (8.,4.)])[category]
            pose = rng.uniform(-1,1,6) * np.array([limits[0]]*3+[limits[1]]*3)
            if index % 3 == 0:
                other = pose * joint.get('radial_contraction', .8)
            elif index % 3 == 1:
                delta = rng.uniform(-1,1,6) * np.array([joint.get('local_rotation_deg',.5)]*3+
                                                      [joint.get('local_translation_mm',.1)]*3)
                if index % 2:
                    keep = rng.randint(6); delta[np.arange(6) != keep] = 0
                other = pose + delta
            else:
                other = rng.uniform(-1,1,6) * np.array([4.]*3+[2.]*3)
            third = pose.copy()
            # In validation use actual micro-motion, independent of training
            # seed augmentation; no original-pose point IDs enter the input.
            if not augment:
                third += rng.uniform(-1,1,6) * np.array([joint.get('micro_rotation_deg',.05)]*3+
                                                       [joint.get('micro_translation_mm',.01)]*3)
            features = []
            initial_ids = None
            for slot, p in enumerate((pose, other, third)):
                sampler = dict(options)
                if slot == 2 and augment:
                    sampler['point_seed'] = options.get('point_seed',42)+1+int(rng.randint(joint.get('candidate_seed_variants',3)))
                rotation = Rotation.from_rotvec(np.deg2rad(p[:3])).as_matrix()
                if options.get('sampling_mode') == 'episode_fixed' and slot == 1:
                    pivot = case['xyz'].mean(0)
                    xyz = ((case['xyz'][initial_ids]-pivot)@rotation.T+pivot+p[3:]).astype(np.float32)
                    normals = case['normal'][initial_ids]@rotation.T
                    features.append(np.column_stack((xyz,normals,case['curvature'][initial_ids],sample_sdf(case,xyz))).astype(np.float32))
                elif options.get('sampling_mode') == 'episode_fixed' and slot == 2 and not augment:
                    pivot = case['xyz'].mean(0)
                    xyz = ((case['xyz'][initial_ids]-pivot)@rotation.T+pivot+p[3:]).astype(np.float32)
                    features.append(np.column_stack((xyz,case['normal'][initial_ids]@rotation.T,case['curvature'][initial_ids],sample_sdf(case,xyz))).astype(np.float32))
                else:
                    feature, selected = features_at_pose(case, rotation, p[3:], t['fps_points'], sampler, True)
                    features.append(feature)
                    if slot == 0:initial_ids = selected
            x.append(features)
            y.append([quality(p, joint['quality_scales']) for p in (pose,other,third)])
            case_ids.append(case_id); kinds.append(category)
        print('bank', case_id, count, flush=True)
    return {'x': np.asarray(x,dtype=np.float32), 'y': np.asarray(y,dtype=np.float32),
            'case': np.asarray(case_ids), 'kind': np.asarray(kinds)}


@torch.no_grad()
def predict(model, bank, batch=64):
    shape = bank['x'].shape
    flat = torch.from_numpy(bank['x'].reshape(-1,*shape[2:]))
    return torch.cat([model(x.to(next(model.parameters()).device)).cpu() for x in flat.split(batch)]).numpy().reshape(-1,3)


def metrics(pred, bank):
    target = bank['y']; gap = target[:,0]-target[:,1]
    valid = np.abs(gap) > 1e-5
    near = valid & (bank['kind'] == 1)
    return {'mae': float(np.abs(pred-target).mean()),
            'spearman': float(spearmanr(pred[:,:2].flatten(), target[:,:2].flatten()).statistic),
            'pair_accuracy': float(((pred[valid,0]-pred[valid,1])*gap[valid]>0).mean()),
            'near_pair_accuracy': float(((pred[near,0]-pred[near,1])*gap[near]>0).mean()),
            'micro_delta_mae': float(np.abs((pred[:,2]-pred[:,0])-(target[:,2]-target[:,0])).mean()),
            'zero_anchor_mae': float(np.abs(pred[bank['kind']==0,0]).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--manifest',default='runs/data11_candidate4096_fps64_more/prepared/manifest.json')
    p.add_argument('--source',default='runs/data11_contact_fps/teacher/calibration/best.pt')
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['regression','joint','consistent','affine'],default='consistent')
    p.add_argument('--bank-dir',default=None)
    p.add_argument('--device',default=None)
    args=p.parse_args(); torch.set_num_threads(4)
    cfg=load_config(args.config); j=cfg['optimization']['teacher']; torch.manual_seed(j['seed'])
    device=resolve_device(cfg,args.device)
    cfg.setdefault('runtime',{})['device']=device
    out=resolve(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'config.snapshot.json').write_text(json.dumps(cfg,indent=2))
    (out/'runtime.json').write_text(json.dumps(runtime_info(device),indent=2))
    paths=read_manifest(resolve(args.manifest))
    cache=resolve(args.bank_dir) if args.bank_dir else out/'banks'; cache.mkdir(parents=True,exist_ok=True)
    banks={}
    for split,count,seed in [('train',j['poses_per_case'],j['seed']),('val',j['validation_per_case'],j['seed']+900000)]:
        f=cache/(split+'.npz')
        provenance={'model':cfg['model'],'teacher':j,'paths':[str(x) for x in paths[split]],'split':split}
        if f.exists():
            saved=json.loads(f.with_suffix('.json').read_text())
            if saved != provenance: raise ValueError('Cached bank configuration mismatch')
            with np.load(f) as loaded: banks[split]={k:loaded[k] for k in loaded.files}
        else:
            banks[split]=build_bank(paths[split],cfg,count,seed,augment=split=='train')
            np.savez_compressed(f,**banks[split]);f.with_suffix('.json').write_text(json.dumps(provenance,indent=2))
    model=OcclusionTeacher(sample_count=cfg['teacher']['fps_points'],**cfg['model']).to(device)
    model.load_state_dict(torch.load(resolve(args.source),map_location='cpu')['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=j['learning_rate'],weight_decay=j.get('weight_decay',1e-4))
    rng=np.random.RandomState(j['seed']); history=[]; best=float('inf'); stale=0
    baseline=metrics(predict(model,banks['val']),banks['val'])
    print('baseline',json.dumps(baseline),flush=True)
    if args.mode=='affine':
        pred=predict(model,banks['train']).flatten();target=banks['train']['y'].flatten()
        slope=max(float(np.cov(pred,target,bias=True)[0,1]/max(np.var(pred),1e-8)),1e-6)
        intercept=float(target.mean()-slope*pred.mean())
        with torch.no_grad():
            model.head[-1].weight.mul_(slope);model.head[-1].bias.mul_(slope).add_(intercept)
        result=metrics(predict(model,banks['val']),banks['val'])
        torch.save({'model':model.state_dict(),'samples':cfg['teacher']['fps_points'],
                    'model_options':cfg['model'],'epoch':0,'affine':[slope,intercept]},out/'best.pt')
        (out/'history.json').write_text(json.dumps([{'epoch':0,'validation':result}],indent=2))
        print(json.dumps(result),flush=True);return
    x=torch.from_numpy(banks['train']['x']); y=torch.from_numpy(banks['train']['y'])
    for epoch in range(j['epochs']):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/j['batch'])))):
            features=x[ids].to(device);target=y[ids].to(device);pred=model(features.flatten(0,1)).reshape(-1,3)
            regression=F.smooth_l1_loss(pred[:,:2],target[:,:2],beta=j.get('regression_beta',.25))
            gap=target[:,0]-target[:,1];valid=gap.abs()>1e-5
            margin=gap.abs().clamp(max=j['margin_cap'])
            if valid.any():
                ranking=(j['rank_temperature']*F.softplus((margin[valid]-gap[valid].sign()*(pred[valid,0]-pred[valid,1]))/j['rank_temperature'])).mean()
            else: ranking=pred.sum()*0
            consistency=F.smooth_l1_loss(pred[:,2]-pred[:,0],target[:,2]-target[:,0],beta=j.get('consistency_beta',.1))
            loss=regression
            if args.mode in ('joint','consistent'):loss=loss+j['ranking_weight']*ranking
            if args.mode=='consistent':loss=loss+j['consistency_weight']*consistency
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),j.get('max_grad_norm',1.));optimizer.step()
            losses.append(float(loss.detach()))
        model.eval();va=metrics(predict(model,banks['val']),banks['val'])
        # Validation quality criteria only; no test-case checkpoint selection.
        selection=va['mae']+j['selection_local_weight']*(1-va['near_pair_accuracy'])+j['selection_noise_weight']*va['micro_delta_mae']
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':va,'selection':selection}
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        checkpoint={'model':model.state_dict(),'samples':cfg['teacher']['fps_points'],
                    'model_options':cfg['model'],'epoch':epoch+1,'mode':args.mode,'validation':va}
        torch.save(checkpoint,out/'latest.pt')
        if selection < best:
            best=selection;stale=0;torch.save(checkpoint,out/'best.pt')
        else:stale+=1
        if stale>=j['patience']:break
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))

if __name__=='__main__':main()
