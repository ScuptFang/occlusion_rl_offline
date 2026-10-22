import tempfile
from pathlib import Path
import torch
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from occlusion_rl_offline.semantic_pipeline.geometry import voxel_sample,extract_region,simplify_region
from occlusion_rl_offline.semantic_pipeline.rewards import weights,trajectory_reward
from occlusion_rl_offline.semantic_pipeline.segmentation import pose_example,load_model
from occlusion_rl_offline.region_segmentation import RegionPointTransformer
from occlusion_rl_offline.semantic_pipeline.__main__ import configuration
from occlusion_rl_offline.data import candidate_indices,features_at_pose
from occlusion_rl_offline.imitation import expert_actions
from occlusion_rl_offline.model import OcclusionTeacher


class SemanticPipelineTests(unittest.TestCase):
    def test_inference_loads_selected_checkpoint_weights(self):
        architecture=dict(width=8,layers=1,k=4)
        sampling=dict(size_mm=1.,max_points=128)
        source=RegionPointTransformer(**architecture)
        with torch.no_grad():
            source.head[-1].bias.fill_(3.25)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'segmentation';target.mkdir()
            torch.save(dict(architecture=architecture,sampling=sampling,model=source.state_dict()),target/'best.pt')
            loaded=load_model(dict(out=folder,runtime={'device':'cpu'},segmentation={'sampling':sampling}))
            for name,value in source.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.state_dict()[name]),name)
            self.assertFalse(loaded.training)

    def test_voxel_inverse_original_attributes_and_translation(self):
        xyz=np.array([[0,0,0],[.1,0,0],[1,0,0],[1.1,0,0],[2,0,0]],dtype=float)
        m=voxel_sample(xyz,.5,10)
        self.assertEqual(len(m['ids']),3)
        np.testing.assert_array_equal(m['inverse'][m['ids']],np.arange(3))
        np.testing.assert_array_equal(m['ids'],voxel_sample(xyz+17,.5,10)['ids'])
        values=np.arange(len(m['ids']))
        np.testing.assert_array_equal(values[m['inverse']][m['ids']],values)
        with self.assertRaises(ValueError):voxel_sample(xyz,0,10)

    def test_rotation_retains_normal_curvature_labels(self):
        mesh=trimesh.creation.icosphere(subdivisions=2)
        item=dict(xyz=mesh.vertices,normal=mesh.vertex_normals,curvature=np.arange(len(mesh.vertices)),
                  label=(mesh.vertices[:,2]>0).astype(np.float32),jaw=0)
        x,y,m,xyz,n=pose_example(item,dict(size_mm=.2,max_points=128),8)
        np.testing.assert_allclose(x[:,:3],xyz[m['ids']],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(x[:,3:6],n[m['ids']],atol=1e-6)
        np.testing.assert_array_equal(x[:,6],item['curvature'][m['ids']])
        np.testing.assert_array_equal(y,item['label'][m['ids']])
        self.assertLessEqual(len(x),128)

    def test_simplification_budget_vertex_features_and_faces(self):
        mesh=trimesh.creation.icosphere(subdivisions=3)
        count=len(mesh.vertices)
        case=dict(xyz=mesh.vertices,normal=mesh.vertex_normals,curvature=np.arange(count),upper_faces=mesh.faces)
        region=extract_region(case,'',np.ones(count))
        simple=simplify_region(region,dict(max_vertices=100,feature_angle_deg=30,lock_boundary_vertices=False,max_rounds=12))
        self.assertLessEqual(len(simple['xyz']),100)
        self.assertTrue((simple['faces'] < len(simple['xyz'])).all())
        for key in ('xyz','normal','curvature'):np.testing.assert_array_equal(simple[key],case[key][simple['vertex_ids']])
        reduced=trimesh.Trimesh(simple['xyz'],simple['faces'],process=False)
        self.assertEqual(reduced.euler_number,mesh.euler_number)
        self.assertTrue(reduced.is_winding_consistent)
        with self.assertRaises(ValueError):extract_region(case,'',np.zeros(count))

    def test_expert_trajectory_reward_and_schedule(self):
        cfg=configuration('configs/semantic_pipeline.json');o=cfg['rl']['trajectory_curriculum']
        self.assertEqual(weights(o,0,100),(1.,0.));self.assertEqual(weights(o,99,100),(0.,1.))
        w=[weights(o,i,100)[0] for i in range(100)];self.assertTrue(np.all(np.diff(w)<=0))
        r=Rotation.from_euler('z',[3],degrees=True).as_matrix();t=np.array([[.5,0,0]])
        a=expert_actions(r,t,2,.4);nr=Rotation.from_rotvec(np.deg2rad(a[:,:3]*2)).as_matrix()@r;nt=t+a[:,3:]*.4
        reward,expert,teacher,ew,tw=trajectory_reward(r,t,nr,nt,np.array([0.]),np.array([-.2]),2,.4,o,0,100)
        self.assertGreater(expert[0],0);np.testing.assert_allclose(reward,expert)
        late,*_=trajectory_reward(r,t,nr,nt,np.array([0.]),np.array([-.2]),2,.4,o,99,100)
        np.testing.assert_allclose(late,[-.4])
        zero,*_=trajectory_reward(r,t,r,t,np.array([0.]),np.array([0.]),2,.4,o,0,100)
        np.testing.assert_allclose(zero,[0.])

    def test_variable_mesh_candidate_count(self):
        case={'xyz':np.random.RandomState(1).randn(300,3)}
        options={'candidate_points':4095,'candidate_source':'all_mesh'}
        np.testing.assert_array_equal(candidate_indices(case,options),np.arange(300))
        model=OcclusionTeacher(sample_count=16,width=8,k=4,encoder_layers=1,**options)
        self.assertEqual(tuple(model(torch.randn(1,16,8)).shape),(1,))
        with self.assertRaises(ValueError):candidate_indices(case,{'candidate_points':4095})

    def test_all_vertices_and_padding_invariance(self):
        from occlusion_rl_offline.data import pad_point_batches
        rng=np.random.RandomState(3)
        xyz=rng.randn(35,3).astype(np.float32)
        case=dict(xyz=xyz,normal=np.tile([0.,0.,1.],(35,1)),curvature=np.arange(35),
                  lower_xyz=xyz-2,lower_normal=np.tile([0.,0.,1.],(35,1)))
        opts=dict(candidate_source='all_mesh',candidate_points=4095)
        points,ids=features_at_pose(case,np.eye(3),np.zeros(3),4095,opts,True)
        np.testing.assert_array_equal(ids,np.arange(35))
        np.testing.assert_allclose(points[:,:3],xyz,atol=1e-6)
        np.testing.assert_array_equal(points[:,6],case['curvature'])
        model=OcclusionTeacher(sample_count=4095,width=8,k=4,encoder_layers=2,**opts)
        a=torch.from_numpy(points[None]);b=a[:,:19].clone()
        padded=torch.from_numpy(np.concatenate(pad_point_batches([a.numpy(),b.numpy()])))
        expected=torch.cat([model(a),model(b)])
        actual=model(padded)
        torch.testing.assert_close(actual,expected)
        actual.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_current_configuration_ranking_and_holdout(self):
        import json
        from occlusion_rl_offline.semantic_pipeline.__main__ import configure
        cfg=configuration('configs/semantic_pipeline.json')
        split=json.loads(Path(cfg['source_manifest']).read_text())['splits']
        self.assertEqual([e['id'] for e in split['test']],['011'])
        self.assertNotIn('011',[e['id'] for k in ('train','val') for e in split[k]])
        with tempfile.TemporaryDirectory() as folder:
            cfg['out']=folder;configure(cfg)
            generated=json.loads((Path(folder)/'rl_config.json').read_text())
            self.assertTrue(generated['paths']['packaged_teacher'].endswith('/ranking/best.pt'))
            self.assertEqual(generated['teacher']['calibration_epochs'],0)
            self.assertEqual(generated['model']['candidate_source'],'all_mesh')
            self.assertEqual(generated['agent']['horizon'],16)

    def test_five_bands_and_smoke_isolation(self):
        full=configuration('configs/semantic_pipeline.json');smoke=configuration('configs/semantic_pipeline.json',True)
        self.assertEqual(len(full['pose_bands']),5)
        self.assertNotEqual(full['out'],smoke['out'])
        self.assertEqual(full['simplification'],smoke['simplification'])
        self.assertNotIn('fps_points', full)


class RecoveryViewerTests(unittest.TestCase):
    def test_start_prefixes_and_zero_band(self):
        from occlusion_rl_offline.semantic_pipeline.recovery_viewer import draw_starts
        band=dict(rotation_deg=8.,translation_mm=4.)
        r,t=draw_starts('010',3,band,2);rr,tt=draw_starts('010',3,band,5)
        np.testing.assert_array_equal(r,rr[:2]);np.testing.assert_array_equal(t,tt[:2])
        r,t=draw_starts('011',0,dict(rotation_deg=0.,translation_mm=0.),1)
        np.testing.assert_array_equal(r,np.eye(3)[None]);np.testing.assert_array_equal(t,np.zeros((1,3)))

    def test_pose_frame_matrix_preserves_vertex_positions(self):
        from occlusion_rl_offline.semantic_pipeline.recovery_viewer import pose_frame
        xyz=np.array([[1.,2.,3.],[4.,5.,6.],[2.,-1.,7.]])
        r=Rotation.from_euler('xyz',[3.,-5.,7.],degrees=True).as_matrix();t=np.array([1.,-.4,.2])
        moved=(xyz-xyz.mean(0))@r.T+xyz.mean(0)+t
        points=np.zeros((3,8));points[:,:3]=moved
        f=pose_frame(dict(xyz=xyz),r,t,points,np.arange(3),{'teacher':-.5})
        matrix=np.array(f['matrix'])
        np.testing.assert_allclose(xyz@matrix[:3,:3].T+matrix[:3,3],moved)
        self.assertAlmostEqual(f['translation_mm'],np.linalg.norm(t))
        self.assertEqual(f['selected_ids'],[0,1,2])

if __name__=='__main__':unittest.main()
