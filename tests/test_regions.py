import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from occlusion_rl_offline.data import load_case, features_at_pose
from occlusion_rl_offline.region_segmentation import (RegionPointTransformer, digest,
                                                       predict_probabilities)
from occlusion_rl_offline.simulate import perturb_case
from occlusion_rl_offline.train_regions import reference_labels, example, train, prepare
from occlusion_rl_offline.cohort import read_manifest


class RegionTests(unittest.TestCase):
    def test_prepare_from_stl_without_old_cache_and_without_label_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = {}
            for index, split in enumerate(('train', 'val', 'test')):
                identity = '%03d' % (index + 1)
                path = root / 'data' / identity; path.mkdir(parents=True)
                mesh = trimesh.creation.icosphere(subdivisions=1)
                mesh.export(path / 'upper.stl')
                mesh.apply_translation([0., 0., 4.])
                mesh.export(path / 'lower.stl')
                splits[split] = [dict(id=identity, case='missing_old_cache/' + identity + '.npz')]
            manifest = root / 'source.json'
            manifest.write_text(json.dumps(dict(version=1, splits=splits)))
            prepare(dict(out=str(root / 'out'), source_manifest=str(manifest),
                         data_root=str(root / 'data'), label_distance_mm=5.))
            for paths in read_manifest(root / 'out/manifest.json').values():
                with np.load(paths[0]) as data:
                    self.assertEqual(set(data.files), {'xyz', 'normal', 'curvature', 'lower_xyz',
                                                     'lower_normal', 'lower_curvature', 'lower_faces'})
                self.assertTrue((paths[0].parent / 'region_labels.npz').is_file())

    def test_training_checkpoint_can_be_used_for_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mesh = trimesh.creation.icosphere(subdivisions=1)
            xyz = np.asarray(mesh.vertices, dtype=np.float32)
            normal = np.asarray(mesh.vertex_normals, dtype=np.float32)
            splits = {}
            for i, split in enumerate(('train', 'val', 'test')):
                path = root / ('%03d' % (i + 1)); path.mkdir()
                np.savez(path / 'case.npz', xyz=xyz, normal=normal, curvature=np.zeros(len(xyz), np.float32),
                         lower_xyz=xyz, lower_normal=normal, lower_curvature=np.zeros(len(xyz), np.float32))
                label = (xyz[:, 2] >= 0).astype(np.float32)
                np.savez(path / 'region_labels.npz', upper=label, lower=label)
                splits[split] = [dict(id=path.name, case=path.name + '/case.npz')]
            (root / 'manifest.json').write_text(json.dumps(dict(version=1, splits=splits)))
            cfg = dict(out=str(root), torch_threads=1, seed=42, runtime=dict(device='cpu'),
                       architecture=dict(width=8, layers=1, k=4), sampling=dict(points=16, candidates=32, seed=42),
                       validation_poses_per_jaw=1, train_poses_per_jaw=1,
                       augmentation=dict(rotation_deg=5., translation_mm=1.),
                       learning_rate=.001, weight_decay=0., epochs=1, batch_size=2, dice_weight=1.)
            train(cfg)
            checkpoint = root / 'segmentation/best.pt'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['train_cases'], ['001'])
            self.assertEqual(saved['validation_cases'], ['002'])
            prob = predict_probabilities(xyz, normal, np.zeros(len(xyz)), 0,
                                         dict(checkpoint=str(checkpoint), sha256=digest(checkpoint)))
            self.assertEqual(prob.shape, (len(xyz),))
            self.assertTrue(np.isfinite(prob).all())

    def test_surface_labels_and_pose_identity(self):
        # Distance must be to the triangle interior, not sparse opposing vertices.
        upper = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        lower = trimesh.Trimesh(vertices=[[-10, -10, 4.9], [10, -10, 4.9], [0, 10, 4.9]], faces=[[0, 1, 2]], process=False)
        uy, ly, ud, ld = reference_labels(upper, lower, 5.)
        np.testing.assert_array_equal(uy, np.ones(3))
        np.testing.assert_array_equal(ly, np.zeros(3))
        np.testing.assert_allclose(ud, 4.9, atol=1e-5)
        lower.apply_translation([0, 0, .2])
        self.assertFalse(reference_labels(upper, lower, 5.)[0].any())
        item = dict(xyz=np.asarray(upper.vertices), normal=np.asarray(upper.vertex_normals),
                    curvature=np.arange(3), label=np.array([1., 0., 1.]), jaw=0)
        f, y, ids = example(item, dict(points=3, candidates=3), 42, 25., 10.)
        np.testing.assert_array_equal(y, item['label'][ids])
        np.testing.assert_array_equal(f[:, 6], item['curvature'][ids])
        self.assertFalse(np.allclose(f[:, :3], item['xyz'][ids]))

    def test_transformer_translation_permutation_and_gradients(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        net = RegionPointTransformer(width=8, layers=2, k=4)
        x = torch.randn(2, 24, 8)
        pred = net(x)
        shifted = x.clone(); shifted[..., :3] += torch.tensor([10., -6., 4.])
        torch.testing.assert_close(net(shifted), pred, atol=1e-5, rtol=1e-5)
        permutation = torch.randperm(24)
        torch.testing.assert_close(net(x[:, permutation]), pred[:, permutation], atol=1e-5, rtol=1e-5)
        pred.square().mean().backward()
        self.assertGreater(float(net.blocks[0].attention[0].weight.grad.abs().sum()), 0.)

    def test_runtime_region_no_ground_truth_labels_and_hash(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model = RegionPointTransformer(width=8, layers=1, k=4)
            checkpoint = folder / 'segment.pt'
            torch.save(dict(model=model.state_dict(), architecture=dict(width=8, layers=1, k=4),
                            sampling=dict(points=16, candidates=32, seed=42)), checkpoint)
            region = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), threshold=.5)
            mesh = trimesh.creation.icosphere(subdivisions=2)
            xyz = np.asarray(mesh.vertices, dtype=np.float32)
            normal = np.asarray(mesh.vertex_normals, dtype=np.float32)
            case_path = folder / 'case.npz'
            np.savez(case_path, xyz=xyz, normal=normal, curvature=np.zeros(len(xyz)),
                     lower_xyz=xyz - [0, 0, 3], lower_normal=normal,
                     lower_faces=mesh.faces, lower_curvature=np.zeros(len(xyz)))
            case = load_case(case_path)
            perturbed = folder / 'observed.npz'
            perturb_case(case_path, perturbed, [1., 2., 3.], [.1, .2, .3])
            np.testing.assert_array_equal(load_case(perturbed)['lower_curvature'], case['lower_curvature'])
            opts = dict(candidate_points=32, distance_method='nearest_surface', segmentation_region=region)
            # Only predictions are permitted; deliberately contradictory labels must have no effect.
            case['region_label'] = np.ones(len(xyz))
            f, ids = features_at_pose(case, np.eye(3), np.zeros(3), 8, opts, True)
            case['region_label'] *= 0
            g, ids2 = features_at_pose(case, np.eye(3), np.zeros(3), 8, opts, True)
            np.testing.assert_array_equal(f, g)
            np.testing.assert_array_equal(ids, ids2)
            self.assertEqual(f.shape, (8, 8))
            self.assertTrue(np.isfinite(f).all())
            r = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
            changed = features_at_pose(case, r, np.ones(3), 8, opts)
            self.assertFalse(np.array_equal(f, changed))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                predict_probabilities(xyz, normal, np.zeros(len(xyz)), 0, dict(region, sha256='0' * 64))


if __name__ == '__main__':
    unittest.main()
