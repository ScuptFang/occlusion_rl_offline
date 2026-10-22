"""Original-vertex voxel maps and feature-aware vertex decimation."""
import numpy as np
from scipy.spatial import cKDTree


def voxel_sample(xyz, size_mm, max_points=4096):
    """One original vertex nearest each occupied voxel's centroid, with exact inverse.

    A centered grid avoids translation-dependent sampling. Rotation changes the
    grid occupancy, intentionally also exercised during segmentation training.
    If necessary increase voxel size, never random-drop occupied voxels.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite nonempty Nx3 coordinates required')
    if not np.isfinite(size_mm) or size_mm <= 0 or max_points < 4:
        raise ValueError('Positive voxel size and max_points >= 4 required')
    centered = xyz.astype(np.float64) - xyz.mean(0, dtype=np.float64)
    size = float(size_mm)
    for _ in range(100):
        keys = np.floor(centered / size).astype(np.int64)
        cells, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        if len(cells) <= max_points:
            break
        size *= max(1.05, (len(cells) / max_points) ** (1 / 3))
    else:
        raise ValueError('Voxel sampling did not meet memory budget')
    centers = np.zeros((len(cells), 3), dtype=np.float64)
    np.add.at(centers, inverse, xyz)
    centers /= counts[:, None]
    distances = ((xyz - centers[inverse]) ** 2).sum(1)
    order = np.lexsort((np.arange(len(xyz)), distances, inverse))
    ids = order[np.r_[True, np.diff(inverse[order]) != 0]]
    assert np.array_equal(inverse[ids], np.arange(len(ids)))
    return dict(ids=ids, inverse=inverse, cells=cells, size_mm=size)


def extract_region(case, prefix, probability, threshold=.5):
    key = 'lower_faces' if prefix else 'upper_faces'
    original_faces = case[key]
    face_ids = np.flatnonzero((probability[original_faces] >= threshold).all(1))
    if not len(face_ids):
        raise ValueError('Segmentation produced no complete faces; train/inspect segmenter first')
    vertex_ids, inverse = np.unique(original_faces[face_ids], return_inverse=True)
    return dict(xyz=case[prefix+'xyz'][vertex_ids], normal=case[prefix+'normal'][vertex_ids],
                curvature=case[prefix+'curvature'][vertex_ids], faces=inverse.reshape(-1, 3),
                vertex_ids=vertex_ids, face_ids=face_ids)


def simplify_region(region, options):
    """Feature-angle constrained vertex decimation with original attribute mapping.

    VTK DecimatePro enables topology preservation, classifies sharp edges by face angle,
    and removes vertices without moving retained vertices. Faces change.
    A strict budget failure never falls back to arbitrary point dropping.
    """
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkDecimatePro, vtkCleanPolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    maximum = options['max_vertices']
    if maximum < 4 or maximum >= 4096:
        raise ValueError('Require 4 <= max_vertices < 4096')
    xyz, faces = region['xyz'], region['faces']
    if len(xyz) <= maximum:
        return dict(region, simplification_rounds=0, source_face_ids=region['face_ids'])
    points = vtkPoints(); points.SetData(numpy_to_vtk(xyz.astype(float), deep=True))
    cells = vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces),3),faces)).astype(np.int64).ravel(), deep=True))
    mesh = vtkPolyData(); mesh.SetPoints(points); mesh.SetPolys(cells)
    iterations = 0
    for _ in range(options.get('max_rounds', 12)):
        if mesh.GetNumberOfPoints() <= maximum: break
        previous = mesh.GetNumberOfPoints()
        dec = vtkDecimatePro(); dec.SetInputData(mesh)
        dec.PreserveTopologyOn(); dec.SplittingOff(); dec.AccumulateErrorOn()
        dec.SetFeatureAngle(options['feature_angle_deg'])
        dec.SetBoundaryVertexDeletion(not options['lock_boundary_vertices'])
        dec.SetTargetReduction(1-min(.85,maximum/previous*.85)); dec.Update()
        clean = vtkCleanPolyData(); clean.SetInputData(dec.GetOutput()); clean.PointMergingOff(); clean.Update()
        mesh = vtkPolyData(); mesh.DeepCopy(clean.GetOutput()); iterations += 1
        if mesh.GetNumberOfPoints() >= previous:
            raise ValueError('Feature/topology constraints prevent reaching vertex budget; inspect mesh or adjust feature_angle_deg')
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    cells = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1,4)
    if not np.all(cells[:,0] == 3): raise ValueError('Nontriangular decimator output')
    if len(points) > maximum:
        raise ValueError('Decimator did not reach vertex budget with feature constraints')
    distance, ids = cKDTree(xyz).query(points)
    if distance.max() > 1e-6 or len(np.unique(ids)) != len(ids):
        raise ValueError('Decimation lost exact original vertex identities')
    result = {key: region[key][ids] for key in ('xyz', 'normal', 'curvature', 'vertex_ids')}
    result.update(faces=cells[:,1:].astype(np.int64), source_face_ids=region['face_ids'], simplification_rounds=iterations)
    return result
