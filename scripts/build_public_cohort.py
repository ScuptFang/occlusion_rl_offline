"""Build a local training cohort from independently downloaded public jaw pairs.

Run from the repository root after placing the numbered pairs in ../data.
The output is local experiment input; it is intentionally not committed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def cases(source, catalog):
    for item in json.loads(catalog.read_text())["pairs"]:
        yield item["case_id"], source / item["number"], item["training_transform"]


TRANSFORMS = {
    "identity": np.eye(4),
    # Available for catalogs that still contain a Y-vertical source pair.
    "y_to_z": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../data"))
    parser.add_argument("--catalog", type=Path, default=Path("docs/data_catalog.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/public_cohort/raw"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    for name, folder, transform_name in cases(args.source, args.catalog):
        target = args.out / name
        target.mkdir(exist_ok=True)
        record = {"case_id": name, "source_dir": str(folder.resolve()), "transform": transform_name}
        meshes = {}
        for jaw in ("upper", "lower"):
            path = folder / (jaw + ".stl")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate scan: {path}")
            seen.add(digest)
            mesh = trimesh.load(path, process=False)
            mesh.apply_transform(TRANSFORMS[transform_name])
            meshes[jaw] = mesh
            mesh.export(target / (jaw + ".stl"))
            record[jaw] = {"sha256_source": digest, "bounds_mm": mesh.bounds.tolist(), "faces": len(mesh.faces)}
        upper, lower = meshes["upper"], meshes["lower"]
        z_delta = float(upper.centroid[2] - lower.centroid[2])
        distance = cKDTree(lower.vertices[::40]).query(upper.vertices[::40])[0]
        near_5pct = float(np.quantile(distance, 0.05))
        if z_delta <= 0 or near_5pct > 2.0:
            raise ValueError(f"Plausibility check failed for {name}: z delta {z_delta:.2f}, near 5% {near_5pct:.2f}")
        record["geometry_qc"] = {"upper_minus_lower_centroid_z_mm": z_delta,
                                 "sampled_nearest_distance_5pct_mm": near_5pct}
        manifest.append(record)
        print(name, flush=True)
    (args.out.parent / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
