"""Verify the numbered STL pairs in ../data against docs/data_catalog.json."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def file_info(path):
    size = path.stat().st_size
    if size < 84:
        raise ValueError(f"STL too short: {path}")
    with path.open("rb") as f:
        header = f.read(84)
        faces = struct.unpack("<I", header[80:84])[0]
        if size != 84 + 50 * faces:
            raise ValueError(f"Invalid binary STL length: {path}")
        f.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": size, "triangles": faces, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--catalog", type=Path, default=Path("docs/data_catalog.json"))
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    pairs = catalog["pairs"]
    expected_dirs = {f"{i:03d}" for i in range(1, len(pairs) + 1)}
    actual_dirs = set()
    for root, dirs, files in os.walk(args.data_dir):
        dirs[:] = [name for name in dirs if name != "tmp"]
        relative = Path(root).relative_to(args.data_dir)
        if relative == Path("."):
            actual_dirs.update(dirs)
            if files:
                raise ValueError(f"Files found directly in data: {files}")
        elif relative.parts[0] in expected_dirs:
            if len(relative.parts) != 1 or dirs or set(files) != {"upper.stl", "lower.stl"}:
                raise ValueError(f"Expected exactly two STL files in {relative}, found {files} and directories {dirs}")
    if actual_dirs != expected_dirs:
        raise ValueError(f"Numbered directories mismatch: expected {sorted(expected_dirs)}, found {sorted(actual_dirs)}")
    seen = set()
    for i, pair in enumerate(pairs, 1):
        number = f"{i:03d}"
        if pair["number"] != number:
            raise ValueError(f"Non-sequential catalog entry: {pair['number']}")
        for jaw in ("upper", "lower"):
            item = pair["files"][jaw]
            relative = f"{number}/{jaw}.stl"
            if item["path"] != relative:
                raise ValueError(f"Path mismatch in catalog: {item['path']}")
            actual = file_info(args.data_dir / relative)
            expected = {key: item[key] for key in ("bytes", "triangles", "sha256")}
            if actual != expected:
                raise ValueError(f"File mismatch: {relative}")
            if actual["sha256"] in seen:
                raise ValueError(f"Duplicate STL: {relative}")
            seen.add(actual["sha256"])
    print(f"Verified {len(pairs)} numbered pairs ({sum(p['role'] == 'training_candidate' for p in pairs)} training candidates, "
          f"{sum(p['role'] == 'demo' for p in pairs)} demo); all {len(seen)} STL files match the catalog.")


if __name__ == "__main__":
    main()
