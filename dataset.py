"""Create reproducible splits from the local wildfire JPEGs."""
import csv
import hashlib
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

CLASSES = ("nofire", "fire")


def source_group(path):
    stem = re.sub(r"\(\d+\)$", "", path.stem.casefold())
    stem = re.sub(r"_f$", "", stem)
    # Some videos have both a frame suffix and a numeric crop suffix.
    match = re.fullmatch(r"(.+?)(?:_\d+)+", stem)
    if match:
        group = match[1].removesuffix("_onlycloud")
        variant = re.fullmatch(r"(pos|neg)(?:video)?0*(\d+)", group)
        if variant:
            return f"{variant[1]}-{int(variant[2])}"
        return group
    # Numeric filenames have no usable recording ID. Keep them out of holdouts.
    return "unknown"


def fingerprint(path):
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        size = str(rgb.size).encode()
        return min(hashlib.sha256(size + pixels).hexdigest()
                   for pixels in (rgb.tobytes(), ImageOps.mirror(rgb).tobytes()))


def read_recordings(path, expected_paths):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "recording_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Recording CSV needs path and recording_id columns")
        groups = {}
        for row in reader:
            image = (row["path"] or "").strip()
            group = (row["recording_id"] or "").strip()
            if not image or not group or image in groups:
                raise ValueError(f"Missing recording ID or duplicate path: {image!r}")
            groups[image] = f"recording:{group}"
    if groups.keys() != expected_paths:
        missing = len(expected_paths - groups.keys())
        extra = len(groups.keys() - expected_paths)
        raise ValueError(f"Recording CSV must cover the dataset exactly: {missing} missing, {extra} extra")
    return groups


def split_dataset(data_dir, seed, recordings_path=None):
    parent = {}

    def find(group):
        parent.setdefault(group, group)
        while parent[group] != group:
            group = parent[group]
        return group

    def merge(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a

    images = [sorted((data_dir / name).glob("*.jpg")) for name in CLASSES]
    recordings = None
    if recordings_path is not None:
        expected = {p.relative_to(data_dir).as_posix() for paths in images for p in paths}
        recordings = read_recordings(recordings_path, expected)
    copies = defaultdict(list)
    for label, (name, paths) in enumerate(zip(CLASSES, images)):
        if not paths:
            raise ValueError(f"No JPEGs found under {data_dir / name}")
        for path in paths:
            group = (recordings[path.relative_to(data_dir).as_posix()] if recordings is not None
                     else source_group(path))
            find(group)
            copies[fingerprint(path)].append((path, label, group))

    for items in copies.values():
        for item in items[1:]:
            merge(items[0][2], item[2])
    retained = []
    duplicates, conflicts = 0, 0
    for digest, items in copies.items():
        if len({item[1] for item in items}) != 1:
            conflicts += len(items)
            continue
        path, label, group = items[0]
        duplicates += len(items) - 1
        retained.append((path, label, find(group), digest))

    groups = sorted({row[2] for row in retained} - {find("unknown")})
    counts = np.array([[sum(row[1] == label and row[2] == g for row in retained)
                        for label in range(2)] for g in groups])
    total = np.bincount([row[1] for row in retained], minlength=2)
    target = total * np.array([[0.15], [0.15]])
    rng = np.random.default_rng(seed)
    best = None
    # Seek class-balanced holdouts by searching whole-group assignments.
    for _ in range(5000):
        categories = rng.choice(3, size=len(groups), p=[0.5, 0.25, 0.25])
        actual = np.array([counts[categories == k].sum(axis=0) for k in (1, 2)])
        if (actual == 0).any() or (total - actual.sum(axis=0) <= 0).any():
            continue
        score = np.sum(((actual - target) / target) ** 2)
        if best is None or score < best[0]:
            best = (score, categories)
    if best is None:
        raise ValueError("Could not form group-disjoint splits containing both classes")
    assignments = dict(zip(groups, best[1]))
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in retained:
        subset = ("train", "validation", "test")[assignments.get(row[2], 0)]
        splits[subset].append(row)
    return splits, dict(duplicates_removed=duplicates, conflicting_images_removed=conflicts,
                        untracked_source_images=sum(row[2] == find("unknown") for row in retained),
                        recording_metadata="manifest" if recordings is not None else "filename")


def write_manifest(path, data_dir, splits):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "path", "class", "source_group", "sha256_pixels_or_mirror"))
        for split, rows in splits.items():
            for image, label, group, digest in rows:
                writer.writerow((split, image.relative_to(data_dir).as_posix(), CLASSES[label],
                                 group, digest))
