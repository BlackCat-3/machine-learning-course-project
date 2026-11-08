"""Validate the competition archive and create a deterministic, leakage-safe cache.

The original competition IDs and label mapping are retained explicitly. Identical
decoded grayscale pixels are always assigned to the same validation partition.
No external labels or data are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


LABEL_MAP = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def pixel_hash(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def grouped_stratified_split(labels, hashes, fraction=0.15, seed=42):
    """Approximately stratify by label without splitting exact-pixel groups.

    Mixed-label groups are allocated as whole groups first. Pure-label groups
    then fill each class's validation target to the nearest attainable size.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie strictly between 0 and 1")
    groups = defaultdict(list)
    for index, digest in enumerate(hashes):
        groups[digest].append(index)
    rng = np.random.default_rng(seed)
    target = np.rint(np.bincount(labels, minlength=6) * fraction).astype(np.int64)
    val_counts = np.zeros(6, dtype=np.int64)
    val_groups = set()
    pure_groups = defaultdict(list)
    mixed_groups = []
    for digest, indices in groups.items():
        group_labels = np.unique(labels[indices])
        if len(group_labels) == 1:
            pure_groups[int(group_labels[0])].append(digest)
        else:
            mixed_groups.append(digest)
    rng.shuffle(mixed_groups)
    for digest in mixed_groups:
        if rng.random() < fraction:
            val_groups.add(digest)
            val_counts += np.bincount(labels[groups[digest]], minlength=6)
    for label in range(6):
        candidates = pure_groups[label]
        rng.shuffle(candidates)
        for digest in candidates:
            count = len(groups[digest])
            old_distance = abs(int(target[label] - val_counts[label]))
            new_distance = abs(int(target[label] - val_counts[label] - count))
            if new_distance < old_distance:
                val_groups.add(digest)
                val_counts[label] += count
    mask = np.fromiter((digest in val_groups for digest in hashes), dtype=bool)
    train_indices = np.flatnonzero(~mask).astype(np.int64)
    val_indices = np.flatnonzero(mask).astype(np.int64)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Unable to produce two nonempty partitions")
    return train_indices, val_indices, groups


def class_counts(labels):
    counts = np.bincount(labels, minlength=6)
    return {name: int(counts[label]) for name, label in LABEL_MAP.items()}


def duplicate_summary(groups, labels=None):
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    summary = {
        "unique_pixel_groups": len(groups),
        "duplicate_pixel_groups": len(duplicates),
        "images_in_duplicate_groups": sum(len(indices) for indices in duplicates),
        "extra_duplicate_images": sum(len(indices) - 1 for indices in duplicates),
        "maximum_group_size": max((len(indices) for indices in groups.values()), default=0),
    }
    if labels is not None:
        conflicts = [indices for indices in duplicates if len(np.unique(labels[indices])) > 1]
        summary["conflicting_label_groups"] = len(conflicts)
        summary["images_in_conflicting_label_groups"] = sum(map(len, conflicts))
    return summary


def prepare(zip_path: Path, submission_path: Path, output_dir: Path, seed: int, val_fraction: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Emotion"]:
            raise ValueError(f"Expected CSV columns ID,Emotion; got {reader.fieldnames!r}")
        template = list(reader)
    test_ids = [row["ID"] for row in template]
    if not test_ids or len(set(test_ids)) != len(test_ids):
        raise ValueError("Submission template has no rows or repeated IDs")
    if any(not value or PurePosixPath(value).name != value for value in test_ids):
        raise ValueError("Submission IDs must be nonempty image basenames")
    audit = {
        "archive": str(zip_path.resolve()),
        "submission_template_source": str(submission_path.resolve()),
        "label_mapping": LABEL_MAP,
        "seed": seed,
        "validation_fraction_requested": val_fraction,
        "split_method": "Exact decoded-pixel groups; mixed-label groups assigned jointly, then class targets filled",
        "bad_images": [],
    }
    bad_images = audit["bad_images"]
    source_sizes = Counter()
    source_modes = Counter()
    source_formats = Counter()
    with zipfile.ZipFile(zip_path) as archive:
        corrupt_entry = archive.testzip()
        audit["zip_crc_ok"] = corrupt_entry is None
        if corrupt_entry is not None:
            raise ValueError(f"Archive CRC failed for {corrupt_entry}")
        train_entries = []
        test_entries = {}
        ignored_entries = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                ignored_entries.append(info.filename)
                continue
            if len(path.parts) >= 3 and path.parts[-3] == "train":
                class_name = path.parts[-2]
                if class_name not in LABEL_MAP:
                    raise ValueError(f"Unexpected training class: {class_name!r}")
                train_entries.append((info.filename, LABEL_MAP[class_name], path.name))
            elif len(path.parts) >= 2 and path.parts[-2] == "test":
                if path.name in test_entries:
                    raise ValueError(f"Repeated test basename: {path.name}")
                test_entries[path.name] = info.filename
            else:
                raise ValueError(f"Image outside expected train/class or test folder: {info.filename}")
        if set(test_entries) != set(test_ids):
            missing = sorted(set(test_ids) - set(test_entries))
            extra = sorted(set(test_entries) - set(test_ids))
            raise ValueError(f"Template/test mismatch; missing={missing[:10]}, extra={extra[:10]}")
        train_entries.sort(key=lambda item: (item[1], item[2]))
        train_ids = [entry[2] for entry in train_entries]
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("Repeated training image basenames")
        if not train_entries:
            raise ValueError("No training images found")
        audit["ignored_archive_entries"] = ignored_entries
        audit["train_count"] = len(train_entries)
        audit["test_count"] = len(test_ids)
        audit["template_ids_match_test"] = True
        audit["train_test_id_overlap"] = len(set(train_ids) & set(test_ids))
        train_images = np.empty((len(train_entries), 48, 48), dtype=np.uint8)
        test_images = np.empty((len(test_ids), 48, 48), dtype=np.uint8)
        train_labels = np.asarray([entry[1] for entry in train_entries], dtype=np.int64)
        train_hashes = []
        test_hashes = []

        def decode(entry, destination, index, hashes):
            try:
                raw = archive.read(entry)
                with Image.open(io.BytesIO(raw)) as picture:
                    picture.load()
                    source_sizes[f"{picture.width}x{picture.height}"] += 1
                    source_modes[picture.mode] += 1
                    source_formats[str(picture.format)] += 1
                    if picture.size != (48, 48):
                        raise ValueError(f"Expected 48x48, got {picture.size}")
                    if picture.mode != "L":
                        raise ValueError(f"Expected grayscale mode L, got {picture.mode}")
                    pixels = np.asarray(picture, dtype=np.uint8)
                    destination[index] = pixels
                    hashes.append(pixel_hash(pixels))
            except Exception as error:
                bad_images.append({"file": entry, "error": f"{type(error).__name__}: {error}"})

        for index, (entry, _, _) in enumerate(train_entries):
            decode(entry, train_images, index, train_hashes)
        for index, identifier in enumerate(test_ids):
            decode(test_entries[identifier], test_images, index, test_hashes)
    audit["source_image_sizes"] = dict(source_sizes)
    audit["source_image_modes"] = dict(source_modes)
    audit["source_image_formats"] = dict(source_formats)
    audit["bad_image_count"] = len(bad_images)
    if bad_images:
        (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Found {len(bad_images)} bad images; see {output_dir / 'audit.json'}")
    train_indices, val_indices, train_groups = grouped_stratified_split(
        train_labels, train_hashes, fraction=val_fraction, seed=seed
    )
    test_groups = defaultdict(list)
    for index, digest in enumerate(test_hashes):
        test_groups[digest].append(index)
    shared_hashes = set(train_groups) & set(test_groups)
    train_partition_hashes = {train_hashes[index] for index in train_indices}
    val_partition_hashes = {train_hashes[index] for index in val_indices}
    split_overlap = len(train_partition_hashes & val_partition_hashes)
    if split_overlap:
        raise AssertionError("Identical pixels leaked across train and validation")
    audit["class_counts"] = class_counts(train_labels)
    audit["train_duplicates"] = duplicate_summary(train_groups, train_labels)
    audit["test_duplicates"] = duplicate_summary(test_groups)
    audit["train_test_exact_pixel_overlap"] = {
        "shared_pixel_groups": len(shared_hashes),
        "training_images": sum(len(train_groups[key]) for key in shared_hashes),
        "test_images": sum(len(test_groups[key]) for key in shared_hashes),
        "shared_groups_with_conflicting_training_labels": sum(
            len(np.unique(train_labels[train_groups[key]])) > 1 for key in shared_hashes
        ),
    }
    audit["split"] = {
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "actual_val_fraction": len(val_indices) / len(train_labels),
        "train_class_counts": class_counts(train_labels[train_indices]),
        "val_class_counts": class_counts(train_labels[val_indices]),
        "train_val_shared_pixel_hash_count": split_overlap,
        "index_overlap_count": len(set(train_indices.tolist()) & set(val_indices.tolist())),
        "every_training_index_assigned_once": len(train_indices) + len(val_indices) == len(train_labels),
    }
    audit["array_contract"] = {
        "train_images": "uint8[N,48,48] grayscale pixels in range 0..255",
        "train_labels": "int64[N], explicit competition mapping",
        "train_ids": "unicode[N], original image filenames; rows sorted by label then ID",
        "test_images": "uint8[M,48,48] in submission template row order",
        "test_ids": "unicode[M], IDs in submission template row order",
        "train_indices": "int64[T], fitting partition indices into train_images/train_labels",
        "val_indices": "int64[V], holdout partition indices into train_images/train_labels",
    }
    np.savez_compressed(
        output_dir / "cache.npz",
        train_images=train_images,
        train_labels=train_labels,
        train_ids=np.asarray(train_ids),
        test_images=test_images,
        test_ids=np.asarray(test_ids),
        train_indices=train_indices,
        val_indices=val_indices,
    )
    destination_template = output_dir / "submission_template.csv"
    if submission_path.resolve() != destination_template.resolve():
        shutil.copyfile(submission_path, destination_template)
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "cache": str((output_dir / "cache.npz").resolve()),
        "train_count": audit["train_count"],
        "test_count": audit["test_count"],
        "bad_image_count": audit["bad_image_count"],
        "class_counts": audit["class_counts"],
        "train_duplicates": audit["train_duplicates"],
        "train_test_exact_pixel_overlap": audit["train_test_exact_pixel_overlap"],
        "split": audit["split"],
    }, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("D:/Downloads/2026task2.zip"))
    parser.add_argument("--submission", type=Path, default=Path("D:/Downloads/submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    prepare(args.zip, args.submission, args.output_dir, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()
