"""Strict data and submission contracts for the fovea localisation task.

Coordinates use the original image pixel frame: ``0 <= x < width`` and
``0 <= y < height``.  Only the pair ``(0, 0)`` is the invisible-fovea
sentinel; a single zero coordinate can be a valid visible image-edge point.
CSV files with CRCRLF or blank lines
are accepted, but malformed rows, duplicate records and missing axes are not.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np


_IMAGE_AXIS = re.compile(r"([1-9][0-9]*)_Fovea_([XY])\Z")
_POSITIVE_ID = re.compile(r"[1-9][0-9]*\Z")


def _rows(path: str | Path, header: Sequence[str]) -> list[list[str]]:
    """Read CSV without mistaking the extra CR in CRCRLF for a data row."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [
            row for row in csv.reader(stream)
            if row and not (len(row) == 1 and not row[0].strip())
        ]
    if not rows or rows[0] != list(header):
        raise ValueError(f"{path}: expected CSV header {list(header)!r}")
    records = rows[1:]
    if not records:
        raise ValueError(f"{path}: CSV contains no records")
    for number, row in enumerate(records, start=2):
        if len(row) != len(header):
            raise ValueError(f"{path}: record {number} has {len(row)} fields; expected {len(header)}")
    return records


def _coordinate_pair(value: Sequence[float], context: str) -> np.ndarray:
    try:
        pair = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: coordinates must be numeric") from exc
    if pair.shape != (2,):
        raise ValueError(f"{context}: expected exactly two coordinates, got shape {pair.shape}")
    if not np.isfinite(pair).all():
        raise ValueError(f"{context}: coordinates must be finite")
    if (pair < 0).any():
        raise ValueError(f"{context}: coordinates cannot be negative")
    return pair


def _validate_bounds(pair: np.ndarray, size: tuple[int, int], context: str) -> None:
    if len(size) != 2 or any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension <= 0
        for dimension in size
    ):
        raise ValueError(f"{context}: image size must be positive integer (width, height)")
    if pair[0] >= size[0] or pair[1] >= size[1]:
        raise ValueError(f"{context}: coordinates {pair.tolist()} exceed image bounds {size}")


def _parse_axis(identifier: str) -> tuple[int, int]:
    match = _IMAGE_AXIS.fullmatch(identifier)
    if match is None:
        raise ValueError(f"Invalid ImageID {identifier!r}; expected '<integer>_Fovea_X/Y'")
    return int(match.group(1)), 0 if match.group(2) == "X" else 1


def _validate_identifiers(identifiers: Sequence[str]) -> None:
    seen: set[tuple[int, int]] = set()
    for identifier in identifiers:
        key = _parse_axis(identifier)
        if key in seen:
            raise ValueError(f"Duplicate ImageID: {identifier}")
        seen.add(key)
    for image_id in {key[0] for key in seen}:
        if (image_id, 0) not in seen or (image_id, 1) not in seen:
            raise ValueError(f"Image {image_id}: both Fovea_X and Fovea_Y rows are required")


def load_labels(path: str | Path) -> dict[int, np.ndarray]:
    """Read unique training IDs and coordinate pairs; tolerate empty CSV lines."""
    labels: dict[int, np.ndarray] = {}
    for row in _rows(path, ("data", "Fovea_X", "Fovea_Y")):
        if _POSITIVE_ID.fullmatch(row[0]) is None:
            raise ValueError(f"{path}: invalid positive integer training ID {row[0]!r}")
        image_id = int(row[0])
        if image_id in labels:
            raise ValueError(f"{path}: duplicate training ID {image_id}")
        labels[image_id] = _coordinate_pair(row[1:], f"Training image {image_id}")
    return labels


def load_template(path: str | Path) -> list[str]:
    """Return the exact template row order after checking IDs and X/Y pairs.

    Placeholder values are ignored: the template is used only as the output
    schema and identifier order, never as labels or predictions.
    """
    identifiers = [row[0] for row in _rows(path, ("ImageID", "value"))]
    _validate_identifiers(identifiers)
    return identifiers


def read_submission(
    path: str | Path,
    image_sizes: Mapping[int, tuple[int, int]] | None = None,
    template_path: str | Path | None = None,
) -> dict[int, np.ndarray]:
    """Read a completed submission and optionally check bounds/template order."""
    rows = _rows(path, ("ImageID", "value"))
    identifiers = [row[0] for row in rows]
    _validate_identifiers(identifiers)
    if template_path is not None and identifiers != load_template(template_path):
        raise ValueError("Submission ImageID order does not exactly match the template")
    predictions: dict[int, np.ndarray] = {}
    for identifier, value in rows:
        image_id, axis = _parse_axis(identifier)
        try:
            numeric_value = float(value)
        except ValueError as exc:
            raise ValueError(f"{identifier}: submission value must be numeric and nonempty") from exc
        predictions.setdefault(image_id, np.empty(2, dtype=np.float64))[axis] = numeric_value
    for image_id, pair in predictions.items():
        _coordinate_pair(pair, f"Image {image_id}")
        if image_sizes is not None:
            if image_id not in image_sizes:
                raise ValueError(f"Image {image_id}: missing image size")
            _validate_bounds(pair, image_sizes[image_id], f"Image {image_id}")
    return predictions


def write_submission(
    template_path: str | Path,
    predictions: Mapping[int, Sequence[float]],
    output_path: str | Path,
    image_sizes: Mapping[int, tuple[int, int]],
) -> Path:
    """Write a checked CSV in template order and verify the numeric round trip.

    Each output row is mapped by its explicit image ID and axis, not by row
    position.  The prediction IDs must match the template exactly.  Validation
    happens before replacing the destination, so an invalid prediction cannot
    overwrite an existing valid submission.
    """
    identifiers = load_template(template_path)
    expected_ids = {_parse_axis(identifier)[0] for identifier in identifiers}
    if any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) for key in predictions):
        raise ValueError("Prediction keys must be integer image IDs")
    supplied_ids = set(predictions)
    if supplied_ids != expected_ids:
        raise ValueError(
            f"Prediction IDs do not match template: missing={sorted(expected_ids - supplied_ids)}, "
            f"extra={sorted(supplied_ids - expected_ids)}"
        )
    checked: dict[int, np.ndarray] = {}
    for image_id in expected_ids:
        pair = _coordinate_pair(predictions[image_id], f"Image {image_id}")
        if image_id not in image_sizes:
            raise ValueError(f"Image {image_id}: missing image size")
        _validate_bounds(pair, image_sizes[image_id], f"Image {image_id}")
        checked[image_id] = pair
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=destination.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("ImageID", "value"))
            for identifier in identifiers:
                image_id, axis = _parse_axis(identifier)
                writer.writerow((identifier, format(float(checked[image_id][axis]), ".17g")))
        round_trip = read_submission(temporary_path, image_sizes, template_path)
        for image_id, pair in checked.items():
            if not np.array_equal(pair, round_trip[image_id]):
                raise ValueError(f"Image {image_id}: CSV round-trip changed coordinates")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def coordinate_metrics(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> dict[str, float]:
    """Return flattened scalar RMSE, and separate X/Y RMSE diagnostics.

    ``rmse = sqrt(mean((prediction - truth)**2))`` over all ``2*N`` scalars.
    It is NOT ``(rmse_x + rmse_y) / 2`` and is NOT mean Euclidean distance.
    Values remain in the supplied coordinate units (original pixels for scoring).
    """
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1:] != (2,) or truth.shape[0] == 0:
        raise ValueError("y_true must be a nonempty array with shape (N, 2)")
    if prediction.shape != truth.shape:
        raise ValueError("y_pred and y_true must have the same shape (N, 2)")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Scoring coordinates must be finite")
    errors = prediction - truth
    # Scaling keeps the RMSE finite for large finite inputs whose squares would overflow.
    def stable_rmse(values: np.ndarray) -> float:
        scale = float(np.abs(values).max())
        return float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0

    return {
        "rmse": stable_rmse(errors),
        "rmse_x": stable_rmse(errors[:, 0]),
        "rmse_y": stable_rmse(errors[:, 1]),
    }
