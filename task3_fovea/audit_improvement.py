"""CPU-only consistency audit of the robust heatmap improvement artifacts.

No training, inference, prediction editing, or online score claim is performed.
Only the requested audit JSON is written. Exit: 0 passed, 1 failed, 2 pending.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from audit import (PendingResults, check_bounds, check_metrics, close, metrics,
                   read_csv, read_json, require, required, sha256)


ROOT = Path(__file__).resolve().parent
BASELINE_SHA256 = "a76ad7db2a9e8b78b4680bcb1630e930a8b57d1651c45d79cc8fc9dbe863377d"
SEEDS = (43, 44)
CANDIDATES = ("robust_only", "robust_baseline_mean")


def check_run(directory, seed, train_ids, val_ids, epochs):
    config = read_json(directory / "config.json")
    status = read_json(directory / "status.json")
    if status.get("state") != "completed":
        raise PendingResults(f"Run is not complete: {directory}")
    expected = {"architecture": "ResNet34 FPN", "size": 448, "seed": seed,
                "epochs_planned": epochs, "schedule_epochs": 40, "ema_decay": 0.95,
                "batch_size": 8, "coordinate_mse_weight": 100,
                "metric": "original-pixel scalar RMSE", "validation_tta": "horizontal flip",
                "augmentation": "affine flip brightness contrast color gamma directional lighting blur",
                "train_ids": train_ids, "validation_ids": val_ids}
    for name, value in expected.items():
        require(config.get(name) == value, f"{directory.name}: config {name} mismatch")
    require(Path(config["pretrained"]).resolve() ==
            (ROOT.parent / ".cache/torch/hub/checkpoints/resnet34-b627a593.pth").resolve(),
            f"{directory.name}: pretrained source mismatch")
    require(len(train_ids) == len(set(train_ids)) and len(val_ids) == len(set(val_ids)),
            f"{directory.name}: duplicate IDs in run")
    require(not set(train_ids).intersection(val_ids), f"{directory.name}: train/validation overlap")
    require(status["completed_epochs"] == epochs, f"{directory.name}: incomplete epoch count")
    with required(directory / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    require([int(row["epoch"]) for row in history] == list(range(1, epochs + 1)),
            f"{directory.name}: history does not prove all completed epochs")
    require(all(np.isfinite(float(row["loss"])) for row in history), f"{directory.name}: nonfinite training loss")
    checkpoint = torch.load(required(directory / "best.pt"), map_location="cpu", weights_only=True)
    require(checkpoint["config"] == config, f"{directory.name}: checkpoint/config mismatch")
    require(checkpoint.get("weight_type") == "ema", f"{directory.name}: checkpoint is not marked EMA")
    require(checkpoint["epoch"] == status["best_epoch"], f"{directory.name}: checkpoint/status epoch mismatch")
    require(1 <= checkpoint["epoch"] <= epochs, f"{directory.name}: checkpoint epoch out of range")
    blocks = defaultdict(set)
    for key in checkpoint["model"]:
        match = re.fullmatch(r"encoder\.layer([1-4])\.(\d+)\.conv1\.weight", key)
        if match:
            blocks[int(match[1])].add(int(match[2]))
    require([blocks[i] for i in range(1, 5)] == [set(range(n)) for n in (3, 4, 6, 3)],
            f"{directory.name}: actual checkpoint is not the expected ResNet34 encoder")
    require(tuple(checkpoint["model"]["head.1.weight"].shape) == (1, 64, 1, 1)
            and "lateral.0.weight" in checkpoint["model"], f"{directory.name}: FPN head mismatch")
    if val_ids:
        require(all(np.isfinite(float(row["rmse"])) for row in history), f"{directory.name}: nonfinite validation RMSE")
        best = min(history, key=lambda row: float(row["rmse"]))
        require(int(best["epoch"]) == checkpoint["epoch"], f"{directory.name}: checkpoint not best history epoch")
        close(status["best_rmse"], float(best["rmse"]), f"{directory.name}: history/status RMSE")
    else:
        require(checkpoint["epoch"] == epochs and status["best_rmse"] is None,
                f"{directory.name}: full refit must use final epoch without validation score")
    return {"best_epoch": checkpoint["epoch"], "completed_epochs": epochs,
            "weight_type": checkpoint["weight_type"], "ema_decay": config["ema_decay"],
            "best_rmse": status["best_rmse"], "checkpoint_sha256": sha256(directory / "best.pt"),
            "train_count": len(train_ids), "validation_count": len(val_ids)}


def audit_robust(output_dir):
    data_dir = ROOT / "data/detection"
    baseline_dir = ROOT / "outputs/study"
    report = {"passed": False, "state": "checking", "method": "robust_heatmap",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": [], "failures": [], "pending": [], "evidence": {}}

    def section(name, function):
        try:
            value = function()
        except PendingResults as exc:
            report["pending"].append({"check": name, "reason": str(exc)})
            return None
        except Exception as exc:
            report["failures"].append({"check": name, "reason": f"{type(exc).__name__}: {exc}"})
            return None
        report["checks"].append(name)
        return value

    def baseline_check():
        feedback = read_json(output_dir.parent / "online_feedback.json")
        path = baseline_dir / "submission.csv"
        backup = output_dir.parent / "baseline_submission_14.73696.csv"
        require(Path(feedback["submission_file"]).resolve() == path.resolve(), "Feedback refers to different baseline")
        hashes = {"original": sha256(path), "backup": sha256(backup), "feedback": feedback["submission_sha256"]}
        require(set(hashes.values()) == {BASELINE_SHA256}, "Original online baseline or backup changed")
        require(read_json(baseline_dir / "result.json")["submission_sha256"] == BASELINE_SHA256,
                "Baseline result hash differs")
        require(feedback["public_rmse"] == 14.73696 and feedback["reported_by"] == "user"
                and feedback["independently_verified"] is False, "Online feedback provenance changed")
        report["evidence"]["baseline_preserved"] = {"sha256": BASELINE_SHA256, "user_reported_public_rmse": 14.73696,
            "independently_verified_online": False, "original_backup_feedback_hashes_match": True}
        with np.load(required(baseline_dir / "oof_predictions.npz"), allow_pickle=False) as saved:
            truth, baseline = saved["truth"].copy(), saved["equal_mean"].copy()
            require(np.array_equal(baseline, (saved["heatmap"] + saved["regression"]) / 2),
                    "Baseline OOF is not the saved two-architecture mean")
        return truth, baseline

    baseline_checked = section("original_online_baseline_and_backup_preserved", baseline_check)

    def data_check():
        manifest = read_json(ROOT / "outputs/baseline/prepared/manifest.json")
        require(manifest["folds"] == 4, "Manifest must retain four folds")
        records = manifest["records"]
        train, test = ([r for r in records if r["split"] == s] for s in ("train", "test"))
        require(len(records) == 100 and [r["id"] for r in train] == list(range(1, 81))
                and [r["id"] for r in test] == list(range(81, 101)), "Manifest IDs or order differ")
        label_rows = read_csv(data_dir / "fovea_localization_train_GT.csv", ("data", "Fovea_X", "Fovea_Y"))
        labels = {int(row[0]): [float(row[1]), float(row[2])] for row in label_rows}
        require(len(labels) == len(label_rows) == 80 and set(labels) == set(range(1, 81)), "CSV IDs invalid")
        with np.load(required(output_dir / "images448.npz"), allow_pickle=False) as saved:
            images = saved["images"].copy()
        require(images.shape == (100, 448, 448, 3) and images.dtype == np.uint8, "448 image cache shape/type invalid")
        groups = defaultdict(list)
        for index, record in enumerate(records):
            path = data_dir / record["split"] / f"{record['id']:04}.jpg"
            require(sha256(path) == record["sha256"], f"Original image changed: {path}")
            with Image.open(path) as image:
                image = image.convert("RGB")
                require(image.size == (record["width"], record["height"]), f"Image size differs: {path}")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                expected = np.asarray(image.crop(record["crop"]).resize((448, 448), Image.Resampling.BILINEAR))
            require(pixel_hash == record["pixel_sha256"], f"Pixel hash mismatch: {path}")
            require(np.array_equal(expected, images[index]), f"Cached input does not match original crop/resize: {path}")
            if record["split"] == "train":
                require(record["fold"] in range(4), f"Invalid fold: {path}")
                require(np.array_equal(np.asarray(record["xy"]), labels[record["id"]]), f"CSV/manifest label mismatch: {path}")
                crop = np.array(record["crop"])
                close(record["xy_crop"], (np.array(record["xy"]) - crop[:2]) / (crop[2:] - crop[:2]),
                      f"Normalized/original coordinates {record['id']}")
                groups[pixel_hash].append(record)
        for group in groups.values():
            require(len({r["fold"] for r in group}) == 1, "Exact duplicate image crosses validation folds")
        duplicates = sorted(sorted(r["id"] for r in group) for group in groups.values() if len(group) > 1)
        require(duplicates == sorted(sorted(g) for g in manifest["exact_duplicate_train_groups"]),
                "Manifest duplicate metadata differs")
        counts = [sum(r["fold"] == fold for r in train) for fold in range(4)]
        require(counts == manifest["fold_counts"] and all(counts), "Fold counts differ or a fold is empty")
        truth = np.array([r["xy"] for r in train], dtype=np.float64)
        check_bounds(truth, train, "Training labels")
        if baseline_checked is not None:
            require(np.array_equal(truth, baseline_checked[0]), "Baseline OOF truth differs from original CSV")
        report["evidence"]["data"] = {"train_count": 80, "test_count": 20, "fold_counts": counts,
            "exact_duplicate_groups": duplicates, "all_448_cached_inputs_match_original_images": True}
        return train, test, truth

    data = section("original_data_manifest_duplicate_groups_and_448_cache", data_check)
    if data is None:
        return finish(report)
    train, test, truth = data
    train_ids = [r["id"] for r in train]
    runs, fold_predictions = {}, {}
    for seed in SEEDS:
        for fold in range(4):
            def validation_check(seed=seed, fold=fold):
                directory = output_dir / f"fold{fold}_seed{seed}"
                fit_ids = [r["id"] for r in train if r["fold"] != fold]
                val_records = [r for r in train if r["fold"] == fold]
                run = check_run(directory, seed, fit_ids, [r["id"] for r in val_records], 40)
                pred = np.load(required(directory / "validation_predictions.npy"), allow_pickle=False)
                check_bounds(pred, val_records, directory.name)
                scored = metrics([r["xy"] for r in val_records], pred)
                close(scored["rmse"], run["best_rmse"], f"{directory.name}: predictions/status RMSE")
                runs[(seed, fold)] = run
                fold_predictions[(seed, fold)] = pred
                report["evidence"][directory.name] = dict(run, metrics=scored)
                return True
            section(f"fold{fold}_seed{seed}_config_history_ema_checkpoint_predictions", validation_check)

    def selection_check():
        selection = read_json(output_dir / "selection.json")
        plan = read_json(output_dir / "plan.json")
        require(selection["plan"] == plan, "Selection/plan mismatch")
        expected = {"seeds": list(SEEDS), "folds": 4, "epochs": 40, "size": 448,
                    "candidates": list(CANDIDATES),
                    "gate": {"relative_rmse_reduction": 0.1, "minimum_improved_folds": 3}}
        for key, value in expected.items():
            require(plan.get(key) == value, f"Fixed plan {key} mismatch")
        if len(runs) != 8 or baseline_checked is None:
            raise PendingResults("All eight validation runs and the original baseline must pass before selection")
        with np.load(required(output_dir / "oof_predictions.npz"), allow_pickle=False) as saved:
            require(set(saved.files) == {"truth", "baseline", "seed43", "seed44", *CANDIDATES}, "Unexpected OOF keys")
            oof = {key: saved[key].copy() for key in saved.files}
        require(np.array_equal(oof["truth"], truth), "Improvement OOF truth differs from original CSV")
        require(np.array_equal(oof["baseline"], baseline_checked[1]), "Baseline comparison OOF changed")
        seed_oof, epoch_lists = {}, {}
        for seed in SEEDS:
            assembled = np.full((80, 2), np.nan)
            coverage = np.zeros(80, dtype=int)
            epoch_lists[str(seed)] = []
            for fold in range(4):
                indices = [i for i, r in enumerate(train) if r["fold"] == fold]
                assembled[indices] = fold_predictions[(seed, fold)]
                coverage[indices] += 1
                epoch_lists[str(seed)].append(runs[(seed, fold)]["best_epoch"])
            require(np.all(coverage == 1), f"Seed {seed}: 80 records not covered exactly once out-of-fold")
            require(np.array_equal(oof[f"seed{seed}"], assembled), f"Seed {seed}: OOF differs from saved fold predictions")
            seed_oof[seed] = assembled
        robust = (seed_oof[43] + seed_oof[44]) / 2
        candidates = {"robust_only": robust, "robust_baseline_mean": (robust + baseline_checked[1]) / 2}
        require(set(selection["candidates"]) == set(CANDIDATES), "Candidate set differs from fixed plan")
        baseline_rmse = metrics(truth, baseline_checked[1])["rmse"]
        fold_indices = [[i for i, r in enumerate(train) if r["fold"] == f] for f in range(4)]
        baseline_folds = [metrics(truth[idx], baseline_checked[1][idx])["rmse"] for idx in fold_indices]
        close(selection["baseline_rmse"], baseline_rmse, "Baseline pooled RMSE")
        close(selection["baseline_fold_rmse"], baseline_folds, "Baseline fold RMSE")
        details, qualified = {}, []
        for name in CANDIDATES:
            pred = candidates[name]
            require(np.array_equal(oof[name], pred), f"{name}: seed/candidate mean differs")
            scored = metrics(truth, pred)
            folds = [metrics(truth[idx], pred[idx])["rmse"] for idx in fold_indices]
            improved = sum(value < old for value, old in zip(folds, baseline_folds))
            passed = scored["rmse"] <= baseline_rmse * 0.9 and improved >= 3
            actual = selection["candidates"][name]
            check_metrics(actual["metrics"], scored, f"{name}: pooled original-pixel metric")
            close(actual["fold_rmse"], folds, f"{name}: fold RMSE")
            require(actual["improved_folds"] == improved and actual["passed_gate"] is passed,
                    f"{name}: fixed candidate gate decision differs")
            details[name] = {"metrics": scored, "fold_rmse": folds, "improved_folds": improved,
                             "passed_gate": passed, "relative_rmse_reduction": 1 - scored["rmse"] / baseline_rmse}
            if passed:
                qualified.append(name)
        chosen = min(qualified, key=lambda name: details[name]["metrics"]["rmse"]) if qualified else None
        require(selection["selected"] == chosen, "Selection does not minimize RMSE among qualified candidates")
        require(selection["best_epochs"] == epoch_lists, "Best epoch provenance differs from actual checkpoints")
        refit_epochs = {seed: int(np.median(values)) for seed, values in epoch_lists.items()}
        require(selection["refit_epochs"] == refit_epochs, "Refit epoch counts are not per-seed integer medians")
        report["evidence"]["selection"] = {"selected": chosen, "recomputed_candidates": details,
            "oof_coverage_each_seed": {str(s): {"records": 80, "times_per_record": 1} for s in SEEDS},
            "baseline_rmse": baseline_rmse, "best_epochs": epoch_lists, "refit_epochs": refit_epochs}
        return selection

    selection = section("oof_truth_coverage_seed_means_candidates_and_fixed_gate", selection_check)
    if selection is not None:
        def final_check():
            if selection["selected"] is None:
                raise PendingResults("No candidate passed the gate; a new final submission is not authorized by the plan")
            result = read_json(output_dir / "result.json")
            require(result["selection"] == selection, "Final result refers to a different selection")
            require(result["train_count"] == 80 and result["rows"] == 40, "Final result record counts differ")
            require([s["seed"] for s in result["models"]] == list(SEEDS), "Final ensemble seeds/order differ")
            predictions = []
            for source in result["models"]:
                seed = source["seed"]
                directory = output_dir / f"full_seed{seed}"
                epochs = selection["refit_epochs"][str(seed)]
                require(source["epochs"] == epochs, "Final model epoch source differs")
                require(Path(source["checkpoint"]).resolve() == (directory / "best.pt").resolve(),
                        "Final checkpoint source path differs")
                run = check_run(directory, seed, train_ids, [], epochs)
                require(source["sha256"] == run["checkpoint_sha256"], "Final model checkpoint SHA256 differs")
                pred = np.load(required(directory / "test_predictions.npy"), allow_pickle=False)
                check_bounds(pred, test, directory.name)
                predictions.append(pred)
                report["evidence"][directory.name] = run
            final = np.mean(predictions, axis=0)
            if selection["selected"] == "robust_baseline_mean":
                original = read_json(baseline_dir / "predictions.json")
                require(set(original) == {str(r["id"]) for r in test}, "Original final prediction IDs differ")
                final = (final + np.array([original[str(r["id"])] for r in test])) / 2
            template = read_csv(data_dir / "sample_submission.csv", ("ImageID", "value"))
            path = output_dir / "submission_robust.csv"
            rows = read_csv(path, ("ImageID", "value"))
            expected_ids = [f"{r['id']}_Fovea_{axis}" for r in test for axis in ("X", "Y")]
            require(len(rows) == len(template) == 40 and [r[0] for r in rows] == [r[0] for r in template] == expected_ids,
                    "Final CSV is not 40 template-ordered rows")
            values = np.array([float(r[1]) for r in rows]).reshape(20, 2)
            require(np.array_equal(values, final), "CSV values differ from exact saved-model mean")
            check_bounds(values, test, "Final submission")
            digest = sha256(path)
            require(result["submission_sha256"] == digest, "Final CSV hash differs from result")
            report["evidence"]["submission"] = {"rows": 40, "test_images": 20, "full_model_count": 2,
                "selected": selection["selected"], "sha256": digest,
                "maximum_csv_mean_absolute_difference": float(np.max(np.abs(values - final)))}
            return True
        section("full80_refits_ema_sources_and_exact_submission_mean", final_check)
    return finish(report)


def finish(report):
    report["passed"] = len(report["checks"]) == 12 and not report["failures"] and not report["pending"]
    report["state"] = "passed" if report["passed"] else "failed" if report["failures"] else "awaiting_results"
    report["scope"] = ("Original data and saved-artifact consistency, including EMA checkpoint metadata; "
                       "not a rerun of model inference, proof of the training trajectory, independent held-out test, or online score verification.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["robust_heatmap"], default="robust_heatmap")
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/improvement_20261024/robust_heatmap")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit_robust(args.directory)
    destination = args.output or args.directory / "audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "state": report["state"], "checks_passed": len(report["checks"]),
                      "failures": report["failures"], "pending": report["pending"], "report": str(destination)},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1 if report["failures"] else 2)


if __name__ == "__main__":
    main()
