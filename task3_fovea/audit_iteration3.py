"""CPU-only independent audit of the shared-epoch ResNet50 experiment.

Only the selected experiment's audit.json is written. No training,
inference, prediction edits, or test-label access. Exit 0: passed; 1: failed;
2: incomplete. A correctly rejected candidate passes the validation audit but
has state completed_not_qualified and final_submission_available false.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageFilter
import torch

from audit import (PendingResults, require, required, read_json, read_csv, sha256,
                   close, metrics, check_metrics, check_bounds)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/iteration3_20261024/fine_heatmap"
MANIFEST = ROOT / "outputs/baseline/prepared/manifest.json"
DATA = ROOT / "data/detection"
BASELINE = ROOT / "outputs/improvement_20261024/robust_heatmap"
SEEDS = (43, 44)
EPOCHS = (6, 10, 16, 24, 32, 40)
BASELINE_HASH = "20d24c207d73e8ef0af3de589e9e3becd0416b14c8b30c02068815451f136a8b"
ILLUMINATION = False
SIZE = 512
CHANNELS = (256, 512, 1024, 2048)
ENCODER_LR = 2.5e-5
CACHE_NAME = "images512.npz"
SUBMISSION_NAME = "submission_fine_v3.csv"
WEIGHTS_NAME = "resnet50-11ad3fa6.pth"


def preprocessing_spec():
    return {"name": "per-image background illumination correction",
            "order": "RGB -> unchanged manifest retina crop -> bilinear resize448 -> correction -> training augmentation",
            "blur": "PIL.ImageFilter.GaussianBlur", "blur_radius": 448 / 30.,
            "corrected": "clip(4 * (RGB - blurred_RGB) + 128, 0, 255)",
            "blend": "0.5 * original_RGB + 0.5 * corrected_RGB",
            "storage": "np.rint then clip to [0,255], uint8",
            "uses_labels": False, "uses_other_images": False, "pil_version": PIL.__version__}


def npz(path):
    with np.load(required(path), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def checkpoint(path, config, epoch):
    saved = torch.load(required(path), map_location="cpu", weights_only=True)
    require(saved["config"] == config, f"{path}: checkpoint config mismatch")
    require(saved["epoch"] == epoch, f"{path}: checkpoint epoch mismatch")
    require(saved.get("weight_type") == "ema", f"{path}: checkpoint is not marked EMA")
    state = saved["model"]
    for index, channels in enumerate(CHANNELS):
        require(tuple(state[f"lateral.{index}.weight"].shape) == (64, channels, 1, 1),
                f"{path}: actual FPN encoder channels mismatch")
        convolution = "conv2" if ILLUMINATION else "conv3"
        blocks = {int(match.group(1)) for name in state
                  if (match := re.fullmatch(rf"encoder\.layer{index + 1}\.(\d+)\.{convolution}\.weight", name))}
        require(blocks == set(range((3, 4, 6, 3)[index])), f"{path}: actual encoder block count/type mismatch")
        if ILLUMINATION:
            require(not any(re.fullmatch(rf"encoder\.layer{index + 1}\.\d+\.conv3\.weight", name) for name in state),
                    f"{path}: unexpected bottleneck encoder")
    require(tuple(state["head.1.weight"].shape) == (1, 64, 1, 1), f"{path}: heatmap head mismatch")
    for name, tensor in state.items():
        require(tensor.device.type == "cpu", f"{path}: auditor loaded non-CPU tensor")
        if tensor.is_floating_point():
            require(bool(torch.isfinite(tensor).all()), f"{path}: nonfinite parameter {name}")
    del saved, state
    return sha256(path)


def fold_metrics(truth, prediction, folds):
    return [metrics(truth[folds == fold], prediction[folds == fold]) for fold in range(4)]


def compare_fold_metrics(actual, expected, context):
    require(len(actual) == len(expected) == 4, f"{context}: fold metric count mismatch")
    for fold, (value, reference) in enumerate(zip(actual, expected)):
        check_metrics(value, reference, f"{context}/fold{fold}")


def audit():
    torch.set_num_threads(2)
    report = {"passed": False, "state": "checking", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": [], "failures": [], "pending": [], "evidence": {},
              "scope": "CPU file/array/checkpoint audit; no model inference or independent test score"}

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

    def dataset():
        manifest = read_json(MANIFEST)
        require(manifest["folds"] == 4 and manifest["seed"] == 20261024, "Original grouped split changed")
        records = manifest["records"]
        train = [r for r in records if r["split"] == "train"]
        test = [r for r in records if r["split"] == "test"]
        require(len(records) == 100 and [r["id"] for r in train] == list(range(1, 81))
                and [r["id"] for r in test] == list(range(81, 101)), "Expected original 80/20 image IDs/order")
        rows = read_csv(DATA / "fovea_localization_train_GT.csv", ("data", "Fovea_X", "Fovea_Y"))
        labels = {int(row[0]): [float(row[1]), float(row[2])] for row in rows}
        require(len(rows) == len(labels) == 80 and set(labels) == set(range(1, 81)), "Original CSV IDs mismatch")
        truth = np.array([labels[r["id"]] for r in train], dtype=np.float64)
        folds = np.array([r["fold"] for r in train])
        require(np.array_equal(np.bincount(folds, minlength=4), [20] * 4), "Four 20-record folds required")
        cache = npz(OUTPUT / CACHE_NAME)
        require(str(cache["manifest_sha256"].item()) == sha256(MANIFEST), "Cache manifest hash mismatch")
        require(cache["ids"].tolist() == [r["id"] for r in records], "Image cache ID/order mismatch")
        require(cache["images"].shape == (100, SIZE, SIZE, 3) and cache["images"].dtype == np.uint8,
                "Image cache shape/type mismatch")
        if ILLUMINATION:
            expected_transform_hash = hashlib.sha256(json.dumps(preprocessing_spec(), sort_keys=True).encode("utf-8")).hexdigest()
            require(str(cache["preprocessing_sha256"].item()) == expected_transform_hash,
                    "Illumination cache preprocessing hash mismatch")
        groups = defaultdict(list)
        for index, record in enumerate(records):
            original = DATA / record["split"] / f"{record['id']:04d}.jpg"
            require(Path(record["file"]).resolve() == original.resolve(), "Image provenance path mismatch")
            require(sha256(original) == record["sha256"], f"Original image changed: {original}")
            with Image.open(original) as image:
                image = image.convert("RGB")
                require(image.size == (record["width"], record["height"]), "Original image dimensions mismatch")
                require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == record["pixel_sha256"],
                        f"Decoded image hash mismatch: {original}")
                crop = np.asarray(record["crop"])
                require(crop.shape == (4,) and 0 <= crop[0] < crop[2] <= image.width
                        and 0 <= crop[1] < crop[3] <= image.height, "Invalid crop geometry")
                resized_image = image.crop(record["crop"]).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                resized = np.asarray(resized_image)
                if ILLUMINATION:
                    foreground = resized.astype(np.float32)
                    background = np.asarray(resized_image.filter(ImageFilter.GaussianBlur(radius=SIZE / 30.)), dtype=np.float32)
                    correction = np.clip(4. * (foreground - background) + 128., 0., 255.)
                    resized = np.rint((foreground + correction) * .5).clip(0, 255).astype(np.uint8)
                require(np.array_equal(cache["images"][index], resized), f"Incorrect cached crop/resize: {original}")
            if record["split"] == "train":
                groups[record["pixel_sha256"]].append(record)
                close(record["xy"], labels[record["id"]], "Manifest truth differs from original CSV")
                close(record["xy_crop"], (np.asarray(record["xy"]) - crop[:2]) / (crop[2:] - crop[:2]),
                      "Normalized crop coordinates mismatch")
        duplicate_groups = []
        for members in groups.values():
            require(len({r["fold"] for r in members}) == 1, "Duplicate image crosses train/validation folds")
            if len(members) > 1:
                duplicate_groups.append([r["id"] for r in members])
        check_bounds(truth, train, "Original training labels")
        report["evidence"]["dataset"] = {"train_count": 80, "test_count": 20, "fold_counts": [20] * 4,
                                         "duplicate_groups": duplicate_groups, "cache_rebuilt_comparison": True}
        return train, test, truth, folds

    def preservation():
        feedback = read_json(ROOT / "outputs/improvement_20261024/online_feedback_improved.json")
        registry = read_json(OUTPUT.parent / "experiment_registry.json")
        require(feedback["public_rmse"] == 14.20909 and feedback["submission_sha256"] == BASELINE_HASH,
                "Current-best feedback provenance changed")
        require(registry["baseline_submission_sha256"] == BASELINE_HASH
                and registry["baseline_public_rmse_user_reported"] == 14.20909, "Iteration baseline changed")
        require(registry["gate"] == {"minimum_relative_rmse_reduction": .05, "minimum_improved_folds": 3},
                "Predeclared iteration gate changed")
        copies = [BASELINE / "submission_robust.csv", ROOT.parent / feedback["submission_file"],
                  ROOT.parent / registry["baseline_submission"]]
        for path in copies:
            require(sha256(path) == BASELINE_HASH, f"Preserved current-best CSV changed: {path}")
        result = read_json(BASELINE / "result.json")
        require(result["submission_sha256"] == BASELINE_HASH, "Baseline result hash mismatch")
        report["evidence"]["preserved_baseline"] = {"sha256": BASELINE_HASH, "public_rmse_user_reported": 14.20909,
                                                    "independently_verified_online": False}
        return True

    def sources():
        plan = read_json(OUTPUT / "plan.json")
        expected = {"architecture": "ResNet34 FPN" if ILLUMINATION else "ResNet50 FPN", "encoder_channels": list(CHANNELS),
                    "decoder_channels": 64, "size": SIZE, "batch_size": 8, "schedule_epochs": 40,
                    "seeds": list(SEEDS), "folds": 4, "candidate_epochs": list(EPOCHS),
                    "encoder_lr": ENCODER_LR, "head_lr": 5e-4, "weight_decay": 2e-4, "eta_min": 1e-6,
                    "ema_decay": .95, "gaussian_sigma_cells": 1.5 if ILLUMINATION else 1., "normalized_coordinate_mse_weight": 100.,
                    "gate": {"relative_rmse_reduction": .05, "minimum_improved_folds": 3}}
        for key, value in expected.items():
            require(plan[key] == value, f"Predeclared plan mismatch: {key}")
        provenance = read_json(OUTPUT / "provenance.json")
        expected_sources = [ROOT / name for name in ("fine_heatmap_v3.py", "model.py", "run.py", "robust_heatmap.py",
                                                    "data.py", "contracts.py")]
        expected_sources += [MANIFEST, ROOT.parent / ".cache/torch/hub/checkpoints" / WEIGHTS_NAME,
                             DATA / "sample_submission.csv", DATA / "fovea_localization_train_GT.csv",
                             BASELINE / "oof_predictions.npz", BASELINE / "result.json", BASELINE / "submission_robust.csv"]
        if ILLUMINATION:
            expected_sources.append(ROOT / "illumination_v3.py")
            require(plan["experiment"] == "illumination_v3" and plan["preprocessing"] == preprocessing_spec(),
                    "Illumination plan/preprocessing mismatch")
            require(provenance["preprocessing"] == preprocessing_spec(), "Illumination provenance transform mismatch")
        require(set(provenance["source_sha256"]) == {str(path.resolve()) for path in expected_sources},
                "Source provenance coverage mismatch")
        for path, digest in provenance["source_sha256"].items():
            require(sha256(Path(path)) == digest, f"Source changed after training started: {path}")
        require(sha256(OUTPUT / CACHE_NAME) == provenance["cache_sha256"], "Cache provenance hash mismatch")
        return plan, provenance

    data = section("original_labels_images_cache_and_grouped_folds", dataset)
    section("online_14_20909_current_best_unchanged", preservation)
    source = section("fixed_plan_and_source_hashes", sources)

    def run_check(directory, seed, fit, validation, epochs, stage):
        plan, provenance = source
        config = read_json(directory / "config.json")
        expected = {**plan, "seed": seed, "epochs_planned": epochs, "train_ids": fit,
                    "validation_ids": validation, "training_stage": stage, "sources": provenance}
        require(config == expected, f"{directory}: exact config mismatch")
        require(len(fit) == len(set(fit)) and len(validation) == len(set(validation))
                and not set(fit).intersection(validation), f"{directory}: overlapping/repeated IDs")
        status = read_json(directory / "status.json")
        if status.get("state") != "completed":
            raise PendingResults(f"Run not completed: {directory.name}, epoch {status.get('completed_epochs')}")
        require(status["completed_epochs"] == epochs, f"{directory}: completed epochs mismatch")
        require(status["candidate_epochs_saved"] == (list(EPOCHS) if validation else []),
                f"{directory}: eligible checkpoint inventory mismatch")
        with required(directory / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        require([int(row["epoch"]) for row in history] == list(range(1, epochs + 1)), "History epoch coverage mismatch")
        for epoch, row in enumerate(history, 1):
            require(all(math.isfinite(float(value)) for value in row.values()), "Nonfinite history values")
            require(float(row["loss"]) >= 0 and float(row["seconds"]) > 0, "Invalid training history values")
            for name, initial in (("encoder_lr", ENCODER_LR), ("head_lr", 5e-4)):
                expected_lr = 1e-6 + (initial - 1e-6) * (1 + math.cos(math.pi * (epoch - 1) / 40)) / 2
                close(float(row[name]), expected_lr, f"{directory}/epoch{epoch}/{name}", atol=1e-12)
        return config, status, history

    runs = {}
    if data is not None and source is not None:
        train, test, truth, folds = data
        ids = [record["id"] for record in train]
        for seed in SEEDS:
            for fold in range(4):
                name = f"fold{fold}_seed{seed}"

                def validation_run(seed=seed, fold=fold, name=name):
                    directory = OUTPUT / name
                    val_records = [r for r in train if r["fold"] == fold]
                    fit_ids = [r["id"] for r in train if r["fold"] != fold]
                    val_ids = [r["id"] for r in val_records]
                    config, status, history = run_check(directory, seed, fit_ids, val_ids, 40, "validation")
                    arrays = {}
                    for epoch, row in enumerate(history, 1):
                        stored = npz(directory / "epoch_predictions" / f"epoch_{epoch:03d}.npz")
                        require(stored["ids"].tolist() == val_ids, f"{name}: epoch prediction IDs/order mismatch")
                        prediction = stored["predictions"]
                        check_bounds(prediction, val_records, f"{name}/epoch{epoch}")
                        actual = metrics(truth[folds == fold], prediction)
                        check_metrics({key: float(row[key]) for key in actual}, actual, f"{name}/epoch{epoch}/history")
                        if epoch in EPOCHS:
                            arrays[epoch] = prediction
                    best = min(history, key=lambda row: float(row["rmse"]))
                    require(status["audit_best_epoch"] == int(best["epoch"]), f"{name}: audit-best epoch mismatch")
                    close(status["audit_best_rmse"], float(best["rmse"]), f"{name}: audit-best score mismatch")
                    checkpoint(directory / "audit_best.pt", config, int(best["epoch"]))
                    digests = {epoch: checkpoint(directory / f"epoch_{epoch:03d}.pt", config, epoch) for epoch in EPOCHS}
                    return {"config": config, "status": status, "arrays": arrays, "hashes": digests}

                value = section(f"{name}_history_predictions_and_actual_ema_checkpoints", validation_run)
                if value is not None:
                    runs[name] = value

    def selection_check():
        train, test, truth, folds = data
        selection = read_json(OUTPUT / "selection.json")
        saved = npz(OUTPUT / "oof_predictions.npz")
        require(selection["train_ids"] == saved["ids"].tolist() == ids, "OOF IDs/order mismatch")
        close(saved["folds"], folds, "OOF fold IDs")
        close(saved["truth"], truth, "OOF original CSV truth")
        require(selection["plan"] == source[0] and selection["provenance"] == source[1], "Selection sources/plan mismatch")
        require(selection["oof_predictions_sha256"] == sha256(OUTPUT / "oof_predictions.npz"), "OOF hash mismatch")
        old = npz(BASELINE / "oof_predictions.npz")
        close(old["truth"], truth, "Baseline original CSV truth")
        baseline = old["robust_only"]
        close(saved["baseline"], baseline, "Baseline OOF values")
        baseline_metrics = metrics(truth, baseline)
        close(baseline_metrics["rmse"], 16.334263567891036, "Previously audited robust OOF reference")
        baseline_folds = fold_metrics(truth, baseline, folds)
        check_metrics(selection["baseline"], baseline_metrics, "Baseline metrics")
        compare_fold_metrics(selection["baseline_folds"], baseline_folds, "Baseline fold metrics")
        require(set(selection["epoch_metrics"]) == set(selection["epoch_fold_metrics"]) == {str(e) for e in EPOCHS},
                "Eligible epoch set changed")
        means, scores = {}, {}
        for epoch in EPOCHS:
            seed_values = []
            for seed in SEEDS:
                prediction = np.full_like(truth, np.nan)
                coverage = np.zeros(80, dtype=int)
                for fold in range(4):
                    prediction[folds == fold] = runs[f"fold{fold}_seed{seed}"]["arrays"][epoch]
                    coverage[folds == fold] += 1
                require(np.all(coverage == 1), "OOF record coverage must be exactly one per seed")
                close(saved[f"epoch_{epoch:03d}_seed{seed}"], prediction, "OOF run prediction assembly")
                seed_values.append(prediction)
            mean = np.mean(seed_values, axis=0)
            means[epoch] = mean
            close(saved[f"epoch_{epoch:03d}_mean"], mean, "OOF equal-seed mean")
            scores[epoch] = metrics(truth, mean)
            check_metrics(selection["epoch_metrics"][str(epoch)], scores[epoch], f"Epoch {epoch} pooled metrics")
            compare_fold_metrics(selection["epoch_fold_metrics"][str(epoch)], fold_metrics(truth, mean, folds),
                                 f"Epoch {epoch} fold metrics")
        chosen_epoch = min(EPOCHS, key=lambda epoch: (scores[epoch]["rmse"], epoch))
        require(selection["shared_epoch"] == chosen_epoch
                and selection["refit_epochs"] == {str(seed): chosen_epoch for seed in SEEDS}, "Shared epoch selection mismatch")
        expected_checkpoints = []
        for seed in SEEDS:
            for fold in range(4):
                name = f"fold{fold}_seed{seed}"
                expected_checkpoints.append({"seed": seed, "fold": fold, "epoch": chosen_epoch,
                                             "path": str((OUTPUT / name / f"epoch_{chosen_epoch:03d}.pt").resolve()),
                                             "sha256": runs[name]["hashes"][chosen_epoch]})
        require(selection["shared_epoch_checkpoints"] == expected_checkpoints, "Selected checkpoint provenance mismatch")
        require(selection["audit_only_fold_best"] == {name: value["status"] for name, value in runs.items()},
                "Audit-only status records mismatch")
        candidates = {"new_only": means[chosen_epoch], "new_robust_mean": (means[chosen_epoch] + baseline) / 2}
        require(set(selection["candidates"]) == set(candidates), "Fixed candidate set changed")
        eligible = []
        for name, prediction in candidates.items():
            close(saved[name], prediction, f"{name}: candidate array")
            score = metrics(truth, prediction)
            per_fold = fold_metrics(truth, prediction, folds)
            improved = sum(a["rmse"] < b["rmse"] for a, b in zip(per_fold, baseline_folds))
            passed = score["rmse"] <= .95 * baseline_metrics["rmse"] and improved >= 3
            value = selection["candidates"][name]
            check_metrics(value["metrics"], score, f"{name}: pooled metrics")
            compare_fold_metrics(value["fold_metrics"], per_fold, f"{name}: fold metrics")
            require(value["improved_folds"] == improved and value["passed_gate"] is passed, f"{name}: gate mismatch")
            if passed:
                eligible.append(name)
        chosen = min(eligible, key=lambda name: metrics(truth, candidates[name])["rmse"]) if eligible else None
        require(selection["selected"] == chosen and selection["gate_passed"] is bool(eligible), "Eligible candidate selection mismatch")
        report["evidence"]["selection"] = {"shared_epoch": chosen_epoch, "selected": chosen,
                                           "epoch_rmse": {str(e): scores[e]["rmse"] for e in EPOCHS},
                                           "candidates": selection["candidates"], "records_per_seed": 80,
                                           "validation_runs": 8, "checkpoint_files_audited": 56}
        return selection

    selection = None
    if len(runs) == 8:
        selection = section("all_80_oof_shared_epoch_fixed_candidates_and_gate", selection_check)
    elif data is not None and source is not None:
        report["pending"].append({"check": "shared_epoch_selection", "reason": "Waiting for all eight completed validation runs"})

    def final_check():
        train, test, truth, folds = data
        epoch = selection["shared_epoch"]
        test_ids = [record["id"] for record in test]
        result = read_json(OUTPUT / "result.json")
        require(result["selection"] == selection, "Final selection provenance mismatch")
        require(result["refit_epochs"] == epoch and result["train_count"] == 80 and result["test_count"] == 20
                and result["submission_rows"] == 40 and result["online_score"] is None, "Final result counts/score mismatch")
        predictions, models = [], []
        for seed in SEEDS:
            directory = OUTPUT / f"full_seed{seed}"
            config, status, history = run_check(directory, seed, ids, [], epoch, "full_data_refit")
            require(status["audit_best_epoch"] is None and status["audit_best_rmse"] is None,
                    "Full-data refit claims validation-based checkpoint selection")
            digest = checkpoint(directory / "final.pt", config, epoch)
            models.append({"seed": seed, "epoch": epoch, "checkpoint": str((directory / "final.pt").resolve()), "sha256": digest})
            saved = npz(directory / "test_predictions.npz")
            require(saved["ids"].tolist() == test_ids, "Saved model test predictions ID/order mismatch")
            check_bounds(saved["predictions"], test, f"full_seed{seed} saved predictions")
            predictions.append(saved["predictions"])
        require(result["models"] == models, "Final model hashes/provenance mismatch")
        final = np.mean(predictions, axis=0)
        template_rows = read_csv(DATA / "sample_submission.csv", ("ImageID", "value"))
        template_ids = [row[0] for row in template_rows]
        require(template_ids == [f"{image_id}_Fovea_{axis}" for image_id in test_ids for axis in "XY"],
                "Original template IDs/order mismatch")
        if selection["selected"] == "new_robust_mean":
            original = read_csv(BASELINE / "submission_robust.csv", ("ImageID", "value"))
            require([row[0] for row in original] == template_ids, "Blending baseline template order mismatch")
            final = (final + np.asarray([float(row[1]) for row in original]).reshape(20, 2)) / 2
        check_bounds(final, test, "Final equal-mean/blended coordinates")
        submission = OUTPUT / SUBMISSION_NAME
        if ILLUMINATION:
            require(result["experiment"] == "illumination_v3" and result["architecture"] == "ResNet34 FPN"
                    and result["preprocessing"] == preprocessing_spec()
                    and Path(result["submission_file"]).resolve() == submission.resolve(), "Illumination final metadata mismatch")
        rows = read_csv(submission, ("ImageID", "value"))
        require(len(rows) == 40 and [row[0] for row in rows] == template_ids, "Submission template rows/order mismatch")
        actual = np.asarray([float(row[1]) for row in rows]).reshape(20, 2)
        close(actual, final, "Submission equals saved full-model mean/blend", atol=1e-7)
        check_bounds(actual, test, "Submission original-pixel bounds")
        mapping = read_json(OUTPUT / "predictions.json")
        require(set(mapping) == {str(image_id) for image_id in test_ids}, "Predictions JSON ID coverage mismatch")
        close(np.asarray([mapping[str(image_id)] for image_id in test_ids]), final, "Predictions JSON mean/blend")
        require(result["submission_sha256"] == sha256(submission), "Final CSV hash mismatch")
        report["evidence"]["final"] = {"submission": str(submission.resolve()), "submission_sha256": sha256(submission),
                                       "models": models, "refit_epochs": epoch, "train_count": 80, "rows": 40,
                                       "prediction_rule": selection["selected"]}
        return True

    final_passed = False
    if selection is not None and selection["gate_passed"]:
        final_passed = section("full_80_refits_actual_ema_models_template_and_saved_prediction_mean", final_check) is True
    elif selection is not None:
        require_no_final = lambda: require(not (OUTPUT / SUBMISSION_NAME).exists(),
                                           "Unqualified candidate unexpectedly has a final CSV")
        section("unqualified_candidate_has_no_final_csv", require_no_final)

    if report["failures"]:
        report["state"] = "failed"
    elif report["pending"]:
        report["state"] = "pending"
    elif selection is not None and not selection["gate_passed"]:
        report["state"] = "completed_not_qualified"
        report["passed"] = True
        report["validation_audit_passed"] = True
        report["final_submission_available"] = False
    elif final_passed:
        report["state"] = "passed"
        report["passed"] = True
        report["validation_audit_passed"] = True
        report["final_submission_available"] = True
    else:
        report["state"] = "pending"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fine", "illumination"), default="fine")
    args = parser.parse_args()
    if args.method == "illumination":
        ILLUMINATION, SIZE, CHANNELS, ENCODER_LR = True, 448, (64, 128, 256, 512), 5e-5
        OUTPUT = ROOT / "outputs/iteration3_20261024/illumination"
        CACHE_NAME, SUBMISSION_NAME, WEIGHTS_NAME = "images448_illumination.npz", "submission_illumination_v3.csv", "resnet34-b627a593.pth"
    report = audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "state": report["state"], "checks": len(report["checks"]),
                      "failures": report["failures"], "pending": report["pending"], "report": str(OUTPUT / "audit.json")},
                     ensure_ascii=False, indent=2))
    raise SystemExit({"passed": 0, "failed": 1, "pending": 2, "completed_not_qualified": 0}[report["state"]])
