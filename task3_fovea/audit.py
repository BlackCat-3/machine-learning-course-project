"""Independently audit saved runs and predictions; never train or edit predictions.

Exit codes: 0 = all checks passed, 1 = a check failed, 2 = results still missing.
The only generated file is the audit report (outputs/study/audit.json by default).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
CANDIDATES = ("heatmap", "regression", "equal_mean")
ARCHITECTURES = CANDIDATES[:2]


class PendingResults(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def required(path):
    if not path.is_file():
        raise PendingResults(f"Not yet available: {path}")
    return path


def read_json(path):
    return json.loads(required(path).read_text(encoding="utf-8"))


def read_csv(path, expected_header):
    with required(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    require(rows and rows[0] == list(expected_header), f"Invalid header: {path}")
    require(all(len(row) == len(expected_header) for row in rows[1:]), f"Malformed CSV: {path}")
    return rows[1:]


def sha256(path):
    return hashlib.sha256(required(path).read_bytes()).hexdigest()


def close(actual, expected, message, atol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"{message}: shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{message}: nonfinite value")
    require(np.allclose(actual, expected, rtol=1e-10, atol=atol), f"{message}: values differ")


def metrics(truth, prediction):
    # Independent implementation: all 2*N original-pixel coordinates are pooled.
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    require(truth.ndim == 2 and truth.shape[1] == 2 and prediction.shape == truth.shape,
            "Metric arrays must be matching (N, 2)")
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), "Nonfinite metric arrays")
    errors = prediction - truth
    return {"rmse": float(np.sqrt(np.mean(errors ** 2))),
            "rmse_x": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "rmse_y": float(np.sqrt(np.mean(errors[:, 1] ** 2)))}


def check_metrics(actual, expected, context):
    require(set(actual) == set(expected), f"{context}: metric names differ")
    for key in expected:
        close(actual[key], expected[key], f"{context}/{key}")


def check_bounds(prediction, records, context):
    prediction = np.asarray(prediction)
    require(prediction.shape == (len(records), 2), f"{context}: incorrect shape")
    require(np.isfinite(prediction).all(), f"{context}: nonfinite coordinates")
    sizes = np.array([[r["width"], r["height"]] for r in records])
    require(((prediction >= 0) & (prediction < sizes)).all(), f"{context}: outside original image bounds")


def architecture_of(checkpoint):
    state = checkpoint["model"]
    if "lateral.0.weight" in state and "head.1.weight" in state:
        require(tuple(state["head.1.weight"].shape) == (1, 64, 1, 1), "Incorrect heatmap head")
        return "heatmap"
    if "head.5.weight" in state:
        require(tuple(state["head.5.weight"].shape) == (2, 128), "Incorrect regression head")
        return "regression"
    raise AssertionError("Unknown checkpoint architecture")


def audit_run(directory, architecture, seed, train_ids, validation_ids, size, expected_epochs=None):
    config = read_json(directory / "config.json")
    status = read_json(directory / "status.json")
    if status.get("state") != "completed":
        raise PendingResults(f"Run not complete: {directory} ({status.get('state')})")
    checkpoint = torch.load(required(directory / "best.pt"), map_location="cpu", weights_only=True)
    require(config == checkpoint["config"], f"{directory}: config differs from checkpoint metadata")
    require(architecture_of(checkpoint) == architecture, f"{directory}: actual model architecture differs")
    # The first heatmap fold predates the explicit architecture field.
    require(config.get("architecture", architecture_of(checkpoint)) == architecture,
            f"{directory}: declared architecture differs")
    require(config["seed"] == seed and config["size"] == size, f"{directory}: seed/size mismatch")
    require(config["train_ids"] == train_ids, f"{directory}: training IDs/order differ")
    require(config["validation_ids"] == validation_ids, f"{directory}: validation IDs/order differ")
    require(len(train_ids) == len(set(train_ids)) and len(validation_ids) == len(set(validation_ids)),
            f"{directory}: duplicate IDs")
    require(not set(train_ids).intersection(validation_ids), f"{directory}: training/validation overlap")
    require(config["schedule_epochs"] == 80, f"{directory}: unexpected learning-rate schedule")
    require(config["tta"] == "horizontal-flip mean", f"{directory}: unexpected TTA")
    require(config["batch_size"] == 16, f"{directory}: unexpected batch size")
    require(Path(config["pretrained"]).resolve() ==
            (ROOT.parent / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth").resolve(),
            f"{directory}: initialization source differs")
    completed = status["completed_epochs"]
    require(isinstance(completed, int) and completed > 0, f"{directory}: invalid completed epoch count")
    # epochs_run in the legacy config is a plan field, never completion evidence.
    if "epochs_planned" in config:
        require(config["epochs_planned"] == completed, f"{directory}: planned/completed epochs differ")
    if expected_epochs is not None:
        require(completed == expected_epochs, f"{directory}: completed epochs differ from selection")
    with required(directory / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    require([int(row["epoch"]) for row in history] == list(range(1, completed + 1)),
            f"{directory}: history does not prove every completed epoch")
    require(int(checkpoint["epoch"]) == status["best_epoch"], f"{directory}: checkpoint/status epoch differs")
    require(1 <= status["best_epoch"] <= completed, f"{directory}: best epoch out of range")
    if validation_ids:
        best = min(history, key=lambda row: float(row["rmse"]))
        require(int(best["epoch"]) == status["best_epoch"], f"{directory}: checkpoint is not best history epoch")
        close(float(best["rmse"]), status["best_rmse"], f"{directory}: best history/status RMSE")
    else:
        require(status["best_epoch"] == completed, f"{directory}: refit checkpoint is not final epoch")
        require(status["best_rmse"] is None, f"{directory}: refit claims a validation RMSE")
    return {"config": config, "status": status, "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256(directory / "best.pt")}


def audit(manifest_path, study_dir, data_dir):
    report = {"passed": False, "state": "checking", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
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

    def manifest_check():
        manifest = read_json(manifest_path)
        require(manifest["folds"] == 4, "Exactly four folds required")
        records = manifest["records"]
        train = [r for r in records if r["split"] == "train"]
        test = [r for r in records if r["split"] == "test"]
        require(len(records) == 100 and len(train) == 80 and len(test) == 20, "Expected 80 train + 20 test")
        require([r["id"] for r in train] == list(range(1, 81)), "Unexpected training IDs/order")
        require([r["id"] for r in test] == list(range(81, 101)), "Unexpected test IDs/order")
        label_rows = read_csv(data_dir / "fovea_localization_train_GT.csv", ("data", "Fovea_X", "Fovea_Y"))
        labels = {int(row[0]): [float(row[1]), float(row[2])] for row in label_rows}
        require(len(label_rows) == len(labels) == 80, "Training CSV IDs not unique")
        duplicate_groups = defaultdict(list)
        for record in records:
            path = data_dir / record["split"] / f"{record['id']:04}.jpg"
            require(sha256(path) == record["sha256"], f"Image file changed: {path}")
            with Image.open(path) as image:
                image = image.convert("RGB")
                require(image.size == (record["width"], record["height"]), f"Image size differs: {path}")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            require(pixel_hash == record["pixel_sha256"], f"Pixel hash differs: {path}")
            crop = record["crop"]
            require(0 <= crop[0] < crop[2] <= record["width"] and
                    0 <= crop[1] < crop[3] <= record["height"], f"Invalid crop: {path}")
            if record["split"] == "train":
                require(record["fold"] in range(4), f"Invalid fold: {path}")
                close(record["xy"], labels[record["id"]], f"Original CSV/manifest label {record['id']}")
                expected_crop = (np.array(record["xy"]) - np.array(crop[:2])) / (np.array(crop[2:]) - crop[:2])
                close(record["xy_crop"], expected_crop, f"Crop coordinates {record['id']}")
                duplicate_groups[pixel_hash].append(record)
        check_bounds([r["xy"] for r in train], train, "Training labels")
        for group in duplicate_groups.values():
            require(len({r["fold"] for r in group}) == 1, "Duplicate image leaked across folds")
        duplicates = sorted(sorted(r["id"] for r in group) for group in duplicate_groups.values() if len(group) > 1)
        require(duplicates == sorted(sorted(g) for g in manifest["exact_duplicate_train_groups"]),
                "Duplicate metadata differs")
        fold_counts = [sum(r["fold"] == fold for r in train) for fold in range(4)]
        require(fold_counts == manifest["fold_counts"], "Fold count metadata differs")
        require(all(count > 0 for count in fold_counts), "Empty validation fold")
        report["evidence"]["data"] = {"train_count": 80, "test_count": 20, "fold_counts": fold_counts,
                                           "exact_duplicate_groups": duplicates}
        return manifest, train, test

    checked = section("manifest_and_original_images_labels", manifest_check)
    if checked is None:
        return finalize(report)
    manifest, train, test = checked
    truth = np.array([r["xy"] for r in train], dtype=np.float64)
    train_ids = [r["id"] for r in train]
    run_evidence, run_predictions = {}, {}
    for arch in ARCHITECTURES:
        for fold in range(4):
            def validation_check(arch=arch, fold=fold):
                directory = ROOT / "outputs" / ("baseline" if arch == "heatmap" else "regression") / f"validation_fold{fold}_seed42"
                train_subset = [r["id"] for r in train if r["fold"] != fold]
                val_records = [r for r in train if r["fold"] == fold]
                run = audit_run(directory, arch, 42, train_subset, [r["id"] for r in val_records], manifest["size"], 80)
                prediction = np.load(required(directory / "validation_predictions.npy"), allow_pickle=False)
                check_bounds(prediction, val_records, f"{arch}/fold{fold} validation predictions")
                scored = metrics([r["xy"] for r in val_records], prediction)
                close(scored["rmse"], run["status"]["best_rmse"], f"{arch}/fold{fold} saved predictions/status RMSE")
                run_predictions[(arch, fold)] = prediction
                run_evidence[(arch, fold)] = run
                report["evidence"][f"{arch}_fold{fold}"] = {"best_epoch": run["checkpoint_epoch"],
                    "completed_epochs": run["status"]["completed_epochs"], "metrics": scored,
                    "train_count": len(train_subset), "validation_count": len(val_records),
                    "checkpoint_sha256": run["checkpoint_sha256"]}
                return True
            section(f"{arch}_fold{fold}_run_checkpoint_predictions", validation_check)

    def selection_check():
        selection = read_json(study_dir / "selection.json")
        plan = read_json(study_dir / "plan.json")
        require(selection["plan"] == plan, "Selection/plan mismatch")
        require(plan["candidate_names"] == list(CANDIDATES) and plan["folds"] == 4 and plan["seed"] == 42,
                "Study is not the fixed three-candidate/four-fold plan")
        require(plan["public_leaderboard_used"] is False, "Selection claims public leaderboard use")
        if len(run_evidence) != 8:
            raise PendingResults("Every validation run must finish and pass before selection can pass")
        with np.load(required(study_dir / "oof_predictions.npz"), allow_pickle=False) as saved:
            require(set(saved.files) == {"truth", "baseline", *CANDIDATES}, "Unexpected OOF arrays")
            oof = {key: saved[key].copy() for key in saved.files}
        close(oof["truth"], truth, "OOF truth/original labels", atol=0)
        reconstructed, epochs = {}, {}
        for arch in ARCHITECTURES:
            coverage = np.zeros(80, dtype=int)
            reconstructed[arch] = np.full((80, 2), np.nan)
            epochs[arch] = []
            require(len(selection["fold_metrics"][arch]) == 4, f"{arch}: missing fold metrics")
            for fold in range(4):
                indices = [i for i, r in enumerate(train) if r["fold"] == fold]
                coverage[indices] += 1
                reconstructed[arch][indices] = run_predictions[(arch, fold)]
                epochs[arch].append(run_evidence[(arch, fold)]["checkpoint_epoch"])
                check_metrics(selection["fold_metrics"][arch][fold], metrics(truth[indices], reconstructed[arch][indices]),
                              f"{arch}/fold{fold} reported metrics")
            require(np.all(coverage == 1), f"{arch}: some records predicted out-of-fold zero or multiple times")
            close(oof[arch], reconstructed[arch], f"{arch} saved/reconstructed OOF", atol=0)
        reconstructed["equal_mean"] = (reconstructed["heatmap"] + reconstructed["regression"]) / 2
        close(oof["equal_mean"], reconstructed["equal_mean"], "Equal-weight OOF mean", atol=0)
        require(set(selection["metrics"]) == set(CANDIDATES), "Candidate set changed")
        rescored = {name: metrics(truth, reconstructed[name]) for name in CANDIDATES}
        for name in CANDIDATES:
            check_metrics(selection["metrics"][name], rescored[name], f"Pooled {name} RMSE")
        chosen = min(CANDIDATES, key=lambda name: rescored[name]["rmse"])
        require(selection["selected"] == chosen, "Selected candidate does not minimize pooled OOF RMSE")
        require(selection["best_epochs"] == epochs, "Selected epoch metadata differs from actual checkpoints")
        refit_epochs = {arch: int(np.median(epochs[arch])) for arch in ARCHITECTURES}
        require(selection["refit_epochs"] == refit_epochs, "Refit epoch count is not the declared integer median")
        baseline = np.empty((80, 2))
        for fold in range(4):
            center = np.mean([r["xy_crop"] for r in train if r["fold"] != fold], axis=0)
            for i, record in enumerate(train):
                if record["fold"] == fold:
                    crop = np.array(record["crop"])
                    baseline[i] = center * (crop[2:] - crop[:2]) + crop[:2]
        close(oof["baseline"], baseline, "OOF baseline uses each fold's training subset")
        check_metrics(selection["constant_baseline"], metrics(truth, baseline), "Constant baseline metrics")
        report["evidence"]["selection"] = {"selected": chosen, "recomputed_metrics": rescored,
            "oof_coverage_per_architecture": {arch: {"records": 80, "count_per_record": 1} for arch in ARCHITECTURES},
            "refit_epochs": refit_epochs}
        return selection

    selection = section("oof_coverage_metrics_and_fixed_candidate_selection", selection_check)
    if selection is not None:
        def final_check():
            result = read_json(study_dir / "result.json")
            require(result["selection"] == selection, "Final result references a different selection")
            require(result["refit_train_count"] == 80 and result["submission_rows"] == 40, "Incorrect final record counts")
            selected_arches = list(ARCHITECTURES) if selection["selected"] == "equal_mean" else [selection["selected"]]
            expected_sources = [(arch, seed) for arch in selected_arches for seed in (43, 44)]
            sources = result["models"]
            require([(s["architecture"], s["seed"]) for s in sources] == expected_sources,
                    "Final model architectures/seeds/order differ from fixed refit plan")
            predictions = []
            for source in sources:
                arch, seed = source["architecture"], source["seed"]
                directory = study_dir / f"full_{arch}_seed{seed}"
                epochs = selection["refit_epochs"][arch]
                require(source["epochs"] == epochs, "Refit source epoch provenance mismatch")
                require(Path(source["checkpoint"]).resolve() == (directory / "best.pt").resolve(),
                        "Refit source checkpoint path mismatch")
                run = audit_run(directory, arch, seed, train_ids, [], manifest["size"], epochs)
                require(source["checkpoint_sha256"] == run["checkpoint_sha256"], "Final checkpoint hash mismatch")
                prediction = np.load(required(directory / "test_predictions.npy"), allow_pickle=False)
                check_bounds(prediction, test, f"{arch}/seed{seed} test predictions")
                predictions.append(prediction)
                report["evidence"][f"full_{arch}_seed{seed}"] = {"train_count": 80, "validation_count": 0,
                    "epochs": epochs, "checkpoint_epoch": run["checkpoint_epoch"],
                    "checkpoint_sha256": run["checkpoint_sha256"]}
            final = np.mean(predictions, axis=0)
            check_bounds(final, test, "Final equal-weight mean")
            prediction_json = read_json(study_dir / "predictions.json")
            require(set(prediction_json) == {str(r["id"]) for r in test}, "Final prediction JSON IDs differ")
            close([prediction_json[str(r["id"])] for r in test], final, "Final JSON/model prediction mean", atol=0)
            template = read_csv(data_dir / "sample_submission.csv", ("ImageID", "value"))
            submission = read_csv(study_dir / "submission.csv", ("ImageID", "value"))
            require(len(submission) == len(template) == 40, "Submission must contain 40 data rows")
            identifiers = [row[0] for row in template]
            require(identifiers == [row[0] for row in submission], "Submission/template row order differs")
            expected_ids = [f"{r['id']}_Fovea_{axis}" for r in test for axis in ("X", "Y")]
            require(identifiers == expected_ids and len(set(identifiers)) == 40, "Template ID/axis pairing differs")
            csv_values = np.array([float(row[1]) for row in submission]).reshape(20, 2)
            require(np.array_equal(csv_values, final), "CSV values differ from saved models' exact equal-weight mean")
            check_bounds(csv_values, test, "CSV coordinates")
            require(sha256(study_dir / "submission.csv") == result["submission_sha256"], "Submission SHA256 mismatch")
            report["evidence"]["submission"] = {"rows": 40, "test_images": 20, "models_in_equal_mean": len(predictions),
                "sha256": result["submission_sha256"], "maximum_csv_mean_absolute_difference": float(np.abs(csv_values - final).max())}
            return True
        section("full_data_refits_model_sources_and_submission", final_check)
    return finalize(report)


def finalize(report):
    required_sections = 1 + 8 + 1 + 1
    report["passed"] = not report["failures"] and not report["pending"] and len(report["checks"]) == required_sections
    report["state"] = "passed" if report["passed"] else "failed" if report["failures"] else "awaiting_results"
    report["scope"] = "Saved artifact consistency audit, not an independent rerun of model inference or an online score verification."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/baseline/prepared/manifest.json")
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/study")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/detection")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(args.manifest, args.study_dir, args.data_dir)
    destination = args.output or args.study_dir / "audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "state": report["state"], "checks_passed": len(report["checks"]),
                      "failures": report["failures"], "pending": report["pending"], "report": str(destination)},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1 if report["failures"] else 2)


if __name__ == "__main__":
    main()
