"""Check the exact competition submission contract without submitting online."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def validate(prediction_path, template_path):
    def read(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["ID", "Emotion"]:
                raise ValueError(f"{path}: expected exactly ID,Emotion; got {reader.fieldnames}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: empty submission")
        if any(set(row) != {"ID", "Emotion"} or not row["ID"] for row in rows):
            raise ValueError(f"{path}: malformed row or missing ID")
        ids = [row["ID"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate IDs")
        return rows, ids

    rows, ids = read(prediction_path)
    _, expected_ids = read(template_path)
    if ids != expected_ids:
        raise ValueError("Prediction IDs or their order differ from the supplied template")
    if any(row["Emotion"] not in {"0", "1", "2", "3", "4", "5"} for row in rows):
        raise ValueError("Emotion must contain only integer labels 0 through 5")
    return {
        "passed": True,
        "prediction_file": str(Path(prediction_path).resolve()),
        "rows": len(rows),
        "columns": ["ID", "Emotion"],
        "ids_match_template_in_order": True,
        "class_counts": dict(sorted(Counter(row["Emotion"] for row in rows).items())),
        "label_map": {"0": "Angry", "1": "Fear", "2": "Happy", "3": "Sad", "4": "Surprise", "5": "Neutral"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--template", type=Path, default=Path("data/submission_template.csv"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = validate(args.submission, args.template)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
