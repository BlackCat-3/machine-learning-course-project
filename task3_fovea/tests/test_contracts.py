"""Boundary tests for competition labels, scalar scoring and output mapping."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from task3_fovea.contracts import coordinate_metrics, load_labels, load_template, read_submission, write_submission


class ContractTests(unittest.TestCase):
    def setUp(self):
        # Stay inside the writable workspace; the desktop sandbox's OS temp
        # location may inherit permissions unsuitable for newly created files.
        self.test_directory = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="_contract_test_", dir=self.test_directory)
        self.root = Path(self.temporary.name)
        self.template = self.root / "sample_submission.csv"
        # Deliberately interleaved, reversed axes and reversed image order.
        self.identifiers = ["82_Fovea_Y", "81_Fovea_X", "82_Fovea_X", "81_Fovea_Y"]
        self.template.write_text(
            "ImageID,value\n" + "\n".join(f"{identifier}," for identifier in self.identifiers) + "\n",
            encoding="utf-8",
        )
        self.output = self.root / "submission.csv"
        self.sizes = {81: (100, 200), 82: (300, 400)}
        self.predictions = {81: [12.345678901234567, 45.5], 82: [222.25, 333.75]}

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.test_directory))
        self.temporary.cleanup()

    def test_crcrlf_blank_lines_and_bom_are_accepted(self):
        labels = self.root / "train.csv"
        labels.write_bytes(b"\xef\xbb\xbfdata,Fovea_X,Fovea_Y\r\r\n1,12.5,20\r\r\n\r\r\n2,0,0\r\r\n")
        result = load_labels(labels)
        self.assertEqual(list(result), [1, 2])
        np.testing.assert_array_equal(result[1], [12.5, 20])
        np.testing.assert_array_equal(result[2], [0, 0])

    def test_mapping_uses_image_id_and_axis_not_adjacent_rows(self):
        write_submission(self.template, self.predictions, self.output, self.sizes)
        with self.output.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["ImageID"] for row in rows], self.identifiers)
        np.testing.assert_array_equal(
            [float(row["value"]) for row in rows], [333.75, self.predictions[81][0], 222.25, 45.5]
        )
        result = read_submission(self.output, self.sizes, self.template)
        for image_id in self.predictions:
            np.testing.assert_array_equal(result[image_id], self.predictions[image_id])

    def test_rmse_is_flattened_not_average_of_axis_rmses(self):
        # X RMSE is 3 and Y RMSE is 4: average would be 3.5, Euclidean error 5.
        metrics = coordinate_metrics([[10, 10], [20, 20]], [[13, 14], [23, 24]])
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(12.5))
        self.assertAlmostEqual(metrics["rmse_x"], 3.0)
        self.assertAlmostEqual(metrics["rmse_y"], 4.0)
        self.assertNotAlmostEqual(metrics["rmse"], 3.5)
        self.assertNotAlmostEqual(metrics["rmse"], 5.0)

    def test_invisible_pair_can_be_written(self):
        self.predictions[82] = [0, 0]
        write_submission(self.template, self.predictions, self.output, self.sizes)
        np.testing.assert_array_equal(read_submission(self.output)[82], [0, 0])

    def test_single_zero_visible_edge_points_are_preserved(self):
        labels = self.root / "train.csv"
        labels.write_text("data,Fovea_X,Fovea_Y\n81,0,12\n82,23,0\n", encoding="utf-8")
        edge_predictions = load_labels(labels)
        write_submission(self.template, edge_predictions, self.output, self.sizes)
        result = read_submission(self.output, self.sizes, self.template)
        np.testing.assert_array_equal(result[81], [0, 12])
        np.testing.assert_array_equal(result[82], [23, 0])

    def test_reject_invalid_coordinates_before_overwriting_output(self):
        self.output.write_text("existing valid file", encoding="utf-8")
        invalid_pairs = [[-1, 12], [np.nan, 1], [1, np.inf], [100, 10], [10, 200], [1], [1, 2, 3]]
        for pair in invalid_pairs:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                write_submission(self.template, {81: pair, 82: [1, 1]}, self.output, self.sizes)
            self.assertEqual(self.output.read_text(encoding="utf-8"), "existing valid file")

    def test_reject_missing_extra_or_non_integer_prediction_ids(self):
        invalid = [{81: [1, 1]}, {**self.predictions, 83: [1, 1]}, {"81": [1, 1], 82: [1, 1]}]
        for predictions in invalid:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                write_submission(self.template, predictions, self.output, self.sizes)

    def test_reject_missing_or_invalid_image_size(self):
        for sizes in [{81: (100, 200)}, {81: (100.5, 200), 82: (300, 400)}, {81: (0, 200), 82: (300, 400)}]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                write_submission(self.template, self.predictions, self.output, sizes)

    def test_template_rejects_missing_duplicate_malformed_axes_and_columns(self):
        bad_contents = [
            "ImageID,value\n81_Fovea_X,\n",
            "ImageID,value\n81_Fovea_X,\n81_Fovea_X,\n81_Fovea_Y,\n",
            "ImageID,value\n81_Fovea_Z,\n81_Fovea_Y,\n",
            "ImageID,value\n81_Fovea_X,,\n81_Fovea_Y,\n",
            "ImageID,other\n81_Fovea_X,\n81_Fovea_Y,\n",
            "ImageID,value\n",
            "ImageID,value\n,\n",
        ]
        for content in bad_contents:
            self.template.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_template(self.template)

    def test_labels_reject_duplicate_invalid_or_missing_values(self):
        labels = self.root / "train.csv"
        for records in ["1,1,2\n1,3,4", "1,nan,2", "1,1,", "1,-1,2", "0,1,2", "1.5,1,2", "1,1,2,3", ",,"]:
            labels.write_text("data,Fovea_X,Fovea_Y\n" + records + "\n", encoding="utf-8")
            with self.subTest(records=records), self.assertRaises(ValueError):
                load_labels(labels)

    def test_read_submission_checks_values_and_order(self):
        write_submission(self.template, self.predictions, self.output, self.sizes)
        lines = self.output.read_text(encoding="utf-8").splitlines()
        self.output.write_text("\n".join([lines[0], *lines[1:][::-1]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "order"):
            read_submission(self.output, self.sizes, self.template)
        self.output.write_text("ImageID,value\n81_Fovea_X,\n81_Fovea_Y,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_submission(self.output)

    def test_metrics_reject_empty_wrong_shapes_and_nonfinite_values(self):
        for truth, prediction in [([], []), ([1, 2], [1, 2]), ([[1, 2]], [[1, 2], [3, 4]]), ([[np.nan, 1]], [[1, 1]]), ([[1, 1]], [[1, np.inf]])]:
            with self.subTest(truth=truth, prediction=prediction), self.assertRaises(ValueError):
                coordinate_metrics(truth, prediction)


if __name__ == "__main__":
    unittest.main()
