"""Small independent mathematical checks; run with python -m unittest discover."""
import importlib
import unittest
import numpy as np
from pathlib import Path
import tempfile


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("recompute")
        except ImportError:
            self.fail("The standalone recompute module has not been implemented.")

    def test_macro_f1_uses_all_declared_classes(self):
        # Class 2 has no support: its F1 is 0, and remains in the macro mean.
        cm = [[3, 1, 0], [2, 4, 0], [0, 0, 0]]
        values = self.module.metrics_from_confusion(cm)
        self.assertAlmostEqual(values["accuracy"], 7 / 10)
        self.assertAlmostEqual(values["macro_f1"], (6 / 9 + 8 / 11) / 3)

    def test_sample_sd_and_student_interval(self):
        values = self.module.describe([1, 2, 3])
        self.assertAlmostEqual(values["mean"], 2)
        self.assertAlmostEqual(values["sample_std"], 1)
        self.assertAlmostEqual(values["ci95_high"] - 2, 4.302652729911275 / np.sqrt(3))

    def test_pairs_match_seed_ids_not_input_order(self):
        # Every seed has an exactly equal effect, despite large seed variability.
        a = {0: 11, 1: 21, 2: 31}
        b = {2: 30, 0: 10, 1: 20}
        result = self.module.paired(a, b)
        self.assertEqual(result["seed_ids"], [0, 1, 2])
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["sample_std"], 0)
        self.assertEqual(result["ci95_low"], 1)
        self.assertEqual(result["ci95_high"], 1)

    def test_incomplete_pairs_raise(self):
        with self.assertRaises(ValueError):
            self.module.paired({0: 1, 1: 2}, {0: 0})

    def test_sparse_confusion_rejects_duplicate_cells(self):
        with self.assertRaises(ValueError):
            self.module.decode_confusion({"n_classes": 2, "confusion_entries": [[0, 0, 1], [0, 0, 2]]})

    def test_confusion_rejects_fractional_counts(self):
        with self.assertRaises(ValueError):
            self.module.metrics_from_confusion([[1.2, 0], [0, 1]])

    def test_absent_optional_results_remove_stale_csv(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "same_text_control.csv"
            path.write_text("stale older results", encoding="utf-8")
            self.module.write_csv(path, [])
            self.assertFalse(path.exists(), "Absent results must not leave an older optional result table.")

    def test_shared_control_uses_reencoded_method_as_comparison_left(self):
        records = []
        for method, entries in (("shared_reencoded_classwise", [[0, 0, 2], [1, 1, 2]]),
                                ("same_text_proker", [[0, 0, 2], [1, 0, 2]]),
                                ("legacy_classwise_reliability", [[0, 1, 2], [1, 0, 2]])):
            for seed in (0, 1):
                records.append({"record_id": f"{method}_{seed}", "shot": 5, "seed": seed,
                    "n_classes": 2, "n_query": 4, "confusion_entries": entries,
                    "uses": [{"scope": "shared_text_control", "dataset": "example", "method": method}]})
        _, summary, comparisons = self.module.calculate({"expected_seed_ids": [0, 1], "records": records})
        self.assertEqual(len(summary), 3)
        self.assertEqual({row["left"] for row in comparisons}, {"shared_reencoded_classwise"})
        by_right = {row["right"]: row for row in comparisons}
        self.assertAlmostEqual(by_right["same_text_proker"]["macro_f1"]["mean"], 2 / 3)
        self.assertAlmostEqual(by_right["legacy_classwise_reliability"]["macro_f1"]["mean"], 1)

    def test_partial_shared_seed_set_is_not_reported(self):
        record = {"record_id": "only_seed_zero", "shot": 5, "seed": 0,
                  "n_classes": 2, "n_query": 2, "confusion_entries": [[0, 0, 1], [1, 1, 1]],
                  "uses": [{"scope": "shared_text_control", "dataset": "example", "method": "shared_reencoded_classwise"}]}
        with self.assertRaises(ValueError):
            self.module.calculate({"expected_seed_ids": [0, 1], "records": [record]})


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("extract_inputs")
        except ImportError:
            self.fail("The count-input extraction module has not been implemented.")

    def test_prediction_csv_counts_use_true_and_predicted_indices(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text("true_index,predicted_index,method_score,correct\n0,1,999,True\n0,0,-999,False\n1,1,0,False\n", encoding="utf-8")
            cm, fields = self.module.cm_from_csv(path, 2)
            np.testing.assert_array_equal(cm, [[1, 1], [0, 1]])
            self.assertIn("true_index", fields)

    def test_out_of_range_prediction_indices_raise(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text("true_index,predicted_index\n0,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.cm_from_csv(path, 2)

    def test_missing_shared_results_are_explicit_and_not_partially_included(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            state = self.module.completed_result_state(root, root / "shared", "evaluation_completed.json")
            self.assertEqual(state["status"], "not_complete")
            self.assertEqual(state["included_tasks"], 0)
            self.assertEqual(state["expected_tasks"], 60)
            self.assertEqual(len(state["missing_expected_files"]), 183)

    def test_completion_receipt_cannot_mask_incorrect_task_count(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            destination = root / "shared"
            for dataset, n_way in (("wm", 8), ("fgvc", 100)):
                for shot in (1, 5, 10):
                    for seed in range(10):
                        stem = f"{n_way}way_{shot}shot_seed{seed}"
                        for child, extension in (("raw", ".json"), ("scores", ".npz"), ("predictions", ".csv")):
                            path = destination / dataset / child / (stem + extension)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text("placeholder content is never read by this status test", encoding="utf-8")
            for name in ("summary.json", "paired_comparisons.json"):
                (destination / name).write_text("[]", encoding="utf-8")
            (destination / "evaluation_completed.json").write_text('{"tasks": 59}', encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.completed_result_state(root, destination, "evaluation_completed.json")
            (destination / "evaluation_completed.json").write_text('{"tasks": 60}', encoding="utf-8")
            self.assertEqual(self.module.completed_result_state(root, destination, "evaluation_completed.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
