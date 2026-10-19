"""Standard-library tests for the independent confusion-matrix recomputation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recompute', Path(__file__).with_name('recompute_bridge.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecomputeTests(unittest.TestCase):
    def test_metrics_include_zero_support_class(self):
        actual = r.matrix_metrics([[8, 2, 0], [1, 9, 0], [0, 0, 0]])
        self.assertAlmostEqual(actual['macro_f1'], (16 / 19 + 18 / 21) / 3)
        self.assertAlmostEqual(actual['accuracy'], .85)
        self.assertAlmostEqual(actual['balanced_accuracy'], (.8 + .9) / 3)

    def test_invalid_counts_are_rejected(self):
        for matrix in ([[1, 0]], [[1, -1], [0, 2]], [[0, 0], [0, 0]], [[1., 0], [0, 1]]):
            with self.assertRaises(ValueError):
                r.matrix_metrics(matrix)

    def test_pair_direction_and_sample_standard_deviation(self):
        a = [i / 100 for i in range(10)]
        b = [0.] * 10
        forward = r.paired(a, b)
        reverse = r.paired(b, a)
        self.assertAlmostEqual(forward['difference_pp'], 4.5)
        self.assertEqual(forward['wins'], 9)
        self.assertEqual(forward['ties'], 1)
        self.assertAlmostEqual(forward['difference_pp'], -reverse['difference_pp'])
        self.assertAlmostEqual(forward['ci95_low_pp'], -reverse['ci95_high_pp'])
        self.assertEqual(forward['nominal_exact_signflip_p'], reverse['nominal_exact_signflip_p'])

    def test_eighty_record_pipeline_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for ds in ('wm', 'fgvc'):
                for shot in (5, 10):
                    for method in ('ours', 'clap'):
                        for seed in range(10):
                            matrix = ([[9, 1], [2, 8]] if method == 'ours' else [[8, 2], [3, 7]])
                            metrics = {**r.matrix_metrics(matrix), 'confusion_matrix': matrix}
                            path = root / 'test' / ds / method / f'{8 if ds == "wm" else 100}way_{shot}shot_seed{seed}.json'
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(dict(dataset=ds, method=method, shot=shot, seed=seed, metrics=metrics)), encoding='utf8')
            rows, seeds, audit = r.recompute(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(seeds), 80)
            self.assertEqual(audit['max_absolute_error'], 0)
            self.assertAlmostEqual(rows[0]['accuracy_difference_pp'], 10.)
            rows[0]['difference_pp'] += .1
            (root / 'summary.json').write_text(json.dumps(rows), encoding='utf8')
            with self.assertRaises(ValueError):
                r.recompute(root)


if __name__ == '__main__':
    unittest.main()
