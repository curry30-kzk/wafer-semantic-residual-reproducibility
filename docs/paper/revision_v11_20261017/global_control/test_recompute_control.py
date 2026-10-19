"""Meaningful edge-case checks for independent replay arithmetic and identity."""
import unittest
from recompute_control import metrics, choose, paired


class ReplayChecks(unittest.TestCase):
    def test_sparse_counts_include_unpredicted_classes(self):
        r=metrics({'class_count':3,'cells':[[0,0,2],[1,0,2],[2,2,2]]})
        self.assertAlmostEqual(r['accuracy'],2/3)
        self.assertAlmostEqual(r['balanced_accuracy'],2/3)
        self.assertAlmostEqual(r['macro_f1'],(2/3+0+1)/3)

    def test_malformed_sparse_counts_rejected(self):
        for cells in ([[0,0,1],[0,0,2]], [[0,2,1]], [[0,0,-1]], [[0,0,1.2]], []):
            with self.subTest(cells=cells):
                with self.assertRaises(ValueError):metrics({'class_count':2,'cells':cells})

    def test_candidate_tie_uses_declared_order(self):
        a={'candidate':0,'mean_macro_f1':.3}
        b={'candidate':1,'mean_macro_f1':.3+5e-13}
        self.assertIs(choose([a,b]),a)
        self.assertIs(choose([b,a]),b)

    def test_paired_unit_difference_and_constant_effect(self):
        r=paired([.2]*10,[.1]*10)
        self.assertAlmostEqual(r['difference_pp'],10.)
        self.assertAlmostEqual(r['ci95_low_pp'],10.)
        self.assertEqual(r['nominal_exact_signflip_p'],2/1024)
        self.assertEqual((r['wins'],r['ties'],r['losses']),(10,0,0))
        with self.assertRaises(ValueError):paired([.2]*9,[.1]*9)


if __name__=='__main__':unittest.main()
