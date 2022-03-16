import unittest
from typing import Dict, List

from woe_scoring.core.binning.functions import _check_diff_woe, _chi2, _find_index_of_diff_flag, _mono_flags


class BaseTests(unittest.TestCase):
    def setUp(self):
        self.test_input_dicts: List[Dict] = [
            {
                "bad": 0,
                "total": 4,
                "bad_rate": 0,
                "woe": 4,
            },
            {
                "bad": 1,
                "total": 4,
                "bad_rate": 1 / 4,
                "woe": 2.5
            },
            {
                "bad": 2,
                "total": 4,
                "bad_rate": 2 / 4,
                "woe": 1
            },
            {
                "bad": 3,
                "total": 4,
                "bad_rate": 3 / 4,
                "woe": -1
            }
        ]
        bad: int = sum(bad_rate["bad"] for bad_rate in self.test_input_dicts)
        total: int = sum(bad_rate["total"] for bad_rate in self.test_input_dicts)
        self.overall_rate: float = bad / total
        self.diff_woe_threshold: float = 0.1

        self.bad_input_dicts: List[Dict] = [{
                "bad": 3,
                "total": 4,
                "bad_rate": 3 / 4,
                "woe": 1.1
            }, {
                "bad": 2,
                "total": 4,
                "bad_rate": 2 / 4,
                "woe": 1.12
            }, {
                "bad": 3,
                "total": 4,
                "bad_rate": 3 / 4,
                "woe": -1
            }, {"bad": 4, "total": 4, "bad_rate": 1, "woe": -3}]

    def test__chi2(self):
        self.assertEqual(3.3333333333333335, _chi2(self.test_input_dicts, self.overall_rate))
        self.assertIsInstance(_chi2(self.test_input_dicts, self.overall_rate), float)

    def test__check_diff_woe(self):
        self.assertEqual(None, _check_diff_woe(self.test_input_dicts, self.diff_woe_threshold))
        self.assertEqual(0, _check_diff_woe(self.bad_input_dicts, self.diff_woe_threshold))

    def test__mono_flags(self):
        self.assertIsInstance(_mono_flags(self.test_input_dicts), bool)
        self.assertEqual(True, _mono_flags(self.test_input_dicts))
        self.assertEqual(False, _mono_flags(self.bad_input_dicts))

    def test__find_index_of_diff_flag(self):
        self.assertIsInstance(_find_index_of_diff_flag(self.bad_input_dicts), int)
        self.assertEqual(0, _find_index_of_diff_flag(self.bad_input_dicts))
