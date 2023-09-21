from __future__ import annotations

import pytest

from upp.stages.normalisation import Normalisation

# Import or define the combine_class_dict function


class TestCombineClassDict:
    @staticmethod
    def test_combine_class_dict_basic():
        # Test when both class_dict_A and class_dict_B have the same structure
        class_dict_A = {
            "name1": {"var1": (["A", "B"], [10, 20])},
            "name2": {"var2": (["X", "Y"], [30, 40])},
        }
        class_dict_B = {
            "name1": {"var1": (["A", "B"], [5, 15])},
            "name2": {"var2": (["X", "Y"], [25, 35])},
        }

        expected_result = {
            "name1": {"var1": (["A", "B"], [15, 35])},
            "name2": {"var2": (["X", "Y"], [55, 75])},
        }

        # Call the function and check the result
        result = Normalisation.combine_class_dict(class_dict_A, class_dict_B)
        assert result == expected_result

    @staticmethod
    def test_combine_class_dict_missing_label():
        # Test when a label is missing in class_dict_A
        class_dict_A = {
            "name1": {"var1": (["A", "B"], [10, 20])},
        }
        class_dict_B = {
            "name1": {"var1": (["A", "B", "C"], [5, 15, 25])},
        }

        expected_result = {
            "name1": {"var1": (["A", "B", "C"], [15, 35, 25])},
        }

        # Call the function and check the result
        result = Normalisation.combine_class_dict(class_dict_A, class_dict_B)
        assert result == expected_result

    @staticmethod
    def test_combine_class_dict_variable_length_mismatch():
        # Test when class_dict_A has arrays of different lengths for the same variable
        class_dict_A = {
            "name1": {"var1": (["A", "B", "C"], [10, 20])},  # Length mismatch
        }
        class_dict_B = {
            "name1": {"var1": (["A", "B", "C"], [5, 15, 25])},
        }

        # Call the function and expect a ValueError to be raised
        with pytest.raises(ValueError):
            Normalisation.combine_class_dict(class_dict_A, class_dict_B)
