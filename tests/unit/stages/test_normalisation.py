from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from ftag import get_mock_file

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.normalisation import Normalisation

CFG_DIR = Path(__file__).parent.parent / "fixtures"

# Import or define the combine_class_dict function


class TestCombineClassDict:
    def generate_mock(self, out_file, N=10):
        _, f = get_mock_file(num_jets=N, fname=out_file)
        f.close()

    def setup_method(self, method):
        os.makedirs("/tmp/upp-tests/integration/temp_workspace/ntuples", exist_ok=True)
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data1.h5")
        self.generate_mock("/tmp/upp-tests/integration/temp_workspace/ntuples/data2.h5")
        print(f"setup_method, method: {method.__name__}")

    def teardown_method(self, method):
        subprocess.run(["rm", "-rf", "/tmp/upp-tests/integration"], check=True)
        print(f"teardown_method, method: {method.__name__}")

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

    @staticmethod
    def test_combine_mean_std():
        # Test combination of mean and std
        mean_A, mean_B = (3, 6)
        std_A, std_B = (1, 5)
        num_A, num_B = (100, 200)
        combined_mean_ref, combined_std_ref = (5, 4.358898943540674)

        # Calculate the combined mean and std
        combined_mean, combined_std = Normalisation.combine_mean_std(
            mean_A=mean_A,
            mean_B=mean_B,
            std_A=std_A,
            std_B=std_B,
            num_A=num_A,
            num_B=num_B,
        )

        # Check that the correct values are returned
        assert combined_mean_ref == combined_mean
        assert combined_std_ref == combined_std

    @staticmethod
    def test_combine_norm_dict():
        # Test combination of mean and std
        norm_A = {
            "jets": {
                "test_var_1": {"mean": 3, "std": 1},
                "test_var_2": {"mean": 3, "std": 1},
            }
        }
        norm_B = {
            "jets": {
                "test_var_1": {"mean": 6, "std": 5},
                "test_var_2": {"mean": 6, "std": 5},
            }
        }
        num_A, num_B = (100, 200)
        combined_dict_ref = {
            "jets": {
                "test_var_1": {"mean": 5, "std": 4.358898943540674},
                "test_var_2": {"mean": 5, "std": 4.358898943540674},
            }
        }

        # Calculate the combined mean and std
        combined_dict = Normalisation.combine_norm_dict(
            self=Normalisation(
                config=PreprocessingConfig.from_file(
                    Path(CFG_DIR / "test_config_pdf_auto_umami.yaml"), "train"
                )
            ),
            norm_A=norm_A,
            norm_B=norm_B,
            num_A=num_A,
            num_B=num_B,
        )

        # Check that the correct values are returned
        assert combined_dict_ref == combined_dict
