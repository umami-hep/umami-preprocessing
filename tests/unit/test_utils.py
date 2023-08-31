import unittest
from pathlib import Path

from upp.utils import path_append


class TestPathAppend(unittest.TestCase):

    def test_path_append(self):
        input_path = Path("file.txt")
        suffix = "new"
        expected_output = Path("file_new.txt")
        
        result = path_append(input_path, suffix)
        
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()