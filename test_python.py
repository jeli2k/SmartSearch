# test_python.py

import unittest

class TestPythonFunctionality(unittest.TestCase):
    
    def test_python_works(self):
        # Descriptive message about what is being tested
        print("\nTesting if Python is able to perform basic addition correctly...")
        
        result = 1 + 1
        expected = 2
        self.assertEqual(result, expected, f"Expected {result} to equal {expected}. Test failed.")
        
        # Confirm success
        print("Python basic addition test passed. Python is working")

if __name__ == "__main__":
    unittest.main()
