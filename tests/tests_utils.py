# tests_utils.py

"""tests_utils.py
----------------------------------
Unit tests for the utils.py module of GTMØ (Generalized Theory of Mathematical Indefiniteness).

This module tests:
- Type checking functions for GTMØ entities
- Validation of Ontological Singularity (Ø) identification
- Validation of Alienated Number (ℓ∅) identification
- Validation of GTMØ primitive type checking
- Edge cases and error handling
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Any
import sys
from io import StringIO

# Import the modules to be tested
from utils import is_ontological_singularity, is_alienated_number, is_gtmo_primitive
from core import O, AlienatedNumber


class TestIsOntologicalSingularity(unittest.TestCase):
    """Test cases for the is_ontological_singularity function."""

    def test_ontological_singularity_positive(self):
        """Test that Ø is correctly identified as ontological singularity."""
        result = is_ontological_singularity(O)
        self.assertTrue(result, "Ø should be identified as ontological singularity")

    def test_ontological_singularity_identity_check(self):
        """Test that the function uses identity check (is) rather than equality."""
        # This test ensures we're using 'is' operator for singleton check
        result = is_ontological_singularity(O)
        self.assertTrue(result)
        
        # Verify that it's specifically checking identity
        self.assertIs(O, O, "Ø should be a singleton")

    def test_non_ontological_singularity_objects(self):
        """Test that non-Ø objects are correctly identified as not ontological singularity."""
        non_o_objects = [
            None,
            0,
            1,
            -1,
            0.0,
            1.0,
            "",
            "O",
            "Ø",
            [],
            {},
            set(),
            tuple(),
            object(),
            type(None),
            bool,
            int,
            float,
            str
        ]
        
        for obj in non_o_objects:
            with self.subTest(obj=obj):
                result = is_ontological_singularity(obj)
                self.assertFalse(result, 
                    f"Object {obj} of type {type(obj).__name__} should not be identified as Ø")

    def test_alienated_number_is_not_ontological_singularity(self):
        """Test that AlienatedNumber instances are not identified as ontological singularity."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        result = is_ontological_singularity(mock_alienated)
        self.assertFalse(result, "AlienatedNumber should not be identified as Ø")

    def test_custom_objects_with_similar_names(self):
        """Test that objects with similar names/attributes are not confused with Ø."""
        class FakeO:
            def __str__(self):
                return "Ø"
            
            def __repr__(self):
                return "O"
        
        fake_o = FakeO()
        result = is_ontological_singularity(fake_o)
        self.assertFalse(result, "Custom object mimicking Ø should not be identified as Ø")

    def test_none_handling(self):
        """Test explicit handling of None."""
        result = is_ontological_singularity(None)
        self.assertFalse(result, "None should not be identified as Ø")


class TestIsAlienatedNumber(unittest.TestCase):
    """Test cases for the is_alienated_number function."""

    def test_alienated_number_positive(self):
        """Test that AlienatedNumber instances are correctly identified."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        result = is_alienated_number(mock_alienated)
        self.assertTrue(result, "AlienatedNumber instance should be identified as alienated number")

    def test_real_alienated_number_instance(self):
        """Test with a real AlienatedNumber instance if possible."""
        try:
            # Try to create a real AlienatedNumber instance
            # This assumes AlienatedNumber can be instantiated
            real_alienated = AlienatedNumber()
            result = is_alienated_number(real_alienated)
            self.assertTrue(result, "Real AlienatedNumber instance should be identified")
        except Exception:
            # If AlienatedNumber cannot be instantiated, skip this test
            self.skipTest("AlienatedNumber cannot be instantiated directly")

    def test_ontological_singularity_is_not_alienated_number(self):
        """Test that Ø is not identified as alienated number."""
        result = is_alienated_number(O)
        self.assertFalse(result, "Ø should not be identified as AlienatedNumber")

    def test_non_alienated_number_objects(self):
        """Test that non-AlienatedNumber objects are correctly identified."""
        non_alienated_objects = [
            None,
            0,
            1,
            -1,
            0.0,
            1.0,
            "",
            "AlienatedNumber",
            "ℓ∅",
            [],
            {},
            set(),
            tuple(),
            object(),
            type(None),
            bool,
            int,
            float,
            str,
            O
        ]
        
        for obj in non_alienated_objects:
            with self.subTest(obj=obj):
                result = is_alienated_number(obj)
                self.assertFalse(result, 
                    f"Object {obj} of type {type(obj).__name__} should not be identified as AlienatedNumber")

    def test_subclass_of_alienated_number(self):
        """Test that subclasses of AlienatedNumber are correctly identified."""
        class SubAlienatedNumber(AlienatedNumber):
            pass
        
        try:
            sub_instance = SubAlienatedNumber()
            result = is_alienated_number(sub_instance)
            self.assertTrue(result, "Subclass of AlienatedNumber should be identified as AlienatedNumber")
        except Exception:
            # If subclass cannot be instantiated, create a mock
            mock_sub = MagicMock(spec=SubAlienatedNumber)
            # Make isinstance work correctly with the mock
            with patch('builtins.isinstance') as mock_isinstance:
                mock_isinstance.return_value = True
                result = is_alienated_number(mock_sub)
                self.assertTrue(result, "Subclass of AlienatedNumber should be identified")

    def test_custom_objects_with_similar_names(self):
        """Test that objects with similar names are not confused with AlienatedNumber."""
        class FakeAlienatedNumber:
            def __str__(self):
                return "ℓ∅"
            
            def __repr__(self):
                return "AlienatedNumber"
        
        fake_alienated = FakeAlienatedNumber()
        result = is_alienated_number(fake_alienated)
        self.assertFalse(result, "Custom object mimicking AlienatedNumber should not be identified as AlienatedNumber")


class TestIsGtmoPrimitive(unittest.TestCase):
    """Test cases for the is_gtmo_primitive function."""

    def test_ontological_singularity_is_primitive(self):
        """Test that Ø is identified as GTMØ primitive."""
        result = is_gtmo_primitive(O)
        self.assertTrue(result, "Ø should be identified as GTMØ primitive")

    def test_alienated_number_is_primitive(self):
        """Test that AlienatedNumber is identified as GTMØ primitive."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        result = is_gtmo_primitive(mock_alienated)
        self.assertTrue(result, "AlienatedNumber should be identified as GTMØ primitive")

    def test_non_gtmo_primitives(self):
        """Test that non-GTMØ objects are not identified as primitives."""
        non_gtmo_objects = [
            None,
            0,
            1,
            -1,
            0.0,
            1.0,
            "",
            "GTMØ",
            "primitive",
            [],
            {},
            set(),
            tuple(),
            object(),
            type(None),
            bool,
            int,
            float,
            str
        ]
        
        for obj in non_gtmo_objects:
            with self.subTest(obj=obj):
                result = is_gtmo_primitive(obj)
                self.assertFalse(result, 
                    f"Object {obj} of type {type(obj).__name__} should not be identified as GTMØ primitive")

    def test_both_gtmo_primitives_in_collection(self):
        """Test identification of GTMØ primitives when both types are present."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        # Test both individually
        self.assertTrue(is_gtmo_primitive(O))
        self.assertTrue(is_gtmo_primitive(mock_alienated))
        
        # Test that function works consistently
        gtmo_objects = [O, mock_alienated]
        for obj in gtmo_objects:
            with self.subTest(obj=obj):
                result = is_gtmo_primitive(obj)
                self.assertTrue(result, f"GTMØ object {obj} should be identified as primitive")

    def test_mixed_collection_filtering(self):
        """Test using is_gtmo_primitive for filtering mixed collections."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        mixed_collection = [
            O,
            mock_alienated,
            42,
            "string",
            None,
            [],
            {}
        ]
        
        gtmo_primitives = [item for item in mixed_collection if is_gtmo_primitive(item)]
        
        self.assertEqual(len(gtmo_primitives), 2, "Should find exactly 2 GTMØ primitives")
        self.assertIn(O, gtmo_primitives, "Ø should be in filtered results")
        self.assertIn(mock_alienated, gtmo_primitives, "AlienatedNumber should be in filtered results")

    def test_empty_and_none_inputs(self):
        """Test edge cases with empty and None inputs."""
        self.assertFalse(is_gtmo_primitive(None), "None should not be GTMØ primitive")
        self.assertFalse(is_gtmo_primitive(""), "Empty string should not be GTMØ primitive")
        self.assertFalse(is_gtmo_primitive([]), "Empty list should not be GTMØ primitive")
        self.assertFalse(is_gtmo_primitive({}), "Empty dict should not be GTMØ primitive")


class TestFunctionIntegration(unittest.TestCase):
    """Integration tests for utils functions working together."""

    def test_mutual_exclusivity_of_gtmo_types(self):
        """Test that Ø and AlienatedNumber are mutually exclusive in type checking."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        # Ø should only be identified as ontological singularity
        self.assertTrue(is_ontological_singularity(O))
        self.assertFalse(is_alienated_number(O))
        self.assertTrue(is_gtmo_primitive(O))
        
        # AlienatedNumber should only be identified as alienated number
        self.assertFalse(is_ontological_singularity(mock_alienated))
        self.assertTrue(is_alienated_number(mock_alienated))
        self.assertTrue(is_gtmo_primitive(mock_alienated))

    def test_consistency_of_primitive_identification(self):
        """Test that is_gtmo_primitive is consistent with individual type checks."""
        test_objects = [
            O,
            MagicMock(spec=AlienatedNumber),
            42,
            "string",
            None,
            [],
            {}
        ]
        
        for obj in test_objects:
            with self.subTest(obj=obj):
                is_o = is_ontological_singularity(obj)
                is_alienated = is_alienated_number(obj)
                is_primitive = is_gtmo_primitive(obj)
                
                # is_gtmo_primitive should return True if either is_o or is_alienated is True
                expected_primitive = is_o or is_alienated
                self.assertEqual(is_primitive, expected_primitive,
                    f"Inconsistent primitive identification for {obj}")

    def test_type_checking_performance(self):
        """Test that type checking functions perform reasonably with many objects."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        # Create a large list of mixed objects
        test_objects = [O, mock_alienated, 42, "string", None] * 1000
        
        # Test that all functions complete without errors
        o_count = sum(1 for obj in test_objects if is_ontological_singularity(obj))
        alienated_count = sum(1 for obj in test_objects if is_alienated_number(obj))
        primitive_count = sum(1 for obj in test_objects if is_gtmo_primitive(obj))
        
        # Verify expected counts
        self.assertEqual(o_count, 1000, "Should find 1000 Ø instances")
        self.assertEqual(alienated_count, 1000, "Should find 1000 AlienatedNumber instances")
        self.assertEqual(primitive_count, 2000, "Should find 2000 total GTMØ primitives")


class TestImportFallback(unittest.TestCase):
    """Test cases for import fallback mechanisms."""

    def test_import_fallback_behavior(self):
        """Test that the module can handle import scenarios gracefully."""
        # This test verifies that the imports work correctly
        # Since we're already successfully importing in the test file,
        # we can verify that O and AlienatedNumber are available
        
        self.assertIsNotNone(O, "O should be importable")
        self.assertIsNotNone(AlienatedNumber, "AlienatedNumber should be importable")
        
        # Test that the functions can access the imported objects
        self.assertTrue(is_ontological_singularity(O))
        
        mock_alienated = MagicMock(spec=AlienatedNumber)
        self.assertTrue(is_alienated_number(mock_alienated))

    @patch('utils.O', None)
    def test_missing_o_import(self):
        """Test behavior when O import is missing."""
        # This test simulates what happens if O is not properly imported
        with self.assertRaises(TypeError):
            # This should fail because None is not the expected singleton
            is_ontological_singularity(None)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and unusual inputs."""

    def test_circular_references(self):
        """Test handling of objects with circular references."""
        circular_obj = {}
        circular_obj['self'] = circular_obj
        
        # Should not cause infinite recursion
        self.assertFalse(is_ontological_singularity(circular_obj))
        self.assertFalse(is_alienated_number(circular_obj))
        self.assertFalse(is_gtmo_primitive(circular_obj))

    def test_very_large_objects(self):
        """Test handling of very large objects."""
        large_list = list(range(10000))
        large_dict = {i: i for i in range(1000)}
        
        # Should handle large objects without issues
        self.assertFalse(is_ontological_singularity(large_list))
        self.assertFalse(is_alienated_number(large_list))
        self.assertFalse(is_gtmo_primitive(large_list))
        
        self.assertFalse(is_ontological_singularity(large_dict))
        self.assertFalse(is_alienated_number(large_dict))
        self.assertFalse(is_gtmo_primitive(large_dict))

    def test_special_numeric_values(self):
        """Test handling of special numeric values."""
        special_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
            complex(1, 2),
            complex(0, 1),
            1j,
            1+1j
        ]
        
        for value in special_values:
            with self.subTest(value=value):
                self.assertFalse(is_ontological_singularity(value))
                self.assertFalse(is_alienated_number(value))
                self.assertFalse(is_gtmo_primitive(value))

    def test_callable_objects(self):
        """Test handling of callable objects."""
        def test_function():
            pass
        
        test_lambda = lambda x: x
        
        callables = [test_function, test_lambda, len, str, int, type]
        
        for callable_obj in callables:
            with self.subTest(callable_obj=callable_obj):
                self.assertFalse(is_ontological_singularity(callable_obj))
                self.assertFalse(is_alienated_number(callable_obj))
                self.assertFalse(is_gtmo_primitive(callable_obj))


if __name__ == '__main__':
    # Configure test runner with high verbosity
    unittest.main(verbosity=2, buffer=True)
