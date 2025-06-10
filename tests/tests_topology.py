# tests_topology.py

"""tests_topology.py
----------------------------------
Unit tests for the topology.py module of GTMØ (Generalized Theory of Mathematical Indefiniteness).

This module tests:
- Trajectory calculations φ(t) for different entity types and time parameters
- Field evaluations E(x) for various GTMØ-specific fields
- Behavior in both strict and non-strict modes
- Error handling for undefined operations
"""

import unittest
from unittest.mock import patch, MagicMock
from typing import Any

# Import the modules to be tested
from topology import get_trajectory_state_phi_t, evaluate_field_E_x, FieldType
from core import O, AlienatedNumber, Singularity, SingularityError, STRICT_MODE


class TestGetTrajectoryStatePhiT(unittest.TestCase):
    """Test cases for the get_trajectory_state_phi_t function."""

    def test_ontological_singularity_fixed_point(self):
        """Test that Ø is a fixed point: φ(Ø, t) = Ø for all t."""
        # Test various time values
        test_times = [-10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        
        for t in test_times:
            with self.subTest(t=t):
                result = get_trajectory_state_phi_t(O, t)
                self.assertIs(result, O, f"Ø should be fixed point at t={t}")

    def test_alienated_number_evolution_positive_time(self):
        """Test that AlienatedNumber collapses to Ø for t > 0."""
        # Create a mock AlienatedNumber
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        positive_times = [0.1, 1.0, 5.0, 100.0]
        
        for t in positive_times:
            with self.subTest(t=t):
                result = get_trajectory_state_phi_t(mock_alienated, t)
                self.assertIs(result, O, f"AlienatedNumber should collapse to Ø at t={t}")

    def test_alienated_number_evolution_zero_and_negative_time(self):
        """Test that AlienatedNumber remains itself for t <= 0."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        non_positive_times = [-10.0, -1.0, 0.0]
        
        for t in non_positive_times:
            with self.subTest(t=t):
                result = get_trajectory_state_phi_t(mock_alienated, t)
                self.assertIs(result, mock_alienated, 
                            f"AlienatedNumber should remain itself at t={t}")

    @patch('topology.STRICT_MODE', True)
    def test_undefined_entity_strict_mode(self):
        """Test that undefined entities raise SingularityError in strict mode."""
        undefined_entities = [42, "string", [1, 2, 3], {"key": "value"}, None]
        
        for entity in undefined_entities:
            with self.subTest(entity=entity):
                with self.assertRaises(SingularityError) as context:
                    get_trajectory_state_phi_t(entity, 1.0)
                
                self.assertIn("Trajectory φ(t) is undefined", str(context.exception))
                self.assertIn(type(entity).__name__, str(context.exception))

    @patch('topology.STRICT_MODE', False)
    def test_undefined_entity_non_strict_mode(self):
        """Test that undefined entities collapse to Ø in non-strict mode."""
        undefined_entities = [42, "string", [1, 2, 3], {"key": "value"}, None]
        
        for entity in undefined_entities:
            with self.subTest(entity=entity):
                result = get_trajectory_state_phi_t(entity, 1.0)
                self.assertIs(result, O, f"Entity {entity} should collapse to Ø in non-strict mode")

    def test_edge_case_very_small_positive_time(self):
        """Test behavior with very small positive time values."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        small_times = [1e-10, 1e-6, 1e-3]
        
        for t in small_times:
            with self.subTest(t=t):
                result = get_trajectory_state_phi_t(mock_alienated, t)
                self.assertIs(result, O, f"AlienatedNumber should collapse to Ø even at small t={t}")


class TestEvaluateFieldEX(unittest.TestCase):
    """Test cases for the evaluate_field_E_x function."""

    def setUp(self):
        """Set up mock AlienatedNumber for testing."""
        self.mock_alienated = MagicMock(spec=AlienatedNumber)
        self.mock_alienated.e_gtm_entropy.return_value = 0.001
        self.mock_alienated.psi_gtm_score.return_value = 0.999

    def test_ontological_singularity_cognitive_entropy(self):
        """Test cognitive entropy field for Ø."""
        result = evaluate_field_E_x(O, "cognitive_entropy")
        self.assertEqual(result, 0.0, "Ø should have cognitive entropy of 0.0")

    def test_ontological_singularity_epistemic_purity(self):
        """Test epistemic purity field for Ø."""
        result = evaluate_field_E_x(O, "epistemic_purity")
        self.assertEqual(result, 1.0, "Ø should have epistemic purity of 1.0")

    def test_ontological_singularity_proximity_to_singularity(self):
        """Test proximity to singularity field for Ø."""
        result = evaluate_field_E_x(O, "proximity_to_singularity")
        self.assertEqual(result, 0.0, "Ø should have proximity to singularity of 0.0")

    def test_alienated_number_cognitive_entropy(self):
        """Test cognitive entropy field for AlienatedNumber."""
        result = evaluate_field_E_x(self.mock_alienated, "cognitive_entropy")
        self.assertEqual(result, 0.001)
        self.mock_alienated.e_gtm_entropy.assert_called_once()

    def test_alienated_number_epistemic_purity(self):
        """Test epistemic purity field for AlienatedNumber."""
        result = evaluate_field_E_x(self.mock_alienated, "epistemic_purity")
        self.assertEqual(result, 0.999)
        self.mock_alienated.psi_gtm_score.assert_called_once()

    def test_alienated_number_proximity_to_singularity(self):
        """Test proximity to singularity field for AlienatedNumber."""
        result = evaluate_field_E_x(self.mock_alienated, "proximity_to_singularity")
        self.assertEqual(result, 1.0 - 0.999)  # 1.0 - psi_gtm_score()
        self.mock_alienated.psi_gtm_score.assert_called_once()

    def test_default_field_parameter(self):
        """Test that cognitive_entropy is the default field."""
        result = evaluate_field_E_x(O)  # No field_name specified
        self.assertEqual(result, 0.0, "Default field should be cognitive_entropy")

    def test_unsupported_field_for_ontological_singularity(self):
        """Test that unsupported field names raise ValueError for Ø."""
        with self.assertRaises(ValueError) as context:
            evaluate_field_E_x(O, "unsupported_field")
        
        self.assertIn("Unsupported field name for O", str(context.exception))
        self.assertIn("unsupported_field", str(context.exception))

    def test_unsupported_field_for_alienated_number(self):
        """Test that unsupported field names raise ValueError for AlienatedNumber."""
        with self.assertRaises(ValueError) as context:
            evaluate_field_E_x(self.mock_alienated, "unsupported_field")
        
        self.assertIn("Unsupported field name for AlienatedNumber", str(context.exception))
        self.assertIn("unsupported_field", str(context.exception))

    @patch('topology.STRICT_MODE', True)
    def test_undefined_entity_strict_mode_all_fields(self):
        """Test that undefined entities raise SingularityError in strict mode for all fields."""
        undefined_entities = [42, "string", [1, 2, 3], {"key": "value"}, None]
        fields = ["cognitive_entropy", "epistemic_purity", "proximity_to_singularity"]
        
        for entity in undefined_entities:
            for field in fields:
                with self.subTest(entity=entity, field=field):
                    with self.assertRaises(SingularityError) as context:
                        evaluate_field_E_x(entity, field)
                    
                    self.assertIn(f"Field '{field}' is undefined", str(context.exception))
                    self.assertIn(type(entity).__name__, str(context.exception))

    @patch('topology.STRICT_MODE', False)
    def test_undefined_entity_non_strict_mode_all_fields(self):
        """Test that undefined entities return Ø in non-strict mode for all fields."""
        undefined_entities = [42, "string", [1, 2, 3], {"key": "value"}, None]
        fields = ["cognitive_entropy", "epistemic_purity", "proximity_to_singularity"]
        
        for entity in undefined_entities:
            for field in fields:
                with self.subTest(entity=entity, field=field):
                    result = evaluate_field_E_x(entity, field)
                    self.assertIs(result, O, 
                                f"Entity {entity} should return Ø for field {field} in non-strict mode")


class TestFieldTypeHints(unittest.TestCase):
    """Test cases for type hints and field validation."""

    def test_field_type_literal_values(self):
        """Test that all expected field types are supported."""
        expected_fields = ["cognitive_entropy", "epistemic_purity", "proximity_to_singularity"]
        
        for field in expected_fields:
            with self.subTest(field=field):
                # Test with Ø - should not raise any errors
                try:
                    result = evaluate_field_E_x(O, field)
                    self.assertIsInstance(result, (float, int))
                except Exception as e:
                    self.fail(f"Field '{field}' should be supported, but raised: {e}")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test cases combining trajectory and field evaluations."""

    def setUp(self):
        """Set up mock AlienatedNumber for integration testing."""
        self.mock_alienated = MagicMock(spec=AlienatedNumber)
        self.mock_alienated.e_gtm_entropy.return_value = 0.01
        self.mock_alienated.psi_gtm_score.return_value = 0.99

    def test_trajectory_then_field_evaluation(self):
        """Test evaluating fields on trajectory results."""
        # Get trajectory state at t=0 (should remain AlienatedNumber)
        trajectory_result = get_trajectory_state_phi_t(self.mock_alienated, 0.0)
        self.assertIs(trajectory_result, self.mock_alienated)
        
        # Evaluate field on the trajectory result
        field_result = evaluate_field_E_x(trajectory_result, "cognitive_entropy")
        self.assertEqual(field_result, 0.01)
        
        # Get trajectory state at t=1 (should collapse to Ø)
        trajectory_result_t1 = get_trajectory_state_phi_t(self.mock_alienated, 1.0)
        self.assertIs(trajectory_result_t1, O)
        
        # Evaluate field on the collapsed state
        field_result_t1 = evaluate_field_E_x(trajectory_result_t1, "cognitive_entropy")
        self.assertEqual(field_result_t1, 0.0)

    def test_field_evolution_over_trajectory(self):
        """Test how field values change as entity evolves along trajectory."""
        # At t=0: AlienatedNumber with specific entropy
        state_t0 = get_trajectory_state_phi_t(self.mock_alienated, 0.0)
        entropy_t0 = evaluate_field_E_x(state_t0, "cognitive_entropy")
        self.assertEqual(entropy_t0, 0.01)
        
        # At t>0: Collapsed to Ø with zero entropy
        state_t1 = get_trajectory_state_phi_t(self.mock_alienated, 1.0)
        entropy_t1 = evaluate_field_E_x(state_t1, "cognitive_entropy")
        self.assertEqual(entropy_t1, 0.0)
        
        # Verify the evolution: entropy decreases from AlienatedNumber to Ø
        self.assertGreater(entropy_t0, entropy_t1)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""

    def test_none_entity_handling(self):
        """Test handling of None as entity."""
        with patch('topology.STRICT_MODE', True):
            with self.assertRaises(SingularityError):
                get_trajectory_state_phi_t(None, 1.0)
            
            with self.assertRaises(SingularityError):
                evaluate_field_E_x(None, "cognitive_entropy")

    def test_extreme_time_values(self):
        """Test handling of extreme time values."""
        mock_alienated = MagicMock(spec=AlienatedNumber)
        
        extreme_times = [float('inf'), float('-inf'), 1e308, -1e308]
        
        for t in extreme_times:
            with self.subTest(t=t):
                try:
                    result = get_trajectory_state_phi_t(mock_alienated, t)
                    # Should either return Ø (for positive) or mock_alienated (for negative/zero)
                    self.assertIn(result, [O, mock_alienated])
                except Exception as e:
                    self.fail(f"Extreme time value {t} should be handled gracefully, but raised: {e}")

    def test_mock_alienated_number_method_errors(self):
        """Test handling when AlienatedNumber methods raise exceptions."""
        failing_alienated = MagicMock(spec=AlienatedNumber)
        failing_alienated.e_gtm_entropy.side_effect = RuntimeError("Mock error")
        failing_alienated.psi_gtm_score.side_effect = RuntimeError("Mock error")
        
        # These should propagate the underlying errors
        with self.assertRaises(RuntimeError):
            evaluate_field_E_x(failing_alienated, "cognitive_entropy")
        
        with self.assertRaises(RuntimeError):
            evaluate_field_E_x(failing_alienated, "epistemic_purity")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
