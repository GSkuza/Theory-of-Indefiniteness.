# tests/tests_gtmo_axioms.py

"""
Comprehensive unit tests for the GTMØ axioms module.

Tests the foundational axiomatic framework for the Generalized Theory
of Mathematical Indefiniteness (GTMØ), including:
- The new Axiom 0 (Systemic Uncertainty) and its implementation via Universe Modes.
- The GTMOSystem simulation controller.
- Core operators (Psi, Entropy), dynamic thresholds, and meta-feedback loops.

Author: Tests for Theory of Indefiniteness by Grzegorz Skuza (Poland)
"""

import unittest
import numpy as np
import math
import random
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List
import logging

# Suppress logging during tests to keep output clean
logging.disable(logging.CRITICAL)

# Import all components from the module to be tested
from gtmo_axioms import (
    GTMOAxiom,
    GTMODefinition,
    OperatorType,
    OperationResult,
    ThresholdManager,
    PsiOperator,
    EntropyOperator,
    MetaFeedbackLoop,
    EmergenceDetector,
    AxiomValidator,
    create_gtmo_system,
    validate_gtmo_system_axioms,
    UniverseMode,
    GTMOSystem,
    O,
    AlienatedNumber
)


###############################################################################
# NEW: Tests for Axiom 0 and GTMOSystem
###############################################################################

class TestAxiomZeroAndSystem(unittest.TestCase):
    """
    Tests for the new Axiom 0, Universe Modes, and the GTMOSystem controller.
    """

    def test_axiom_zero_definition(self):
        """Test that Axiom 0 is correctly defined."""
        self.assertIn("Systemic Uncertainty", GTMOAxiom.AX0)
        self.assertIn(GTMOAxiom.AX0, GTMOAxiom.ALL_AXIOMS)
        self.assertEqual(len(GTMOAxiom.ALL_AXIOMS), 11)

    def test_universe_mode_enum(self):
        """Test the UniverseMode enumeration."""
        self.assertIn(UniverseMode.INDEFINITE_STILLNESS, UniverseMode)
        self.assertIn(UniverseMode.ETERNAL_FLUX, UniverseMode)
        self.assertNotEqual(UniverseMode.INDEFINITE_STILLNESS, UniverseMode.ETERNAL_FLUX)

    def test_gtmo_system_initialization(self):
        """Test the initialization of the GTMOSystem."""
        # Test Stillness mode initialization
        still_system = GTMOSystem(mode=UniverseMode.INDEFINITE_STILLNESS, initial_fragments=["A"])
        self.assertEqual(still_system.mode, UniverseMode.INDEFINITE_STILLNESS)
        self.assertEqual(len(still_system.fragments), 1)
        self.assertEqual(still_system.system_time, 0.0)
        self.assertIsInstance(still_system.psi_op, PsiOperator)

        # Test Flux mode initialization
        flux_system = GTMOSystem(mode=UniverseMode.ETERNAL_FLUX)
        self.assertEqual(flux_system.mode, UniverseMode.ETERNAL_FLUX)
        self.assertEqual(len(flux_system.fragments), 0)

    def test_step_advances_time_and_runs_loop(self):
        """Test that a system step advances time and attempts to run the feedback loop."""
        system = GTMOSystem(mode=UniverseMode.ETERNAL_FLUX, initial_fragments=["Test"])
        
        # Mock the feedback loop to see if it's called
        with patch.object(system.meta_loop, 'run', return_value={'final_state': {'final_classification_ratios': {}}}) as mock_run:
            system.step()
            self.assertEqual(system.system_time, 1.0)
            mock_run.assert_called_once()
            
            system.step()
            self.assertEqual(system.system_time, 2.0)
            self.assertEqual(mock_run.call_count, 2)

    @patch('gtmo_axioms.random.random')
    def test_stillness_mode_genesis(self, mock_random):
        """Test fragment genesis in INDEFINITE_STILLNESS mode."""
        system = GTMOSystem(mode=UniverseMode.INDEFINITE_STILLNESS, initial_fragments=[])
        
        # Scenario 1: No genesis (random > 1e-6)
        mock_random.return_value = 0.5
        system.step()
        self.assertEqual(len(system.fragments), 0, "Genesis should not occur when random value is high.")
        
        # Scenario 2: Genesis occurs (random < 1e-6)
        mock_random.return_value = 1e-7
        system.step()
        self.assertEqual(len(system.fragments), 1, "A rare genesis event should have occurred.")
        self.assertIn("Spontaneous genesis", system.fragments[0])

    @patch('gtmo_axioms.random.random')
    def test_flux_mode_genesis(self, mock_random):
        """Test fragment genesis in ETERNAL_FLUX mode."""
        system = GTMOSystem(mode=UniverseMode.ETERNAL_FLUX, initial_fragments=[])
        genesis_rate = 0.4 # from the implementation

        # Scenario 1: Genesis occurs (random < genesis_rate)
        mock_random.return_value = genesis_rate - 0.1
        system.step()
        self.assertEqual(len(system.fragments), 1, "Genesis should occur when random value is low.")
        self.assertIn("Chaotic flux particle", system.fragments[0])
        
        # Scenario 2: No genesis (random > genesis_rate)
        mock_random.return_value = genesis_rate + 0.1
        system.step()
        self.assertEqual(len(system.fragments), 1, "Genesis should not occur when random value is high.")

    def test_get_system_state(self):
        """Test the system state reporting method."""
        fragments = ["A", "B"]
        system = GTMOSystem(mode=UniverseMode.ETERNAL_FLUX, initial_fragments=fragments)
        system.system_time = 5.0

        state = system.get_system_state()
        
        self.assertEqual(state['mode'], 'ETERNAL_FLUX')
        self.assertEqual(state['system_time'], 5.0)
        self.assertEqual(state['fragment_count'], 2)
        self.assertEqual(state['fragments'], ["A", "B"])


###############################################################################
# Existing Tests for Operators and Other Components (Still part of the file)
###############################################################################

class TestCoreOperators(unittest.TestCase):
    """Test Ψ_GTMØ and E_GTMØ operators."""
    
    def setUp(self):
        """Set up test environment."""
        self.threshold_manager = ThresholdManager()
        self.psi_operator = PsiOperator(self.threshold_manager)
        self.entropy_operator = EntropyOperator()

    def test_psi_operator_on_singularity(self):
        """Test Ψ_GTMØ processing of ontological singularity (Ø)."""
        result = self.psi_operator(O)
        self.assertEqual(result.operator_type, OperatorType.META)
        self.assertEqual(result.value['score'], 1.0)
        self.assertEqual(result.value['classification'], 'Ø')

    def test_psi_operator_on_alienated_number(self):
        """Test Ψ_GTMØ processing of alienated numbers (ℓ∅)."""
        alien = AlienatedNumber("test")
        result = self.psi_operator(alien)
        self.assertEqual(result.operator_type, OperatorType.META)
        self.assertAlmostEqual(result.value['score'], alien.psi_gtm_score())
        self.assertEqual(result.value['classification'], 'ℓ∅')

    def test_psi_operator_on_general_fragment(self):
        """Test Ψ_GTMØ processing of a general knowledge fragment."""
        fragment = "A mathematical theorem about prime numbers."
        result = self.psi_operator(fragment, {'all_scores': np.random.rand(20).tolist()})
        self.assertEqual(result.operator_type, OperatorType.STANDARD)
        self.assertIn(result.value['classification'], ['Ψᴷ', 'Ψʰ', 'Ψᴧ'])

    def test_entropy_operator_on_singularity(self):
        """Test E_GTMØ processing for Ø (AX6 compliance)."""
        result = self.entropy_operator(O)
        self.assertEqual(result.operator_type, OperatorType.META)
        self.assertEqual(result.value['total_entropy'], 0.0)
        self.assertTrue(result.axiom_compliance['AX6'])

    def test_entropy_operator_on_alienated_number(self):
        """Test E_GTMØ processing for alienated numbers."""
        alien = AlienatedNumber("test")
        result = self.entropy_operator(alien)
        self.assertEqual(result.operator_type, OperatorType.META)
        self.assertAlmostEqual(result.value['total_entropy'], alien.e_gtm_entropy())

    def test_entropy_calculation_for_general_fragment(self):
        """Test E_GTMØ entropy calculation for a general fragment."""
        fragment = "This statement is somewhat certain but also a bit vague."
        result = self.entropy_operator(fragment)
        self.assertEqual(result.operator_type, OperatorType.STANDARD)
        self.assertGreater(result.value['total_entropy'], 0)
        self.assertLess(result.value['total_entropy'], math.log2(3)) # Max entropy for 3 partitions


class TestAdvancedSystems(unittest.TestCase):
    """Test meta-feedback loop, emergence detection, and validation."""

    def setUp(self):
        """Set up a full GTMØ system for testing."""
        self.psi_op, self.entropy_op, self.meta_loop = create_gtmo_system()
        self.emergence_detector = self.meta_loop.emergence_detector
        self.validator = AxiomValidator()

    def test_meta_feedback_loop_run(self):
        """Test that the meta-feedback loop runs and produces a valid report."""
        fragments = ["Stable fact", "Uncertain idea", "Paradoxical statement"]
        initial_scores = [0.9, 0.2, 0.5]
        result = self.meta_loop.run(fragments, initial_scores, iterations=2)
        
        self.assertIn('history', result)
        self.assertIn('final_state', result)
        self.assertEqual(len(result['history']), 2)
        self.assertIn('system_stability', result['final_state'])

    def test_emergence_detection(self):
        """Test the EmergenceDetector logic."""
        # A fragment likely to be emergent
        fragment = "A novel, emergent, meta-cognitive pattern"
        psi_result = OperationResult(value={'score': 0.7}, operator_type=OperatorType.STANDARD)
        entropy_result = OperationResult(value={'total_entropy': 0.6}, operator_type=OperatorType.STANDARD)
        
        detection = self.emergence_detector.detect_emergence(fragment, psi_result, entropy_result)
        self.assertTrue(detection['is_emergent'])
        self.assertIsNotNone(detection['emergent_type'])
        
        # A non-emergent fragment
        fragment_stable = "This is a simple fact."
        psi_result_stable = OperationResult(value={'score': 0.9}, operator_type=OperatorType.STANDARD)
        entropy_result_stable = OperationResult(value={'total_entropy': 0.1}, operator_type=OperatorType.STANDARD)
        
        detection_stable = self.emergence_detector.detect_emergence(fragment_stable, psi_result_stable, entropy_result_stable)
        self.assertFalse(detection_stable['is_emergent'])

    def test_axiom_validator(self):
        """Test the AxiomValidator functionality."""
        # Compliant operation
        compliant_result = OperationResult(value={'score': 1.0}, operator_type=OperatorType.META)
        compliance = self.validator.validate_operation('Ψ_GTMØ', [O], compliant_result, ['AX9', 'AX10'])
        self.assertTrue(compliance['AX9'])
        self.assertTrue(compliance['AX10'])
        
        # Non-compliant operation (hypothetical)
        non_compliant_result = OperationResult(value=0.5, operator_type=OperatorType.STANDARD)
        compliance_fail = self.validator.validate_operation('Ψ_GTMØ', [O], non_compliant_result, ['AX9'])
        self.assertFalse(compliance_fail['AX9'])
        
        report = self.validator.get_compliance_report()
        self.assertEqual(report['total_validations'], 2)
        self.assertLess(report['overall_compliance'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
