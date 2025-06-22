"""
Test suite for gtmo_axioms_v2.py
Tests for enhanced GTMØ axioms with v2 integration
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from enum import Enum

# Import components to test
try:
    from gtmo_axioms_v2 import (
        GTMOAxiom, GTMODefinition, OperatorType, OperationResult,
        EnhancedPsiOperator, EnhancedEntropyOperator, 
        EnhancedMetaFeedbackLoop, UniverseMode, EnhancedGTMOSystem,
        EmergenceDetector, create_enhanced_gtmo_system
    )
except ImportError:
    print("Warning: gtmo_axioms_v2.py not found in path")


class TestGTMOAxiom(unittest.TestCase):
    """Test GTMOAxiom class"""
    
    def test_axiom_compliance_ax0(self):
        """Test AX0 always returns True"""
        result = GTMOAxiom.validate_axiom_compliance(None, "AX0")
        self.assertTrue(result)
    
    def test_axiom_compliance_ax1(self):
        """Test AX1 validates result not in {0, 1, inf}"""
        self.assertFalse(GTMOAxiom.validate_axiom_compliance(0, "AX1"))
        self.assertFalse(GTMOAxiom.validate_axiom_compliance(1, "AX1"))
        self.assertFalse(GTMOAxiom.validate_axiom_compliance(float('inf'), "AX1"))
        self.assertTrue(GTMOAxiom.validate_axiom_compliance(0.5, "AX1"))
    
    def test_axiom_compliance_ax6(self):
        """Test AX6 checks minimal entropy"""
        obj_low_entropy = Mock(entropy=0.0001)
        obj_high_entropy = Mock(entropy=0.5)
        self.assertTrue(GTMOAxiom.validate_axiom_compliance(obj_low_entropy, "AX6"))
        self.assertFalse(GTMOAxiom.validate_axiom_compliance(obj_high_entropy, "AX6"))


class TestOperationResult(unittest.TestCase):
    """Test OperationResult class"""
    
    def test_operation_result_creation(self):
        """Test creating OperationResult"""
        result = OperationResult(
            value=42,
            operator_type=OperatorType.STANDARD,
            axiom_compliance={'AX1': True},
            metadata={'test': 'data'}
        )
        self.assertEqual(result.value, 42)
        self.assertEqual(result.operator_type, OperatorType.STANDARD)
        self.assertEqual(result.axiom_compliance['AX1'], True)


class TestEnhancedPsiOperator(unittest.TestCase):
    """Test EnhancedPsiOperator"""
    
    def setUp(self):
        self.psi_op = EnhancedPsiOperator(classifier=None)
    
    def test_process_general_fragment_fallback(self):
        """Test fallback processing when v2 not available"""
        result = self.psi_op("test fragment")
        self.assertIsInstance(result, OperationResult)
        self.assertEqual(result.operator_type, OperatorType.STANDARD)
        self.assertIn('score', result.value)
        self.assertIn('classification', result.value)
    
    def test_epistemic_purity_calculation(self):
        """Test epistemic purity score calculation"""
        score1 = self.psi_op._calculate_epistemic_purity_fallback("theorem proof axiom")
        score2 = self.psi_op._calculate_epistemic_purity_fallback("maybe perhaps might")
        self.assertGreater(score1, score2)
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)


class TestEnhancedEntropyOperator(unittest.TestCase):
    """Test EnhancedEntropyOperator"""
    
    def setUp(self):
        self.entropy_op = EnhancedEntropyOperator()
    
    def test_semantic_partitions_fallback(self):
        """Test semantic partition calculation"""
        partitions = self.entropy_op._calculate_semantic_partitions_fallback("always theorem")
        self.assertEqual(len(partitions), 3)
        self.assertAlmostEqual(sum(partitions), 1.0, places=5)
        self.assertTrue(all(p > 0 for p in partitions))
    
    def test_process_general_entropy(self):
        """Test general entropy processing"""
        result = self.entropy_op("test fragment")
        self.assertIsInstance(result, OperationResult)
        self.assertIn('total_entropy', result.value)
        self.assertIn('partitions', result.value)


class TestEmergenceDetector(unittest.TestCase):
    """Test EmergenceDetector"""
    
    def setUp(self):
        self.detector = EmergenceDetector()
    
    def test_detect_emergence_basic(self):
        """Test basic emergence detection"""
        psi_result = Mock(value={'score': 0.75})
        entropy_result = Mock(value={'total_entropy': 0.5})
        
        result = self.detector.detect_emergence("emergent novel meta-pattern", 
                                               psi_result, entropy_result)
        self.assertIn('is_emergent', result)
        self.assertIn('emergence_score', result)
        self.assertIn('indicators', result)
    
    def test_novelty_keywords_detection(self):
        """Test detection of novelty keywords"""
        psi_result = Mock(value={'score': 0.8})
        entropy_result = Mock(value={'total_entropy': 0.6})
        
        result = self.detector.detect_emergence("paradox contradiction emergent", 
                                               psi_result, entropy_result)
        self.assertGreater(result['emergence_score'], 0)
        self.assertTrue(any('novelty' in ind for ind in result['indicators']))


class TestEnhancedGTMOSystem(unittest.TestCase):
    """Test EnhancedGTMOSystem"""
    
    @patch('gtmo_axioms_v2.V2_AVAILABLE', False)
    def test_basic_mode_initialization(self):
        """Test system initialization in basic mode"""
        system = EnhancedGTMOSystem(UniverseMode.INDEFINITE_STILLNESS)
        self.assertEqual(system.mode, UniverseMode.INDEFINITE_STILLNESS)
        self.assertIsNone(system.psi_op)
        self.assertIsNone(system.entropy_op)
    
    def test_genesis_stillness_mode(self):
        """Test genesis in stillness mode"""
        system = EnhancedGTMOSystem(UniverseMode.INDEFINITE_STILLNESS, enable_v2_features=False)
        initial_count = len(system.fragments)
        
        # Run many times due to low probability
        with patch('random.random', return_value=0.0):
            system._handle_genesis()
        
        # Should create fragment with very low probability
        self.assertGreater(len(system.fragments), initial_count)
    
    def test_genesis_flux_mode(self):
        """Test genesis in flux mode"""
        system = EnhancedGTMOSystem(UniverseMode.ETERNAL_FLUX, enable_v2_features=False)
        initial_count = len(system.fragments)
        
        with patch('random.random', return_value=0.3):
            system._handle_genesis()
        
        self.assertGreater(len(system.fragments), initial_count)
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation"""
        system = EnhancedGTMOSystem(UniverseMode.ETERNAL_FLUX, enable_v2_features=False)
        report = system.get_comprehensive_report()
        
        self.assertIn('system_time', report)
        self.assertIn('universe_mode', report)
        self.assertIn('fragment_count', report)
        self.assertEqual(report['universe_mode'], 'ETERNAL_FLUX')


class TestMetaFeedbackLoop(unittest.TestCase):
    """Test EnhancedMetaFeedbackLoop"""
    
    def setUp(self):
        self.psi_op = EnhancedPsiOperator()
        self.entropy_op = EnhancedEntropyOperator()
        self.loop = EnhancedMetaFeedbackLoop(self.psi_op, self.entropy_op)
    
    def test_adaptation_weights(self):
        """Test adaptation weight adjustment"""
        current = {'average_score': 0.7, 'average_entropy': 0.5}
        previous = {'average_score': 0.6, 'average_entropy': 0.5}
        
        initial_weights = self.loop.adaptation_weights.copy()
        self.loop._adapt_processing_weights(current, previous)
        
        # Weights should change due to score improvement
        self.assertFalse(np.array_equal(initial_weights, self.loop.adaptation_weights))
        self.assertAlmostEqual(np.sum(self.loop.adaptation_weights), 1.0, places=5)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    @patch('gtmo_axioms_v2.V2_AVAILABLE', False)
    def test_create_system_basic_mode(self):
        """Test system creation in basic mode"""
        system, psi, entropy, loop = create_enhanced_gtmo_system(enable_v2=False)
        self.assertIsInstance(system, EnhancedGTMOSystem)
        self.assertIsNone(psi)
        self.assertIsNone(entropy)
        self.assertIsNone(loop)


if __name__ == '__main__':
    unittest.main()
