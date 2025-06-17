import unittest

# Importuj wymagane klasy/funkcje z repozytorium.
from gtmo.gtmo_axioms import (
    create_gtmo_system,
    validate_gtmo_system_axioms,
    PsiOperator,
    EntropyOperator,
    MetaFeedbackLoop,
    ThresholdManager,
)
# Załóżmy, że O i AlienatedNumber są dostępne:
from gtmo.gtmo_axioms import O, AlienatedNumber

class TestGTMOSystemFactory(unittest.TestCase):
    def test_default_parameters(self):
        psi, entropy, meta = create_gtmo_system()
        self.assertIsInstance(psi, PsiOperator)
        self.assertIsInstance(entropy, EntropyOperator)
        self.assertIsInstance(meta, MetaFeedbackLoop)

    def test_custom_parameters(self):
        psi, entropy, meta = create_gtmo_system(99.0, 1.0, 0.001)
        self.assertIsInstance(psi, PsiOperator)
        self.assertIsInstance(entropy, EntropyOperator)
        self.assertIsInstance(meta, MetaFeedbackLoop)

    def test_edge_adaptation_rate(self):
        psi, entropy, meta = create_gtmo_system(85.0, 15.0, 0.0)
        self.assertIsInstance(meta, MetaFeedbackLoop)
        psi, entropy, meta = create_gtmo_system(85.0, 15.0, 1.0)
        self.assertIsInstance(meta, MetaFeedbackLoop)

    def test_invalid_parameters(self):
        with self.assertRaises(Exception):
            create_gtmo_system(-1, 200, -0.5)


class TestGTMOOperators(unittest.TestCase):
    def setUp(self):
        self.psi, self.entropy, self.meta = create_gtmo_system()

    def test_psi_operator_with_O(self):
        result = self.psi(O)
        self.assertTrue(hasattr(result, 'value'))
        self.assertEqual(result.value.get('score', -1), 1.0)

    def test_entropy_operator_with_alienated(self):
        alien = AlienatedNumber("alien")
        result = self.entropy(alien)
        self.assertIn('total_entropy', result.value)
        self.assertAlmostEqual(result.value['total_entropy'], 1e-9, places=10)

    def test_psi_operator_with_arbitrary(self):
        result = self.psi("arbitrary_input")
        self.assertIsInstance(result.value.get('score'), float)

    def test_entropy_operator_with_arbitrary(self):
        result = self.entropy("arbitrary_input")
        self.assertIsInstance(result.value.get('total_entropy'), float)


class TestMetaFeedbackLoop(unittest.TestCase):
    def setUp(self):
        self.psi, self.entropy, self.meta = create_gtmo_system()

    def test_meta_feedback_loop_runs(self):
        fragments = ["fragment1", "fragment2"]
        scores = [0.6, 0.8]
        output = self.meta.run(fragments, scores, iterations=2)
        self.assertIn('final_state', output)
        self.assertIn('system_stability', output['final_state'])

    def test_meta_feedback_extreme_iterations(self):
        fragments = ["f"]
        scores = [0.99]
        output = self.meta.run(fragments, scores, iterations=100)
        self.assertIn('final_state', output)


class TestAxiomValidation(unittest.TestCase):
    def setUp(self):
        self.psi, self.entropy, _ = create_gtmo_system()

    def test_validate_gtmo_system_axioms(self):
        report = validate_gtmo_system_axioms(self.psi, self.entropy)
        self.assertIn('overall_report', report)
        self.assertIn('axiom_compliance', report['overall_report'])
        self.assertGreaterEqual(report['overall_report']['overall_compliance'], 0.0)
        self.assertLessEqual(report['overall_report']['overall_compliance'], 1.0)


class TestEdgeCases(unittest.TestCase):
    def test_create_gtmo_system_extreme_parameters(self):
        psi, entropy, meta = create_gtmo_system(100.0, 0.0, 0.99)
        self.assertIsInstance(meta, MetaFeedbackLoop)
        psi, entropy, meta = create_gtmo_system(0.0, 100.0, 0.01)
        self.assertIsInstance(meta, MetaFeedbackLoop)

    def test_operator_with_none(self):
        psi, entropy, _ = create_gtmo_system()
        result_psi = psi(None)
        self.assertIsNotNone(result_psi)
        result_entropy = entropy(None)
        self.assertIsNotNone(result_entropy)

    def test_operator_with_number(self):
        psi, entropy, _ = create_gtmo_system()
        result_psi = psi(42)
        self.assertIsInstance(result_psi.value.get('score'), float)
        result_entropy = entropy(42)
        self.assertIsInstance(result_entropy.value.get('total_entropy'), float)


if __name__ == "__main__":
    unittest.main()
