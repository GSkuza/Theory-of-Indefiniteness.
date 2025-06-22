"""Unit tests for gtmo_core_v2.py
Testing core GTMØ v2 components including Singularity, AlienatedNumber,
TopologicalClassifier, ExecutableAxioms, AdaptiveGTMONeuron, 
EpistemicParticle, and GTMOSystemV2.
"""

import unittest
import numpy as np
from gtmo_core_v2 import (
    O, Singularity, AlienatedNumber, get_singularity,
    TopologicalClassifier, KnowledgeEntity, KnowledgeType,
    AX0_SystemicUncertainty, AX1_OntologicalDifference, AX6_MinimalEntropy,
    AdaptiveGTMONeuron, EpistemicParticle, EpistemicState, GTMOSystemV2
)


class TestSingularity(unittest.TestCase):
    """Test Singularity (O) behavior and singleton pattern."""
    
    def test_singleton_pattern(self):
        """Test that only one instance of Singularity exists."""
        s1 = get_singularity()
        s2 = get_singularity()
        s3 = Singularity()
        self.assertIs(s1, s2)
        self.assertIs(s2, s3)
        self.assertIs(s1, O)
    
    def test_absorbing_operations(self):
        """Test that arithmetic operations with O return O."""
        self.assertIs(O + 5, O)
        self.assertIs(5 + O, O)
        self.assertIs(O * 10, O)
        self.assertIs(O - 3, O)
        self.assertIs(O / 2, O)
    
    def test_boolean_value(self):
        """Test that O evaluates to False."""
        self.assertFalse(bool(O))
    
    def test_representation(self):
        """Test string representation."""
        self.assertEqual(repr(O), "O_empty_singularity")


class TestAlienatedNumber(unittest.TestCase):
    """Test AlienatedNumber with dynamic context-aware properties."""
    
    def test_context_aware_psi_score(self):
        """Test that PSI score varies based on context."""
        alien1 = AlienatedNumber("test1", context={
            'temporal_distance': 1.0,
            'volatility': 0.1,
            'predictability': 0.9
        })
        alien2 = AlienatedNumber("test2", context={
            'temporal_distance': 10.0,
            'volatility': 0.9,
            'predictability': 0.1
        })
        
        psi1 = alien1.psi_gtm_score()
        psi2 = alien2.psi_gtm_score()
        
        # PSI score should be higher for more predictable, less volatile contexts
        self.assertGreater(psi1, psi2)
        self.assertGreater(psi1, 0.001)
        self.assertLess(psi1, 0.999)
    
    def test_context_aware_entropy(self):
        """Test that entropy calculation uses context factors."""
        alien = AlienatedNumber("future_event", context={
            'temporal_distance': 5.0,
            'volatility': 0.8,
            'predictability': 0.2
        })
        
        entropy = alien.e_gtm_entropy()
        self.assertGreater(entropy, 0.5)  # High uncertainty factors should yield high entropy
        self.assertLessEqual(entropy, 1.0)
    
    def test_semantic_distance_calculation(self):
        """Test semantic distance for different identifiers."""
        alien_future = AlienatedNumber("bitcoin_future_prediction")
        alien_paradox = AlienatedNumber("russell_paradox")
        
        # Force calculation
        psi_future = alien_future.psi_gtm_score()
        psi_paradox = alien_paradox.psi_gtm_score()
        
        # Both should have valid PSI scores
        self.assertIsInstance(psi_future, float)
        self.assertIsInstance(psi_paradox, float)


class TestTopologicalClassifier(unittest.TestCase):
    """Test topological phase space classification."""
    
    def setUp(self):
        self.classifier = TopologicalClassifier()
    
    def test_singularity_classification(self):
        """Test that high determinacy/stability, low entropy → SINGULARITY."""
        entity = KnowledgeEntity(
            content="Absolute truth",
            determinacy=0.95,
            stability=0.95,
            entropy=0.05
        )
        classification = self.classifier.classify(entity)
        self.assertEqual(classification, KnowledgeType.SINGULARITY)
    
    def test_particle_classification(self):
        """Test knowledge particle classification."""
        entity = KnowledgeEntity(
            content="Mathematical theorem",
            determinacy=0.85,
            stability=0.85,
            entropy=0.15
        )
        classification = self.classifier.classify(entity)
        self.assertEqual(classification, KnowledgeType.PARTICLE)
    
    def test_shadow_classification(self):
        """Test knowledge shadow classification."""
        entity = KnowledgeEntity(
            content="Vague notion",
            determinacy=0.15,
            stability=0.15,
            entropy=0.85
        )
        classification = self.classifier.classify(entity)
        self.assertEqual(classification, KnowledgeType.SHADOW)
    
    def test_void_classification(self):
        """Test void fragment classification."""
        entity = KnowledgeEntity(
            content="Empty",
            determinacy=0.0,
            stability=0.0,
            entropy=0.5
        )
        classification = self.classifier.classify(entity)
        self.assertEqual(classification, KnowledgeType.VOID)
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        p1 = (0.1, 0.2, 0.3)
        p2 = (0.4, 0.6, 0.8)
        distance = self.classifier._wasserstein_distance(p1, p2)
        expected = np.sqrt((0.3**2) + (0.4**2) + (0.5**2))
        self.assertAlmostEqual(distance, expected, places=5)
    
    def test_attractor_adaptation(self):
        """Test that attractors can be adapted based on feedback."""
        initial_center = self.classifier.attractors['particle']['center']
        
        # Create feedback data
        entities = [
            KnowledgeEntity("test1", 0.9, 0.9, 0.1),
            KnowledgeEntity("test2", 0.88, 0.92, 0.08),
            KnowledgeEntity("test3", 0.91, 0.89, 0.12),
            KnowledgeEntity("test4", 0.87, 0.93, 0.09)
        ]
        feedback = [(e, KnowledgeType.PARTICLE) for e in entities]
        
        self.classifier.adapt_attractors(feedback)
        new_center = self.classifier.attractors['particle']['center']
        
        # Center should have moved slightly
        self.assertNotEqual(initial_center, new_center)


class TestExecutableAxioms(unittest.TestCase):
    """Test executable axiom implementations."""
    
    def test_ax0_systemic_uncertainty(self):
        """Test AX0 introduces quantum superposition."""
        ax0 = AX0_SystemicUncertainty()
        
        # Create mock system with neurons
        class MockSystem:
            def __init__(self):
                self.neurons = [AdaptiveGTMONeuron("test", (0, 0, 0))]
        
        system = MockSystem()
        ax0.apply(system)
        
        # Should add quantum state to neurons
        self.assertTrue(hasattr(system.neurons[0], 'quantum_state'))
        self.assertIsNotNone(system.neurons[0].quantum_state)
        
        # Verify axiom
        self.assertTrue(ax0.verify(system))
    
    def test_ax6_minimal_entropy(self):
        """Test AX6 entropy minimization for singularity."""
        ax6 = AX6_MinimalEntropy()
        
        # Create neuron approaching singularity
        neuron = AdaptiveGTMONeuron("test", (0, 0, 0))
        neuron.determinacy = 0.95
        neuron.stability = 0.95
        neuron.entropy = 0.1
        
        class MockSystem:
            def __init__(self):
                self.neurons = [neuron]
        
        system = MockSystem()
        initial_entropy = neuron.entropy
        ax6.apply(system)
        
        # Entropy should decrease when approaching singularity
        self.assertLessEqual(neuron.entropy, initial_entropy)


class TestAdaptiveNeuron(unittest.TestCase):
    """Test adaptive learning neuron functionality."""
    
    def test_defense_strategies(self):
        """Test that neuron can execute defense strategies."""
        neuron = AdaptiveGTMONeuron("test_neuron", (0, 0, 0))
        
        attack_result = neuron.experience_attack(
            attack_type='anti_paradox',
            attack_vector={'semantic_attack': 0.5, 'logical_attack': 0.5},
            intensity=0.8
        )
        
        self.assertIn('defense_used', attack_result)
        self.assertIn('success', attack_result)
        self.assertIn(attack_result['defense_used'], 
                     ['absorb', 'deflect', 'rigidify', 'dissolve'])
    
    def test_learning_from_experience(self):
        """Test that neuron updates defense strategies based on success."""
        neuron = AdaptiveGTMONeuron("learner", (0, 0, 0))
        initial_strategies = neuron.defense_strategies.copy()
        
        # Simulate multiple attacks
        for _ in range(3):
            neuron.experience_attack(
                'overflow',
                {'semantic_attack': 2.0, 'logical_attack': 2.0},
                intensity=1.0
            )
        
        # Defense strategies should have changed
        self.assertNotEqual(neuron.defense_strategies, initial_strategies)
        # Total probability should still be 1
        total_prob = sum(neuron.defense_strategies.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_state_vector(self):
        """Test state vector generation."""
        neuron = AdaptiveGTMONeuron("state_test", (1, 2, 3))
        state = neuron.get_state_vector()
        
        self.assertEqual(len(state), 6)
        self.assertTrue(all(0 <= val <= 1 for val in state[:5]))
    
    def test_memory_consolidation(self):
        """Test that experiences are stored in long-term memory."""
        neuron = AdaptiveGTMONeuron("memory_test", (0, 0, 0))
        
        # Successful defense
        neuron.determinacy = 0.8
        result = neuron.experience_attack(
            'anti_paradox',
            {'semantic_attack': 0.3, 'logical_attack': 0.3},
            intensity=0.5
        )
        
        patterns = neuron.get_learned_patterns()
        self.assertGreater(patterns['total_experiences'], 0)


class TestEpistemicParticle(unittest.TestCase):
    """Test epistemic particle functionality."""
    
    def test_initialization(self):
        """Test particle initialization with epistemic state."""
        particle = EpistemicParticle(
            content="Quantum knowledge",
            determinacy=0.6,
            stability=0.5,
            entropy=0.7,
            epistemic_state=EpistemicState.INDEFINITE
        )
        
        self.assertEqual(particle.epistemic_state, EpistemicState.INDEFINITE)
        self.assertEqual(particle.determinacy, 0.6)
        self.assertIsNone(particle.quantum_state)
    
    def test_evolution(self):
        """Test particle evolution over time."""
        particle = EpistemicParticle(
            content="Evolving concept",
            determinacy=0.5,
            stability=0.5,
            entropy=0.5
        )
        
        # Evolve particle
        particle.evolve(0.1)
        
        # Should have trajectory history
        self.assertGreater(len(particle.trajectory_history), 0)
        self.assertIn('determinacy', particle.trajectory_history[0])
    
    def test_epistemic_state_transitions(self):
        """Test transitions between epistemic states."""
        # High determinacy → ONE
        particle1 = EpistemicParticle("certain", 0.95, 0.95, 0.05)
        particle1._update_epistemic_state()
        self.assertEqual(particle1.epistemic_state, EpistemicState.ONE)
        
        # Low everything → ZERO
        particle2 = EpistemicParticle("nothing", 0.05, 0.05, 0.1)
        particle2._update_epistemic_state()
        self.assertEqual(particle2.epistemic_state, EpistemicState.ZERO)
        
        # High entropy → INFINITY
        particle3 = EpistemicParticle("chaos", 0.5, 0.4, 0.9)
        particle3._update_epistemic_state()
        self.assertEqual(particle3.epistemic_state, EpistemicState.INFINITY)
    
    def test_phase_transitions(self):
        """Test detection of phase transitions."""
        particle = EpistemicParticle("test", 0.5, 0.5, 0.5)
        
        # First evolution
        particle.evolve(0.1)
        
        # Force a phase transition
        particle.determinacy = 0.95
        particle.evolve(0.2)
        
        # Check if phase transition was detected
        if 'phase_transition' in particle.metadata:
            self.assertIn('from_state', particle.metadata['phase_transition'])
            self.assertIn('to_state', particle.metadata['phase_transition'])


class TestGTMOSystemV2(unittest.TestCase):
    """Test the main GTMØ v2 system."""
    
    def setUp(self):
        self.system = GTMOSystemV2()
    
    def test_system_initialization(self):
        """Test system initializes with all components."""
        self.assertIsNotNone(self.system.classifier)
        self.assertEqual(len(self.system.axioms), 3)
        self.assertEqual(self.system.iteration, 0)
        self.assertIsInstance(self.system.neurons, list)
        self.assertIsInstance(self.system.epistemic_particles, list)
    
    def test_add_neuron(self):
        """Test adding adaptive neurons to the system."""
        neuron = AdaptiveGTMONeuron("test_neuron", (0, 0, 0))
        self.system.add_neuron(neuron)
        
        self.assertEqual(len(self.system.neurons), 1)
        self.assertIs(self.system.neurons[0], neuron)
    
    def test_add_particle(self):
        """Test adding epistemic particles to the system."""
        particle = EpistemicParticle("test_particle", 0.5, 0.5, 0.5)
        self.system.add_particle(particle)
        
        self.assertEqual(len(self.system.epistemic_particles), 1)
        self.assertIs(self.system.epistemic_particles[0], particle)
    
    def test_evolution(self):
        """Test system evolution step."""
        # Add some particles
        for i in range(3):
            particle = EpistemicParticle(f"particle_{i}", 0.5, 0.5, 0.5)
            self.system.add_particle(particle)
        
        initial_iteration = self.system.iteration
        self.system.evolve()
        
        self.assertEqual(self.system.iteration, initial_iteration + 1)
        self.assertGreater(len(self.system.phase_space_history), 0)
    
    def test_adversarial_attack_simulation(self):
        """Test simulating attacks on neurons."""
        # Add neurons
        for i in range(3):
            neuron = AdaptiveGTMONeuron(f"neuron_{i}", (i, 0, 0))
            self.system.add_neuron(neuron)
        
        results = self.system.simulate_attack('anti_paradox', [0, 1], intensity=0.7)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('neuron_id', result)
            self.assertIn('result', result['result'])
            self.assertIn('defense_used', result['result']['result'])
    
    def test_system_report(self):
        """Test comprehensive system report generation."""
        # Add components
        self.system.add_neuron(AdaptiveGTMONeuron("n1", (0, 0, 0)))
        self.system.add_particle(EpistemicParticle("p1", 0.7, 0.7, 0.3))
        
        # Evolve once
        self.system.evolve()
        
        report = self.system.get_system_report()
        
        self.assertIn('iteration', report)
        self.assertIn('total_neurons', report)
        self.assertIn('total_particles', report)
        self.assertIn('phase_distribution', report)
        self.assertIn('axiom_compliance', report)
        self.assertIn('learning_summary', report)


if __name__ == '__main__':
    unittest.main()
