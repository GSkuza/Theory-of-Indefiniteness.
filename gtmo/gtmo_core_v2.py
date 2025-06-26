"""
GTMO Core v2 - FIXED VERSION with Enhanced Documentation
=======================================================

Generalized Theory of Mathematical Indefiniteness (GTMO) Implementation
----------------------------------------------------------------------

This module implements a deterministic version of the GTMO system, which models
epistemological uncertainty through quantum-inspired neural networks. All random
functions have been replaced with deterministic calculations based on the
system's epistemological state.

Key Concepts:
    - Epistemological Coordinates: determinacy, stability, entropy
    - Quantum Superposition: representing undefined knowledge states
    - Adaptive Defense: learning-based response to adversarial attacks
    - Foundational Modes: system-wide phase states (stillness/flux)

Architecture:
    - QuantumStateCalculator: Manages quantum state calculations
    - DeterministicDefenseSelector: Selects defense strategies based on experience
    - AdaptiveGTMONeuron: Core processing unit with quantum properties
    - Executable Axioms: Active rules that shape system behavior

Author: GTMO Research Team
Version: 2.0 (Fixed/Deterministic)
Date: 2024
"""

import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging for system monitoring
logger = logging.getLogger(__name__)


# === QUANTUM STATE CALCULATIONS ===

class QuantumStateCalculator:
    """
    Handles deterministic quantum state calculations for GTMO neurons.
    
    This class replaces random quantum state generation with deterministic
    calculations based on the neuron's epistemological properties. The quantum
    states represent superpositions of knowledge states: defined, undefined,
    and indefinite.
    
    Mathematical Foundation:
        The quantum state |ψ⟩ = α|defined⟩ + β|undefined⟩ + γ|indefinite⟩
        where α, β, γ are complex amplitudes derived from neuron properties.
    """
    
    @staticmethod
    def create_superposition(neuron) -> Dict[str, complex]:
        """
        Create a quantum superposition state based on neuron properties.
        
        This method generates a normalized quantum state where amplitudes
        and phases are deterministically derived from the neuron's
        epistemological coordinates (determinacy, stability, entropy).
        
        Args:
            neuron: An AdaptiveGTMONeuron instance with properties:
                - determinacy (float): How well-defined the knowledge is [0,1]
                - stability (float): How stable the knowledge state is [0,1]
                - entropy (float): Amount of uncertainty/disorder [0,1]
        
        Returns:
            Dict[str, complex]: Quantum state dictionary with keys:
                - 'defined': Complex amplitude for defined state
                - 'undefined': Complex amplitude for undefined state
                - 'indefinite': Complex amplitude for indefinite state
        
        Example:
            >>> neuron = AdaptiveGTMONeuron("test", (0.5, 0.5, 0.5))
            >>> state = QuantumStateCalculator.create_superposition(neuron)
            >>> total_prob = sum(abs(amp)**2 for amp in state.values())
            >>> assert abs(total_prob - 1.0) < 1e-10  # Normalized
        
        Note:
            Previously used: np.exp(1j * np.random.rand() * 2 * np.pi)
            Now uses: deterministic phase = 2π * epistemological_property
        """
        # Calculate phases from neuron's intrinsic epistemological properties
        # Phase encodes the "angle" of knowledge in complex plane
        determinacy_phase = 2 * math.pi * neuron.determinacy
        stability_phase = 2 * math.pi * neuron.stability
        entropy_phase = 2 * math.pi * neuron.entropy
        
        # Calculate amplitude magnitudes based on epistemological state
        # Higher determinacy → stronger "defined" component
        alpha_magnitude = math.sqrt(neuron.determinacy)
        # Lower determinacy → stronger "undefined" component
        beta_magnitude = math.sqrt(1 - neuron.determinacy)
        # Entropy contributes to "indefinite" component (scaled down)
        gamma_magnitude = math.sqrt(neuron.entropy) * 0.1
        
        # Construct complex amplitudes using Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
        alpha = alpha_magnitude * np.exp(1j * determinacy_phase)
        beta = beta_magnitude * np.exp(1j * stability_phase)
        gamma = gamma_magnitude * np.exp(1j * entropy_phase)
        
        # Normalize to ensure total probability = 1 (quantum mechanics requirement)
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2 + abs(gamma)**2)
        
        return {
            'defined': alpha / norm,
            'undefined': beta / norm,
            'indefinite': gamma / norm
        }
    
    @staticmethod
    def evolve_superposition(quantum_state: Dict[str, complex], time_step: float) -> Dict[str, complex]:
        """
        Apply unitary time evolution to quantum state using Hamiltonian dynamics.
        
        This implements the Schrödinger equation: |ψ(t)⟩ = e^(-iHt)|ψ(0)⟩
        where H is the Hamiltonian (energy operator) of the system.
        
        Args:
            quantum_state: Current quantum state dictionary
            time_step: Evolution time parameter (typically 0.1)
        
        Returns:
            Dict[str, complex]: Evolved quantum state (still normalized)
        
        Mathematical Details:
            - Each basis state has an associated energy level
            - Time evolution operator: U(t) = exp(-iEt/ℏ) (ℏ=1 in our units)
            - Higher energy states evolve faster (more phase rotation)
        
        Note:
            Previously used: rotation = np.exp(1j * 0.1) (constant rotation)
            Now uses: proper energy-based evolution
        """
        # Define Hamiltonian eigenvalues (energy levels) for each basis state
        # These represent the "cost" or "tension" of being in each state
        E_defined = 1.0      # Defined state: highest energy (most constrained)
        E_undefined = 0.5    # Undefined state: medium energy
        E_indefinite = 0.0   # Indefinite state: ground state (most relaxed)
        
        # Calculate time evolution operators for each eigenstate
        # U(t) = exp(-iEt) applies phase rotation proportional to energy
        U_defined = np.exp(-1j * E_defined * time_step)
        U_undefined = np.exp(-1j * E_undefined * time_step)
        U_indefinite = np.exp(-1j * E_indefinite * time_step)
        
        # Apply unitary evolution to each component
        return {
            'defined': quantum_state['defined'] * U_defined,
            'undefined': quantum_state['undefined'] * U_undefined,
            'indefinite': quantum_state['indefinite'] * U_indefinite
        }


class FoundationalModeCalculator:
    """
    Calculates the foundational mode of the GTMO system based on aggregate properties.
    
    Foundational modes represent system-wide phase states that emerge from
    the collective behavior of neurons. These modes influence how the system
    processes information and responds to perturbations.
    
    Modes:
        - definite_stillness: High determinacy, high stability (crystallized knowledge)
        - indefinite_stillness: Low activity, uncertain but stable
        - definite_flux: High determinacy but changing (learning phase)
        - eternal_flux: High entropy, constant change (exploration phase)
    """
    
    @staticmethod
    def calculate_mode(system_state: Any) -> str:
        """
        Calculate system's foundational mode from aggregate neuron properties.
        
        This method analyzes the statistical distribution of epistemological
        properties across all neurons to determine the system's phase state.
        
        Args:
            system_state: System object containing a list of neurons
        
        Returns:
            str: One of ['definite_stillness', 'indefinite_stillness',
                        'definite_flux', 'eternal_flux']
        
        Algorithm:
            1. Calculate average epistemological properties across neurons
            2. Compute flux and stillness indicators
            3. Select mode based on indicator comparison and thresholds
        
        Note:
            Previously used: np.random.choice(['stillness', 'flux'])
            Now uses: deterministic calculation from system properties
        """
        # Handle edge case: no neurons in system
        if not hasattr(system_state, 'neurons'):
            return 'indefinite_stillness'  # Default safe state
        
        # Aggregate epistemological properties across all neurons
        total_entropy = 0.0
        total_stability = 0.0
        total_determinacy = 0.0
        neuron_count = 0
        
        for neuron in system_state.neurons:
            if hasattr(neuron, 'entropy'):
                total_entropy += neuron.entropy
                neuron_count += 1
            if hasattr(neuron, 'stability'):
                total_stability += neuron.stability
            if hasattr(neuron, 'determinacy'):
                total_determinacy += neuron.determinacy
        
        # Handle empty system
        if neuron_count == 0:
            return 'indefinite_stillness'
        
        # Calculate system-wide averages
        avg_entropy = total_entropy / neuron_count
        avg_stability = total_stability / neuron_count
        avg_determinacy = total_determinacy / neuron_count
        
        # Calculate mode indicators
        # Flux: driven by entropy and instability
        flux_indicator = avg_entropy + (1 - avg_stability)
        # Stillness: driven by determinacy and stability
        stillness_indicator = avg_determinacy + avg_stability
        
        # Determine foundational mode based on indicators
        if flux_indicator > stillness_indicator:
            # System is in flux state
            if avg_entropy > 0.7:
                return 'eternal_flux'  # High entropy → constant change
            else:
                return 'definite_flux'  # Moderate flux → directed change
        else:
            # System is in stillness state
            if avg_determinacy > 0.7:
                return 'definite_stillness'  # High determinacy → crystallized
            else:
                return 'indefinite_stillness'  # Low determinacy → dormant


# === DEFENSE STRATEGY SELECTION ===

class DeterministicDefenseSelector:
    """
    Implements experience-based defense strategy selection for neurons.
    
    This class maintains a history of attack patterns and strategy effectiveness,
    allowing neurons to learn optimal defense strategies over time. The selection
    process is completely deterministic, based on neuron state and historical data.
    
    Defense Strategies:
        - absorption: Absorb and neutralize the attack (best for entropy attacks)
        - deflection: Redirect attack away (best for semantic attacks)
        - transformation: Transform attack into useful signal (best for logical attacks)
        - synthesis: Integrate attack into knowledge (best for complex attacks)
    """
    
    def __init__(self):
        """
        Initialize the defense selector with empty history.
        
        Attributes:
            attack_history: Maps attack signatures to strategy performance
            strategy_effectiveness: Global strategy success rates
        """
        self.attack_history = {}  # Track attack patterns
        self.strategy_effectiveness = {}  # Track strategy success rates
        
    def select_strategy(self, neuron, attack_type: str, attack_vector: Dict[str, float]) -> str:
        """
        Select optimal defense strategy based on neuron state and experience.
        
        This method combines multiple factors to deterministically choose
        the best defense strategy:
        1. Neuron's current epistemological state
        2. Attack vector characteristics
        3. Historical performance data
        
        Args:
            neuron: AdaptiveGTMONeuron being attacked
            attack_type: String identifier for attack category
            attack_vector: Dictionary with attack components:
                - semantic_attack: [0,1] strength of semantic confusion
                - logical_attack: [0,1] strength of logical paradox
                - entropy_attack: [0,1] strength of chaos injection
        
        Returns:
            str: Selected strategy name
        
        Algorithm:
            1. Calculate base strategy scores from neuron state
            2. Adjust scores based on attack characteristics
            3. Apply historical learning multipliers
            4. Select highest-scoring strategy
        
        Note:
            Previously used: np.random.choice(strategies, p=probabilities)
            Now uses: deterministic scoring and selection
        """
        available_strategies = ['absorption', 'deflection', 'transformation', 'synthesis']
        
        # Extract neuron's current epistemological state
        determinacy = getattr(neuron, 'determinacy', 0.5)
        stability = getattr(neuron, 'stability', 0.5)
        entropy = getattr(neuron, 'entropy', 0.5)
        
        # Initialize strategy scores based on neuron state
        strategy_scores = {}
        
        # Absorption: effective when stable and low entropy (can contain attacks)
        # High stability allows containment, low entropy prevents amplification
        strategy_scores['absorption'] = stability * (1 - entropy) + 0.1
        
        # Deflection: effective when deterministic (clear boundaries)
        # High determinacy provides clear attack surface for redirection
        strategy_scores['deflection'] = determinacy + stability * 0.5
        
        # Transformation: effective when balanced (can reshape attacks)
        # Balance allows flexibility without losing coherence
        balance_score = 1 - abs(determinacy - 0.5) - abs(stability - 0.5)
        strategy_scores['transformation'] = balance_score + entropy * 0.3
        
        # Synthesis: effective with high entropy (can integrate contradictions)
        # Entropy allows holding multiple states simultaneously
        strategy_scores['synthesis'] = entropy + (1 - determinacy) * 0.5
        
        # Adjust scores based on attack vector characteristics
        semantic_attack = attack_vector.get('semantic_attack', 0)
        logical_attack = attack_vector.get('logical_attack', 0)
        entropy_attack = attack_vector.get('entropy_attack', 0)
        
        # Semantic attacks target meaning → deflect or synthesize
        if semantic_attack > 0.5:
            strategy_scores['deflection'] += semantic_attack
            strategy_scores['synthesis'] += semantic_attack * 0.7
        
        # Logical attacks target consistency → transform or absorb
        if logical_attack > 0.5:
            strategy_scores['transformation'] += logical_attack
            strategy_scores['absorption'] += logical_attack * 0.8
        
        # Entropy attacks inject chaos → absorb to contain
        if entropy_attack > 0.5:
            strategy_scores['absorption'] += entropy_attack * 1.2
        
        # Apply historical learning: boost strategies that worked before
        attack_key = f"{attack_type}_{hash(str(attack_vector)) % 1000}"
        if attack_key in self.attack_history:
            for strategy, success_rate in self.attack_history[attack_key].items():
                strategy_scores[strategy] *= (1 + success_rate)
        
        # Select strategy with highest score (deterministic)
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        return best_strategy
    
    def update_strategy_effectiveness(self, attack_type: str, attack_vector: Dict[str, float], 
                                    strategy: str, success: bool):
        """
        Update learning history based on defense outcome.
        
        Uses exponential moving average to track strategy effectiveness,
        allowing the system to adapt to changing attack patterns while
        maintaining memory of past experiences.
        
        Args:
            attack_type: Type of attack experienced
            attack_vector: Attack characteristics
            strategy: Defense strategy used
            success: Whether defense was successful
        
        Learning Rate:
            α = 0.2 provides balance between stability and adaptability
        """
        # Create unique key for this attack pattern
        attack_key = f"{attack_type}_{hash(str(attack_vector)) % 1000}"
        
        # Initialize history for new attack patterns
        if attack_key not in self.attack_history:
            self.attack_history[attack_key] = {}
        
        # Initialize strategy performance at neutral (0.5)
        if strategy not in self.attack_history[attack_key]:
            self.attack_history[attack_key][strategy] = 0.5
        
        # Update using exponential moving average
        # This allows recent experiences to have more weight while preserving history
        alpha = 0.2  # Learning rate
        current_rate = self.attack_history[attack_key][strategy]
        new_sample = 1.0 if success else 0.0
        
        # EMA formula: new_value = (1-α)*old_value + α*new_sample
        self.attack_history[attack_key][strategy] = (1 - alpha) * current_rate + alpha * new_sample


# === ADAPTIVE GTMO NEURON ===

class AdaptiveGTMONeuron:
    """
    Core processing unit of the GTMO system with quantum properties and learning.
    
    Each neuron represents a knowledge processing node with:
    - Epistemological coordinates (determinacy, stability, entropy)
    - Quantum superposition state
    - Adaptive defense mechanisms
    - Experience-based learning
    
    The neuron's behavior emerges from the interaction between its quantum
    state, epistemological properties, and learned defense patterns.
    """
    
    def __init__(self, neuron_id: str, position: Tuple[float, float, float]):
        """
        Initialize an adaptive GTMO neuron at given position.
        
        Args:
            neuron_id: Unique identifier for the neuron
            position: 3D coordinates in knowledge space (x, y, z)
        
        The position in 3D space determines the neuron's initial
        epistemological properties through mathematical transformations.
        """
        self.id = neuron_id
        self.position = np.array(position)
        
        # Derive epistemological coordinates from spatial position
        # This creates a mapping from physical to epistemological space
        self.determinacy = self._calculate_determinacy_from_position()
        self.stability = self._calculate_stability_from_position()
        self.entropy = self._calculate_entropy_from_position()
        
        # Initialize learning components
        self.defense_selector = DeterministicDefenseSelector()
        self.experience_count = 0
        self.success_count = 0
        
        # Quantum state (initialized on first use)
        self.quantum_state = None
        
    def _calculate_determinacy_from_position(self) -> float:
        """
        Calculate determinacy based on distance from origin.
        
        Neurons closer to origin have higher determinacy (more certain knowledge).
        Uses exponential decay to create smooth gradient.
        
        Returns:
            float: Determinacy value in [0.01, 0.99]
        """
        # Euclidean distance from origin
        distance_from_origin = np.linalg.norm(self.position)
        # Exponential decay: closer to origin → higher determinacy
        return min(0.99, max(0.01, math.exp(-distance_from_origin / 2.0)))
    
    def _calculate_stability_from_position(self) -> float:
        """
        Calculate stability based on position coordinate variance.
        
        Neurons with more "regular" positions (low variance) are more stable.
        This creates regions of stability in knowledge space.
        
        Returns:
            float: Stability value in [0.01, 0.99]
        """
        # Variance measures how "spread out" the coordinates are
        coord_variance = np.var(self.position)
        # Low variance → high stability
        return min(0.99, max(0.01, math.exp(-coord_variance)))
    
    def _calculate_entropy_from_position(self) -> float:
        """
        Calculate entropy based on position irregularity patterns.
        
        Uses trigonometric functions to create complex entropy landscapes
        in knowledge space, with peaks and valleys of uncertainty.
        
        Returns:
            float: Entropy value in [0.01, 0.99]
        """
        x, y, z = self.position
        # Create interference pattern using different frequencies
        # This generates complex entropy topology
        pattern_score = abs(math.sin(x * 4)) + abs(math.cos(y * 3)) + abs(math.sin(z * 2))
        # Normalize to [0, 1] range
        return min(0.99, max(0.01, pattern_score / 3.0))
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as vector for neural network processing.
        
        Returns:
            np.ndarray: 5-dimensional state vector containing:
                [determinacy, stability, entropy, experience_ratio, success_rate]
        """
        return np.array([
            self.determinacy,
            self.stability,
            self.entropy,
            self.experience_count / 100.0,  # Normalized experience
            self.success_count / max(self.experience_count, 1)  # Success rate
        ])
    
    def experience_attack(self, attack_type: str, attack_vector: Dict[str, float], 
                         intensity: float) -> Dict[str, Any]:
        """
        Process an attack and learn from the experience.
        
        This method:
        1. Selects appropriate defense strategy
        2. Applies the defense
        3. Updates learning based on outcome
        4. Evolves quantum state
        
        Args:
            attack_type: Category of attack (e.g., 'semantic_confusion')
            attack_vector: Attack strength components
            intensity: Overall attack intensity [0, 1]
        
        Returns:
            Dict containing:
                - defense_used: Selected strategy name
                - success: Success level [0, 1]
                - learning_update: Whether learning occurred
                - quantum_evolution: Whether quantum state evolved
        """
        self.experience_count += 1
        
        # Select defense strategy based on current knowledge
        strategy = self.defense_selector.select_strategy(self, attack_type, attack_vector)
        
        # Apply the selected defense strategy
        defense_result = self._apply_defense_strategy(strategy, attack_type, attack_vector, intensity)
        
        # Update learning based on outcome
        success = defense_result['success'] > 0.5
        if success:
            self.success_count += 1
        
        # Store experience for future strategy selection
        self.defense_selector.update_strategy_effectiveness(
            attack_type, attack_vector, strategy, success
        )
        
        # Initialize or evolve quantum state based on experience
        if self.quantum_state is None:
            # First experience: create initial superposition
            self.quantum_state = QuantumStateCalculator.create_superposition(self)
        else:
            # Subsequent experiences: evolve the quantum state
            self.quantum_state = QuantumStateCalculator.evolve_superposition(
                self.quantum_state, 0.1
            )
        
        return {
            'defense_used': strategy,
            'success': defense_result['success'],
            'learning_update': True,
            'quantum_evolution': True
        }
    
    def _apply_defense_strategy(self, strategy: str, attack_type: str, 
                              attack_vector: Dict[str, float], intensity: float) -> Dict[str, Any]:
        """
        Apply specific defense strategy and calculate success.
        
        Each strategy has different effectiveness based on:
        - Neuron's epistemological state
        - Attack vector characteristics
        - Attack intensity
        
        Args:
            strategy: Name of defense strategy to apply
            attack_type: Type of attack
            attack_vector: Attack components
            intensity: Attack strength
        
        Returns:
            Dict with defense results including success rate
        """
        # Base success rate (neutral starting point)
        base_success = 0.5
        
        # Extract attack components
        semantic_attack = attack_vector.get('semantic_attack', 0)
        logical_attack = attack_vector.get('logical_attack', 0)
        entropy_attack = attack_vector.get('entropy_attack', 0)
        
        if strategy == 'absorption':
            # Absorption: contain and neutralize the attack
            # Effectiveness scales with stability (container strength)
            success_rate = self.stability * 0.8 + 0.2
            # Particularly effective against entropy (chaos containment)
            if entropy_attack > 0.5:
                success_rate += 0.2
                
        elif strategy == 'deflection':
            # Deflection: redirect attack away from core
            # Effectiveness scales with determinacy (clear boundaries)
            success_rate = self.determinacy * 0.8 + 0.2
            # Particularly effective against semantic attacks
            if semantic_attack > 0.5:
                success_rate += 0.2
                
        elif strategy == 'transformation':
            # Transformation: convert attack into useful signal
            # Effectiveness scales with balance (flexibility)
            balance = 1 - abs(self.determinacy - 0.5) - abs(self.stability - 0.5)
            success_rate = balance * 0.8 + 0.2
            # Particularly effective against logical attacks
            if logical_attack > 0.5:
                success_rate += 0.2
                
        elif strategy == 'synthesis':
            # Synthesis: integrate attack into knowledge structure
            # Effectiveness scales with entropy (can hold contradictions)
            success_rate = self.entropy * 0.6 + 0.4
            # Effective against complex multi-component attacks
            if sum(attack_vector.values()) > 1.0:
                success_rate += 0.2
        else:
            success_rate = base_success
        
        # Apply intensity penalty: stronger attacks are harder to defend
        # Formula ensures high intensity reduces success rate
        success_rate *= (2.0 - intensity)
        # Clamp to valid probability range
        success_rate = max(0.0, min(1.0, success_rate))
        
        return {
            'success': success_rate,
            'strategy_effectiveness': success_rate,
            'intensity_handled': intensity
        }
    
    def get_learned_patterns(self) -> Dict[str, Any]:
        """
        Summarize the neuron's learned defense patterns.
        
        Returns:
            Dict containing:
                - total_experiences: Number of attacks experienced
                - success_rate: Overall defense success rate
                - preferred_strategies: Strategy effectiveness map
                - quantum_state_magnitude: Quantum coherence measure
        """
        return {
            'total_experiences': self.experience_count,
            'success_rate': self.success_count / max(self.experience_count, 1),
            'preferred_strategies': dict(self.defense_selector.strategy_effectiveness),
            'quantum_state_magnitude': (
                sum(abs(amp)**2 for amp in self.quantum_state.values()) 
                if self.quantum_state else 0.0
            )
        }


# === EXECUTABLE AXIOMS ===

class AX0_SystemicUncertainty:
    """
    Axiom 0: There is no proof that the GTMØ system is fully definable.
    
    This foundational axiom ensures the system maintains quantum uncertainty
    at all levels. It actively creates and maintains superposition states
    in neurons and determines the system's foundational mode.
    
    Mathematical Interpretation:
        ∄ proof: GTMØ → Defined
        The system resists complete crystallization into definite states.
    """
    
    def apply(self, system_state: Any) -> Any:
        """
        Apply systemic uncertainty transformation to system.
        
        This method:
        1. Ensures all neurons have quantum superposition states
        2. Evolves existing quantum states
        3. Determines system's foundational mode
        
        Args:
            system_state: System object containing neurons
        
        Returns:
            Modified system_state with quantum properties
        """
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if not hasattr(neuron, 'quantum_state') or neuron.quantum_state is None:
                    # Create initial quantum superposition
                    neuron.quantum_state = QuantumStateCalculator.create_superposition(neuron)
                else:
                    # Evolve existing quantum state
                    neuron.quantum_state = QuantumStateCalculator.evolve_superposition(
                        neuron.quantum_state, 0.1
                    )
        
        # Calculate and set system's foundational mode
        if not hasattr(system_state, 'foundational_mode'):
            system_state.foundational_mode = FoundationalModeCalculator.calculate_mode(system_state)
        
        return system_state
    
    def verify(self, system_state: Any) -> bool:
        """
        Verify that system maintains foundational uncertainty.
        
        Args:
            system_state: System to verify
        
        Returns:
            bool: True if system has quantum uncertainty
        """
        if not hasattr(system_state, 'neurons'):
            return True  # Trivially satisfied for empty systems
        
        # Check that at least some neurons maintain quantum superposition
        quantum_neurons = sum(1 for n in system_state.neurons 
                            if hasattr(n, 'quantum_state') and n.quantum_state is not None)
        
        return quantum_neurons > 0
    
    @property
    def description(self) -> str:
        """Get human-readable description of the axiom."""
        return "There is no proof that the GTMØ system is fully definable"


def demonstrate_fixed_core():
    """
    Comprehensive demonstration of the fixed GTMO core functionality.
    
    This function showcases:
    1. Deterministic quantum state creation
    2. Defense strategy selection
    3. Foundational mode calculation
    4. Learning accumulation over time
    
    All operations are deterministic and reproducible.
    """
    print("=" * 80)
    print("GTMO CORE v2 - FIXED VERSION DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. TESTING DETERMINISTIC QUANTUM STATE CREATION")
    print("-" * 50)
    
    # Create test neuron at specific position
    neuron = AdaptiveGTMONeuron("test_neuron", (0.3, 0.7, 0.2))
    print(f"Neuron position: {neuron.position}")
    print(f"Derived determinacy: {neuron.determinacy:.3f}")
    print(f"Derived stability: {neuron.stability:.3f}")
    print(f"Derived entropy: {neuron.entropy:.3f}")
    
    # Create and verify quantum state
    quantum_state = QuantumStateCalculator.create_superposition(neuron)
    print(f"\nQuantum state amplitudes:")
    for state, amplitude in quantum_state.items():
        print(f"  |{state}⟩: {amplitude:.3f}")
    
    # Verify quantum state normalization
    total_prob = sum(abs(amp)**2 for amp in quantum_state.values())
    print(f"Total probability: {total_prob:.6f} (should be 1.0)")
    
    print("\n2. TESTING DETERMINISTIC DEFENSE STRATEGY SELECTION")
    print("-" * 50)
    
    # Define test attack scenarios
    attack_scenarios = [
        ("semantic_confusion", {'semantic_attack': 0.8, 'logical_attack': 0.1, 'entropy_attack': 0.1}),
        ("logical_paradox", {'semantic_attack': 0.2, 'logical_attack': 0.9, 'entropy_attack': 0.3}),
        ("entropy_flood", {'semantic_attack': 0.1, 'logical_attack': 0.2, 'entropy_attack': 0.9})
    ]
    
    # Test defense against each attack type
    for attack_type, attack_vector in attack_scenarios:
        result = neuron.experience_attack(attack_type, attack_vector, 0.7)
        print(f"\nAttack: {attack_type}")
        print(f"  Vector: {attack_vector}")
        print(f"  Defense: {result['defense_used']}")
        print(f"  Success: {result['success']:.3f}")
    
    print("\n3. TESTING FOUNDATIONAL MODE CALCULATION")
    print("-" * 50)
    
    # Create mock system with multiple neurons
    class MockSystem:
        def __init__(self):
            self.neurons = [
                AdaptiveGTMONeuron("n1", (0.1, 0.2, 0.3)),  # Near origin: high determinacy
                AdaptiveGTMONeuron("n2", (0.8, 0.9, 0.1)),  # Far from origin: low determinacy
                AdaptiveGTMONeuron("n3", (0.5, 0.4, 0.6))   # Moderate position
            ]
    
    system = MockSystem()
    mode = FoundationalModeCalculator.calculate_mode(system)
    print(f"Calculated foundational mode: {mode}")
    
    # Show calculation details
    avg_det = np.mean([n.determinacy for n in system.neurons])
    avg_stab = np.mean([n.stability for n in system.neurons])
    avg_ent = np.mean([n.entropy for n in system.neurons])
    
    print(f"System averages: det={avg_det:.3f}, stab={avg_stab:.3f}, ent={avg_ent:.3f}")
    
    print("\n4. TESTING LEARNING ACCUMULATION")
    print("-" * 50)
    
    # Simulate repeated attacks to demonstrate learning
    for i in range(5):
        attack_type = "repeated_test"
        attack_vector = {'semantic_attack': 0.6, 'logical_attack': 0.4, 'entropy_attack': 0.2}
        result = neuron.experience_attack(attack_type, attack_vector, 0.5)
        
        patterns = neuron.get_learned_patterns()
        print(f"Experience {i+1}: success_rate={patterns['success_rate']:.3f}, "
              f"total_exp={patterns['total_experiences']}")
    
    print("\n" + "=" * 80)
    print("GTMO CORE IS NOW FULLY DETERMINISTIC")
    print("All random functions replaced with epistemologically-grounded calculations")
    print("=" * 80)


if __name__ == "__main__":
    # Run demonstration when module is executed directly
    demonstrate_fixed_core()
