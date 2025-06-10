"""epistemic_particles.py
----------------------------------
Extension of GTMØ theory implementing EpistemicParticles (Ψᴱ) with
full integration with GTMØ axioms and operators.

This module implements Theorem TΨᴱ and the additional hypothesis about
cognitive trajectories that can be independent of temporal dimension.
"""

from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import GTMØ core components
from core import O, AlienatedNumber, Singularity, STRICT_MODE, SingularityError
from classification import KnowledgeEntity, KnowledgeType, GTMOClassifier
from topology import get_trajectory_state_phi_t, evaluate_field_E_x

# Import advanced GTMØ operators from axioms
from gtmo_axioms import (
    PsiOperator, EntropyOperator, ThresholdManager,
    MetaFeedbackLoop, OperatorType, create_gtmo_system
)


class EpistemicState(Enum):
    """Possible epistemic states for EpistemicParticles."""
    
    ZERO = 0                    # Minimal epistemic content
    ONE = 1                     # Maximal epistemic determinacy
    INFINITY = float('inf')     # Unbounded epistemic expansion
    INDEFINITE = 'Ø'           # Epistemic indefiniteness


class EpistemicDimension(Enum):
    """Available epistemic dimensions for trajectory evolution."""
    
    TEMPORAL = auto()        # Standard time-based evolution
    ENTROPIC = auto()        # Entropy-based evolution
    DETERMINACY = auto()     # Determinacy-based evolution
    COMPLEXITY = auto()      # Complexity-based evolution
    COHERENCE = auto()       # Coherence-based evolution
    EMERGENCE = auto()       # Emergence-based evolution
    QUANTUM = auto()         # Quantum superposition dimension
    TOPOLOGICAL = auto()     # Topological transformation dimension


@dataclass
class EpistemicParticle(KnowledgeEntity):
    """
    Extended knowledge entity representing an EpistemicParticle (Ψᴱ).
    
    Implements Theorem TΨᴱ: adaptive state changes based on cognitive
    trajectory dynamics and epistemic entropy.
    """
    
    # Current epistemic state
    epistemic_state: EpistemicState = EpistemicState.ONE
    
    # Trajectory function φ(t) - can be customized per particle
    trajectory_function: Optional[Callable[[float], Any]] = None
    
    # Selected epistemic dimension for evolution
    epistemic_dimension: EpistemicDimension = EpistemicDimension.TEMPORAL
    
    # History of state transitions
    state_history: List[Tuple[float, EpistemicState]] = field(default_factory=list)
    
    # Alienated number representation if in indefinite state
    alienated_representation: Optional[AlienatedNumber] = None
    
    # GTMØ operator scores
    psi_score: float = 0.5
    cognitive_entropy: float = 0.5
    
    # Meta-cognitive properties
    emergence_potential: float = 0.0
    coherence_factor: float = 1.0
    
    def __post_init__(self):
        """Initialize particle and validate state."""
        super().__post_init__()
        self._update_epistemic_state()
        self._calculate_gtmo_metrics()
        
    def _calculate_gtmo_metrics(self) -> None:
        """Calculate GTMØ-specific metrics using operators."""
        # This would integrate with actual GTMØ operators
        # For now, using approximations
        self.psi_score = self.determinacy * self.stability
        self.cognitive_entropy = -self.psi_score * math.log2(self.psi_score + 0.001)
        
    def _update_epistemic_state(self) -> None:
        """Update epistemic state based on current properties and GTMØ operators."""
        # Apply GTMØ thresholds for state determination
        if self.entropy > 0.9 or (self.determinacy < 0.1 and self.stability < 0.1):
            self.epistemic_state = EpistemicState.ZERO
        elif self.determinacy > 0.9 and self.stability > 0.9 and self.entropy < 0.1:
            self.epistemic_state = EpistemicState.ONE
        elif self.determinacy < 0.3 or self.stability < 0.3:
            self.epistemic_state = EpistemicState.INDEFINITE
            # Create alienated representation
            if not self.alienated_representation:
                self.alienated_representation = AlienatedNumber(
                    f"particle_{id(self)}"
                )
        else:
            # Check for unbounded expansion patterns
            if len(self.state_history) > 5:
                recent_states = [s[1] for s in self.state_history[-5:]]
                if self._is_expanding_pattern(recent_states):
                    self.epistemic_state = EpistemicState.INFINITY
                    
    def _is_expanding_pattern(self, states: List[EpistemicState]) -> bool:
        """Detect if the particle shows unbounded expansion behavior."""
        # Enhanced detection using GTMØ principles
        unique_states = len(set(states))
        state_entropy = self._calculate_state_entropy(states)
        
        # High state entropy indicates expansion
        return unique_states >= 3 and state_entropy > 0.7
        
    def _calculate_state_entropy(self, states: List[EpistemicState]) -> float:
        """Calculate entropy of state sequence."""
        if not states:
            return 0.0
            
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        total = len(states)
        entropy = 0.0
        for count in state_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
                
        # Normalize
        max_entropy = math.log2(len(EpistemicState))
        return entropy / max_entropy if max_entropy > 0 else 0.0
        
    def evolve(
        self,
        parameter: float,
        operators: Optional[Dict[str, Any]] = None
    ) -> 'EpistemicParticle':
        """
        Evolve the particle along its cognitive trajectory using GTMØ operators.
        
        Args:
            parameter: Evolution parameter (interpretation depends on epistemic_dimension)
            operators: Optional GTMØ operators for advanced evolution
            
        Returns:
            Evolved EpistemicParticle
        """
        # Store current state in history
        self.state_history.append((parameter, self.epistemic_state))
        
        # Apply GTMØ operators if provided
        if operators:
            self._apply_gtmo_operators(operators, parameter)
        
        # Apply trajectory evolution based on selected dimension
        if self.epistemic_dimension == EpistemicDimension.TEMPORAL:
            self._evolve_temporal(parameter)
        elif self.epistemic_dimension == EpistemicDimension.ENTROPIC:
            self._evolve_entropic(parameter)
        elif self.epistemic_dimension == EpistemicDimension.DETERMINACY:
            self._evolve_determinacy_dimension(parameter)
        elif self.epistemic_dimension == EpistemicDimension.COMPLEXITY:
            self._evolve_complexity(parameter)
        elif self.epistemic_dimension == EpistemicDimension.COHERENCE:
            self._evolve_coherence_dimension(parameter)
        elif self.epistemic_dimension == EpistemicDimension.EMERGENCE:
            self._evolve_emergence(parameter)
        elif self.epistemic_dimension == EpistemicDimension.QUANTUM:
            self._evolve_quantum(parameter)
        elif self.epistemic_dimension == EpistemicDimension.TOPOLOGICAL:
            self._evolve_topological(parameter)
                
        # Update epistemic state based on new properties
        self._update_epistemic_state()
        self._calculate_gtmo_metrics()
        
        # Handle collapse to singularity (AX1, AX5)
        if self.epistemic_state == EpistemicState.INDEFINITE and parameter > 1.0:
            if self._should_collapse_to_singularity():
                self.content = O
                self.epistemic_state = EpistemicState.INDEFINITE
            
        return self
        
    def _apply_gtmo_operators(self, operators: Dict[str, Any], parameter: float) -> None:
        """Apply GTMØ operators to evolve particle properties."""
        if 'psi' in operators:
            psi_op = operators['psi']
            context = {'all_scores': operators.get('scores', []), 'timestamp': parameter}
            result = psi_op(self.content, context)
            self.psi_score = result['score']
            
        if 'entropy' in operators:
            entropy_op = operators['entropy']
            context = {'parameter': parameter}
            result = entropy_op(self.content, context)
            self.cognitive_entropy = result['total_entropy']
            
    def _should_collapse_to_singularity(self) -> bool:
        """Determine if particle should collapse to Ø based on GTMØ axioms."""
        # Implements AX6: Ø has minimal entropy
        # If particle reaches minimal entropy threshold, it approaches Ø
        entropy_threshold = 0.01
        determinacy_threshold = 0.99
        
        return (
            self.cognitive_entropy < entropy_threshold or
            (self.entropy < entropy_threshold and self.determinacy > determinacy_threshold)
        )
        
    def _evolve_temporal(self, parameter: float) -> None:
        """Standard time-based evolution."""
        if self.trajectory_function:
            new_state = self.trajectory_function(parameter)
        else:
            # Default temporal evolution towards Ø
            new_state = get_trajectory_state_phi_t(self.content, parameter)
            
        # Update properties based on trajectory
        if new_state is O:
            self.epistemic_state = EpistemicState.INDEFINITE
            
    def _evolve_entropic(self, parameter: float) -> None:
        """Entropy-based evolution following GTMØ entropy principles."""
        # Implements entropy dynamics from E_GTMØ operator
        delta_entropy = 0.1 * math.sin(parameter) * self.coherence_factor
        self.entropy = max(0.0, min(1.0, self.entropy + delta_entropy))
        self.determinacy = 1.0 - self.entropy
        
    def _evolve_determinacy_dimension(self, parameter: float) -> None:
        """Determinacy-based evolution."""
        # Oscillating determinacy with decay
        decay_factor = math.exp(-0.1 * parameter)
        self.determinacy = 0.5 + 0.5 * math.sin(parameter) * decay_factor
        self.entropy = 1.0 - self.determinacy
        
    def _evolve_complexity(self, parameter: float) -> None:
        """Complexity-based evolution."""
        # Complexity affects stability inversely
        complexity = self._calculate_complexity(parameter)
        self.stability = 1.0 / (1.0 + complexity)
        self.emergence_potential = min(1.0, complexity / 10.0)
        
    def _evolve_coherence_dimension(self, parameter: float) -> None:
        """Coherence-based evolution."""
        # Coherence decay with quantum fluctuations
        base_decay = math.exp(-0.5 * parameter)
        fluctuation = 0.1 * math.sin(10 * parameter)
        self.stability = max(0.0, self.stability * base_decay + fluctuation)
        self.coherence_factor = self.stability
        
    def _evolve_emergence(self, parameter: float) -> None:
        """Emergence-based evolution - implements Ψᴺ detection."""
        # Threshold-based emergence with hysteresis
        emergence_threshold = 0.5
        if parameter > emergence_threshold and self.emergence_potential > 0.7:
            self.epistemic_state = EpistemicState.INFINITY
            self.metadata['emergence_triggered'] = True
            
    def _evolve_quantum(self, parameter: float) -> None:
        """Quantum superposition evolution."""
        # Quantum state superposition
        phase = parameter * 2 * math.pi
        self.determinacy = (math.cos(phase) ** 2)
        self.stability = (math.sin(phase) ** 2)
        # Quantum coherence
        self.coherence_factor = abs(math.cos(phase) * math.sin(phase))
        
    def _evolve_topological(self, parameter: float) -> None:
        """Topological transformation evolution."""
        # Implements topological boundary conditions (AX5)
        # As particle approaches boundary ∂(CognitiveSpace), properties change
        boundary_distance = abs(1.0 - parameter)
        if boundary_distance < 0.1:
            # Near boundary, increase indefiniteness
            self.determinacy *= boundary_distance
            self.stability *= boundary_distance
            self.entropy = 1.0 - boundary_distance
        
    def _calculate_complexity(self, parameter: float) -> float:
        """Calculate complexity metric using GTMØ principles."""
        # Complexity as emergent property
        base_complexity = math.exp(parameter) - 1.0
        state_complexity = len(self.state_history) * 0.1
        return base_complexity + state_complexity
        
    def get_current_representation(self) -> Union[float, AlienatedNumber, Singularity]:
        """
        Get the current mathematical representation of the particle.
        
        Returns:
            Numerical value, AlienatedNumber, or Singularity based on state
        """
        if self.epistemic_state == EpistemicState.ZERO:
            return 0.0
        elif self.epistemic_state == EpistemicState.ONE:
            return 1.0
        elif self.epistemic_state == EpistemicState.INFINITY:
            return float('inf')
        elif self.epistemic_state == EpistemicState.INDEFINITE:
            if self.content is O:
                return O
            return self.alienated_representation or AlienatedNumber("undefined")
        else:
            return self.determinacy  # Default to determinacy value
            
    def to_gtmo_classification(self) -> str:
        """Convert particle to GTMØ classification (Ψᴷ, Ψʰ, Ψᴺ, Ø)."""
        if self.epistemic_state == EpistemicState.INDEFINITE:
            return "Ø"
        elif self.epistemic_state == EpistemicState.INFINITY:
            return "Ψᴺ"
        elif self.determinacy > 0.7 and self.stability > 0.7:
            return "Ψᴷ"
        elif self.determinacy < 0.3 or self.stability < 0.3:
            return "Ψʰ"
        else:
            return "Ψᴧ"


class AdvancedCognitiveTrajectory(CognitiveTrajectory):
    """
    Advanced cognitive trajectory implementing GTMØ operators.
    
    Integrates with Ψ_GTMØ and E_GTMØ operators for trajectory calculation.
    """
    
    def __init__(
        self,
        psi_operator: PsiOperator,
        entropy_operator: EntropyOperator,
        smoothing_factor: float = 0.1,
        dimension: EpistemicDimension = EpistemicDimension.TEMPORAL
    ):
        self.psi_operator = psi_operator
        self.entropy_operator = entropy_operator
        self.smoothing_factor = smoothing_factor
        self.dimension = dimension
        
    def __call__(self, particle: EpistemicParticle, parameter: float) -> Any:
        """Apply trajectory transformation using GTMØ operators."""
        # Get current GTMØ measurements
        context = {'all_scores': [], 'parameter': parameter}
        psi_result = self.psi_operator(particle.content, context)
        entropy_result = self.entropy_operator(particle.content, context)
        
        # Apply smooth transition based on GTMØ metrics
        target_determinacy = psi_result['score']
        target_entropy = entropy_result['total_entropy']
        
        # Smooth interpolation
        particle.determinacy = (
            (1 - self.smoothing_factor) * particle.determinacy +
            self.smoothing_factor * target_determinacy
        )
        particle.entropy = (
            (1 - self.smoothing_factor) * particle.entropy +
            self.smoothing_factor * target_entropy
        )
        
        # Update cognitive entropy
        particle.cognitive_entropy = entropy_result['total_entropy']
        particle.psi_score = psi_result['score']
        
        return particle


class IntegratedEpistemicSystem(EpistemicParticleSystem):
    """
    Enhanced system integrating EpistemicParticles with full GTMØ framework.
    """
    
    def __init__(self, strict_mode: Optional[bool] = None):
        super().__init__(strict_mode)
        
        # Create GTMØ operators
        self.psi_op, self.entropy_op, self.meta_loop = create_gtmo_system()
        self.threshold_manager = self.meta_loop.threshold_manager
        
        # System metrics
        self.total_entropy_history: List[float] = []
        self.emergence_events: List[Tuple[float, EpistemicParticle]] = []
        
    def evolve_system(self, delta: float = 0.1) -> None:
        """
        Evolve all particles using GTMØ operators and meta-feedback.
        
        Args:
            delta: Evolution parameter increment
        """
        self.system_time += delta
        
        # Collect current scores for threshold calculation
        all_scores = [p.psi_score for p in self.particles]
        
        # Create operator context
        operators = {
            'psi': self.psi_op,
            'entropy': self.entropy_op,
            'scores': all_scores
        }
        
        # Evolve each particle with GTMØ operators
        for particle in self.particles:
            particle.evolve(self.system_time, operators)
            
        # Update system metrics
        self._update_system_metrics()
        
        # Check for emergent phenomena using GTMØ criteria
        emergent = self._detect_gtmo_emergence()
        if emergent:
            self.emergence_events.append((self.system_time, emergent))
            
        # Apply meta-feedback if needed
        if len(self.particles) % 10 == 0:
            self._apply_meta_feedback()
            
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics."""
        if self.particles:
            total_entropy = sum(p.cognitive_entropy for p in self.particles) / len(self.particles)
            self.total_entropy_history.append(total_entropy)
            
    def _detect_gtmo_emergence(self) -> Optional[EpistemicParticle]:
        """Detect emergence using GTMØ principles (Ψᴺ detection)."""
        # Calculate system-wide metrics
        if not self.particles:
            return None
            
        coherence = self._calculate_system_coherence()
        avg_entropy = sum(p.cognitive_entropy for p in self.particles) / len(self.particles)
        
        # GTMØ emergence conditions
        # Low entropy + high coherence = emergence potential
        if coherence > 0.8 and avg_entropy < 0.2:
            # Check for critical mass of high-determinacy particles
            high_det_count = sum(1 for p in self.particles if p.determinacy > 0.8)
            if high_det_count / len(self.particles) > 0.6:
                # Create emergent particle (Ψᴺ)
                emergent = EpistemicParticle(
                    content=f"emergent_phenomenon_{self.system_time}",
                    determinacy=0.9,
                    stability=0.85,
                    entropy=0.05,
                    epistemic_state=EpistemicState.INFINITY,
                    epistemic_dimension=EpistemicDimension.EMERGENCE,
                    metadata={
                        'emerged_at': self.system_time,
                        'type': 'Ψᴺ',
                        'parent_coherence': coherence
                    }
                )
                self.add_particle(emergent)
                return emergent
                
        return None
        
    def _apply_meta_feedback(self) -> None:
        """Apply GTMØ meta-feedback loop principles."""
        # Collect all particles as fragments
        fragments = [p.content for p in self.particles]
        scores = [p.psi_score for p in self.particles]
        
        # Run meta-feedback
        results = self.meta_loop.run(fragments, scores, iterations=3)
        
        # Update thresholds based on feedback
        if results['final_state']['thresholds']:
            new_thresholds = results['final_state']['thresholds']
            # Thresholds are automatically updated in threshold_manager
            
    def get_detailed_state(self) -> Dict[str, Any]:
        """Get detailed system state including GTMØ metrics."""
        base_state = self.get_system_state()
        
        # Add GTMØ-specific metrics
        gtmo_metrics = {
            'particle_classifications': {},
            'entropy_evolution': self.total_entropy_history[-10:] if self.total_entropy_history else [],
            'emergence_count': len(self.emergence_events),
            'current_thresholds': self.threshold_manager.history[-1] if self.threshold_manager.history else (0.5, 0.5)
        }
        
        # Classify particles according to GTMØ
        for particle in self.particles:
            classification = particle.to_gtmo_classification()
            gtmo_metrics['particle_classifications'][classification] = \
                gtmo_metrics['particle_classifications'].get(classification, 0) + 1
                
        base_state['gtmo_metrics'] = gtmo_metrics
        return base_state


# Enhanced utility functions

def create_epistemic_particle_with_gtmo(
    content: Any,
    dimension: EpistemicDimension = EpistemicDimension.TEMPORAL,
    psi_operator: Optional[PsiOperator] = None,
    **kwargs
) -> EpistemicParticle:
    """
    Factory function creating EpistemicParticle with GTMØ integration.
    
    Args:
        content: The content to encapsulate
        dimension: The epistemic dimension for evolution
        psi_operator: Optional Ψ_GTMØ operator for initial scoring
        **kwargs: Additional parameters
        
    Returns:
        Configured EpistemicParticle with GTMØ properties
    """
    # Determine initial properties based on content type
    if content is O or isinstance(content, Singularity):
        particle = EpistemicParticle(
            content=content,
            determinacy=1.0,
            stability=1.0,
            entropy=0.0,
            epistemic_state=EpistemicState.INDEFINITE,
            epistemic_dimension=dimension,
            psi_score=1.0,
            cognitive_entropy=0.0,
            **kwargs
        )
    elif isinstance(content, AlienatedNumber):
        particle = EpistemicParticle(
            content=content,
            determinacy=1.0 - content.e_gtm_entropy(),
            stability=content.psi_gtm_score(),
            entropy=content.e_gtm_entropy(),
            epistemic_state=EpistemicState.INDEFINITE,
            alienated_representation=content,
            epistemic_dimension=dimension,
            psi_score=content.psi_gtm_score(),
            cognitive_entropy=content.e_gtm_entropy(),
            **kwargs
        )
    else:
        # Generic content - use GTMØ operator if available
        if psi_operator:
            context = {'all_scores': []}
            result = psi_operator(content, context)
            psi_score = result['score']
        else:
            psi_score = 0.5
            
        particle = EpistemicParticle(
            content=content,
            determinacy=0.5,
            stability=0.5,
            entropy=0.5,
            epistemic_dimension=dimension,
            psi_score=psi_score,
            **kwargs
        )
        
    return particle


def demonstrate_integrated_evolution():
    """Demonstrate the evolution of EpistemicParticles with full GTMØ integration."""
    print("=" * 80)
    print("INTEGRATED EPISTEMIC PARTICLES + GTMØ DEMONSTRATION")
    print("=" * 80)
    
    # Create integrated system
    system = IntegratedEpistemicSystem()
    
    # Create GTMØ operators for particle creation
    psi_op, _, _ = create_gtmo_system()
    
    # Create diverse particles
    test_contents = [
        "The fundamental theorem of calculus",
        "This statement is paradoxical",
        AlienatedNumber("undefined_concept"),
        "Meta-knowledge about knowledge",
        O,
        "Quantum superposition principle",
        "Emergent pattern in complex systems"
    ]
    
    dimensions = [
        EpistemicDimension.TEMPORAL,
        EpistemicDimension.EMERGENCE,
        EpistemicDimension.ENTROPIC,
        EpistemicDimension.QUANTUM,
        EpistemicDimension.TOPOLOGICAL,
        EpistemicDimension.COMPLEXITY,
        EpistemicDimension.COHERENCE
    ]
    
    # Create and add particles
    for content, dimension in zip(test_contents, dimensions):
        particle = create_epistemic_particle_with_gtmo(
            content=content,
            dimension=dimension,
            psi_operator=psi_op
        )
        system.add_particle(particle)
        
    # Initial state
    print("\n## INITIAL STATE")
    print("-" * 40)
    initial_state = system.get_detailed_state()
    print(f"Particles: {initial_state['particle_count']}")
    print(f"Classifications: {initial_state['gtmo_metrics']['particle_classifications']}")
    print(f"Average Entropy: {initial_state['average_entropy']:.3f}")
    
    # Evolution
    print("\n## EVOLUTION PROCESS")
    print("-" * 40)
    
    for i in range(10):
        system.evolve_system(0.2)
        
        if i % 2 == 0:  # Print every other step
            state = system.get_detailed_state()
            print(f"\nStep {i+1}:")
            print(f"  Classifications: {state['gtmo_metrics']['particle_classifications']}")
            print(f"  System Coherence: {state['system_coherence']:.3f}")
            print(f"  Average Entropy: {state['average_entropy']:.3f}")
            print(f"  Emergent Events: {state['gtmo_metrics']['emergence_count']}")
            
    # Final analysis
    print("\n## FINAL ANALYSIS")
    print("-" * 40)
    final_state = system.get_detailed_state()
    
    print(f"\nFinal Classifications:")
    for class_type, count in final_state['gtmo_metrics']['particle_classifications'].items():
        print(f"  {class_type}: {count}")
        
    print(f"\nSystem Metrics:")
    print(f"  Total Particles: {final_state['particle_count']}")
    print(f"  Alienated Particles: {final_state['alienated_count']}")
    print(f"  System Coherence: {final_state['system_coherence']:.3f}")
    print(f"  Final Entropy: {final_state['average_entropy']:.3f}")
    print(f"  Total Emergent Events: {final_state['gtmo_metrics']['emergence_count']}")
    
    if system.emergence_events:
        print(f"\nEmergence Timeline:")
        for time, particle in system.emergence_events:
            print(f"  t={time:.1f}: {particle.content}")
            
    # Particle state details
    print("\n## PARTICLE STATES")
    print("-" * 40)
    for i, particle in enumerate(system.particles[:5]):  # First 5 particles
        print(f"\nParticle {i+1} ({particle.epistemic_dimension.name}):")
        print(f"  Content: {str(particle.content)[:50]}...")
        print(f"  State: {particle.epistemic_state.name}")
        print(f"  Classification: {particle.to_gtmo_classification()}")
        print(f"  Ψ_GTMØ score: {particle.psi_score:.3f}")
        print(f"  Cognitive Entropy: {particle.cognitive_entropy:.3f}")
        
    return system


if __name__ == "__main__":
    # Run integrated demonstration
    system = demonstrate_integrated_evolution()
    
    print("\n" + "=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)
