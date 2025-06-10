"""epistemic_particles.py
----------------------------------
Extension of GTMØ theory implementing EpistemicParticles (Ψᴱ) with
adaptive epistemic states and dimension-independent cognitive trajectories.

This module implements Theorem TΨᴱ and the additional hypothesis about
cognitive trajectories that can be independent of temporal dimension.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import GTMØ core components
from core import O, AlienatedNumber, Singularity, STRICT_MODE, SingularityError
from classification import KnowledgeEntity, KnowledgeType, GTMOClassifier
from topology import get_trajectory_state_phi_t, evaluate_field_E_x


class EpistemicState(Enum):
    """Possible epistemic states for EpistemicParticles."""
    
    ZERO = 0         # Minimal epistemic content
    ONE = 1          # Maximal epistemic determinacy
    INFINITY = float('inf')  # Unbounded epistemic expansion
    INDEFINITE = 'Ø'         # Epistemic indefiniteness


class EpistemicDimension(Enum):
    """Available epistemic dimensions for trajectory evolution."""
    
    TEMPORAL = auto()        # Standard time-based evolution
    ENTROPIC = auto()        # Entropy-based evolution
    DETERMINACY = auto()     # Determinacy-based evolution
    COMPLEXITY = auto()      # Complexity-based evolution
    COHERENCE = auto()       # Coherence-based evolution
    EMERGENCE = auto()       # Emergence-based evolution


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
    
    def __post_init__(self):
        """Initialize particle and validate state."""
        super().__post_init__()
        self._update_epistemic_state()
        
    def _update_epistemic_state(self) -> None:
        """Update epistemic state based on current properties."""
        # Map numerical properties to epistemic states
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
        # Simple heuristic: frequent state changes indicate expansion
        unique_states = len(set(states))
        return unique_states >= 3
        
    def evolve(self, parameter: float) -> 'EpistemicParticle':
        """
        Evolve the particle along its cognitive trajectory.
        
        Args:
            parameter: Evolution parameter (interpretation depends on epistemic_dimension)
            
        Returns:
            Evolved EpistemicParticle
        """
        # Store current state in history
        self.state_history.append((parameter, self.epistemic_state))
        
        # Apply trajectory evolution based on selected dimension
        if self.epistemic_dimension == EpistemicDimension.TEMPORAL:
            # Standard time-based evolution
            if self.trajectory_function:
                new_state = self.trajectory_function(parameter)
            else:
                # Default temporal evolution towards Ø
                new_state = get_trajectory_state_phi_t(self.content, parameter)
                
        elif self.epistemic_dimension == EpistemicDimension.ENTROPIC:
            # Entropy-based evolution
            self.entropy = self._evolve_entropy(parameter)
            self.determinacy = 1.0 - self.entropy
            
        elif self.epistemic_dimension == EpistemicDimension.DETERMINACY:
            # Determinacy-based evolution
            self.determinacy = self._evolve_determinacy(parameter)
            self.entropy = 1.0 - self.determinacy
            
        elif self.epistemic_dimension == EpistemicDimension.COMPLEXITY:
            # Complexity-based evolution
            complexity = self._calculate_complexity(parameter)
            self.stability = 1.0 / (1.0 + complexity)
            
        elif self.epistemic_dimension == EpistemicDimension.COHERENCE:
            # Coherence-based evolution
            self.stability = self._evolve_coherence(parameter)
            
        elif self.epistemic_dimension == EpistemicDimension.EMERGENCE:
            # Emergence-based evolution
            if parameter > 0.5:  # Threshold for emergence
                self.epistemic_state = EpistemicState.INFINITY
                
        # Update epistemic state based on new properties
        self._update_epistemic_state()
        
        # Handle collapse to singularity
        if self.epistemic_state == EpistemicState.INDEFINITE and parameter > 1.0:
            self.content = O
            self.epistemic_state = EpistemicState.INDEFINITE
            
        return self
        
    def _evolve_entropy(self, parameter: float) -> float:
        """Entropy evolution function."""
        # Entropy increases with parameter (second law of thermodynamics analog)
        return min(1.0, self.entropy + 0.1 * parameter)
        
    def _evolve_determinacy(self, parameter: float) -> float:
        """Determinacy evolution function."""
        # Oscillating determinacy
        return 0.5 + 0.5 * math.sin(parameter)
        
    def _calculate_complexity(self, parameter: float) -> float:
        """Calculate complexity metric."""
        # Complexity grows exponentially
        return math.exp(parameter) - 1.0
        
    def _evolve_coherence(self, parameter: float) -> float:
        """Coherence evolution function."""
        # Coherence decays over parameter
        return max(0.0, self.stability * math.exp(-0.5 * parameter))
        
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


class CognitiveTrajectory(ABC):
    """Abstract base class for cognitive trajectories φ(t)."""
    
    @abstractmethod
    def __call__(self, particle: EpistemicParticle, parameter: float) -> Any:
        """Apply trajectory transformation to particle."""
        pass


class SmoothTrajectory(CognitiveTrajectory):
    """
    Smooth cognitive trajectory with continuous transitions.
    
    Implements the hypothesis that trajectories can be smooth and
    independent of temporal dimension.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.1,
        dimension: EpistemicDimension = EpistemicDimension.TEMPORAL
    ):
        self.smoothing_factor = smoothing_factor
        self.dimension = dimension
        
    def __call__(self, particle: EpistemicParticle, parameter: float) -> Any:
        """Apply smooth transformation based on selected dimension."""
        # Save current state
        current_determinacy = particle.determinacy
        current_stability = particle.stability
        current_entropy = particle.entropy
        
        # Apply smooth transition
        if self.dimension == EpistemicDimension.ENTROPIC:
            # Evolution based on entropy gradient
            target_entropy = evaluate_field_E_x(particle.content, "cognitive_entropy")
            if isinstance(target_entropy, float):
                particle.entropy = (
                    (1 - self.smoothing_factor) * current_entropy +
                    self.smoothing_factor * target_entropy
                )
                
        elif self.dimension == EpistemicDimension.DETERMINACY:
            # Evolution based on epistemic purity
            target_purity = evaluate_field_E_x(particle.content, "epistemic_purity")
            if isinstance(target_purity, float):
                particle.determinacy = (
                    (1 - self.smoothing_factor) * current_determinacy +
                    self.smoothing_factor * target_purity
                )
                
        # Ensure smooth transitions in stability
        particle.stability = (
            (1 - self.smoothing_factor) * current_stability +
            self.smoothing_factor * particle.stability
        )
        
        return particle


class EpistemicParticleSystem:
    """
    System for managing collections of EpistemicParticles.
    
    Implements collective behaviors and emergent phenomena.
    """
    
    def __init__(self, strict_mode: Optional[bool] = None):
        self.particles: List[EpistemicParticle] = []
        self.classifier = GTMOClassifier(strict_mode=strict_mode)
        self.system_time = 0.0
        self.emergence_threshold = 0.7
        
    def add_particle(self, particle: EpistemicParticle) -> None:
        """Add a particle to the system."""
        self.particles.append(particle)
        self.classifier.add_to_knowledge_base(particle)
        
    def evolve_system(self, delta: float = 0.1) -> None:
        """
        Evolve all particles in the system.
        
        Args:
            delta: Evolution parameter increment
        """
        self.system_time += delta
        
        # Evolve each particle
        for particle in self.particles:
            particle.evolve(self.system_time)
            
        # Check for emergent phenomena
        self._detect_emergence()
        
    def _detect_emergence(self) -> Optional[EpistemicParticle]:
        """Detect and handle emergent particles."""
        # Calculate system-wide metrics
        total_entropy = sum(p.entropy for p in self.particles) / len(self.particles)
        coherence = self._calculate_system_coherence()
        
        # Emergence condition
        if coherence > self.emergence_threshold and total_entropy < 0.3:
            # Create emergent particle
            emergent = EpistemicParticle(
                content="emergent_phenomenon",
                determinacy=0.8,
                stability=0.9,
                entropy=0.1,
                epistemic_state=EpistemicState.INFINITY,
                metadata={'emerged_at': self.system_time}
            )
            self.add_particle(emergent)
            return emergent
            
        return None
        
    def _calculate_system_coherence(self) -> float:
        """Calculate overall system coherence."""
        if len(self.particles) < 2:
            return 0.0
            
        # Coherence based on state similarity
        state_counts = {}
        for particle in self.particles:
            state = particle.epistemic_state
            state_counts[state] = state_counts.get(state, 0) + 1
            
        # Higher coherence when particles share similar states
        max_count = max(state_counts.values())
        return max_count / len(self.particles)
        
    def get_alienated_particles(self) -> List[EpistemicParticle]:
        """Get all particles in indefinite/alienated state."""
        return [
            p for p in self.particles
            if p.epistemic_state == EpistemicState.INDEFINITE
        ]
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state information."""
        state_distribution = {}
        for particle in self.particles:
            state = particle.epistemic_state.name
            state_distribution[state] = state_distribution.get(state, 0) + 1
            
        return {
            'particle_count': len(self.particles),
            'system_time': self.system_time,
            'state_distribution': state_distribution,
            'average_entropy': sum(p.entropy for p in self.particles) / len(self.particles) if self.particles else 0,
            'system_coherence': self._calculate_system_coherence(),
            'alienated_count': len(self.get_alienated_particles()),
            'classifier_stats': self.classifier.get_statistics()
        }


# Utility functions for working with EpistemicParticles

def create_epistemic_particle_from_content(
    content: Any,
    dimension: EpistemicDimension = EpistemicDimension.TEMPORAL,
    **kwargs
) -> EpistemicParticle:
    """
    Factory function to create EpistemicParticle from arbitrary content.
    
    Args:
        content: The content to encapsulate
        dimension: The epistemic dimension for evolution
        **kwargs: Additional parameters
        
    Returns:
        Configured EpistemicParticle
    """
    # Determine initial properties based on content type
    if content is O or isinstance(content, Singularity):
        return EpistemicParticle(
            content=content,
            determinacy=1.0,
            stability=1.0,
            entropy=0.0,
            epistemic_state=EpistemicState.INDEFINITE,
            epistemic_dimension=dimension,
            **kwargs
        )
    elif isinstance(content, AlienatedNumber):
        return EpistemicParticle(
            content=content,
            determinacy=1.0 - content.e_gtm_entropy(),
            stability=content.psi_gtm_score(),
            entropy=content.e_gtm_entropy(),
            epistemic_state=EpistemicState.INDEFINITE,
            alienated_representation=content,
            epistemic_dimension=dimension,
            **kwargs
        )
    else:
        # Generic content
        return EpistemicParticle(
            content=content,
            determinacy=0.5,
            stability=0.5,
            entropy=0.5,
            epistemic_dimension=dimension,
            **kwargs
        )


def demonstrate_epistemic_evolution():
    """Demonstrate the evolution of EpistemicParticles."""
    print("=== EpistemicParticles Evolution Demo ===\n")
    
    # Create a system
    system = EpistemicParticleSystem()
    
    # Create particles with different dimensions
    particles = [
        create_epistemic_particle_from_content(
            "temporal_knowledge",
            EpistemicDimension.TEMPORAL
        ),
        create_epistemic_particle_from_content(
            "entropic_knowledge",
            EpistemicDimension.ENTROPIC
        ),
        create_epistemic_particle_from_content(
            AlienatedNumber("alien_42"),
            EpistemicDimension.DETERMINACY
        ),
        create_epistemic_particle_from_content(
            O,
            EpistemicDimension.EMERGENCE
        )
    ]
    
    # Add particles to system
    for p in particles:
        system.add_particle(p)
        
    # Evolve system
    print("Initial state:")
    print(system.get_system_state())
    print()
    
    for i in range(5):
        system.evolve_system(0.3)
        print(f"After evolution step {i+1}:")
        state = system.get_system_state()
        print(f"  State distribution: {state['state_distribution']}")
        print(f"  Average entropy: {state['average_entropy']:.3f}")
        print(f"  System coherence: {state['system_coherence']:.3f}")
        print(f"  Alienated particles: {state['alienated_count']}")
        print()
        
    return system


if __name__ == "__main__":
    # Run demonstration
    demonstrate_epistemic_evolution()
