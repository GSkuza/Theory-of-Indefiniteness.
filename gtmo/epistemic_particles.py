"""enhanced_epistemic_particles.py
----------------------------------
Enhanced extension of GTMØ theory implementing EpistemicParticles (Ψᴱ) with
inexhaustible dimension discovery, precognitive trajectories, and meta-reflection
about system unknowability.

This module implements Theorem TΨᴱ with extended capabilities:
- Dynamic dimension discovery during evolution
- Precognitive trajectory adaptation
- Unknown pattern observation and classification
- Meta-reflection about system's inherent unknowability
"""

from __future__ import annotations

import math
import time
import random
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Import GTMØ core components
from core import O, AlienatedNumber, Singularity, STRICT_MODE, SingularityError
# Assuming these exist in the system - create minimal stubs if needed
try:
    from classification import KnowledgeEntity, KnowledgeType, GTMOClassifier
except ImportError:
    # Minimal stub implementations
    class KnowledgeEntity:
        def __init__(self, content: Any = None, determinacy: float = 0.5, 
                     stability: float = 0.5, entropy: float = 0.5, 
                     metadata: Dict[str, Any] = None):
            self.content = content
            self.determinacy = determinacy
            self.stability = stability
            self.entropy = entropy
            self.metadata = metadata or {}

from topology import get_trajectory_state_phi_t, evaluate_field_E_x

# Import advanced GTMØ operators from axioms
from gtmo_axioms import (
    PsiOperator, EntropyOperator, ThresholdManager,
    MetaFeedbackLoop, OperatorType, create_gtmo_system
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # New dimensions (Point 1)
    PHYSICAL = auto()        # Physical reality dimension
    LOGICAL = auto()         # Logical consistency dimension
    
    # Special values for unknown dimensions
    UNKNOWN = auto()         # Nieznany, ale zaobserwowany wymiar
    EMERGENT = auto()        # Wymiar w trakcie odkrywania


class DynamicDimension:
    """Reprezentacja wymiarów odkrywanych w runtime (Point 1)"""
    
    def __init__(self, identifier: str, properties: Dict[str, Any]):
        self.id = identifier
        self.properties = properties
        self.discovery_time = time.time()
        self.observation_count = 0
        self.confidence_level = 0.0
        
    def __hash__(self):
        return hash(self.id)
        
    def __eq__(self, other):
        return isinstance(other, DynamicDimension) and self.id == other.id
        
    def __repr__(self):
        return f"DynamicDimension(id={self.id}, confidence={self.confidence_level:.3f})"


@dataclass
class EpistemicParticle(KnowledgeEntity):
    """
    Extended knowledge entity representing an EpistemicParticle (Ψᴱ).
    
    Implements Theorem TΨᴱ: adaptive state changes based on cognitive
    trajectory dynamics and epistemic entropy with inexhaustible dimension discovery.
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
    
    # New fields for inexhaustibility (Point 2)
    discovered_dimensions: Set[Union[EpistemicDimension, DynamicDimension]] = field(default_factory=set)
    dimension_discovery_potential: float = 0.5  # Prawdopodobieństwo odkrycia nowego wymiaru
    unknown_evolution_traces: List[Tuple[float, Any]] = field(default_factory=list)
    
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
        
    def _detect_unknown_dimension(self, parameter: float) -> Optional[DynamicDimension]:
        """Wykrywa potencjalnie nowy wymiar ewolucji (Point 2)"""
        if random.random() < self.dimension_discovery_potential:
            # Analiza śladów ewolucji
            pattern = self._analyze_evolution_pattern()
            if pattern and not self._matches_known_dimension(pattern):
                return DynamicDimension(
                    identifier=f"dim_{hash(str(pattern))}_{int(parameter*1000)}",
                    properties=pattern
                )
        return None
        
    def _analyze_evolution_pattern(self) -> Optional[Dict[str, Any]]:
        """Analizuje wzorce ewolucji dla wykrycia nowych wymiarów"""
        if len(self.state_history) < 3:
            return None
            
        # Analiza ostatnich 5 stanów
        recent_history = self.state_history[-5:]
        
        # Wykryj wzorce zmian
        determinacy_changes = []
        entropy_changes = []
        
        for i in range(1, len(recent_history)):
            det_change = abs(self.determinacy - 0.5)  # Simplified analysis
            ent_change = abs(self.entropy - 0.5)
            determinacy_changes.append(det_change)
            entropy_changes.append(ent_change)
            
        # Jeśli wzorzec nie pasuje do znanych wymiarów
        pattern_signature = {
            'determinacy_variance': np.var(determinacy_changes) if determinacy_changes else 0,
            'entropy_variance': np.var(entropy_changes) if entropy_changes else 0,
            'oscillation_frequency': self._calculate_oscillation_frequency(),
            'emergence_indicators': self._detect_emergence_indicators()
        }
        
        return pattern_signature
        
    def _calculate_oscillation_frequency(self) -> float:
        """Oblicza częstotliwość oscylacji w historii stanów"""
        if len(self.state_history) < 4:
            return 0.0
            
        # Simplified frequency analysis
        state_values = [hash(str(state)) % 100 for _, state in self.state_history[-10:]]
        if len(state_values) < 2:
            return 0.0
            
        # Count transitions
        transitions = sum(1 for i in range(1, len(state_values)) 
                         if state_values[i] != state_values[i-1])
        return transitions / len(state_values)
        
    def _detect_emergence_indicators(self) -> List[str]:
        """Wykrywa wskaźniki emergencji w ewolucji cząstki"""
        indicators = []
        
        if self.emergence_potential > 0.7:
            indicators.append('high_emergence_potential')
        if self.coherence_factor < 0.3:
            indicators.append('low_coherence')
        if len(self.discovered_dimensions) > 2:
            indicators.append('multi_dimensional')
            
        return indicators
        
    def _matches_known_dimension(self, pattern: Dict[str, Any]) -> bool:
        """Sprawdza, czy wzorzec pasuje do znanych wymiarów"""
        # Prosta heurystyka - w rzeczywistej implementacji byłaby bardziej złożona
        variance_threshold = 0.1
        frequency_threshold = 0.3
        
        det_var = pattern.get('determinacy_variance', 0)
        ent_var = pattern.get('entropy_variance', 0)
        freq = pattern.get('oscillation_frequency', 0)
        
        # Sprawdź charakterystyki znanych wymiarów
        if det_var < variance_threshold and ent_var < variance_threshold:
            return True  # Pasuje do TEMPORAL
        if freq > frequency_threshold:
            return True  # Pasuje do QUANTUM
            
        return False
        
    def _capture_state(self) -> Dict[str, Any]:
        """Przechwytuje aktualny stan cząstki do analizy"""
        return {
            'determinacy': self.determinacy,
            'stability': self.stability,
            'entropy': self.entropy,
            'epistemic_state': self.epistemic_state,
            'psi_score': self.psi_score,
            'cognitive_entropy': self.cognitive_entropy,
            'emergence_potential': self.emergence_potential,
            'coherence_factor': self.coherence_factor,
            'discovered_dimensions_count': len(self.discovered_dimensions)
        }

    def evolve(
        self,
        parameter: float,
        operators: Optional[Dict[str, Any]] = None
    ) -> 'EpistemicParticle':
        """
        Evolve the particle along its cognitive trajectory using GTMØ operators.
        Enhanced with dimension discovery (Point 3).
        
        Args:
            parameter: Evolution parameter (interpretation depends on epistemic_dimension)
            operators: Optional GTMØ operators for advanced evolution
            
        Returns:
            Evolved EpistemicParticle
        """
        # Store current state in history
        self.state_history.append((parameter, self.epistemic_state))
        
        # Sprawdzenie możliwości odkrycia nowego wymiaru (Point 3)
        new_dimension = self._detect_unknown_dimension(parameter)
        if new_dimension:
            self.discovered_dimensions.add(new_dimension)
            logger.info(f"Discovered new dimension: {new_dimension.id}")
            # Ewolucja w nowo odkrytym wymiarze
            self._evolve_in_unknown_dimension(parameter, new_dimension)
        
        # Zapisywanie śladów nieznanych ewolucji (Point 3)
        self.unknown_evolution_traces.append((parameter, self._capture_state()))
        
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
        elif self.epistemic_dimension == EpistemicDimension.PHYSICAL:
            self._evolve_physical(parameter)
        elif self.epistemic_dimension == EpistemicDimension.LOGICAL:
            self._evolve_logical(parameter)
        elif self.epistemic_dimension == EpistemicDimension.UNKNOWN:
            self._evolve_unknown_dimension(parameter)
                
        # Update epistemic state based on new properties
        self._update_epistemic_state()
        self._calculate_gtmo_metrics()
        
        # Handle collapse to singularity (AX1, AX5)
        if self.epistemic_state == EpistemicState.INDEFINITE and parameter > 1.0:
            if self._should_collapse_to_singularity():
                self.content = O
                self.epistemic_state = EpistemicState.INDEFINITE
            
        return self
        
    def _evolve_in_unknown_dimension(self, parameter: float, dimension: DynamicDimension) -> None:
        """Ewolucja w nowo odkrytym wymiarze"""
        # Eksperymentalna ewolucja na podstawie właściwości wymiaru
        properties = dimension.properties
        
        variance_factor = properties.get('determinacy_variance', 0.1)
        frequency_factor = properties.get('oscillation_frequency', 0.1)
        
        # Modyfikacja właściwości na podstawie charakterystyk wymiaru
        self.determinacy += variance_factor * math.sin(parameter * frequency_factor)
        self.stability += variance_factor * math.cos(parameter * frequency_factor)
        self.entropy = max(0.0, min(1.0, 1.0 - (self.determinacy + self.stability) / 2))
        
        # Zwiększ potencjał emergencji dla nieznanych wymiarów
        self.emergence_potential = min(1.0, self.emergence_potential + 0.1)
        
    def _apply_gtmo_operators(self, operators: Dict[str, Any], parameter: float) -> None:
        """Apply GTMØ operators to evolve particle properties."""
        if 'psi' in operators:
            psi_op = operators['psi']
            context = {'all_scores': operators.get('scores', []), 'timestamp': parameter}
            result = psi_op(self.content, context)
            self.psi_score = result.value['score']
            
        if 'entropy' in operators:
            entropy_op = operators['entropy']
            context = {'parameter': parameter}
            result = entropy_op(self.content, context)
            self.cognitive_entropy = result.value['total_entropy']
            
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
            
    def _evolve_physical(self, parameter: float) -> None:
        """Physical dimension evolution - new dimension (Point 1)"""
        # Physical constraints affect stability
        physical_resistance = 0.9  # Physical reality resistance to change
        self.stability *= physical_resistance
        # Physical entropy increases over time
        self.entropy = min(1.0, self.entropy + 0.01 * parameter)
        
    def _evolve_logical(self, parameter: float) -> None:
        """Logical dimension evolution - new dimension (Point 1)"""
        # Logical consistency affects determinacy
        logical_consistency = abs(math.sin(parameter * math.pi))
        self.determinacy = max(0.0, min(1.0, self.determinacy * logical_consistency))
        # Update entropy based on logical consistency
        self.entropy = 1.0 - self.determinacy
        
    def _evolve_unknown_dimension(self, parameter: float) -> None:
        """Evolution in unknown dimension"""
        # Chaotic evolution for unknown dimensions
        random_factor = random.uniform(-0.1, 0.1)
        self.determinacy = max(0.0, min(1.0, self.determinacy + random_factor))
        self.stability = max(0.0, min(1.0, self.stability + random_factor))
        self.entropy = max(0.0, min(1.0, self.entropy + random_factor))
        
    def _calculate_complexity(self, parameter: float) -> float:
        """Calculate complexity metric using GTMØ principles."""
        # Complexity as emergent property
        base_complexity = math.exp(parameter) - 1.0
        state_complexity = len(self.state_history) * 0.1
        dimension_complexity = len(self.discovered_dimensions) * 0.2
        return base_complexity + state_complexity + dimension_complexity
        
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


class TrajectoryObserver:
    """Obserwator trajektorii w nieznanych wymiarach (Point 4)"""
    
    def __init__(self):
        self.unclassified_patterns = []
        self.potential_dimensions = {}
        self.pattern_threshold = 0.7
        
    def observe(self, particle: EpistemicParticle, state_before: Any, state_after: Any):
        """Obserwuje zmiany, które nie pasują do znanych wymiarów"""
        delta = self._calculate_unexplained_delta(state_before, state_after)
        if delta:
            self.unclassified_patterns.append({
                'particle_id': id(particle),
                'delta': delta,
                'timestamp': time.time(),
                'parameter_context': len(particle.state_history)
            })
            
    def _calculate_unexplained_delta(self, state_before: Dict[str, Any], 
                                   state_after: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Oblicza niewyjaśnione zmiany między stanami"""
        if not state_before or not state_after:
            return None
            
        delta = {}
        for key in ['determinacy', 'stability', 'entropy']:
            if key in state_before and key in state_after:
                change = abs(state_after[key] - state_before[key])
                if change > 0.1:  # Znacząca zmiana
                    delta[key] = change
                    
        return delta if delta else None
            
    def hypothesize_dimensions(self) -> List[DynamicDimension]:
        """Próbuje zidentyfikować nowe wymiary na podstawie wzorców"""
        if len(self.unclassified_patterns) < 5:
            return []
            
        # Clustering niewyjaśnionych zmian
        dimension_candidates = {}
        
        for pattern in self.unclassified_patterns[-20:]:  # Ostatnie 20 wzorców
            signature = self._create_pattern_signature(pattern)
            signature_key = str(sorted(signature.items()))
            
            if signature_key not in dimension_candidates:
                dimension_candidates[signature_key] = {
                    'count': 0,
                    'properties': signature,
                    'particles': set()
                }
                
            dimension_candidates[signature_key]['count'] += 1
            dimension_candidates[signature_key]['particles'].add(pattern['particle_id'])
            
        # Identyfikacja powtarzających się wzorców
        new_dimensions = []
        for signature_key, candidate in dimension_candidates.items():
            if (candidate['count'] >= 3 and 
                len(candidate['particles']) >= 2):  # Minimum 3 obserwacje, 2 cząstki
                
                dimension_id = f"discovered_{hash(signature_key) % 10000}"
                new_dimension = DynamicDimension(
                    identifier=dimension_id,
                    properties=candidate['properties']
                )
                new_dimension.confidence_level = min(1.0, candidate['count'] / 10.0)
                new_dimensions.append(new_dimension)
                
        return new_dimensions
        
    def _create_pattern_signature(self, pattern: Dict[str, Any]) -> Dict[str, float]:
        """Tworzy sygnaturę wzorca dla klasyfikacji"""
        delta = pattern['delta']
        return {
            'determinacy_change_magnitude': delta.get('determinacy', 0.0),
            'stability_change_magnitude': delta.get('stability', 0.0),
            'entropy_change_magnitude': delta.get('entropy', 0.0),
            'total_change': sum(delta.values()),
            'change_ratio': max(delta.values()) / (min(delta.values()) + 0.001)
        }


class PrecognitiveAdapter:
    """Adaptuje system do trajektorii, które jeszcze nie mają danych (Point 5)"""
    
    def __init__(self):
        self.potential_trajectories = {}
        self.confidence_threshold = 0.3
        self.shadow_detection_sensitivity = 0.5
        
    def anticipate_trajectory(self, particle: EpistemicParticle) -> List[Tuple[Any, float]]:
        """Antycypuje możliwe trajektorie przed pojawieniem się danych"""
        # Analiza stanów cząstek i cieni
        shadows = self._identify_knowledge_shadows(particle)
        
        # Ekstrapolacja możliwych ścieżek
        potential_paths = []
        for shadow in shadows:
            path = self._extrapolate_from_shadow(shadow, particle)
            confidence = self._calculate_path_confidence(path, particle)
            if confidence > self.confidence_threshold:
                potential_paths.append((path, confidence))
                
        return potential_paths
        
    def _identify_knowledge_shadows(self, particle: EpistemicParticle) -> List[Dict[str, Any]]:
        """Identyfikuje cienie wiedzy - potencjalne trajektorie"""
        shadows = []
        
        # Analiza niedookreślonych obszarów w historii cząstki
        if len(particle.state_history) >= 3:
            # Szukaj luk w ewolucji
            for i in range(1, len(particle.state_history)):
                prev_time, prev_state = particle.state_history[i-1]
                curr_time, curr_state = particle.state_history[i]
                
                time_gap = curr_time - prev_time
                if time_gap > 0.5:  # Znacząca luka czasowa
                    shadow = {
                        'type': 'temporal_gap',
                        'start_time': prev_time,
                        'end_time': curr_time,
                        'start_state': prev_state,
                        'end_state': curr_state,
                        'gap_magnitude': time_gap,
                        'potential_states': self._interpolate_missing_states(prev_state, curr_state)
                    }
                    shadows.append(shadow)
        
        # Analiza niewyjaśnionych zmian w właściwościach
        for trace_time, trace_state in particle.unknown_evolution_traces[-5:]:
            if self._is_anomalous_state(trace_state, particle):
                shadow = {
                    'type': 'anomalous_evolution',
                    'time': trace_time,
                    'state': trace_state,
                    'anomaly_indicators': self._identify_anomaly_indicators(trace_state),
                    'potential_dimensions': self._suggest_missing_dimensions(trace_state)
                }
                shadows.append(shadow)
                
        return shadows
        
    def _interpolate_missing_states(self, start_state: EpistemicState, 
                                  end_state: EpistemicState) -> List[EpistemicState]:
        """Interpoluje brakujące stany między znanymi punktami"""
        if start_state == end_state:
            return [start_state]
            
        # Dla różnych stanów, sugeruj możliwe przejścia
        possible_transitions = []
        
        if start_state == EpistemicState.ONE and end_state == EpistemicState.INDEFINITE:
            possible_transitions = [EpistemicState.INFINITY, EpistemicState.ZERO]
        elif start_state == EpistemicState.ZERO and end_state == EpistemicState.INFINITY:
            possible_transitions = [EpistemicState.ONE, EpistemicState.INDEFINITE]
        else:
            # Domyślna interpolacja przez wszystkie możliwe stany
            all_states = [EpistemicState.ZERO, EpistemicState.ONE, 
                         EpistemicState.INFINITY, EpistemicState.INDEFINITE]
            possible_transitions = [s for s in all_states if s not in [start_state, end_state]]
            
        return possible_transitions
        
    def _is_anomalous_state(self, state: Dict[str, Any], particle: EpistemicParticle) -> bool:
        """Sprawdza, czy stan jest anomalny względem historii cząstki"""
        if not particle.state_history:
            return False
            
        # Porównaj z średnimi wartościami z historii
        avg_determinacy = particle.determinacy  # Simplified - should calculate from history
        avg_stability = particle.stability
        avg_entropy = particle.entropy
        
        current_det = state.get('determinacy', 0.5)
        current_stab = state.get('stability', 0.5)
        current_ent = state.get('entropy', 0.5)
        
        # Sprawdź odchylenia
        det_deviation = abs(current_det - avg_determinacy)
        stab_deviation = abs(current_stab - avg_stability)
        ent_deviation = abs(current_ent - avg_entropy)
        
        return (det_deviation > 0.3 or stab_deviation > 0.3 or ent_deviation > 0.3)
        
    def _identify_anomaly_indicators(self, state: Dict[str, Any]) -> List[str]:
        """Identyfikuje wskaźniki anomalii w stanie"""
        indicators = []
        
        determinacy = state.get('determinacy', 0.5)
        stability = state.get('stability', 0.5)
        entropy = state.get('entropy', 0.5)
        
        if determinacy > 0.9 and entropy > 0.9:
            indicators.append('paradoxical_high_determinacy_entropy')
        if stability < 0.1 and determinacy > 0.8:
            indicators.append('unstable_high_determinacy')
        if entropy < 0.1 and stability < 0.1:
            indicators.append('low_entropy_low_stability')
            
        return indicators
        
    def _suggest_missing_dimensions(self, state: Dict[str, Any]) -> List[str]:
        """Sugeruje brakujące wymiary na podstawie anomalnego stanu"""
        suggestions = []
        
        determinacy = state.get('determinacy', 0.5)
        stability = state.get('stability', 0.5)
        entropy = state.get('entropy', 0.5)
        
        if determinacy > 0.8 and entropy > 0.8:
            suggestions.append('paradox_resolution_dimension')
        if stability < 0.2:
            suggestions.append('stability_enhancement_dimension')
        if entropy < 0.1:
            suggestions.append('entropy_injection_dimension')
            
        return suggestions
        
    def _extrapolate_from_shadow(self, shadow: Dict[str, Any], 
                                particle: EpistemicParticle) -> Dict[str, Any]:
        """Ekstrapoluje trajektorię z cienia wiedzy"""
        shadow_type = shadow['type']
        
        if shadow_type == 'temporal_gap':
            return self._extrapolate_temporal_gap(shadow, particle)
        elif shadow_type == 'anomalous_evolution':
            return self._extrapolate_anomalous_evolution(shadow, particle)
        else:
            return {'type': 'unknown_extrapolation', 'confidence': 0.1}
            
    def _extrapolate_temporal_gap(self, shadow: Dict[str, Any], 
                                particle: EpistemicParticle) -> Dict[str, Any]:
        """Ekstrapoluje trajektorię dla luki czasowej"""
        start_state = shadow['start_state']
        end_state = shadow['end_state']
        gap_magnitude = shadow['gap_magnitude']
        
        # Sugerowana trajektoria na podstawie znanego początku i końca
        trajectory = {
            'type': 'temporal_interpolation',
            'start_state': start_state,
            'end_state': end_state,
            'suggested_path': shadow['potential_states'],
            'evolution_rate': gap_magnitude,
            'interpolation_method': 'state_transition_analysis'
        }
        
        return trajectory
        
    def _extrapolate_anomalous_evolution(self, shadow: Dict[str, Any], 
                                       particle: EpistemicParticle) -> Dict[str, Any]:
        """Ekstrapoluje trajektorię dla anomalnej ewolucji"""
        anomaly_indicators = shadow['anomaly_indicators']
        potential_dimensions = shadow['potential_dimensions']
        
        trajectory = {
            'type': 'anomaly_resolution',
            'anomaly_indicators': anomaly_indicators,
            'suggested_dimensions': potential_dimensions,
            'resolution_strategy': self._suggest_resolution_strategy(anomaly_indicators),
            'dimensional_intervention': len(potential_dimensions) > 0
        }
        
        return trajectory
        
    def _suggest_resolution_strategy(self, indicators: List[str]) -> str:
        """Sugeruje strategię rozwiązania anomalii"""
        if 'paradoxical_high_determinacy_entropy' in indicators:
            return 'paradox_resolution_via_dimensional_expansion'
        elif 'unstable_high_determinacy' in indicators:
            return 'stability_injection_via_coherence_enhancement'
        elif 'low_entropy_low_stability' in indicators:
            return 'entropy_stability_rebalancing'
        else:
            return 'general_dimensional_exploration'
            
    def _calculate_path_confidence(self, path: Dict[str, Any], 
                                 particle: EpistemicParticle) -> float:
        """Oblicza zaufanie do przewidywanej ścieżki"""
        base_confidence = 0.5
        
        # Zwiększ zaufanie na podstawie dostępnych danych
        if len(particle.state_history) > 5:
            base_confidence += 0.2
        if len(particle.unknown_evolution_traces) > 3:
            base_confidence += 0.1
        if len(particle.discovered_dimensions) > 0:
            base_confidence += 0.15
            
        # Dostosuj na podstawie typu ścieżki
        path_type = path.get('type', 'unknown')
        if path_type == 'temporal_interpolation':
            base_confidence += 0.1
        elif path_type == 'anomaly_resolution':
            base_confidence += 0.05
            
        return min(1.0, base_confidence)


# Minimal EpistemicParticleSystem stub for inheritance
class EpistemicParticleSystem:
    """Base system for managing epistemic particles"""
    
    def __init__(self, strict_mode: Optional[bool] = None):
        self.particles: List[EpistemicParticle] = []
        self.system_time: float = 0.0
        self.strict_mode = strict_mode if strict_mode is not None else STRICT_MODE
        
    def add_particle(self, particle: EpistemicParticle) -> None:
        """Add particle to system"""
        self.particles.append(particle)
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get basic system state"""
        if not self.particles:
            return {
                'particle_count': 0,
                'average_entropy': 0.0,
                'alienated_count': 0,
                'system_coherence': 0.0
            }
            
        total_entropy = sum(p.entropy for p in self.particles) / len(self.particles)
        alienated_count = sum(1 for p in self.particles 
                            if p.epistemic_state == EpistemicState.INDEFINITE)
        
        # Simple coherence calculation
        determinacy_values = [p.determinacy for p in self.particles]
        coherence = 1.0 - np.var(determinacy_values) if len(determinacy_values) > 1 else 1.0
        
        return {
            'particle_count': len(self.particles),
            'average_entropy': total_entropy,
            'alienated_count': alienated_count,
            'system_coherence': coherence
        }
        
    def _calculate_system_coherence(self) -> float:
        """Calculate system-wide coherence"""
        if len(self.particles) < 2:
            return 1.0
            
        coherence_factors = [p.coherence_factor for p in self.particles]
        return sum(coherence_factors) / len(coherence_factors)


class IntegratedEpistemicSystem(EpistemicParticleSystem):
    """
    Enhanced system integrating EpistemicParticles with full GTMØ framework.
    Enhanced with inexhaustible dimension discovery (Point 6).
    """
    
    def __init__(self, strict_mode: Optional[bool] = None):
        super().__init__(strict_mode)
        
        # Create GTMØ operators
        self.psi_op, self.entropy_op, self.meta_loop = create_gtmo_system()
        self.threshold_manager = self.meta_loop.threshold_manager
        
        # System metrics
        self.total_entropy_history: List[float] = []
        self.emergence_events: List[Tuple[float, EpistemicParticle]] = []
        
        # New components (Point 6)
        self.trajectory_observer = TrajectoryObserver()
        self.precognitive_adapter = PrecognitiveAdapter()
        self.discovered_dimensions = set()
        self.dimension_emergence_history = []
        
    def evolve_system(self, delta: float = 0.1) -> None:
        """
        Evolve all particles using GTMØ operators and meta-feedback.
        Enhanced with trajectory observation (Point 6).
        
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
        
        # Obserwacja nieznanych trajektorii (Point 6)
        for particle in self.particles:
            state_before = particle._capture_state()
            particle.evolve(self.system_time, operators)
            state_after = particle._capture_state()
            self.trajectory_observer.observe(particle, state_before, state_after)
            
        # Update system metrics
        self._update_system_metrics()
        
        # Hipotezowanie nowych wymiarów (Point 6)
        new_dimensions = self.trajectory_observer.hypothesize_dimensions()
        for dim in new_dimensions:
            if dim not in self.discovered_dimensions:
                self.discovered_dimensions.add(dim)
                self.dimension_emergence_history.append((self.system_time, dim))
                logger.info(f"System discovered new dimension: {dim.id} at t={self.system_time:.2f}")
        
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
        if results['final_state']['final_thresholds']:
            new_thresholds = results['final_state']['final_thresholds']
            # Thresholds are automatically updated in threshold_manager
            
    def get_detailed_state(self) -> Dict[str, Any]:
        """Get detailed system state including GTMØ metrics."""
        base_state = self.get_system_state()
        
        # Add GTMØ-specific metrics
        gtmo_metrics = {
            'particle_classifications': {},
            'entropy_evolution': self.total_entropy_history[-10:] if self.total_entropy_history else [],
            'emergence_count': len(self.emergence_events),
            'current_thresholds': self.threshold_manager.history[-1] if self.threshold_manager.history else (0.5, 0.5),
            'discovered_dimensions_count': len(self.discovered_dimensions),
            'dimension_emergence_timeline': [(t, d.id) for t, d in self.dimension_emergence_history[-5:]]
        }
        
        # Classify particles according to GTMØ
        for particle in self.particles:
            classification = particle.to_gtmo_classification()
            gtmo_metrics['particle_classifications'][classification] = \
                gtmo_metrics['particle_classifications'].get(classification, 0) + 1
                
        base_state['gtmo_metrics'] = gtmo_metrics
        return base_state
        
    def get_unknowability_metrics(self) -> Dict[str, Any]:
        """Metryki tego, czego system NIE wie o sobie (Point 7)"""
        known_dimensions = len([d for d in self.discovered_dimensions 
                              if isinstance(d, EpistemicDimension)])
        discovered_dimensions = len([d for d in self.discovered_dimensions 
                                   if isinstance(d, DynamicDimension)])
        
        return {
            'known_dimensions': len(EpistemicDimension),
            'discovered_dynamic_dimensions': discovered_dimensions,
            'unclassified_patterns': len(self.trajectory_observer.unclassified_patterns),
            'unknowability_index': self._calculate_unknowability_index(),
            'potential_undiscovered': 'infinite',  # Zgodnie z teorią
            'system_completeness': 'undecidable',  # Zgodnie z AX0
            'precognitive_trajectory_count': len(self.precognitive_adapter.potential_trajectories),
            'meta_uncertainty_level': self._calculate_meta_uncertainty(),
            'dimensional_exhaustion_impossibility': True,  # Fundamentalna właściwość
            'theoretical_dimension_bound': None  # Brak teoretycznego ograniczenia
        }
        
    def _calculate_unknowability_index(self) -> float:
        """Oblicza indeks niewiedzy systemu"""
        if not self.particles:
            return 1.0
            
        # Stosunek nieznanych wzorców do obserwowanych
        total_patterns = len(self.trajectory_observer.unclassified_patterns)
        total_observations = sum(len(p.state_history) for p in self.particles)
        
        if total_observations == 0:
            return 1.0
            
        unknowability = total_patterns / total_observations
        
        # Zwiększ indeks na podstawie odkrytych wymiarów (paradoks: więcej wiedzy = więcej niewiedzy)
        dimension_factor = len(self.discovered_dimensions) * 0.1
        
        return min(1.0, unknowability + dimension_factor)
        
    def _calculate_meta_uncertainty(self) -> float:
        """Oblicza meta-niepewność - niepewność o własnej niepewności"""
        # Im więcej system wie o swojej niewiedzy, tym większa meta-niepewność
        unknowability = self._calculate_unknowability_index()
        
        # Paradoks samoświadomości: świadomość granic zwiększa niepewność o granicach
        meta_uncertainty = unknowability * (1 - unknowability) * 4  # Maksimum przy 0.5
        
        return min(1.0, meta_uncertainty)


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
            psi_score = result.value['score']
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


def demonstrate_enhanced_evolution():
    """Demonstrate the enhanced evolution with inexhaustible dimensions."""
    print("=" * 80)
    print("ENHANCED EPISTEMIC PARTICLES WITH INEXHAUSTIBLE DIMENSIONS")
    print("=" * 80)
    
    # Create enhanced system
    system = IntegratedEpistemicSystem()
    
    # Create GTMØ operators for particle creation
    psi_op, _, _ = create_gtmo_system()
    
    # Create diverse particles with focus on dimension discovery
    test_contents = [
        "The fundamental theorem of calculus",
        "This statement is paradoxical and self-referential",
        AlienatedNumber("undefined_concept"),
        "Meta-knowledge about the limits of knowledge",
        O,
        "Quantum superposition in cognitive space",
        "Emergent pattern transcending known dimensions",
        "Unknown phenomenon requiring new framework",
        "Precognitive trajectory in unexplored space"
    ]
    
    dimensions = [
        EpistemicDimension.TEMPORAL,
        EpistemicDimension.EMERGENCE,
        EpistemicDimension.ENTROPIC,
        EpistemicDimension.QUANTUM,
        EpistemicDimension.TOPOLOGICAL,
        EpistemicDimension.COMPLEXITY,
        EpistemicDimension.COHERENCE,
        EpistemicDimension.UNKNOWN,
        EpistemicDimension.LOGICAL
    ]
    
    # Create and add particles
    for content, dimension in zip(test_contents, dimensions):
        particle = create_epistemic_particle_with_gtmo(
            content=content,
            dimension=dimension,
            psi_operator=psi_op
        )
        # Increase discovery potential for some particles
        if "unknown" in str(content).lower() or "emergent" in str(content).lower():
            particle.dimension_discovery_potential = 0.8
        system.add_particle(particle)
        
    # Initial state
    print("\n## INITIAL STATE")
    print("-" * 40)
    initial_state = system.get_detailed_state()
    unknowability = system.get_unknowability_metrics()
    
    print(f"Particles: {initial_state['particle_count']}")
    print(f"Classifications: {initial_state['gtmo_metrics']['particle_classifications']}")
    print(f"Average Entropy: {initial_state['average_entropy']:.3f}")
    print(f"Known Dimensions: {unknowability['known_dimensions']}")
    print(f"Unknowability Index: {unknowability['unknowability_index']:.3f}")
    
    # Evolution with dimension discovery
    print("\n## ENHANCED EVOLUTION PROCESS")
    print("-" * 40)
    
    for i in range(15):  # More steps to allow dimension discovery
        system.evolve_system(0.15)
        
        if i % 3 == 0:  # Print every third step
            state = system.get_detailed_state()
            unknowability = system.get_unknowability_metrics()
            
            print(f"\nStep {i+1}:")
            print(f"  Classifications: {state['gtmo_metrics']['particle_classifications']}")
            print(f"  System Coherence: {state['system_coherence']:.3f}")
            print(f"  Average Entropy: {state['average_entropy']:.3f}")
            print(f"  Emergent Events: {state['gtmo_metrics']['emergence_count']}")
            print(f"  Discovered Dimensions: {unknowability['discovered_dynamic_dimensions']}")
            print(f"  Unclassified Patterns: {unknowability['unclassified_patterns']}")
            print(f"  Unknowability Index: {unknowability['unknowability_index']:.3f}")
            print(f"  Meta-Uncertainty: {unknowability['meta_uncertainty_level']:.3f}")
            
    # Final comprehensive analysis
    print("\n## COMPREHENSIVE FINAL ANALYSIS")
    print("-" * 40)
    final_state = system.get_detailed_state()
    final_unknowability = system.get_unknowability_metrics()
    
    print(f"\nFinal Classifications:")
    for class_type, count in final_state['gtmo_metrics']['particle_classifications'].items():
        print(f"  {class_type}: {count}")
        
    print(f"\nSystem Metrics:")
    print(f"  Total Particles: {final_state['particle_count']}")
    print(f"  Alienated Particles: {final_state['alienated_count']}")
    print(f"  System Coherence: {final_state['system_coherence']:.3f}")
    print(f"  Final Entropy: {final_state['average_entropy']:.3f}")
    print(f"  Total Emergent Events: {final_state['gtmo_metrics']['emergence_count']}")
    
    print(f"\nDimension Discovery:")
    print(f"  Known Dimensions: {final_unknowability['known_dimensions']}")
    print(f"  Discovered Dynamic Dimensions: {final_unknowability['discovered_dynamic_dimensions']}")
    print(f"  Dimension Timeline: {final_state['gtmo_metrics']['dimension_emergence_timeline']}")
    
    print(f"\nUnknowability Analysis:")
    print(f"  Unknowability Index: {final_unknowability['unknowability_index']:.3f}")
    print(f"  Meta-Uncertainty Level: {final_unknowability['meta_uncertainty_level']:.3f}")
    print(f"  System Completeness: {final_unknowability['system_completeness']}")
    print(f"  Potential Undiscovered: {final_unknowability['potential_undiscovered']}")
    print(f"  Dimensional Exhaustion: {'Impossible' if final_unknowability['dimensional_exhaustion_impossibility'] else 'Possible'}")
    
    if system.emergence_events:
        print(f"\nEmergence Timeline:")
        for time, particle in system.emergence_events:
            print(f"  t={time:.1f}: {str(particle.content)[:60]}...")
            
    # Enhanced particle state details
    print("\n## ENHANCED PARTICLE STATES")
    print("-" * 40)
    for i, particle in enumerate(system.particles[:7]):  # First 7 particles
        print(f"\nParticle {i+1} ({particle.epistemic_dimension.name}):")
        print(f"  Content: {str(particle.content)[:50]}...")
        print(f"  State: {particle.epistemic_state.name}")
        print(f"  Classification: {particle.to_gtmo_classification()}")
        print(f"  Ψ_GTMØ score: {particle.psi_score:.3f}")
        print(f"  Cognitive Entropy: {particle.cognitive_entropy:.3f}")
        print(f"  Discovered Dimensions: {len(particle.discovered_dimensions)}")
        print(f"  Unknown Traces: {len(particle.unknown_evolution_traces)}")
        print(f"  Discovery Potential: {particle.dimension_discovery_potential:.3f}")
        
    return system


if __name__ == "__main__":
    # Run enhanced demonstration
    system = demonstrate_enhanced_evolution()
    
    print("\n" + "=" * 80)
    print("ENHANCED DEMONSTRATION COMPLETED!")
    print("Key Features Implemented:")
    print("  ✓ Inexhaustible dimension discovery")
    print("  ✓ Precognitive trajectory adaptation")
    print("  ✓ Unknown pattern observation")
    print("  ✓ Meta-reflection about unknowability")
    print("  ✓ Dynamic dimension emergence")
    print("  ✓ Paradoxical self-awareness of limits")
    print("=" * 80)
