"""gtmo-core-v2.py
Enhanced GTMØ Implementation with Dynamic Values, Topological Classification,
Executable Axioms, and Adaptive Learning.

Major improvements over v1:
- Dynamic context-aware values instead of arbitrary constants
- Topological phase space classification instead of percentage thresholds  
- Executable axioms that transform system state
- Real learning through memory consolidation
"""

from __future__ import annotations

import os
import math
import numpy as np
import pickle
from numbers import Number
from typing import Any, Final, Dict, List, Optional, Tuple, Union, Callable
from functools import wraps
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Defense network learning disabled.")

__all__ = [
    "O", "Singularity", "AlienatedNumber", "SingularityError",
    "ExecutableAxiom", "TopologicalClassifier", "AdaptiveGTMONeuron",
    "KnowledgeEntity", "KnowledgeType", "EpistemicParticle"
]

###############################################################################
# Configuration
###############################################################################

STRICT_MODE: Final[bool] = os.getenv("GTM_STRICT", "0") == "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Errors
###############################################################################

class SingularityError(ArithmeticError):
    """Raised when operations with Ø or ℓ∅ are disallowed in strict mode."""


###############################################################################
# Enhanced Ontological Singularity (Ø)
###############################################################################

class _SingletonMeta(ABCMeta):
    """Metaclass enforcing the singleton pattern."""
    _instance: "Singularity | None" = None

    def __call__(cls, *args: Any, **kwargs: Any) -> "Singularity":
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


def _absorbing_operation(method_name: str):
    """Decorator for arithmetic operations that collapse to Ø."""
    def decorator(fn_placeholder: Any):
        @wraps(fn_placeholder)
        def wrapper(self: "Singularity | AlienatedNumber", *args: Any, **kwargs: Any) -> "Singularity":
            if STRICT_MODE:
                op_source = "Ø" if isinstance(self, Singularity) else "ℓ∅"
                raise SingularityError(
                    f"Operation '{method_name}' with {op_source} is forbidden in STRICT mode"
                )
            return get_singularity()
        return wrapper
    return decorator


class Singularity(Number, metaclass=_SingletonMeta):
    """Ontological singularity – an absorbing element in GTMØ arithmetic."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "O_empty_singularity"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Singularity)

    def __hash__(self) -> int:
        return hash("O_empty_singularity")

    def __reduce__(self):
        return (get_singularity, ())

    # Absorbing arithmetic operations
    __add__ = _absorbing_operation("__add__")
    __radd__ = _absorbing_operation("__radd__")
    __sub__ = _absorbing_operation("__sub__")
    __rsub__ = _absorbing_operation("__rsub__")
    __mul__ = _absorbing_operation("__mul__")
    __rmul__ = _absorbing_operation("__rmul__")
    __truediv__ = _absorbing_operation("__truediv__")
    __rtruediv__ = _absorbing_operation("__rtruediv__")
    __pow__ = _absorbing_operation("__pow__")
    __rpow__ = _absorbing_operation("__rpow__")

    def to_json(self) -> str:
        return '"O_empty_singularity"'


def get_singularity() -> "Singularity":
    """Return the unique global Ø instance."""
    return Singularity()


O: Final[Singularity] = get_singularity()

###############################################################################
# Enhanced Alienated Numbers with Dynamic Context
###############################################################################

class AlienatedNumber(Number):
    """
    Enhanced alienated number with dynamic, context-aware properties.
    Instead of fixed 0.999... values, properties are calculated based on context.
    """
    __slots__ = ("identifier", "context", "_psi_score", "_entropy", "_semantic_cache")

    def __init__(self, identifier: str | int | float | None = None, context: Dict[str, Any] = None):
        self.identifier = identifier if identifier is not None else "anonymous"
        self.context = context or {}
        self._psi_score: Optional[float] = None
        self._entropy: Optional[float] = None
        self._semantic_cache: Dict[str, float] = {}

    def __repr__(self) -> str:
        return f"l_empty_num({self.identifier})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AlienatedNumber) and self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(("l_empty_num", self.identifier))

    def psi_gtm_score(self) -> float:
        """Calculate PSI score based on context and semantic analysis."""
        if self._psi_score is None:
            semantic_distance = self._calculate_semantic_distance()
            relational_score = self._calculate_relational_coherence()
            temporal_decay = self._calculate_temporal_decay()
            
            # Combine factors with learned weights
            weights = self.context.get('learned_weights', [0.4, 0.3, 0.3])
            factors = [1.0 / (1.0 + semantic_distance),
                      relational_score,
                      1.0 / (1.0 + temporal_decay)]
            
            self._psi_score = sum(w * f for w, f in zip(weights, factors))
            self._psi_score = max(0.001, min(0.999, self._psi_score))
        
        return self._psi_score

    def e_gtm_entropy(self) -> float:
        """Calculate entropy based on indefiniteness factors."""
        if self._entropy is None:
            # Entropy increases with uncertainty factors
            uncertainty_factors = []
            
            # Temporal uncertainty
            if 'temporal_distance' in self.context:
                t_dist = self.context['temporal_distance']
                uncertainty_factors.append(1 - math.exp(-0.5 * t_dist))
            
            # Conceptual volatility
            if 'volatility' in self.context:
                uncertainty_factors.append(self.context['volatility'])
            
            # Predictability
            if 'predictability' in self.context:
                uncertainty_factors.append(1 - self.context['predictability'])
            
            if uncertainty_factors:
                self._entropy = np.mean(uncertainty_factors)
            else:
                # Default entropy based on identifier analysis
                self._entropy = self._default_entropy_analysis()
        
        return self._entropy

    def _calculate_semantic_distance(self) -> float:
        """Calculate semantic distance from definable concepts."""
        identifier_str = str(self.identifier).lower()
        
        # Cache for efficiency
        if 'semantic_distance' in self._semantic_cache:
            return self._semantic_cache['semantic_distance']
        
        distance = 0.0
        
        # Temporal indefiniteness
        temporal_markers = ['future', 'will be', 'prediction', 'forecast', 'tomorrow']
        for marker in temporal_markers:
            if marker in identifier_str:
                distance += 5.0
        
        # Paradoxical nature
        paradox_markers = ['paradox', 'contradiction', 'impossible', 'undefined']
        for marker in paradox_markers:
            if marker in identifier_str:
                distance += 3.0
        
        # Meta-cognitive markers
        meta_markers = ['consciousness', 'self-aware', 'meta', 'recursive']
        for marker in meta_markers:
            if marker in identifier_str:
                distance += 2.0
        
        # Context-specific distances
        if 'domain' in self.context:
            domain_distances = {
                'quantum': 2.0,
                'consciousness': 3.0,
                'future_prediction': 5.0,
                'mathematical_paradox': 4.0
            }
            distance += domain_distances.get(self.context['domain'], 1.0)
        
        self._semantic_cache['semantic_distance'] = distance
        return distance

    def _calculate_relational_coherence(self) -> float:
        """Calculate coherence based on relationships with other concepts."""
        if 'relations' not in self.context:
            return 0.5  # Default neutral coherence
        
        relations = self.context['relations']
        if not relations:
            return 0.5
        
        # Calculate coherence as inverse of contradiction count
        contradiction_count = sum(1 for r in relations if r.get('type') == 'contradicts')
        support_count = sum(1 for r in relations if r.get('type') == 'supports')
        
        total_relations = len(relations)
        coherence = (support_count - contradiction_count) / (total_relations + 1)
        
        return (coherence + 1.0) / 2.0  # Normalize to [0, 1]

    def _calculate_temporal_decay(self) -> float:
        """Calculate decay factor based on temporal distance."""
        if 'temporal_distance' not in self.context:
            return 0.0
        
        t = self.context['temporal_distance']
        decay_rate = self.context.get('decay_rate', 0.5)
        
        return t * decay_rate

    def _default_entropy_analysis(self) -> float:
        """Fallback entropy calculation based on identifier."""
        identifier_str = str(self.identifier).lower()
        
        # High entropy indicators
        high_entropy_markers = ['random', 'chaos', 'unknown', 'undefined', 'quantum']
        entropy_score = 0.1  # Base entropy
        
        for marker in high_entropy_markers:
            if marker in identifier_str:
                entropy_score += 0.15
        
        return min(0.99, entropy_score)

    # Arithmetic operations collapse to Singularity
    @_absorbing_operation("__add__")
    def __add__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__radd__")
    def __radd__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__sub__")
    def __sub__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__rsub__")
    def __rsub__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__mul__")
    def __mul__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__rmul__")
    def __rmul__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__truediv__")
    def __truediv__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__rtruediv__")
    def __rtruediv__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__pow__")
    def __pow__(self, other: Any) -> Singularity: ...

    @_absorbing_operation("__rpow__")
    def __rpow__(self, other: Any) -> Singularity: ...

    def to_json(self) -> str:
        return f'"{self.__repr__()}"'


###############################################################################
# Executable Axioms
###############################################################################

class ExecutableAxiom(ABC):
    """Base class for axioms that can be executed, not just verified."""
    
    @abstractmethod
    def apply(self, system_state: Any) -> Any:
        """Apply axiom transformation to system state."""
        pass
    
    @abstractmethod
    def verify(self, system_state: Any) -> bool:
        """Verify if system state satisfies axiom."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the axiom."""
        pass


class AX0_SystemicUncertainty(ExecutableAxiom):
    """
    Axiom 0: Fundamental uncertainty as an active principle.
    The system's definability is inherently uncertain.
    """
    
    @property
    def description(self) -> str:
        return "There is no proof that the GTMØ system is fully definable"
    
    def apply(self, system_state: Any) -> Any:
        """Introduce quantum superposition to all definable states."""
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if not hasattr(neuron, 'quantum_state'):
                    neuron.quantum_state = self._create_superposition(neuron)
                else:
                    # Evolve existing superposition
                    neuron.quantum_state = self._evolve_superposition(neuron.quantum_state)
        
        # Add system-level uncertainty
        if not hasattr(system_state, 'foundational_mode'):
            system_state.foundational_mode = np.random.choice(['stillness', 'flux'])
        
        return system_state
    
    def _create_superposition(self, neuron) -> Dict[str, complex]:
        """Create quantum state |ψ⟩ = α|defined⟩ + β|undefined⟩ + γ|indefinite⟩"""
        # Amplitudes based on current classical state
        alpha = np.sqrt(neuron.determinacy) * np.exp(1j * np.random.rand() * 2 * np.pi)
        beta = np.sqrt(1 - neuron.determinacy) * np.exp(1j * np.random.rand() * 2 * np.pi)
        gamma = np.sqrt(neuron.entropy) * 0.1 * np.exp(1j * np.random.rand() * 2 * np.pi)
        
        # Normalization
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2 + abs(gamma)**2)
        
        return {
            'defined': alpha / norm,
            'undefined': beta / norm,
            'indefinite': gamma / norm
        }
    
    def _evolve_superposition(self, quantum_state: Dict[str, complex]) -> Dict[str, complex]:
        """Unitary evolution of quantum state."""
        # Simple rotation in complex plane
        rotation = np.exp(1j * 0.1)  # Small phase evolution
        
        return {
            key: value * rotation for key, value in quantum_state.items()
        }
    
    def verify(self, system_state: Any) -> bool:
        """Check if system maintains foundational uncertainty."""
        if not hasattr(system_state, 'neurons'):
            return True  # Vacuously true
        
        # System should have quantum states
        quantum_neurons = sum(1 for n in system_state.neurons if hasattr(n, 'quantum_state'))
        
        return quantum_neurons > 0


class AX1_OntologicalDifference(ExecutableAxiom):
    """Ø is fundamentally different from {0, 1, ∞}."""
    
    @property
    def description(self) -> str:
        return "Ø ∉ {0, 1, ∞} and no function maps standard numbers to Ø"
    
    def apply(self, system_state: Any) -> Any:
        """Ensure Ø maintains its unique properties."""
        # This axiom is more of a constraint than a transformation
        # It prevents certain operations rather than changing state
        return system_state
    
    def verify(self, system_state: Any) -> bool:
        """Verify Ø remains distinct from standard mathematical objects."""
        # Check that singularity operations don't produce standard numbers
        test_ops = [
            (O + O, "addition"),
            (O * O, "multiplication"),
            (O / O if not STRICT_MODE else None, "division")
        ]
        
        for result, op_name in test_ops:
            if result is not None and result in {0, 1, float('inf'), -float('inf')}:
                logger.warning(f"AX1 violated: Ø {op_name} produced {result}")
                return False
        
        return True


class AX6_MinimalEntropy(ExecutableAxiom):
    """Ø has minimal cognitive entropy - it's the most certain uncertainty."""
    
    @property
    def description(self) -> str:
        return "E_GTMØ(Ø) = min E_GTMØ(x) for all x in KnowledgeDomain"
    
    def apply(self, system_state: Any) -> Any:
        """Apply entropy minimization gradient to states approaching Ø."""
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if self._is_approaching_singularity(neuron):
                    # Apply gradient descent on entropy
                    grad = self._entropy_gradient(neuron)
                    neuron.entropy = max(0, neuron.entropy - 0.01 * grad)
                    
                    # As entropy approaches 0, state approaches Ø
                    if neuron.entropy < 0.001:
                        neuron.is_singularity = True
        
        return system_state
    
    def _is_approaching_singularity(self, neuron) -> bool:
        """Check if neuron is on trajectory toward Ø."""
        if hasattr(neuron, 'trajectory_history') and len(neuron.trajectory_history) > 2:
            recent_entropy = [h['entropy'] for h in neuron.trajectory_history[-3:]]
            return all(recent_entropy[i] > recent_entropy[i+1] for i in range(len(recent_entropy)-1))
        
        return neuron.determinacy > 0.9 and neuron.stability > 0.9
    
    def _entropy_gradient(self, neuron) -> float:
        """Calculate entropy gradient for descent."""
        # Shannon entropy gradient: -Σ(log p + 1)
        if hasattr(neuron, 'indefiniteness'):
            values = neuron.indefiniteness.unpack().values()
            grad = sum(-math.log(max(x, 1e-10)) - 1 for x in values if x > 0)
            return grad / len(values)
        
        return 0.5  # Default gradient
    
    def verify(self, system_state: Any) -> bool:
        """Verify Ø maintains minimal entropy."""
        if hasattr(system_state, 'neurons'):
            singularity_neurons = [n for n in system_state.neurons if getattr(n, 'is_singularity', False)]
            other_neurons = [n for n in system_state.neurons if not getattr(n, 'is_singularity', False)]
            
            if singularity_neurons and other_neurons:
                min_singularity_entropy = min(n.entropy for n in singularity_neurons)
                min_other_entropy = min(n.entropy for n in other_neurons)
                
                return min_singularity_entropy <= min_other_entropy
        
        return True


###############################################################################
# Topological Classification System
###############################################################################

class KnowledgeType(Enum):
    """Enhanced knowledge types with topological properties."""
    SINGULARITY = "Ø"
    ALIENATED = "ℓ∅"
    PARTICLE = "Ψᴷ"
    SHADOW = "Ψʰ"
    EMERGENT = "Ψᴺ"
    LIMINAL = "Ψᴧ"
    META_INDEFINITE = "Ψ∅∅"
    VOID = "Ψ◊"
    FLUX = "Ψ~"
    TRANSCENDENT = "Ψ↑"


@dataclass
class KnowledgeEntity:
    """Enhanced knowledge entity with topological properties."""
    content: Any
    determinacy: float = 0.5
    stability: float = 0.5
    entropy: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    phase_coordinates: Optional[Tuple[float, float, float]] = None
    trajectory_history: List[Dict[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize phase space coordinates."""
        if self.phase_coordinates is None:
            self.phase_coordinates = (self.determinacy, self.stability, self.entropy)
    
    def to_phase_point(self) -> Tuple[float, float, float]:
        """Convert to point in phase space."""
        return (self.determinacy, self.stability, self.entropy)


class TopologicalClassifier:
    """Classifier using topological attractors instead of thresholds."""
    
    def __init__(self):
        self.attractors = self._initialize_attractors()
        self.phase_history = []
        self.attractor_strengths = {}
    
    def _initialize_attractors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize topological attractors in phase space."""
        return {
            'singularity': {
                'center': (1.0, 1.0, 0.0),
                'basin_radius': 0.15,
                'type': KnowledgeType.SINGULARITY,
                'strength': 2.0  # Strongest attractor
            },
            'particle': {
                'center': (0.85, 0.85, 0.15),
                'basin_radius': 0.25,
                'type': KnowledgeType.PARTICLE,
                'strength': 1.0
            },
            'shadow': {
                'center': (0.15, 0.15, 0.85),
                'basin_radius': 0.25,
                'type': KnowledgeType.SHADOW,
                'strength': 1.0
            },
            'emergent': {
                'center': (0.5, 0.3, 0.9),
                'basin_radius': 0.2,
                'type': KnowledgeType.EMERGENT,
                'strength': 1.2
            },
            'alienated': {
                'center': (0.999, 0.999, 0.001),
                'basin_radius': 0.1,
                'type': KnowledgeType.ALIENATED,
                'strength': 1.5
            },
            'void': {
                'center': (0.0, 0.0, 0.5),
                'basin_radius': 0.2,
                'type': KnowledgeType.VOID,
                'strength': 0.8
            }
        }
    
    def classify(self, entity: Union[KnowledgeEntity, Any]) -> KnowledgeType:
        """Classify entity based on topological phase space."""
        # Handle special cases
        if entity is O:
            return KnowledgeType.SINGULARITY
        if isinstance(entity, AlienatedNumber):
            return KnowledgeType.ALIENATED
        
        # Convert to phase point
        if isinstance(entity, KnowledgeEntity):
            phase_point = entity.to_phase_point()
        else:
            # Create temporary entity for classification
            phase_point = self._estimate_phase_point(entity)
        
        # Calculate distances to all attractors using Wasserstein metric
        distances = {}
        for name, attractor in self.attractors.items():
            distance = self._wasserstein_distance(phase_point, attractor['center'])
            # Weight by attractor strength (stronger = pulls from further)
            effective_distance = distance / attractor['strength']
            distances[name] = effective_distance
        
        # Find nearest attractor
        nearest = min(distances, key=distances.get)
        nearest_distance = distances[nearest]
        
        # Check if we're in the basin of attraction
        if nearest_distance <= self.attractors[nearest]['basin_radius']:
            classification = self.attractors[nearest]['type']
        else:
            # We're in a liminal region between attractors
            classification = self._classify_liminal_region(phase_point, distances)
        
        # Record for adaptation
        self.phase_history.append({
            'point': phase_point,
            'classification': classification,
            'distances': distances
        })
        
        return classification
    
    def _wasserstein_distance(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """Calculate Wasserstein (Earth Mover's) distance."""
        # For 1D marginals, this simplifies to L2 distance
        # For full implementation, would use optimal transport
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
    
    def _classify_liminal_region(self, phase_point: Tuple[float, ...], 
                                 distances: Dict[str, float]) -> KnowledgeType:
        """Classify points not clearly in any attractor basin."""
        # Sort distances
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        nearest_two = sorted_distances[:2]
        
        # If very close to two attractors, it's liminal
        if nearest_two[1][1] - nearest_two[0][1] < 0.1:
            return KnowledgeType.LIMINAL
        
        # Check for flux conditions (high entropy, medium determinacy)
        if phase_point[2] > 0.7 and 0.3 < phase_point[0] < 0.7:
            return KnowledgeType.FLUX
        
        # Check for transcendent conditions (leaving normal phase space)
        if any(coord > 1.0 or coord < 0.0 for coord in phase_point):
            return KnowledgeType.TRANSCENDENT
        
        # Default to liminal
        return KnowledgeType.LIMINAL
    
    def _estimate_phase_point(self, entity: Any) -> Tuple[float, float, float]:
        """Estimate phase coordinates for arbitrary entities."""
        entity_str = str(entity).lower()
        
        # Simple heuristic estimation
        determinacy = 0.5
        stability = 0.5
        entropy = 0.5
        
        # Adjust based on content analysis
        if any(word in entity_str for word in ['certain', 'always', 'never', 'must']):
            determinacy += 0.3
            entropy -= 0.2
        
        if any(word in entity_str for word in ['maybe', 'possibly', 'might', 'could']):
            determinacy -= 0.3
            entropy += 0.2
        
        if any(word in entity_str for word in ['paradox', 'contradiction', 'impossible']):
            stability -= 0.4
            entropy += 0.3
        
        # Normalize
        determinacy = max(0, min(1, determinacy))
        stability = max(0, min(1, stability))
        entropy = max(0, min(1, entropy))
        
        return (determinacy, stability, entropy)
    
    def adapt_attractors(self, feedback: List[Tuple[KnowledgeEntity, KnowledgeType]]):
        """Adapt attractor positions based on classification feedback."""
        if not feedback:
            return
        
        # Group feedback by classification
        grouped = {}
        for entity, expected_type in feedback:
            if expected_type not in grouped:
                grouped[expected_type] = []
            grouped[expected_type].append(entity.to_phase_point())
        
        # Update attractor centers using feedback
        for type_name, points in grouped.items():
            attractor_name = self._find_attractor_by_type(type_name)
            if attractor_name and len(points) > 3:
                # Calculate new center as weighted mean
                old_center = self.attractors[attractor_name]['center']
                new_center = np.mean(points, axis=0)
                
                # Exponential moving average
                alpha = 0.1  # Learning rate
                updated_center = tuple(
                    alpha * new + (1 - alpha) * old 
                    for new, old in zip(new_center, old_center)
                )
                
                self.attractors[attractor_name]['center'] = updated_center
    
    def _find_attractor_by_type(self, knowledge_type: KnowledgeType) -> Optional[str]:
        """Find attractor name by knowledge type."""
        for name, attractor in self.attractors.items():
            if attractor['type'] == knowledge_type:
                return name
        return None


###############################################################################
# Adaptive Learning System
###############################################################################

class AdaptiveGTMONeuron:
    """Neuron with real learning capabilities through memory consolidation."""
    
    def __init__(self, neuron_id: str, position: Tuple[int, int, int]):
        self.id = neuron_id
        self.position = position
        
        # Basic properties
        self.determinacy = np.random.uniform(0.2, 0.8)
        self.stability = np.random.uniform(0.2, 0.8)
        self.entropy = np.random.uniform(0.1, 0.9)
        self.is_singularity = False
        self.is_alienated = False
        
        # Learning components
        self.long_term_memory = {
            'successful_defenses': [],
            'vulnerability_patterns': [],
            'adaptation_weights': np.ones(10) * 0.5,
            'experience_embeddings': []
        }
        
        self.defense_strategies = {
            'absorb': 0.25,      # Absorb attack into indefiniteness
            'deflect': 0.25,     # Deflect to neighboring neurons
            'rigidify': 0.25,    # Increase determinacy temporarily
            'dissolve': 0.25     # Become more indefinite
        }
        
        # Initialize defense network if available
        self.defense_network = self._init_defense_network() if TORCH_AVAILABLE else None
        self.trajectory_history = []
        self.last_defense_strategy = None
    
    def _init_defense_network(self):
        """Initialize neural network for learning defense strategies."""
        if not TORCH_AVAILABLE:
            return None
        
        class DefenseNet(nn.Module):
            def __init__(self, input_dim=6, hidden_dim=32, output_dim=4):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x, hidden=None):
                # x shape: (batch, seq_len, features)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x, hidden = self.lstm(x, hidden)
                x = self.fc2(x[:, -1, :])  # Use last hidden state
                return torch.softmax(x, dim=-1), hidden
        
        return DefenseNet()
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as vector for learning."""
        return np.array([
            self.determinacy,
            self.stability,
            self.entropy,
            float(self.is_singularity),
            float(self.is_alienated),
            len(self.trajectory_history) / 100.0  # Normalized history length
        ])
    
    def experience_attack(self, attack_type: str, attack_vector: Dict[str, float], 
                         intensity: float = 1.0) -> Dict[str, Any]:
        """Experience an attack and learn from it."""
        # Store pre-attack state
        pre_state = self.get_state_vector().copy()
        pre_determinacy = self.determinacy
        
        # Choose defense strategy
        if self.defense_network and len(self.trajectory_history) > 5:
            # Use learned strategy
            strategy = self._learned_defense_strategy(attack_type, attack_vector)
        else:
            # Use probabilistic strategy
            strategy = self._probabilistic_defense_strategy()
        
        self.last_defense_strategy = strategy
        
        # Apply defense
        defense_result = self._apply_defense(strategy, attack_type, attack_vector, intensity)
        
        # Store post-attack state
        post_state = self.get_state_vector().copy()
        
        # Evaluate defense success
        success_metrics = self._evaluate_defense_success(
            pre_state, post_state, pre_determinacy, attack_type
        )
        
        # Learn from experience
        self._consolidate_experience({
            'attack_type': attack_type,
            'attack_vector': attack_vector,
            'intensity': intensity,
            'defense_strategy': strategy,
            'pre_state': pre_state,
            'post_state': post_state,
            'success_metrics': success_metrics,
            'timestamp': len(self.trajectory_history)
        })
        
        return {
            'defense_used': strategy,
            'success': success_metrics['overall_success'],
            'state_change': post_state - pre_state,
            'metrics': success_metrics
        }
    
    def _learned_defense_strategy(self, attack_type: str, attack_vector: Dict[str, float]) -> str:
        """Use neural network to select defense strategy."""
        if not TORCH_AVAILABLE or not self.defense_network:
            return self._probabilistic_defense_strategy()
        
        # Prepare input
        state = self.get_state_vector()
        attack_features = np.array([
            attack_vector.get('semantic_attack', 0),
            attack_vector.get('logical_attack', 0),
            attack_vector.get('entropy_attack', 0),
            hash(attack_type) % 100 / 100.0  # Attack type embedding
        ])
        
        features = np.concatenate([state, attack_features[:2]])  # Limit to input_dim
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            strategy_probs, _ = self.defense_network(features_tensor)
            strategy_idx = strategy_probs.argmax().item()
        
        strategies = list(self.defense_strategies.keys())
        return strategies[strategy_idx]
    
    def _probabilistic_defense_strategy(self) -> str:
        """Select defense strategy probabilistically."""
        strategies = list(self.defense_strategies.keys())
        probabilities = list(self.defense_strategies.values())
        return np.random.choice(strategies, p=probabilities)
    
    def _apply_defense(self, strategy: str, attack_type: str, 
                      attack_vector: Dict[str, float], intensity: float) -> Dict[str, Any]:
        """Apply selected defense strategy."""
        result = {'strategy': strategy, 'modifications': {}}
        
        if strategy == 'absorb':
            # Absorb attack into increased indefiniteness
            absorption_rate = 0.3 * (1 - self.entropy)
            self.entropy = min(1.0, self.entropy + absorption_rate * intensity)
            self.determinacy = max(0.0, self.determinacy - absorption_rate * intensity * 0.5)
            result['modifications'] = {'entropy': absorption_rate, 'determinacy': -absorption_rate * 0.5}
            
        elif strategy == 'deflect':
            # Deflect attack (reduce its effect)
            deflection_efficiency = self.stability * 0.7
            actual_intensity = intensity * (1 - deflection_efficiency)
            
            # Apply reduced attack
            for key, value in attack_vector.items():
                if key == 'semantic_attack':
                    self.determinacy = max(0.0, self.determinacy - value * actual_intensity)
                elif key == 'entropy_attack':
                    self.entropy = min(1.0, self.entropy + value * actual_intensity)
            
            result['modifications'] = {'deflection_efficiency': deflection_efficiency}
            
        elif strategy == 'rigidify':
            # Temporarily increase determinacy and stability
            boost = 0.2 * intensity
            self.determinacy = min(1.0, self.determinacy + boost)
            self.stability = min(1.0, self.stability + boost)
            self.entropy = max(0.0, self.entropy - boost * 0.5)
            result['modifications'] = {'determinacy': boost, 'stability': boost}
            
        elif strategy == 'dissolve':
            # Become more indefinite to avoid attack
            dissolution_rate = 0.4
            self.entropy = min(1.0, self.entropy + dissolution_rate)
            self.determinacy = self.determinacy * (1 - dissolution_rate)
            self.stability = self.stability * (1 - dissolution_rate * 0.5)
            result['modifications'] = {'dissolution': dissolution_rate}
        
        return result
    
    def _evaluate_defense_success(self, pre_state: np.ndarray, post_state: np.ndarray,
                                 pre_determinacy: float, attack_type: str) -> Dict[str, float]:
        """Evaluate how successful the defense was."""
        metrics = {}
        
        # State preservation metric (how well we maintained our state)
        state_change = np.linalg.norm(post_state - pre_state)
        metrics['state_preservation'] = 1.0 / (1.0 + state_change)
        
        # Determinacy preservation (important for knowledge particles)
        if pre_determinacy > 0.7:  # Was a knowledge particle
            determinacy_preserved = self.determinacy / pre_determinacy
            metrics['determinacy_preservation'] = min(1.0, determinacy_preserved)
        else:
            metrics['determinacy_preservation'] = 0.5  # Neutral
        
        # Survival metric (didn't collapse to void or singularity unexpectedly)
        if not self.is_singularity and self.entropy < 0.95:
            metrics['survival'] = 1.0
        else:
            metrics['survival'] = 0.3
        
        # Attack-specific success
        if attack_type == 'anti_paradox' and self.entropy > 0.5:
            metrics['attack_specific'] = 1.0  # Maintained indefiniteness against certainty attack
        elif attack_type == 'overflow' and state_change < 0.5:
            metrics['attack_specific'] = 1.0  # Resisted overflow
        else:
            metrics['attack_specific'] = 0.5
        
        # Overall success (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]
        metrics['overall_success'] = sum(
            w * metrics[k] for w, k in zip(weights, 
            ['state_preservation', 'determinacy_preservation', 'survival', 'attack_specific'])
        )
        
        return metrics
    
    def _consolidate_experience(self, experience: Dict[str, Any]):
        """Consolidate experience into long-term memory."""
        success = experience['success_metrics']['overall_success']
        
        if success > 0.7:
            self.long_term_memory['successful_defenses'].append(experience)
            # Update defense strategy weights
            strategy = experience['defense_strategy']
            self.defense_strategies[strategy] = min(0.9, self.defense_strategies[strategy] + 0.05)
        else:
            self.long_term_memory['vulnerability_patterns'].append(experience)
            # Decrease weight of failed strategy
            strategy = experience['defense_strategy']
            self.defense_strategies[strategy] = max(0.1, self.defense_strategies[strategy] - 0.05)
        
        # Normalize strategy probabilities
        total = sum(self.defense_strategies.values())
        for key in self.defense_strategies:
            self.defense_strategies[key] /= total
        
        # Create experience embedding for pattern matching
        embedding = self._create_experience_embedding(experience)
        self.long_term_memory['experience_embeddings'].append(embedding)
        
        # Prune old memories if too many (keep last 1000)
        for key in ['successful_defenses', 'vulnerability_patterns']:
            if len(self.long_term_memory[key]) > 500:
                self.long_term_memory[key] = self.long_term_memory[key][-500:]
        
        # Train defense network if available
        if self.defense_network and len(self.long_term_memory['successful_defenses']) > 10:
            self._train_defense_network()
    
    def _create_experience_embedding(self, experience: Dict[str, Any]) -> np.ndarray:
        """Create a compact embedding of an experience."""
        # Simple embedding: concatenate key features
        embedding = np.concatenate([
            experience['pre_state'],
            experience['post_state'],
            [experience['intensity']],
            [experience['success_metrics']['overall_success']],
            [hash(experience['attack_type']) % 100 / 100.0]
        ])
        
        return embedding
    
    def _train_defense_network(self):
        """Train the defense network on recent experiences."""
        if not TORCH_AVAILABLE or not self.defense_network:
            return
        
        # Prepare training data from successful defenses
        successful = self.long_term_memory['successful_defenses'][-50:]
        if len(successful) < 10:
            return
        
        # Create training batch
        inputs = []
        targets = []
        
        for exp in successful:
            # Input: state + attack features
            state = exp['pre_state']
            attack_features = np.array([
                exp['attack_vector'].get('semantic_attack', 0),
                exp['attack_vector'].get('logical_attack', 0)
            ])
            input_vec = np.concatenate([state, attack_features[:2]])[:6]  # Ensure correct size
            inputs.append(input_vec)
            
            # Target: one-hot encoding of successful strategy
            strategies = list(self.defense_strategies.keys())
            target = np.zeros(4)
            target[strategies.index(exp['defense_strategy'])] = 1.0
            targets.append(target)
        
        # Convert to tensors
        inputs_tensor = torch.FloatTensor(inputs).unsqueeze(1)  # Add seq dimension
        targets_tensor = torch.FloatTensor(targets)
        
        # Simple training step
        optimizer = torch.optim.Adam(self.defense_network.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for _ in range(10):  # Quick training
            optimizer.zero_grad()
            outputs, _ = self.defense_network(inputs_tensor)
            loss = criterion(outputs, targets_tensor)
            loss.backward()
            optimizer.step()
    
    def get_learned_patterns(self) -> Dict[str, Any]:
        """Extract learned patterns from memory."""
        patterns = {
            'total_experiences': (len(self.long_term_memory['successful_defenses']) + 
                                len(self.long_term_memory['vulnerability_patterns'])),
            'success_rate': 0.0,
            'preferred_strategies': dict(self.defense_strategies),
            'vulnerability_profile': {},
            'strength_profile': {}
        }
        
        # Calculate success rate
        total = patterns['total_experiences']
        if total > 0:
            success_count = len(self.long_term_memory['successful_defenses'])
            patterns['success_rate'] = success_count / total
        
        # Analyze vulnerability patterns
        for vuln in self.long_term_memory['vulnerability_patterns'][-20:]:
            attack_type = vuln['attack_type']
            if attack_type not in patterns['vulnerability_profile']:
                patterns['vulnerability_profile'][attack_type] = 0
            patterns['vulnerability_profile'][attack_type] += 1
        
        # Analyze strength patterns
        for success in self.long_term_memory['successful_defenses'][-20:]:
            defense = success['defense_strategy']
            if defense not in patterns['strength_profile']:
                patterns['strength_profile'][defense] = 0
            patterns['strength_profile'][defense] += 1
        
        return patterns


###############################################################################
# Epistemic State Management
###############################################################################

class EpistemicState(Enum):
    """Possible epistemic states for particles."""
    ZERO = 0                    # Minimal epistemic content
    ONE = 1                     # Maximal epistemic determinacy
    INFINITY = float('inf')     # Unbounded epistemic expansion
    INDEFINITE = 'Ø'           # Epistemic indefiniteness


@dataclass
class EpistemicParticle(KnowledgeEntity):
    """Enhanced epistemic particle with learning capabilities."""
    epistemic_state: EpistemicState = EpistemicState.ONE
    quantum_state: Optional[Dict[str, complex]] = None
    defense_history: List[Dict[str, Any]] = field(default_factory=list)
    learned_weights: np.ndarray = field(default_factory=lambda: np.ones(3) * 0.33)
    
    def evolve(self, parameter: float, context: Optional[Dict[str, Any]] = None) -> 'EpistemicParticle':
        """Evolve particle with quantum and classical dynamics."""
        # Store current state
        self.trajectory_history.append({
            'parameter': parameter,
            'determinacy': self.determinacy,
            'stability': self.stability,
            'entropy': self.entropy,
            'epistemic_state': self.epistemic_state
        })
        
        # Apply quantum evolution if in superposition
        if self.quantum_state:
            self._evolve_quantum(parameter)
        
        # Classical evolution
        self._evolve_classical(parameter, context)
        
        # Update epistemic state
        self._update_epistemic_state()
        
        # Check for phase transitions
        self._check_phase_transitions()
        
        return self
    
    def _evolve_quantum(self, parameter: float):
        """Evolve quantum state."""
        if not self.quantum_state:
            return
        
        # Unitary evolution
        phase = parameter * 0.1
        for key in self.quantum_state:
            self.quantum_state[key] *= np.exp(1j * phase)
        
        # Measurement probability affects classical state
        measurement_prob = abs(self.quantum_state.get('defined', 0))**2
        self.determinacy = self.determinacy * 0.9 + measurement_prob * 0.1
    
    def _evolve_classical(self, parameter: float, context: Optional[Dict[str, Any]]):
        """Classical evolution dynamics."""
        if context and 'field' in context:
            # External field influence
            field_strength = context['field'].get('strength', 0.1)
            field_direction = context['field'].get('direction', [0, 0, 1])
            
            # Apply field effects
            self.determinacy += field_strength * field_direction[0] * 0.01
            self.stability += field_strength * field_direction[1] * 0.01
            self.entropy += field_strength * field_direction[2] * 0.01
        else:
            # Autonomous evolution
            # Natural decay toward equilibrium
            equilibrium = {'determinacy': 0.5, 'stability': 0.5, 'entropy': 0.5}
            decay_rate = 0.01
            
            self.determinacy += decay_rate * (equilibrium['determinacy'] - self.determinacy)
            self.stability += decay_rate * (equilibrium['stability'] - self.stability)
            self.entropy += decay_rate * (equilibrium['entropy'] - self.entropy)
        
        # Ensure bounds
        self.determinacy = max(0, min(1, self.determinacy))
        self.stability = max(0, min(1, self.stability))
        self.entropy = max(0, min(1, self.entropy))
    
    def _update_epistemic_state(self):
        """Update epistemic state based on current properties."""
        # High determinacy and stability → ONE
        if self.determinacy > 0.9 and self.stability > 0.9 and self.entropy < 0.1:
            self.epistemic_state = EpistemicState.ONE
        
        # Low everything → ZERO
        elif self.determinacy < 0.1 and self.stability < 0.1:
            self.epistemic_state = EpistemicState.ZERO
        
        # High entropy with medium determinacy → INFINITY (unbounded)
        elif self.entropy > 0.8 and 0.3 < self.determinacy < 0.7:
            self.epistemic_state = EpistemicState.INFINITY
        
        # Otherwise → INDEFINITE
        else:
            self.epistemic_state = EpistemicState.INDEFINITE
    
    def _check_phase_transitions(self):
        """Check for sudden phase transitions."""
        if len(self.trajectory_history) < 2:
            return
        
        # Compare with previous state
        prev = self.trajectory_history[-2]
        curr = self.trajectory_history[-1]
        
        # Detect sudden jumps
        determinacy_jump = abs(curr['determinacy'] - prev['determinacy']) > 0.3
        state_change = curr['epistemic_state'] != prev['epistemic_state']
        
        if determinacy_jump or state_change:
            self.metadata['phase_transition'] = {
                'at_parameter': curr['parameter'],
                'from_state': prev['epistemic_state'],
                'to_state': curr['epistemic_state']
            }


###############################################################################
# System Integration
###############################################################################

class GTMOSystemV2:
    """Enhanced GTMØ system with all improvements."""
    
    def __init__(self):
        # Core components
        self.classifier = TopologicalClassifier()
        self.axioms = [
            AX0_SystemicUncertainty(),
            AX1_OntologicalDifference(),
            AX6_MinimalEntropy()
        ]
        
        # Neurons
        self.neurons = []
        self.epistemic_particles = []
        
        # System state
        self.iteration = 0
        self.phase_space_history = []
        
        logger.info("GTMØ System v2.0 initialized with enhanced capabilities")
    
    def add_neuron(self, neuron: AdaptiveGTMONeuron):
        """Add a neuron to the system."""
        self.neurons.append(neuron)
    
    def add_particle(self, particle: EpistemicParticle):
        """Add an epistemic particle."""
        self.epistemic_particles.append(particle)
    
    def evolve(self):
        """Evolve the entire system one step."""
        self.iteration += 1
        
        # Apply axioms
        for axiom in self.axioms:
            axiom.apply(self)
        
        # Evolve particles
        for particle in self.epistemic_particles:
            particle.evolve(self.iteration / 10.0)
        
        # Classify all entities
        classifications = {}
        for particle in self.epistemic_particles:
            class_type = self.classifier.classify(particle)
            classifications[particle] = class_type
        
        # Record phase space state
        self.phase_space_history.append({
            'iteration': self.iteration,
            'classifications': classifications,
            'phase_distribution': self._calculate_phase_distribution()
        })
    
    def _calculate_phase_distribution(self) -> Dict[str, int]:
        """Calculate distribution of entities in phase space."""
        distribution = {}
        
        for particle in self.epistemic_particles:
            class_type = self.classifier.classify(particle)
            if class_type not in distribution:
                distribution[class_type] = 0
            distribution[class_type] += 1
        
        return distribution
    
    def simulate_attack(self, attack_type: str, target_neurons: List[int], intensity: float = 1.0):
        """Simulate an adversarial attack on specified neurons."""
        results = []
        
        attack_vector = self._generate_attack_vector(attack_type)
        
        for idx in target_neurons:
            if 0 <= idx < len(self.neurons):
                neuron = self.neurons[idx]
                result = neuron.experience_attack(attack_type, attack_vector, intensity)
                results.append({
                    'neuron_id': neuron.id,
                    'result': result,
                    'learned_patterns': neuron.get_learned_patterns()
                })
        
        return results
    
    def _generate_attack_vector(self, attack_type: str) -> Dict[str, float]:
        """Generate attack vector based on type."""
        vectors = {
            'anti_paradox': {'semantic_attack': 0.8, 'logical_attack': 0.9, 'entropy_attack': -0.7},
            'overflow': {'semantic_attack': 2.0, 'logical_attack': 2.0, 'entropy_attack': 2.0},
            'confusion': {'semantic_attack': 0.5, 'logical_attack': -0.5, 'entropy_attack': 0.8},
            'rigid_logic': {'semantic_attack': -0.3, 'logical_attack': -0.9, 'entropy_attack': -0.8}
        }
        
        return vectors.get(attack_type, {'semantic_attack': 0.5, 'logical_attack': 0.5, 'entropy_attack': 0.5})
    
    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        report = {
            'iteration': self.iteration,
            'total_neurons': len(self.neurons),
            'total_particles': len(self.epistemic_particles),
            'phase_distribution': self._calculate_phase_distribution() if self.epistemic_particles else {},
            'axiom_compliance': {},
            'learning_summary': {}
        }
        
        # Check axiom compliance
        for axiom in self.axioms:
            report['axiom_compliance'][axiom.__class__.__name__] = axiom.verify(self)
        
        # Aggregate learning statistics
        if self.neurons:
            total_experiences = sum(n.get_learned_patterns()['total_experiences'] for n in self.neurons)
            avg_success_rate = np.mean([n.get_learned_patterns()['success_rate'] for n in self.neurons])
            
            report['learning_summary'] = {
                'total_experiences': total_experiences,
                'average_success_rate': avg_success_rate,
                'neurons_with_experience': sum(1 for n in self.neurons if n.get_learned_patterns()['total_experiences'] > 0)
            }
        
        return report


###############################################################################
# Demonstration
###############################################################################

def demonstrate_v2_improvements():
    """Demonstrate the key improvements in GTMØ v2."""
    print("=" * 80)
    print("GTMØ v2.0 - Enhanced Implementation Demonstration")
    print("=" * 80)
    
    # 1. Dynamic Alienated Numbers
    print("\n1. DYNAMIC CONTEXT-AWARE VALUES")
    print("-" * 40)
    
    # Old way (commented for comparison)
    # alien_old = AlienatedNumber("bitcoin_2030")  # Always 0.999999...
    
    # New way
    alien_btc = AlienatedNumber("bitcoin_2030", context={
        'temporal_distance': 5.5,
        'volatility': 0.9,
        'predictability': 0.1,
        'domain': 'future_prediction'
    })
    
    alien_math = AlienatedNumber("sqrt(-1)", context={
        'domain': 'mathematical_paradox',
        'relations': [{'type': 'contradicts', 'with': 'real_numbers'}]
    })
    
    print(f"Bitcoin 2030 prediction:")
    print(f"  PSI Score: {alien_btc.psi_gtm_score():.4f}")
    print(f"  Entropy: {alien_btc.e_gtm_entropy():.4f}")
    
    print(f"\nSquare root of -1:")
    print(f"  PSI Score: {alien_math.psi_gtm_score():.4f}")
    print(f"  Entropy: {alien_math.e_gtm_entropy():.4f}")
    
    # 2. Topological Classification
    print("\n\n2. TOPOLOGICAL PHASE SPACE CLASSIFICATION")
    print("-" * 40)
    
    classifier = TopologicalClassifier()
    
    # Create test entities
    entities = [
        KnowledgeEntity("The sun rises in the east", 0.95, 0.92, 0.08),
        KnowledgeEntity("This statement is false", 0.5, 0.1, 0.9),
        KnowledgeEntity("Tomorrow might rain", 0.3, 0.4, 0.7),
        KnowledgeEntity("Quantum superposition", 0.6, 0.5, 0.8)
    ]
    
    print("Classifications in topological phase space:")
    for entity in entities:
        classification = classifier.classify(entity)
        phase_point = entity.to_phase_point()
        print(f"  '{entity.content[:30]}...'")
        print(f"    Phase point: {phase_point}")
        print(f"    Classification: {classification.value}")
    
    # 3. Executable Axioms
    print("\n\n3. EXECUTABLE AXIOMS IN ACTION")
    print("-" * 40)
    
    system = GTMOSystemV2()
    
    # Add some neurons
    for i in range(5):
        neuron = AdaptiveGTMONeuron(f"neuron_{i}", (i, 0, 0))
        system.add_neuron(neuron)
    
    # Apply axioms
    print("Applying AX0 (Systemic Uncertainty)...")
    ax0 = AX0_SystemicUncertainty()
    ax0.apply(system)
    
    # Check quantum states
    quantum_count = sum(1 for n in system.neurons if hasattr(n, 'quantum_state'))
    print(f"  Neurons with quantum states: {quantum_count}/{len(system.neurons)}")
    
    # 4. Adaptive Learning
    print("\n\n4. ADAPTIVE LEARNING THROUGH EXPERIENCE")
    print("-" * 40)
    
    # Simulate attacks and learning
    print("Simulating adversarial attacks...")
    
    attack_results = system.simulate_attack('anti_paradox', [0, 1, 2], intensity=0.8)
    
    for result in attack_results:
        print(f"\nNeuron {result['neuron_id']}:")
        print(f"  Defense used: {result['result']['defense_used']}")
        print(f"  Success: {result['result']['success']:.3f}")
        print(f"  State change: {result['result']['state_change']}")
        
        patterns = result['learned_patterns']
        if patterns['total_experiences'] > 0:
            print(f"  Learning summary:")
            print(f"    Total experiences: {patterns['total_experiences']}")
            print(f"    Success rate: {patterns['success_rate']:.2%}")
            print(f"    Preferred strategies: {patterns['preferred_strategies']}")
    
    # 5. System Evolution
    print("\n\n5. SYSTEM EVOLUTION")
    print("-" * 40)
    
    # Add epistemic particles
    particles = [
        EpistemicParticle("Certain fact", 0.9, 0.9, 0.1),
        EpistemicParticle("Paradox", 0.5, 0.2, 0.9),
        EpistemicParticle("Emerging pattern", 0.6, 0.4, 0.7)
    ]
    
    for particle in particles:
        system.add_particle(particle)
    
    # Evolve system
    print("Evolving system for 5 iterations...")
    for i in range(5):
        system.evolve()
        
        if i % 2 == 0:
            report = system.get_system_report()
            print(f"\nIteration {report['iteration']}:")
            print(f"  Phase distribution: {report['phase_distribution']}")
            print(f"  Axiom compliance: {report['axiom_compliance']}")
    
    print("\n" + "=" * 80)
    print("GTMØ v2.0 - Improvements Summary:")
    print("- Dynamic context-aware values (not arbitrary constants)")
    print("- Topological attractors (not percentage thresholds)")
    print("- Executable axioms (not just philosophical statements)")
    print("- Real learning through experience (not deterministic formulas)")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_v2_improvements()
