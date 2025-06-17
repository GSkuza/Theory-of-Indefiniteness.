"""
GTMØ Topology - Implementation of AX5 "Ø ∈ ∂(CognitiveSpace)"
Boundary detection, trajectory evolution φ(t), and field evaluation E(x)
"""
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

class BoundaryType(Enum):
    """Types of cognitive space boundaries"""
    DEFINABILITY_BOUNDARY = "definability"      # Border between definable/indefinable
    COMPREHENSION_BOUNDARY = "comprehension"    # Limits of understanding
    LOGIC_BOUNDARY = "logic"                   # Where logic breaks down
    PARADOX_BOUNDARY = "paradox"               # Self-reference barriers
    EMERGENCE_BOUNDARY = "emergence"           # Novel pattern formation

@dataclass
class TrajectoryPoint:
    """Point along cognitive trajectory φ(t)"""
    time: float
    state: Any
    position: np.ndarray
    boundary_distance: float
    field_strength: float

class CognitiveSpaceTopology:
    """Complete topological framework for GTMØ"""
    
    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions  # 6D indefiniteness space
        self.boundary_points = []
        self.trajectory_cache = {}
        self.field_cache = {}
        self.omega_regions = []  # Regions where Ø appears
        
    def find_boundary_points(self, cognitive_system: List[Any]) -> List[Dict[str, Any]]:
        """Find ∂(CognitiveSpace) - where Ø emerges (AX5)"""
        boundary_points = []
        
        for i, fragment in enumerate(cognitive_system):
            # Calculate position in 6D space
            position = self._map_to_cognitive_coordinates(fragment)
            
            # Check for boundary indicators
            boundary_indicators = self._detect_boundary_indicators(fragment)
            
            if boundary_indicators['total_score'] > 0.7:
                boundary_point = {
                    'index': i,
                    'fragment': fragment,
                    'position': position,
                    'boundary_type': boundary_indicators['primary_type'],
                    'distance_to_omega': boundary_indicators['omega_proximity'],
                    'field_strength': self._evaluate_field_strength(position)
                }
                boundary_points.append(boundary_point)
                
                # If very close to boundary, mark as Ω region
                if boundary_indicators['omega_proximity'] < 0.1:
                    self.omega_regions.append(boundary_point)
        
        self.boundary_points = boundary_points
        return boundary_points
    
    def trajectory_phi_t(self, initial_state: Any, time_span: Tuple[float, float], 
                        steps: int = 100) -> List[TrajectoryPoint]:
        """Calculate cognitive trajectory φ(t) evolution"""
        t_start, t_end = time_span
        dt = (t_end - t_start) / steps
        trajectory = []
        
        current_state = initial_state
        
        for i in range(steps + 1):
            t = t_start + i * dt
            
            # Update state based on GTMØ evolution rules
            current_state = self._evolve_state(current_state, dt)
            
            # Calculate position in cognitive space
            position = self._map_to_cognitive_coordinates(current_state)
            
            # Calculate distance to nearest boundary
            boundary_distance = self._distance_to_boundary(position)
            
            # Evaluate field strength
            field_strength = self._evaluate_field_strength(position)
            
            trajectory_point = TrajectoryPoint(
                time=t,
                state=current_state,
                position=position,
                boundary_distance=boundary_distance,
                field_strength=field_strength
            )
            trajectory.append(trajectory_point)
            
            # Check for boundary crossing (approach to Ø)
            if boundary_distance < 0.05:
                # Approaching singularity
                current_state = self._handle_boundary_crossing(current_state, position)
        
        return trajectory
    
    def evaluate_field_E_x(self, entity: Any, field_type: str = "cognitive_entropy") -> float:
        """Evaluate GTMØ field E(x) at given entity"""
        cache_key = f"{field_type}_{str(entity)[:50]}"
        
        if cache_key in self.field_cache:
            return self.field_cache[cache_key]
        
        # Map entity to position
        position = self._map_to_cognitive_coordinates(entity)
        
        # Calculate field based on type
        if field_type == "cognitive_entropy":
            field_value = self._calculate_entropy_field(entity, position)
        elif field_type == "epistemic_purity":
            field_value = self._calculate_purity_field(entity, position)
        elif field_type == "proximity_to_singularity":
            field_value = self._calculate_proximity_field(position)
        elif field_type == "indefiniteness_gradient":
            field_value = self._calculate_indefiniteness_gradient(position)
        else:
            field_value = 0.0
        
        self.field_cache[cache_key] = field_value
        return field_value
    
    def boundary_crossing_detector(self, trajectory: List[TrajectoryPoint]) -> List[Dict[str, Any]]:
        """Detect when trajectory crosses cognitive boundaries"""
        crossings = []
        
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # Check for boundary crossing
            if (prev_point.boundary_distance > 0.1 and curr_point.boundary_distance <= 0.1):
                crossing = {
                    'time': curr_point.time,
                    'crossing_type': 'approach_to_boundary',
                    'position': curr_point.position,
                    'pre_state': prev_point.state,
                    'post_state': curr_point.state,
                    'field_change': curr_point.field_strength - prev_point.field_strength
                }
                crossings.append(crossing)
            
            # Check for emergence events (field strength spikes)
            field_change = abs(curr_point.field_strength - prev_point.field_strength)
            if field_change > 0.5:
                crossing = {
                    'time': curr_point.time,
                    'crossing_type': 'field_discontinuity',
                    'position': curr_point.position,
                    'field_jump': field_change,
                    'state': curr_point.state
                }
                crossings.append(crossing)
        
        return crossings
    
    def _map_to_cognitive_coordinates(self, entity: Any) -> np.ndarray:
        """Map cognitive entity to 6D coordinates"""
        entity_str = str(entity).lower()
        
        # Calculate 6 coordinates based on indefiniteness dimensions
        coords = np.zeros(6)
        
        # Semantic coordinate
        semantic_words = ["meaning", "concept", "idea", "definition"]
        coords[0] = sum(0.2 for word in semantic_words if word in entity_str)
        
        # Ontological coordinate  
        existence_words = ["exist", "being", "reality", "what is"]
        coords[1] = sum(0.2 for word in existence_words if word in entity_str)
        
        # Logical coordinate
        logic_words = ["if", "then", "therefore", "because", "implies"]
        coords[2] = sum(0.2 for word in logic_words if word in entity_str)
        
        # Temporal coordinate
        time_words = ["when", "always", "never", "sometimes", "eternal"]
        coords[3] = sum(0.2 for word in time_words if word in entity_str)
        
        # Paradox coordinate
        paradox_words = ["paradox", "contradiction", "self-reference"]
        coords[4] = sum(0.3 for word in paradox_words if word in entity_str)
        
        # Definability coordinate
        definable_words = ["define", "precise", "exact", "clear"]
        indefinable_words = ["undefined", "unclear", "vague", "mysterious"]
        coords[5] = sum(0.2 for word in definable_words if word in entity_str) - \
                   sum(0.2 for word in indefinable_words if word in entity_str)
        
        # Normalize to [0, 1] range
        coords = np.clip(coords, 0, 1)
        
        return coords
    
    def _detect_boundary_indicators(self, fragment: Any) -> Dict[str, Any]:
        """Detect indicators that fragment is near cognitive boundary"""
        text = str(fragment).lower()
        
        # Initialize scores
        scores = {
            BoundaryType.DEFINABILITY_BOUNDARY: 0.0,
            BoundaryType.COMPREHENSION_BOUNDARY: 0.0,
            BoundaryType.LOGIC_BOUNDARY: 0.0,
            BoundaryType.PARADOX_BOUNDARY: 0.0,
            BoundaryType.EMERGENCE_BOUNDARY: 0.0
        }
        
        # Definability boundary indicators
        if any(word in text for word in ["undefined", "indefinable", "meaningless"]):
            scores[BoundaryType.DEFINABILITY_BOUNDARY] += 0.4
        
        # Comprehension boundary indicators
        if any(word in text for word in ["incomprehensible", "beyond understanding", "cannot grasp"]):
            scores[BoundaryType.COMPREHENSION_BOUNDARY] += 0.4
        
        # Logic boundary indicators
        if any(word in text for word in ["illogical", "contradiction", "absurd"]):
            scores[BoundaryType.LOGIC_BOUNDARY] += 0.4
        
        # Paradox boundary indicators
        if any(phrase in text for phrase in ["this statement", "self-referential", "liar paradox"]):
            scores[BoundaryType.PARADOX_BOUNDARY] += 0.5
        
        # Emergence boundary indicators
        if any(word in text for word in ["emergent", "novel", "unprecedented"]):
            scores[BoundaryType.EMERGENCE_BOUNDARY] += 0.3
        
        # Find primary boundary type
        primary_type = max(scores.keys(), key=lambda k: scores[k])
        total_score = sum(scores.values()) / len(scores)
        
        # Calculate proximity to Ω (closer to 0 = closer to singularity)
        omega_proximity = 1.0 - total_score
        
        return {
            'scores': scores,
            'primary_type': primary_type,
            'total_score': total_score,
            'omega_proximity': omega_proximity
        }
    
    def _evolve_state(self, state: Any, dt: float) -> Any:
        """Evolve cognitive state according to GTMØ dynamics"""
        state_str = str(state)
        
        # Simple evolution rules
        if "undefined" in state_str.lower():
            # Undefined concepts may evolve toward AlienatedNumbers
            if np.random.random() < 0.1 * dt:
                return f"ℓ∅({state_str})"
        
        if "paradox" in state_str.lower():
            # Paradoxes may collapse to Ø
            if np.random.random() < 0.05 * dt:
                return "Ø"
        
        return state  # No evolution by default
    
    def _distance_to_boundary(self, position: np.ndarray) -> float:
        """Calculate distance to nearest cognitive boundary"""
        if not self.boundary_points: