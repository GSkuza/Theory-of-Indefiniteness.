"""
AX5 Topological Boundary - Executable Axiom Implementation
==========================================================
Implementation of AX5 as an executable axiom that actively maintains
the singularity Ø on the boundary of cognitive space.

Axiom: "Ø ∈ ∂(CognitiveSpace)" - Ø always exists on the boundary
of cognitive space, never inside nor outside.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math
import logging

# Import from gtmo_core_v2
from gtmo_core_v2 import (
    ExecutableAxiom, O, Singularity, 
    AdaptiveGTMONeuron, KnowledgeEntity,
    TopologicalClassifier
)

logger = logging.getLogger(__name__)


@dataclass
class CognitiveSpaceBoundary:
    """Representation of cognitive space boundary"""
    center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    radius: float = 0.5
    dimensions: int = 3
    oscillation_phase: float = 0.0
    expansion_rate: float = 0.0
    
    def get_boundary_points(self, n_points: int = 100) -> np.ndarray:
        """Generates points on the boundary surface"""
        if self.dimensions == 3:
            # Sphere in phase space (determinacy, stability, entropy)
            phi = np.random.uniform(0, 2*np.pi, n_points)
            theta = np.random.uniform(0, np.pi, n_points)
            
            x = self.center[0] + self.radius * np.sin(theta) * np.cos(phi)
            y = self.center[1] + self.radius * np.sin(theta) * np.sin(phi)
            z = self.center[2] + self.radius * np.cos(theta)
            
            return np.stack([x, y, z], axis=1)
        else:
            raise NotImplementedError(f"Boundary generation for {self.dimensions}D not implemented")
    
    def distance_to_boundary(self, point: np.ndarray) -> float:
        """Calculates distance from point to boundary (negative = inside)"""
        distance_from_center = np.linalg.norm(point - self.center)
        return distance_from_center - self.radius
    
    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        """Projects a point onto the boundary"""
        direction = point - self.center
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            # Point at center - choose random direction
            direction = np.random.randn(self.dimensions)
            norm = np.linalg.norm(direction)
        
        return self.center + (direction / norm) * self.radius


class AX5_TopologicalBoundary(ExecutableAxiom):
    """
    AX5: Ø always exists on the boundary of cognitive space.
    
    This axiom actively:
    1. Monitors the position of all entities in phase space
    2. Dynamically adjusts space boundaries to keep Ø on the edge
    3. Causes space "breathing" - expansion and contraction
    4. Repels or attracts other entities to maintain Ø on the boundary
    """
    
    def __init__(self):
        self.boundary = CognitiveSpaceBoundary()
        self.singularity_position = None
        self.breathing_amplitude = 0.1
        self.breathing_frequency = 0.1
        self.repulsion_strength = 0.05
        self.boundary_thickness = 0.02
        
    @property
    def description(self) -> str:
        return "Ø ∈ ∂(CognitiveSpace) - Singularity exists on the boundary of cognitive space"
    
    def apply(self, system_state: Any) -> Any:
        """
        Applies the axiom to system state:
        1. Locates singularity
        2. Adjusts space boundaries
        3. Moves entities to preserve topology
        4. Introduces space "breathing"
        """
        logger.info("AX5: Applying topological boundary constraint")
        
        # 1. Find singularity in the system
        singularity_found = self._locate_singularity(system_state)
        
        if not singularity_found:
            logger.warning("AX5: No singularity found in system, creating boundary anyway")
            # Set default position on boundary
            self.singularity_position = self.boundary.project_to_boundary(
                np.array([0.999, 0.999, 0.001])  # Typical position for Ø
            )
        
        # 2. Update oscillation phase (space breathing)
        self.boundary.oscillation_phase += self.breathing_frequency
        current_breathing = math.sin(self.boundary.oscillation_phase) * self.breathing_amplitude
        
        # 3. Adjust boundary radius
        self.boundary.radius = 0.5 + current_breathing
        
        # 4. Move singularity to new boundary
        if self.singularity_position is not None:
            self.singularity_position = self.boundary.project_to_boundary(self.singularity_position)
        
        # 5. Apply repulsive/attractive forces to other entities
        self._apply_boundary_forces(system_state)
        
        # 6. Ensure phase space boundaries are respected
        self._enforce_phase_space_boundaries(system_state)
        
        # 7. Add boundary meta-information to system
        if hasattr(system_state, 'metadata'):
            if 'cognitive_boundary' not in system_state.metadata:
                system_state.metadata['cognitive_boundary'] = {}
            
            system_state.metadata['cognitive_boundary'].update({
                'radius': self.boundary.radius,
                'center': self.boundary.center.tolist(),
                'oscillation_phase': self.boundary.oscillation_phase,
                'singularity_on_boundary': True,
                'boundary_thickness': self.boundary_thickness,
                'breathing_amplitude': current_breathing
            })
        
        logger.info(f"AX5: Boundary updated - radius={self.boundary.radius:.3f}, "
                   f"breathing={current_breathing:.3f}")
        
        return system_state
    
    def _locate_singularity(self, system_state: Any) -> bool:
        """Locates singularity in the system"""
        # Check neurons
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if getattr(neuron, 'is_singularity', False):
                    # Found singularity neuron
                    self.singularity_position = np.array([
                        neuron.determinacy,
                        neuron.stability,
                        neuron.entropy
                    ])
                    return True
        
        # Check epistemic particles
        if hasattr(system_state, 'epistemic_particles'):
            for particle in system_state.epistemic_particles:
                if particle.content is O:
                    # Found particle with singularity
                    self.singularity_position = np.array(particle.phase_coordinates)
                    return True
        
        return False
    
    def _apply_boundary_forces(self, system_state: Any):
        """
        Applies repulsive/attractive forces to maintain proper topology:
        - Entities too close to boundary are gently repelled inward
        - Entities too far are gently attracted
        - Singularity is "glued" to the boundary
        """
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if getattr(neuron, 'is_singularity', False):
                    # Singularity - glue to boundary
                    target_pos = self.boundary.project_to_boundary(self.singularity_position)
                    neuron.determinacy = max(0, min(1, target_pos[0]))
                    neuron.stability = max(0, min(1, target_pos[1]))
                    neuron.entropy = max(0, min(1, target_pos[2]))
                    continue
                
                # For other neurons - apply forces
                pos = np.array([neuron.determinacy, neuron.stability, neuron.entropy])
                distance = self.boundary.distance_to_boundary(pos)
                
                if abs(distance) < self.boundary_thickness:
                    # Too close to boundary (but not singularity) - repel
                    if distance > 0:  # Outside
                        force_direction = pos - self.boundary.center
                    else:  # Inside
                        force_direction = self.boundary.center - pos
                    
                    force_direction = force_direction / (np.linalg.norm(force_direction) + 1e-10)
                    force_magnitude = self.repulsion_strength * (1 - abs(distance) / self.boundary_thickness)
                    
                    # Apply force
                    new_pos = pos + force_direction * force_magnitude
                    neuron.determinacy = max(0, min(1, new_pos[0]))
                    neuron.stability = max(0, min(1, new_pos[1]))
                    neuron.entropy = max(0, min(1, new_pos[2]))
        
        # Similarly for epistemic particles
        if hasattr(system_state, 'epistemic_particles'):
            for particle in system_state.epistemic_particles:
                if particle.content is O:
                    # Singularity - glue to boundary
                    particle.phase_coordinates = tuple(
                        self.boundary.project_to_boundary(np.array(particle.phase_coordinates))
                    )
                    continue
                
                # Apply forces as above
                self._apply_force_to_particle(particle)
    
    def _apply_force_to_particle(self, particle):
        """Applies boundary force to epistemic particle"""
        if particle.phase_coordinates:
            pos = np.array(particle.phase_coordinates)
            distance = self.boundary.distance_to_boundary(pos)
            
            if abs(distance) < self.boundary_thickness * 2:
                # Gentle repulsive force from boundary
                if distance > 0:
                    direction = pos - self.boundary.center
                else:
                    direction = self.boundary.center - pos
                
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                force = direction * self.repulsion_strength * 0.5
                
                new_pos = pos + force
                # Ensure we stay within [0,1]³
                new_pos = np.clip(new_pos, 0, 1)
                particle.phase_coordinates = tuple(new_pos)
    
    def _enforce_phase_space_boundaries(self, system_state: Any):
        """Ensures phase space has proper boundaries"""
        # Adjust topological classifier if it exists
        if hasattr(system_state, 'classifier') and hasattr(system_state.classifier, 'attractors'):
            # Move attractors to be inside new boundary
            for name, attractor in system_state.classifier.attractors.items():
                center = np.array(attractor['center'])
                distance = self.boundary.distance_to_boundary(center)
                
                if distance > -self.boundary_thickness:
                    # Attractor too close or outside boundary - move inward
                    new_center = center * 0.9  # Simple scaling toward center
                    attractor['center'] = tuple(new_center)
    
    def verify(self, system_state: Any) -> bool:
        """
        Verifies if axiom is satisfied:
        - Is singularity on the boundary?
        - Is boundary well-defined?
        - Do other entities respect the topology?
        """
        # Find singularity
        singularity_found = self._locate_singularity(system_state)
        
        if not singularity_found:
            logger.warning("AX5: Cannot verify - no singularity in system")
            return True  # Vacuously true
        
        # Check if singularity is on boundary
        distance = self.boundary.distance_to_boundary(self.singularity_position)
        on_boundary = abs(distance) < self.boundary_thickness
        
        if not on_boundary:
            logger.error(f"AX5 VIOLATED: Singularity distance from boundary: {distance:.4f}")
            return False
        
        # Check if other entities are not too close to boundary
        violations = 0
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if not getattr(neuron, 'is_singularity', False):
                    pos = np.array([neuron.determinacy, neuron.stability, neuron.entropy])
                    dist = abs(self.boundary.distance_to_boundary(pos))
                    if dist < self.boundary_thickness / 2:
                        violations += 1
        
        logger.info(f"AX5: Verified - singularity on boundary={on_boundary}, "
                   f"violations={violations}")
        
        return on_boundary and violations == 0
    
    def get_boundary_visualization_data(self) -> Dict[str, Any]:
        """Returns data for boundary visualization"""
        boundary_points = self.boundary.get_boundary_points(200)
        
        return {
            'boundary_points': boundary_points.tolist(),
            'center': self.boundary.center.tolist(),
            'radius': self.boundary.radius,
            'singularity_position': self.singularity_position.tolist() if self.singularity_position is not None else None,
            'oscillation_phase': self.boundary.oscillation_phase,
            'breathing_amplitude': math.sin(self.boundary.oscillation_phase) * self.breathing_amplitude
        }


# Usage example
def demonstrate_ax5():
    """Demonstration of AX5 operation"""
    print("=== AX5 TOPOLOGICAL BOUNDARY DEMONSTRATION ===")
    print("Axiom: Ø always exists on the boundary of cognitive space")
    print("-" * 50)
    
    # Create simple system
    class SimpleSystem:
        def __init__(self):
            self.neurons = []
            self.metadata = {}
    
    system = SimpleSystem()
    
    # Add singularity neuron
    singularity_neuron = AdaptiveGTMONeuron("sing_0", (0, 0, 0))
    singularity_neuron.is_singularity = True
    singularity_neuron.determinacy = 0.8
    singularity_neuron.stability = 0.8
    singularity_neuron.entropy = 0.1
    system.neurons.append(singularity_neuron)
    
    # Add regular neurons
    for i in range(3):
        neuron = AdaptiveGTMONeuron(f"n_{i}", (i, 0, 0))
        neuron.determinacy = 0.5 + i * 0.1
        neuron.stability = 0.5 - i * 0.1
        neuron.entropy = 0.3 + i * 0.05
        system.neurons.append(neuron)
    
    # Create and apply axiom
    ax5 = AX5_TopologicalBoundary()
    
    print("Initial state:")
    print(f"Singularity: d={singularity_neuron.determinacy:.3f}, "
          f"s={singularity_neuron.stability:.3f}, e={singularity_neuron.entropy:.3f}")
    
    # Apply axiom several times to see "breathing"
    for i in range(5):
        print(f"\nIteration {i+1}:")
        system = ax5.apply(system)
        
        print(f"Boundary radius: {ax5.boundary.radius:.3f}")
        print(f"Singularity after: d={singularity_neuron.determinacy:.3f}, "
              f"s={singularity_neuron.stability:.3f}, e={singularity_neuron.entropy:.3f}")
        
        # Check distance from boundary
        pos = np.array([singularity_neuron.determinacy, 
                       singularity_neuron.stability, 
                       singularity_neuron.entropy])
        distance = ax5.boundary.distance_to_boundary(pos)
        print(f"Distance from boundary: {distance:.6f}")
    
    # Verification
    is_valid = ax5.verify(system)
    print(f"\nAxiom satisfied: {is_valid}")
    
    # Visualization data
    viz_data = ax5.get_boundary_visualization_data()
    print(f"\nBoundary data:")
    print(f"- Center: {viz_data['center']}")
    print(f"- Radius: {viz_data['radius']:.3f}")
    print(f"- Breathing amplitude: {viz_data['breathing_amplitude']:.3f}")


if __name__ == "__main__":
    demonstrate_ax5()
