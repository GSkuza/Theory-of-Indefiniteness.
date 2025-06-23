"""
AX5 Fundamental Algorithms for AI Systems
========================================
Five fundamental algorithms based on AX5 theorems
with practical applications for AI systems

author: Grzegorz Skuza (Poland)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
import logging

from gtmo_core_v2 import (
    O, AlienatedNumber, AdaptiveGTMONeuron,
    KnowledgeEntity, TopologicalClassifier
)

logger = logging.getLogger(__name__)


# ============================================================================
# COGNITIVE SPACE BOUNDARY IMPLEMENTATION
# ============================================================================

class CognitiveSpaceBoundary:
    """
    Implementation of cognitive space boundary according to AX5.
    Represents the topological boundary where Ø resides.
    """
    
    def __init__(self, centre: np.ndarray = None, radius: float = 0.5):
        self.centre = centre if centre is not None else np.array([0.5, 0.5, 0.5])
        self.radius = radius
        self.boundary_thickness = 0.01
        
    def distance_to_boundary(self, point: np.ndarray) -> float:
        """Calculate distance from point to boundary (negative = inside)"""
        distance_from_centre = np.linalg.norm(point - self.centre)
        return distance_from_centre - self.radius
    
    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        """Project point onto the boundary surface"""
        direction = point - self.centre
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-10:
            # Point at centre, choose arbitrary direction
            direction = np.array([1.0, 0.0, 0.0])
            direction_norm = 1.0
        
        unit_direction = direction / direction_norm
        boundary_point = self.centre + unit_direction * self.radius
        return boundary_point
    
    def get_boundary_points(self, n_points: int) -> List[np.ndarray]:
        """Generate points on the boundary surface"""
        points = []
        
        for i in range(n_points):
            # Spherical coordinates
            phi = np.arccos(1 - 2 * i / n_points)  # Uniform distribution
            theta = np.pi * (1 + np.sqrt(5)) * i  # Golden angle
            
            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            boundary_point = self.centre + self.radius * np.array([x, y, z])
            points.append(boundary_point)
        
        return points


class AX5_TopologicalBoundary:
    """
    AX5 implementation: Ø ∈ ∂(CognitiveSpace)
    Provides verification and analysis of topological boundary properties.
    """
    
    def __init__(self):
        self.boundary = CognitiveSpaceBoundary()
        self.verification_tolerance = 1e-6
    
    def verify_boundary_membership(self, entity: Any) -> bool:
        """Verify if entity belongs to cognitive space boundary"""
        if entity is O:
            return True
        
        if isinstance(entity, KnowledgeEntity):
            pos = np.array(entity.to_phase_point())
            distance = abs(self.boundary.distance_to_boundary(pos))
            return distance < self.verification_tolerance
        
        return False


# ============================================================================
# ALGORITHM 1: Boundary-Aware Knowledge Clustering (BAKC)
# ============================================================================

class BoundaryAwareKnowledgeClustering:
    """
    Knowledge clustering algorithm aware of cognitive boundaries.
    Based on Theorem 1 (Topological Invariance).
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.boundary = CognitiveSpaceBoundary()
        self.clusters = []
        self.singularity_cluster = None
        
    def cluster_knowledge(self, entities: List[KnowledgeEntity]) -> Dict[int, List[KnowledgeEntity]]:
        """
        Clusters knowledge whilst respecting cognitive boundaries.
        
        AI Applications:
        - Organisation of knowledge bases in chatbots
        - Hierarchisation of information in expert systems
        - Detection of knowledge gaps (areas near boundary)
        """
        # 1. Find singularities
        singularities = []
        regular_entities = []
        
        for entity in entities:
            if self._is_singularity(entity):
                singularities.append(entity)
            else:
                regular_entities.append(entity)
        
        # 2. Place singularities on boundary
        if singularities:
            self.singularity_cluster = {
                'entities': singularities,
                'position': self.boundary.project_to_boundary(
                    np.array([0.999, 0.999, 0.001])
                )
            }
        
        # 3. Cluster remaining entities whilst respecting boundary
        clusters = self._adaptive_kmeans(regular_entities)
        
        # 4. Ensure minimal separation from boundary
        for cluster_id, cluster in clusters.items():
            self._ensure_boundary_separation(cluster)
        
        return clusters
    
    def _is_singularity(self, entity: KnowledgeEntity) -> bool:
        """Check if entity is a singularity"""
        return (entity.content is O or 
                isinstance(entity.content, AlienatedNumber) or
                (entity.determinacy > 0.99 and entity.entropy < 0.01))
    
    def _adaptive_kmeans(self, entities: List[KnowledgeEntity]) -> Dict[int, List[KnowledgeEntity]]:
        """K-means adapted to cognitive space topology"""
        if not entities:
            return {}
        
        # Initialise centroids away from boundary
        centroids = []
        for i in range(self.n_clusters):
            # Distribute centroids inside space
            angle = 2 * np.pi * i / self.n_clusters
            r = self.boundary.radius * 0.5  # 50% of radius
            centroid = self.boundary.centre + r * np.array([
                np.cos(angle), np.sin(angle), 0
            ])
            centroids.append(centroid)
        
        # Iterative assignment
        clusters = {i: [] for i in range(self.n_clusters)}
        
        for entity in entities:
            pos = np.array(entity.phase_coordinates or entity.to_phase_point())
            
            # Find nearest centroid with boundary distance weighting
            best_cluster = 0
            best_score = float('inf')
            
            for i, centroid in enumerate(centroids):
                distance = np.linalg.norm(pos - centroid)
                boundary_penalty = 1.0 / (abs(self.boundary.distance_to_boundary(pos)) + 0.1)
                score = distance + 0.3 * boundary_penalty
                
                if score < best_score:
                    best_score = score
                    best_cluster = i
            
            clusters[best_cluster].append(entity)
        
        return clusters
    
    def _ensure_boundary_separation(self, cluster: List[KnowledgeEntity]):
        """Ensure minimal cluster separation from boundary"""
        for entity in cluster:
            if entity.phase_coordinates:
                pos = np.array(entity.phase_coordinates)
                dist = self.boundary.distance_to_boundary(pos)
                
                if abs(dist) < 0.05:  # Too close to boundary
                    # Move towards centre
                    direction = self.boundary.centre - pos
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    new_pos = pos + direction * 0.05
                    entity.phase_coordinates = tuple(new_pos)


# ============================================================================
# ALGORITHM 2: Cognitive Breathing Optimiser (CBO)
# ============================================================================

class CognitiveBbreathingOptimiser:
    """
    Optimiser utilising cognitive boundary oscillations.
    Based on Theorem 2 (Oscillatory Dynamics).
    """
    
    def __init__(self, breathing_rate: float = 0.1):
        self.breathing_rate = breathing_rate
        self.phase = 0.0
        self.amplitude = 0.1
        self.exploration_history = []
        
    def optimise_exploration(self, current_knowledge: List[KnowledgeEntity], 
                           target_areas: List[str]) -> List[Tuple[float, float, float]]:
        """
        Optimises knowledge space exploration using oscillations.
        
        AI Applications:
        - Dynamic exploration/exploitation adjustment in RL
        - Cyclical search through hypothesis spaces
        - Adaptive sampling in generative models
        """
        # 1. Calculate current oscillation phase
        self.phase += self.breathing_rate
        expansion = math.sin(self.phase) * self.amplitude
        
        # 2. Adjust exploration strategy to phase
        if expansion > 0:  # Expansion phase - explore boundaries
            exploration_points = self._explore_boundaries(current_knowledge, expansion)
        else:  # Contraction phase - consolidate knowledge
            exploration_points = self._consolidate_knowledge(current_knowledge, -expansion)
        
        # 3. Prioritise target areas
        prioritised_points = self._prioritise_targets(exploration_points, target_areas)
        
        # 4. Record history for learning
        self.exploration_history.append({
            'phase': self.phase,
            'expansion': expansion,
            'points_explored': len(prioritised_points),
            'coverage': self._calculate_coverage(prioritised_points)
        })
        
        return prioritised_points
    
    def _explore_boundaries(self, knowledge: List[KnowledgeEntity], 
                          expansion: float) -> List[Tuple[float, float, float]]:
        """Generate exploration points near expanded boundary"""
        boundary = CognitiveSpaceBoundary()
        boundary.radius = 0.5 + expansion
        
        # Generate points on expanded boundary
        n_points = max(10, int(50 * expansion))
        boundary_points = boundary.get_boundary_points(n_points)
        
        # Filter points that are 'novel' relative to current knowledge
        novel_points = []
        for point in boundary_points:
            if self._is_novel_area(point, knowledge):
                novel_points.append(tuple(point))
        
        return novel_points
    
    def _consolidate_knowledge(self, knowledge: List[KnowledgeEntity], 
                             contraction: float) -> List[Tuple[float, float, float]]:
        """Consolidate knowledge during contraction phase"""
        # Find gaps between existing entities
        gaps = []
        
        for i, entity1 in enumerate(knowledge):
            for j, entity2 in enumerate(knowledge[i+1:], i+1):
                pos1 = np.array(entity1.to_phase_point())
                pos2 = np.array(entity2.to_phase_point())
                
                # Midpoint
                midpoint = (pos1 + pos2) / 2
                
                # Check if there's a gap
                if self._is_knowledge_gap(midpoint, knowledge):
                    gaps.append(tuple(midpoint))
        
        return gaps
    
    def _is_novel_area(self, point: np.ndarray, knowledge: List[KnowledgeEntity]) -> bool:
        """Check if area is novel"""
        min_distance = float('inf')
        
        for entity in knowledge:
            pos = np.array(entity.to_phase_point())
            distance = np.linalg.norm(point - pos)
            min_distance = min(min_distance, distance)
        
        return min_distance > 0.1  # Novelty threshold
    
    def _is_knowledge_gap(self, point: np.ndarray, knowledge: List[KnowledgeEntity]) -> bool:
        """Detect knowledge gaps"""
        nearby_count = 0
        
        for entity in knowledge:
            pos = np.array(entity.to_phase_point())
            if np.linalg.norm(point - pos) < 0.15:
                nearby_count += 1
        
        return nearby_count < 2  # Gap if fewer than 2 neighbours
    
    def _prioritise_targets(self, points: List[Tuple[float, float, float]], 
                          targets: List[str]) -> List[Tuple[float, float, float]]:
        """Prioritise points relative to targets"""
        # Simplified: return first N points
        return points[:min(len(points), len(targets) * 3)]
    
    def _calculate_coverage(self, points: List[Tuple[float, float, float]]) -> float:
        """Calculate space coverage"""
        if not points:
            return 0.0
        
        # Simplified: convex hull volume / total space volume
        points_array = np.array(points)
        volume = np.prod(np.max(points_array, axis=0) - np.min(points_array, axis=0))
        return min(1.0, volume)


# ============================================================================
# ALGORITHM 3: Boundary Repulsion Learning (BRL)
# ============================================================================

class BoundaryRepulsionLearning:
    """
    Learning with repulsion from cognitive boundary.
    Based on Theorem 3 (Minimal Separation).
    """
    
    def __init__(self, min_separation: float = 0.01):
        self.min_separation = min_separation
        self.repulsion_strength = 0.05
        self.learning_rate = 0.01
        self.boundary = CognitiveSpaceBoundary()
        
    def train_with_boundary_awareness(self, neurons: List[AdaptiveGTMONeuron], 
                                     training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train neurons whilst maintaining boundary separation.
        
        AI Applications:
        - Regularisation in neural networks
        - Overfitting prevention through topological constraints
        - Diversity preservation in genetic populations
        """
        results = {
            'iterations': 0,
            'boundary_violations': 0,
            'convergence_achieved': False,
            'final_positions': []
        }
        
        for iteration in range(100):  # Max iterations
            results['iterations'] = iteration + 1
            
            # 1. Standard learning step
            for neuron, data in zip(neurons, training_data):
                self._learning_step(neuron, data)
            
            # 2. Apply repulsive forces from boundary
            violations = self._apply_boundary_forces(neurons)
            results['boundary_violations'] += violations
            
            # 3. Check convergence
            if self._check_convergence(neurons):
                results['convergence_achieved'] = True
                break
        
        # 4. Record final positions
        for neuron in neurons:
            results['final_positions'].append({
                'id': neuron.id,
                'position': [neuron.determinacy, neuron.stability, neuron.entropy],
                'distance_to_boundary': self.boundary.distance_to_boundary(
                    np.array([neuron.determinacy, neuron.stability, neuron.entropy])
                )
            })
        
        return results
    
    def _learning_step(self, neuron: AdaptiveGTMONeuron, data: Dict[str, Any]):
        """Single neuron learning step"""
        # Simulate learning - adjust neuron parameters
        target = data.get('target', [0.5, 0.5, 0.5])
        current = np.array([neuron.determinacy, neuron.stability, neuron.entropy])
        
        # Gradient descent with momentum
        gradient = (np.array(target) - current) * self.learning_rate
        
        # Update with constraints
        new_pos = current + gradient
        new_pos = np.clip(new_pos, 0, 1)
        
        neuron.determinacy = new_pos[0]
        neuron.stability = new_pos[1]
        neuron.entropy = new_pos[2]
    
    def _apply_boundary_forces(self, neurons: List[AdaptiveGTMONeuron]) -> int:
        """Apply repulsive forces from boundary"""
        violations = 0
        
        for neuron in neurons:
            if getattr(neuron, 'is_singularity', False):
                continue  # Singularities may be on boundary
            
            pos = np.array([neuron.determinacy, neuron.stability, neuron.entropy])
            distance = self.boundary.distance_to_boundary(pos)
            
            if abs(distance) < self.min_separation:
                violations += 1
                
                # Calculate repulsive force
                if distance > 0:  # Outside
                    direction = pos - self.boundary.centre
                else:  # Inside
                    direction = self.boundary.centre - pos
                
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                force = direction * self.repulsion_strength * (
                    1 - abs(distance) / self.min_separation
                )
                
                # Apply force
                new_pos = pos + force
                new_pos = np.clip(new_pos, 0, 1)
                
                neuron.determinacy = new_pos[0]
                neuron.stability = new_pos[1]
                neuron.entropy = new_pos[2]
        
        return violations
    
    def _check_convergence(self, neurons: List[AdaptiveGTMONeuron]) -> bool:
        """Check if learning has converged"""
        # Simplified: check if all neurons are stable
        for neuron in neurons:
            if len(neuron.trajectory_history) < 2:
                return False
            
            # Compare last 2 positions
            last_pos = neuron.trajectory_history[-1]
            prev_pos = neuron.trajectory_history[-2]
            
            change = abs(last_pos.get('determinacy', 0) - prev_pos.get('determinacy', 0))
            change += abs(last_pos.get('stability', 0) - prev_pos.get('stability', 0))
            change += abs(last_pos.get('entropy', 0) - prev_pos.get('entropy', 0))
            
            if change > 0.001:
                return False
        
        return True


# ============================================================================
# ALGORITHM 4: Singularity-Guided Exploration (SGE)
# ============================================================================

class SingularityGuidedExploration:
    """
    Exploration guided by singularity on boundary.
    Utilises all 3 theorems.
    """
    
    def __init__(self):
        self.boundary = CognitiveSpaceBoundary()
        self.singularity_position = None
        self.exploration_radius = 0.2
        self.visited_regions = []
        
    def explore_unknown_regions(self, current_knowledge: List[KnowledgeEntity],
                              max_steps: int = 50) -> List[Dict[str, Any]]:
        """
        Explore unknown regions using singularity as guide.
        
        AI Applications:
        - Directed reinforcement learning
        - Intelligent sampling in high-dimensional spaces
        - Anomaly and edge case detection
        """
        explorations = []
        
        # 1. Locate singularity
        self._locate_singularity(current_knowledge)
        
        for step in range(max_steps):
            # 2. Choose exploration direction relative to singularity
            direction = self._choose_exploration_direction(step)
            
            # 3. Generate exploration point
            exploration_point = self._generate_exploration_point(direction)
            
            # 4. Evaluate exploration value
            value = self._evaluate_exploration_value(exploration_point, current_knowledge)
            
            # 5. Record exploration
            exploration = {
                'step': step,
                'point': exploration_point,
                'direction_from_singularity': direction,
                'value': value,
                'distance_to_boundary': self.boundary.distance_to_boundary(
                    np.array(exploration_point)
                )
            }
            
            explorations.append(exploration)
            self.visited_regions.append(exploration_point)
            
            # 6. Adaptively adjust exploration radius
            self._adapt_exploration_radius(value)
        
        return explorations
    
    def _locate_singularity(self, knowledge: List[KnowledgeEntity]):
        """Locate singularity in knowledge"""
        for entity in knowledge:
            if self._is_singularity_entity(entity):
                self.singularity_position = np.array(entity.to_phase_point())
                # Ensure it's on boundary
                self.singularity_position = self.boundary.project_to_boundary(
                    self.singularity_position
                )
                return
        
        # Default singularity position
        self.singularity_position = self.boundary.project_to_boundary(
            np.array([0.999, 0.999, 0.001])
        )
    
    def _is_singularity_entity(self, entity: KnowledgeEntity) -> bool:
        """Check if entity is singularity"""
        return (entity.content is O or
                (entity.determinacy > 0.99 and entity.entropy < 0.01))
    
    def _choose_exploration_direction(self, step: int) -> np.ndarray:
        """Choose exploration direction"""
        # Spiral exploration around singularity
        angle = step * 0.2  # Angular step
        
        # Tangent vector to boundary at singularity point
        tangent = np.cross(
            self.singularity_position - self.boundary.centre,
            np.array([0, 0, 1])  # Z axis
        )
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        # Rotate tangent vector
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        direction = cos_a * tangent + sin_a * np.cross(
            self.singularity_position - self.boundary.centre, tangent
        )
        
        return direction / (np.linalg.norm(direction) + 1e-10)
    
    def _generate_exploration_point(self, direction: np.ndarray) -> Tuple[float, float, float]:
        """Generate exploration point"""
        # Point in direction from singularity, but inside space
        base_point = self.singularity_position + direction * self.exploration_radius
        
        # Ensure it's inside boundary
        if self.boundary.distance_to_boundary(base_point) > 0:
            # Beyond boundary - move towards centre
            base_point = base_point * 0.9
        
        # Add small randomness
        noise = np.random.randn(3) * 0.01
        final_point = base_point + noise
        
        # Constrain to [0,1]
        final_point = np.clip(final_point, 0, 1)
        
        return tuple(final_point)
    
    def _evaluate_exploration_value(self, point: Tuple[float, float, float],
                                  knowledge: List[KnowledgeEntity]) -> float:
        """Evaluate exploration point value"""
        point_array = np.array(point)
        
        # Factors affecting value:
        # 1. Novelty (distance from known knowledge)
        novelty = min(np.linalg.norm(point_array - np.array(k.to_phase_point()))
                     for k in knowledge) if knowledge else 1.0
        
        # 2. Distance from singularity (closer = more information)
        sing_distance = np.linalg.norm(point_array - self.singularity_position)
        sing_value = 1.0 / (1.0 + sing_distance)
        
        # 3. Position relative to boundary (optimal distance)
        boundary_distance = abs(self.boundary.distance_to_boundary(point_array))
        boundary_value = 1.0 - np.exp(-10 * boundary_distance)  # Penalise being too close
        
        # Combined value
        value = 0.4 * novelty + 0.4 * sing_value + 0.2 * boundary_value
        
        return value
    
    def _adapt_exploration_radius(self, last_value: float):
        """Adaptively adjust exploration radius"""
        if last_value > 0.7:
            # Good value - increase radius
            self.exploration_radius = min(0.4, self.exploration_radius * 1.1)
        elif last_value < 0.3:
            # Poor value - decrease radius
            self.exploration_radius = max(0.05, self.exploration_radius * 0.9)


# ============================================================================
# ALGORITHM 5: Phase Space Navigation System (PSNS)
# ============================================================================

class PhaseSpaceNavigationSystem:
    """
    Phase space navigation system with boundary consideration.
    Integrates all aspects of AX5.
    """
    
    def __init__(self):
        self.ax5 = AX5_TopologicalBoundary()
        self.navigation_graph = {}
        self.safe_paths = []
        self.forbidden_zones = []
        
    def plan_knowledge_path(self, start: KnowledgeEntity, goal: KnowledgeEntity,
                          obstacles: List[KnowledgeEntity] = None) -> List[Tuple[float, float, float]]:
        """
        Plan path in knowledge space whilst avoiding boundary.
        
        AI Applications:
        - Learning path planning in curriculum learning
        - Concept space navigation for reasoning systems
        - Knowledge transfer optimisation between domains
        """
        obstacles = obstacles or []
        
        # 1. Convert to phase space
        start_pos = np.array(start.to_phase_point())
        goal_pos = np.array(goal.to_phase_point())
        
        # 2. Check if goal is not singularity
        if self._is_singularity_position(goal_pos):
            logger.warning("Goal is singularity - adjusting target")
            goal_pos = self._find_nearest_safe_position(goal_pos)
        
        # 3. Generate navigation graph
        self._build_navigation_graph(start_pos, goal_pos, obstacles)
        
        # 4. Find optimal path avoiding boundary
        path = self._find_optimal_path(start_pos, goal_pos)
        
        # 5. Smooth path whilst maintaining safe distance
        smooth_path = self._smooth_path(path)
        
        # 6. Verify path safety
        safe_path = self._ensure_path_safety(smooth_path)
        
        return safe_path
    
    def _is_singularity_position(self, pos: np.ndarray) -> bool:
        """Check if position is near singularity"""
        # Typical singularity position
        sing_pos = np.array([0.999, 0.999, 0.001])
        return np.linalg.norm(pos - sing_pos) < 0.1
    
    def _find_nearest_safe_position(self, pos: np.ndarray) -> np.ndarray:
        """Find nearest safe position"""
        # Move towards centre whilst maintaining direction
        direction = self.ax5.boundary.centre - pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        safe_distance = 0.1  # Minimum distance from boundary
        return pos + direction * safe_distance
    
    def _build_navigation_graph(self, start: np.ndarray, goal: np.ndarray,
                              obstacles: List[KnowledgeEntity]):
        """Build navigation graph in phase space"""
        # Navigation point grid
        grid_size = 10
        self.navigation_graph = {}
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    point = np.array([i/(grid_size-1), j/(grid_size-1), k/(grid_size-1)])
                    
                    # Check if point is safe
                    if self._is_safe_point(point, obstacles):
                        neighbours = self._get_safe_neighbours(point, grid_size)
                        self.navigation_graph[tuple(point)] = neighbours
    
    def _is_safe_point(self, point: np.ndarray, obstacles: List[KnowledgeEntity]) -> bool:
        """Check if point is safe"""
        # Distance from boundary
        boundary_dist = abs(self.ax5.boundary.distance_to_boundary(point))
        if boundary_dist < self.ax5.boundary.boundary_thickness:
            return False
        
        # Distance from obstacles
        for obstacle in obstacles:
            obs_pos = np.array(obstacle.to_phase_point())
            if np.linalg.norm(point - obs_pos) < 0.05:
                return False
        
        return True
    
    def _get_safe_neighbours(self, point: np.ndarray, grid_size: int) -> List[Tuple[float, float, float]]:
        """Find safe neighbours of point"""
        neighbours = []
        step = 1.0 / (grid_size - 1)
        
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                for dz in [-step, 0, step]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    neighbour = point + np.array([dx, dy, dz])
                    
                    # Check [0,1] bounds
                    if np.all(neighbour >= 0) and np.all(neighbour <= 1):
                        # Check safety
                        if abs(self.ax5.boundary.distance_to_boundary(neighbour)) > self.ax5.boundary.boundary_thickness:
                            neighbours.append(tuple(neighbour))
        
        return neighbours
    
    def _find_optimal_path(self, start: np.ndarray, goal: np.ndarray) -> List[Tuple[float, float, float]]:
        """Find optimal path using A*"""
        # Simplified A* - returns straight path
        # In full implementation would use proper A* on graph
        
        path = []
        current = start
        
        # Linear interpolation with boundary avoidance
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            point = (1 - t) * start + t * goal
            
            # Check if point is safe
            if abs(self.ax5.boundary.distance_to_boundary(point)) < 0.05:
                # Avoid boundary
                detour = self._create_detour(point)
                path.extend(detour)
            else:
                path.append(tuple(point))
        
        return path
    
    def _create_detour(self, unsafe_point: np.ndarray) -> List[Tuple[float, float, float]]:
        """Create detour around unsafe point"""
        # Move point to safe distance
        direction = self.ax5.boundary.centre - unsafe_point
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        safe_point = unsafe_point + direction * 0.1
        return [tuple(safe_point)]
    
    def _smooth_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Smooth path"""
        if len(path) < 3:
            return path
        
        smooth_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_point = np.array(path[i-1])
            curr_point = np.array(path[i])
            next_point = np.array(path[i+1])
            
            # Weighted average
            smooth_point = 0.25 * prev_point + 0.5 * curr_point + 0.25 * next_point
            smooth_path.append(tuple(smooth_point))
        
        smooth_path.append(path[-1])
        return smooth_path
    
    def _ensure_path_safety(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Ensure entire path safety"""
        safe_path = []
        
        for point in path:
            point_array = np.array(point)
            
            # Check safety
            if abs(self.ax5.boundary.distance_to_boundary(point_array)) < self.ax5.boundary.boundary_thickness:
                # Correct point
                safe_point = self._find_nearest_safe_position(point_array)
                safe_path.append(tuple(safe_point))
            else:
                safe_path.append(point)
        
        return safe_path


# ============================================================================
# Demonstration and Tests
# ============================================================================

def demonstrate_ax5_algorithms():
    """Demonstration of all 5 AX5 algorithms"""
    print("=== DEMONSTRATION OF 5 AX5 ALGORITHMS ===\n")
    
    # Prepare test data
    test_entities = [
        KnowledgeEntity("Certain fact", 0.9, 0.9, 0.1),
        KnowledgeEntity("Uncertain hypothesis", 0.3, 0.4, 0.7),
        KnowledgeEntity("Paradox", 0.5, 0.2, 0.9),
        KnowledgeEntity("Emerging pattern", 0.6, 0.5, 0.6),
        KnowledgeEntity(O, 0.999, 0.999, 0.001)  # Singularity
    ]
    
    # 1. Boundary-Aware Knowledge Clustering
    print("1. BOUNDARY-AWARE KNOWLEDGE CLUSTERING")
    print("-" * 40)
    bakc = BoundaryAwareKnowledgeClustering(n_clusters=3)
    clusters = bakc.cluster_knowledge(test_entities)
    print(f"Created {len(clusters)} clusters")
    if bakc.singularity_cluster:
        print(f"Singularity cluster at position: {bakc.singularity_cluster['position']}")
    print()
    
    # 2. Cognitive Breathing Optimiser
    print("2. COGNITIVE BREATHING OPTIMISER")
    print("-" * 40)
    cbo = CognitiveBbreathingOptimiser()
    exploration_points = cbo.optimise_exploration(
        test_entities[:4],  # Without singularity
        target_areas=["quantum", "consciousness", "emergence"]
    )
    print(f"Generated {len(exploration_points)} exploration points")
    print(f"Space coverage: {cbo._calculate_coverage(exploration_points):.3f}")
    print()
    
    # 3. Boundary Repulsion Learning
    print("3. BOUNDARY REPULSION LEARNING")
    print("-" * 40)
    brl = BoundaryRepulsionLearning()
    test_neurons = [
        AdaptiveGTMONeuron(f"n_{i}", (i, 0, 0))
        for i in range(3)
    ]
    training_data = [
        {'target': [0.7, 0.7, 0.3]},
        {'target': [0.6, 0.8, 0.2]},
        {'target': [0.8, 0.6, 0.4]}
    ]
    results = brl.train_with_boundary_awareness(test_neurons, training_data)
    print(f"Learning iterations: {results['iterations']}")
    print(f"Boundary violations: {results['boundary_violations']}")
    print(f"Convergence achieved: {results['convergence_achieved']}")
    print()
    
    # 4. Singularity-Guided Exploration
    print("4. SINGULARITY-GUIDED EXPLORATION")
    print("-" * 40)
    sge = SingularityGuidedExploration()
    explorations = sge.explore_unknown_regions(test_entities, max_steps=10)
    print(f"Conducted {len(explorations)} exploration steps")
    avg_value = np.mean([e['value'] for e in explorations])
    print(f"Average exploration value: {avg_value:.3f}")
    print()
    
    # 5. Phase Space Navigation System
    print("5. PHASE SPACE NAVIGATION SYSTEM")
    print("-" * 40)
    psns = PhaseSpaceNavigationSystem()
    start_entity = test_entities[0]  # Certain fact
    goal_entity = test_entities[2]   # Paradox
    path = psns.plan_knowledge_path(start_entity, goal_entity, obstacles=[test_entities[4]])
    print(f"Planned path has {len(path)} points")
    print(f"Start: {start_entity.to_phase_point()}")
    print(f"Goal: {goal_entity.to_phase_point()}")
    if path:
        print(f"First step: {path[0]}")
        print(f"Last step: {path[-1]}")


if __name__ == "__main__":
    demonstrate_ax5_algorithms()
