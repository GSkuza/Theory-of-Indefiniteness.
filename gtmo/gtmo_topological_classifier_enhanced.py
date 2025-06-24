"""
Enhanced GTMØ Topological Classification System
==============================================

An improved implementation of the GTMØ topological classifier with:
- True Wasserstein distance metric
- Spatial indexing for scalability
- Uncertainty quantification
- Interactive 3D visualization

Author: GTMØ Research Team
Version: 2.0
License: MIT
"""

from __future__ import annotations

import numpy as np
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict
import warnings
import time

# Import optional dependencies with graceful fallback
try:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import cdist
    from scipy.stats import bootstrap
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Using fallback distance metrics.")

try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Spatial indexing disabled.")

try:
    import ot  # Python Optimal Transport
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Visualization features disabled.")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Knowledge Type Enumeration
# =============================================================================

class KnowledgeType(Enum):
    """Enhanced knowledge types with topological properties."""
    SINGULARITY = "Ø"          # Ontological singularity
    ALIENATED = "ℓ∅"           # Alienated numbers
    PARTICLE = "Ψᴷ"            # Knowledge particles
    SHADOW = "Ψʰ"              # Knowledge shadows
    EMERGENT = "Ψᴺ"            # Emergent patterns
    LIMINAL = "Ψᴧ"             # Liminal fragments
    META_INDEFINITE = "Ψ∅∅"    # Meta-indefinite
    VOID = "Ψ◊"               # Void fragments
    FLUX = "Ψ~"               # Fluctuating
    TRANSCENDENT = "Ψ↑"        # Transcendent


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AttractorConfig:
    """Configuration for a topological attractor."""
    center: Tuple[float, float, float]
    basin_radius: float
    type: KnowledgeType
    strength: float = 1.0
    adaptive: bool = True
    
    def to_array(self) -> np.ndarray:
        """Convert center to numpy array."""
        return np.array(self.center)


@dataclass
class ClassificationResult:
    """Enhanced classification result with uncertainty quantification."""
    type: KnowledgeType
    confidence: float
    confidence_interval: Optional[Tuple[float, float]] = None
    nearest_attractors: List[Tuple[str, float]] = field(default_factory=list)
    phase_point: Optional[Tuple[float, float, float]] = None
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class KnowledgeEntity:
    """Enhanced knowledge entity with topological properties."""
    content: Any
    determinacy: float = 0.5
    stability: float = 0.5
    entropy: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_phase_point(self) -> Tuple[float, float, float]:
        """Convert to point in phase space."""
        return (
            np.clip(self.determinacy, 0, 1),
            np.clip(self.stability, 0, 1),
            np.clip(self.entropy, 0, 1)
        )


# =============================================================================
# Distance Metrics
# =============================================================================

class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""
    
    @abstractmethod
    def compute(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute distance between two points."""
        pass


class L2Distance(DistanceMetric):
    """Standard Euclidean distance."""
    
    def compute(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute L2 distance."""
        return np.linalg.norm(p1 - p2)


class WassersteinDistance(DistanceMetric):
    """True Wasserstein distance implementation."""
    
    def __init__(self, use_pot: bool = True):
        """
        Initialize Wasserstein distance calculator.
        
        Args:
            use_pot: Whether to use Python Optimal Transport library
        """
        self.use_pot = use_pot and POT_AVAILABLE
        
    def compute(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute Wasserstein distance.
        
        Interprets points as probability distributions.
        """
        if not SCIPY_AVAILABLE:
            # Fallback to L2
            return np.linalg.norm(p1 - p2)
            
        if len(p1.shape) == 1:
            # 1D case - use scipy
            return wasserstein_distance(p1, p2)
        
        if self.use_pot:
            # Use optimal transport
            # Create uniform weights
            n = len(p1)
            m = len(p2)
            a = np.ones(n) / n
            b = np.ones(m) / m
            
            # Compute cost matrix
            M = cdist(p1.reshape(-1, 1), p2.reshape(-1, 1))
            
            # Solve optimal transport
            return ot.emd2(a, b, M)
        else:
            # Fallback to scipy 1D projection
            return wasserstein_distance(p1.flatten(), p2.flatten())


# =============================================================================
# Uncertainty Quantification
# =============================================================================

class UncertaintyQuantifier:
    """Quantifies classification uncertainty using various methods."""
    
    def __init__(self, n_bootstrap: int = 100):
        """
        Initialize uncertainty quantifier.
        
        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        self.n_bootstrap = n_bootstrap
        
    def compute_confidence(
        self, 
        distances: Dict[str, float],
        basin_radii: Dict[str, float]
    ) -> float:
        """
        Compute classification confidence based on distances.
        
        Args:
            distances: Distances to each attractor
            basin_radii: Basin radii for each attractor
            
        Returns:
            Confidence score in [0, 1]
        """
        # Find nearest attractor
        nearest = min(distances.items(), key=lambda x: x[1])
        nearest_name, nearest_dist = nearest
        
        # Confidence based on how deep we are in the basin
        basin_radius = basin_radii[nearest_name]
        if nearest_dist >= basin_radius:
            return 0.0
        
        # Linear decay from center to edge
        confidence = 1.0 - (nearest_dist / basin_radius)
        
        # Penalize if close to multiple attractors
        sorted_distances = sorted(distances.values())
        if len(sorted_distances) > 1:
            gap = sorted_distances[1] - sorted_distances[0]
            confidence *= min(1.0, gap / basin_radius)
            
        return confidence
    
    def compute_entropy_uncertainty(
        self,
        distances: Dict[str, float],
        temperature: float = 1.0
    ) -> float:
        """
        Compute uncertainty using entropy of probability distribution.
        
        Args:
            distances: Distances to attractors
            temperature: Softmax temperature
            
        Returns:
            Entropy-based uncertainty
        """
        # Convert distances to probabilities using softmax
        neg_distances = [-d / temperature for d in distances.values()]
        max_val = max(neg_distances)
        exp_vals = [np.exp(v - max_val) for v in neg_distances]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]
        
        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(distances))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def bootstrap_confidence_interval(
        self,
        phase_point: np.ndarray,
        classify_func: Callable,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using bootstrap.
        
        Args:
            phase_point: Point to classify
            classify_func: Classification function
            confidence_level: Confidence level for interval
            
        Returns:
            (lower, upper) confidence bounds
        """
        if not SCIPY_AVAILABLE:
            return (0.0, 1.0)  # Maximum uncertainty
            
        # Add small noise and reclassify
        results = []
        for _ in range(self.n_bootstrap):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, size=3)
            noisy_point = np.clip(phase_point + noise, 0, 1)
            
            # Classify and get confidence
            result = classify_func(noisy_point)
            results.append(result.confidence)
            
        # Compute percentile interval
        alpha = 1 - confidence_level
        lower = np.percentile(results, 100 * alpha / 2)
        upper = np.percentile(results, 100 * (1 - alpha / 2))
        
        return (lower, upper)


# =============================================================================
# Spatial Index for Scalability
# =============================================================================

class SpatialIndex:
    """Spatial indexing for efficient nearest neighbor queries."""
    
    def __init__(self, use_kdtree: bool = True):
        """
        Initialize spatial index.
        
        Args:
            use_kdtree: Whether to use KD-tree (requires sklearn)
        """
        self.use_kdtree = use_kdtree and SKLEARN_AVAILABLE
        self.tree = None
        self.points = None
        self.labels = None
        
    def build(self, points: np.ndarray, labels: List[str]):
        """
        Build spatial index from points.
        
        Args:
            points: Array of shape (n, 3) with phase space coordinates
            labels: List of attractor names
        """
        self.points = points
        self.labels = labels
        
        if self.use_kdtree:
            self.tree = KDTree(points)
            
    def query_nearest(
        self, 
        point: np.ndarray, 
        k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors.
        
        Args:
            point: Query point
            k: Number of neighbors (None for all)
            
        Returns:
            List of (label, distance) tuples
        """
        if k is None:
            k = len(self.points)
            
        if self.use_kdtree and self.tree is not None:
            distances, indices = self.tree.query(
                point.reshape(1, -1), 
                k=min(k, len(self.points))
            )
            distances = distances.flatten()
            indices = indices.flatten()
        else:
            # Fallback to brute force
            distances = cdist(
                point.reshape(1, -1), 
                self.points
            ).flatten()
            indices = np.argsort(distances)[:k]
            distances = distances[indices]
            
        results = [
            (self.labels[idx], float(dist))
            for idx, dist in zip(indices, distances)
        ]
        
        return results


# =============================================================================
# Visualization Engine
# =============================================================================

class VisualizationEngine:
    """3D visualization for phase space and attractors."""
    
    def __init__(self):
        """Initialize visualization engine."""
        self.fig = None
        
    def create_phase_space_plot(
        self,
        attractors: Dict[str, AttractorConfig],
        entities: Optional[List[KnowledgeEntity]] = None,
        trajectories: Optional[List[List[Tuple[float, float, float]]]] = None
    ) -> go.Figure:
        """
        Create interactive 3D visualization of phase space.
        
        Args:
            attractors: Dictionary of attractor configurations
            entities: Optional list of entities to plot
            trajectories: Optional list of trajectories
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create visualization.")
            return None
            
        fig = go.Figure()
        
        # Plot attractors
        for name, config in attractors.items():
            center = config.center
            
            # Attractor center
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(
                    size=10 * config.strength,
                    color='red',
                    symbol='diamond'
                ),
                name=f"{name} ({config.type.value})",
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Type: {config.type.value}<br>"
                    f"Determinacy: %{{x:.3f}}<br>"
                    f"Stability: %{{y:.3f}}<br>"
                    f"Entropy: %{{z:.3f}}<br>"
                    f"Strength: {config.strength:.2f}"
                )
            ))
            
            # Basin of attraction (wireframe sphere)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = center[0] + config.basin_radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + config.basin_radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + config.basin_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.2,
                colorscale='Viridis',
                showscale=False,
                name=f"{name} basin"
            ))
        
        # Plot entities
        if entities:
            points = np.array([e.to_phase_point() for e in entities])
            
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=points[:, 2],  # Color by entropy
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Entropy")
                ),
                name='Entities',
                hovertemplate=(
                    "Determinacy: %{x:.3f}<br>"
                    "Stability: %{y:.3f}<br>"
                    "Entropy: %{z:.3f}"
                )
            ))
        
        # Plot trajectories
        if trajectories:
            for i, traj in enumerate(trajectories):
                points = np.array(traj)
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='lines+markers',
                    line=dict(width=3),
                    marker=dict(size=3),
                    name=f'Trajectory {i}'
                ))
        
        # Update layout
        fig.update_layout(
            title="GTMØ Phase Space Visualization",
            scene=dict(
                xaxis_title="Determinacy",
                yaxis_title="Stability",
                zaxis_title="Entropy",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1])
            ),
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_attractor_adaptation_plot(
        self,
        adaptation_history: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create visualization of attractor adaptation over time.
        
        Args:
            adaptation_history: History of attractor positions
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE or not adaptation_history:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Determinacy', 'Stability', 'Entropy', '3D Evolution'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter3d'}]]
        )
        
        # Extract time series for each attractor
        attractors = {}
        for step in adaptation_history:
            for name, pos in step['positions'].items():
                if name not in attractors:
                    attractors[name] = {'det': [], 'stab': [], 'ent': []}
                attractors[name]['det'].append(pos[0])
                attractors[name]['stab'].append(pos[1])
                attractors[name]['ent'].append(pos[2])
        
        # Plot time series
        for name, data in attractors.items():
            steps = list(range(len(data['det'])))
            
            # Determinacy
            fig.add_trace(
                go.Scatter(x=steps, y=data['det'], name=name, legendgroup=name),
                row=1, col=1
            )
            
            # Stability
            fig.add_trace(
                go.Scatter(x=steps, y=data['stab'], name=name, legendgroup=name, showlegend=False),
                row=1, col=2
            )
            
            # Entropy
            fig.add_trace(
                go.Scatter(x=steps, y=data['ent'], name=name, legendgroup=name, showlegend=False),
                row=2, col=1
            )
            
            # 3D trajectory
            fig.add_trace(
                go.Scatter3d(
                    x=data['det'], 
                    y=data['stab'], 
                    z=data['ent'],
                    mode='lines+markers',
                    name=name,
                    legendgroup=name,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title="Attractor Adaptation History")
        return fig


# =============================================================================
# Enhanced Topological Classifier
# =============================================================================

class EnhancedTopologicalClassifier:
    """
    Enhanced GTMØ topological classifier with improved features.
    
    Features:
        - True Wasserstein distance metric
        - Spatial indexing for scalability
        - Uncertainty quantification
        - Interactive 3D visualization
        - Adaptive learning with performance tracking
    """
    
    def __init__(
        self,
        enhanced_mode: bool = True,
        distance_metric: str = 'wasserstein',
        use_spatial_index: bool = True,
        enable_uncertainty: bool = True,
        enable_visualization: bool = True,
        cache_distances: bool = True
    ):
        """
        Initialize enhanced topological classifier.
        
        Args:
            enhanced_mode: Enable all enhancements
            distance_metric: 'wasserstein' or 'l2'
            use_spatial_index: Enable KD-tree indexing
            enable_uncertainty: Enable uncertainty quantification
            enable_visualization: Enable visualization features
            cache_distances: Cache distance calculations
        """
        self.enhanced_mode = enhanced_mode
        self.attractors = self._initialize_attractors()
        self.phase_history = []
        self.adaptation_history = []
        
        # Initialize distance metric
        if enhanced_mode and distance_metric == 'wasserstein':
            self.distance_metric = WassersteinDistance()
        else:
            self.distance_metric = L2Distance()
            
        # Initialize components
        self.spatial_index = SpatialIndex() if use_spatial_index and enhanced_mode else None
        self.uncertainty_quantifier = UncertaintyQuantifier() if enable_uncertainty and enhanced_mode else None
        self.visualization_engine = VisualizationEngine() if enable_visualization and enhanced_mode else None
        
        # Distance cache
        self.cache_enabled = cache_distances and enhanced_mode
        self.distance_cache = {}
        
        # Build spatial index
        self._rebuild_spatial_index()
        
        logger.info(f"Enhanced Topological Classifier initialized (enhanced_mode={enhanced_mode})")
        
    def _initialize_attractors(self) -> Dict[str, AttractorConfig]:
        """Initialize topological attractors in phase space."""
        return {
            'singularity': AttractorConfig(
                center=(1.0, 1.0, 0.0),
                basin_radius=0.15,
                type=KnowledgeType.SINGULARITY,
                strength=2.0
            ),
            'particle': AttractorConfig(
                center=(0.85, 0.85, 0.15),
                basin_radius=0.25,
                type=KnowledgeType.PARTICLE,
                strength=1.0
            ),
            'shadow': AttractorConfig(
                center=(0.15, 0.15, 0.85),
                basin_radius=0.25,
                type=KnowledgeType.SHADOW,
                strength=1.0
            ),
            'emergent': AttractorConfig(
                center=(0.5, 0.3, 0.9),
                basin_radius=0.2,
                type=KnowledgeType.EMERGENT,
                strength=1.2
            ),
            'alienated': AttractorConfig(
                center=(0.999, 0.999, 0.001),
                basin_radius=0.1,
                type=KnowledgeType.ALIENATED,
                strength=1.5
            ),
            'void': AttractorConfig(
                center=(0.0, 0.0, 0.5),
                basin_radius=0.2,
                type=KnowledgeType.VOID,
                strength=0.8
            ),
            'flux': AttractorConfig(
                center=(0.5, 0.5, 0.8),
                basin_radius=0.3,
                type=KnowledgeType.FLUX,
                strength=0.9
            ),
            'transcendent': AttractorConfig(
                center=(0.7, 0.7, 0.3),
                basin_radius=0.15,
                type=KnowledgeType.TRANSCENDENT,
                strength=1.1
            )
        }
        
    def _rebuild_spatial_index(self):
        """Rebuild spatial index after attractor adaptation."""
        if self.spatial_index is None:
            return
            
        points = np.array([a.center for a in self.attractors.values()])
        labels = list(self.attractors.keys())
        self.spatial_index.build(points, labels)
        
        # Clear distance cache
        if self.cache_enabled:
            self.distance_cache.clear()
            
    def _compute_distance(
        self, 
        p1: Union[Tuple[float, float, float], np.ndarray],
        p2: Union[Tuple[float, float, float], np.ndarray]
    ) -> float:
        """
        Compute distance with caching.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Distance value
        """
        # Convert to arrays
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        
        # Check cache
        if self.cache_enabled:
            cache_key = (tuple(p1_arr), tuple(p2_arr))
            if cache_key in self.distance_cache:
                return self.distance_cache[cache_key]
                
        # Compute distance
        distance = self.distance_metric.compute(p1_arr, p2_arr)
        
        # Store in cache
        if self.cache_enabled:
            self.distance_cache[cache_key] = distance
            
        return distance
        
    def classify(self, entity: Union[KnowledgeEntity, Any]) -> Union[KnowledgeType, ClassificationResult]:
        """
        Classify entity with backward compatibility.
        
        Args:
            entity: Entity to classify
            
        Returns:
            KnowledgeType (basic mode) or ClassificationResult (enhanced mode)
        """
        if self.enhanced_mode:
            return self.classify_with_confidence(entity)
        else:
            # Backward compatible mode
            return self._classify_basic(entity)
            
    def _classify_basic(self, entity: Any) -> KnowledgeType:
        """Basic classification for backward compatibility."""
        # Convert to phase point
        if isinstance(entity, KnowledgeEntity):
            phase_point = entity.to_phase_point()
        else:
            phase_point = self._estimate_phase_point(entity)
            
        # Calculate distances
        distances = {}
        for name, attractor in self.attractors.items():
            distance = self._compute_distance(phase_point, attractor.center)
            effective_distance = distance / attractor.strength
            distances[name] = effective_distance
            
        # Find nearest
        nearest = min(distances, key=distances.get)
        nearest_distance = distances[nearest]
        
        # Check basin
        if nearest_distance <= self.attractors[nearest].basin_radius:
            return self.attractors[nearest].type
        else:
            return self._classify_liminal_region(phase_point, distances)
            
    def classify_with_confidence(self, entity: Union[KnowledgeEntity, Any]) -> ClassificationResult:
        """
        Enhanced classification with uncertainty quantification.
        
        Args:
            entity: Entity to classify
            
        Returns:
            ClassificationResult with confidence metrics
        """
        # Convert to phase point
        if isinstance(entity, KnowledgeEntity):
            phase_point = entity.to_phase_point()
        else:
            phase_point = self._estimate_phase_point(entity)
            
        phase_array = np.array(phase_point)
        
        # Use spatial index if available
        if self.spatial_index and self.spatial_index.tree is not None:
            nearest_results = self.spatial_index.query_nearest(phase_array)
            distances = {name: dist for name, dist in nearest_results}
        else:
            # Compute all distances
            distances = {}
            for name, attractor in self.attractors.items():
                distance = self._compute_distance(phase_point, attractor.center)
                effective_distance = distance / attractor.strength
                distances[name] = effective_distance
                
        # Find classification
        nearest = min(distances, key=distances.get)
        nearest_distance = distances[nearest]
        
        if nearest_distance <= self.attractors[nearest].basin_radius:
            classification = self.attractors[nearest].type
        else:
            classification = self._classify_liminal_region(phase_point, distances)
            
        # Compute confidence
        basin_radii = {name: att.basin_radius for name, att in self.attractors.items()}
        confidence = 1.0
        
        if self.uncertainty_quantifier:
            confidence = self.uncertainty_quantifier.compute_confidence(distances, basin_radii)
            
        # Create result
        result = ClassificationResult(
            type=classification,
            confidence=confidence,
            phase_point=phase_point,
            nearest_attractors=sorted(distances.items(), key=lambda x: x[1])[:3]
        )
        
        # Add uncertainty metrics if enabled
        if self.uncertainty_quantifier:
            result.uncertainty_metrics['entropy'] = self.uncertainty_quantifier.compute_entropy_uncertainty(distances)
            
            # Bootstrap confidence interval (expensive, so optional)
            if confidence < 0.8:  # Only for uncertain cases
                ci = self.uncertainty_quantifier.bootstrap_confidence_interval(
                    phase_array,
                    lambda p: self._classify_point_for_bootstrap(p)
                )
                result.confidence_interval = ci
                
        # Record history
        self.phase_history.append({
            'point': phase_point,
            'classification': classification,
            'distances': distances,
            'confidence': confidence
        })
        
        return result
        
    def _classify_point_for_bootstrap(self, point: np.ndarray) -> ClassificationResult:
        """Helper method for bootstrap confidence calculation."""
        entity = KnowledgeEntity(
            content="bootstrap_sample",
            determinacy=point[0],
            stability=point[1],
            entropy=point[2]
        )
        return self.classify_with_confidence(entity)
        
    def batch_classify(
        self, 
        entities: List[Union[KnowledgeEntity, Any]],
        parallel: bool = True,
        batch_size: int = 1000
    ) -> List[Union[KnowledgeType, ClassificationResult]]:
        """
        Efficiently classify multiple entities.
        
        Args:
            entities: List of entities to classify
            parallel: Use parallel processing (if available)
            batch_size: Size of processing batches
            
        Returns:
            List of classification results
        """
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            
            if self.enhanced_mode and parallel:
                # TODO: Implement parallel processing with multiprocessing
                batch_results = [self.classify(e) for e in batch]
            else:
                batch_results = [self.classify(e) for e in batch]
                
            results.extend(batch_results)
            
        return results
        
    def _classify_liminal_region(
        self, 
        phase_point: Tuple[float, float, float],
        distances: Dict[str, float]
    ) -> KnowledgeType:
        """Classify points not clearly in any attractor basin."""
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        nearest_two = sorted_distances[:2]
        
        # If very close to two attractors, it's liminal
        if len(nearest_two) > 1 and nearest_two[1][1] - nearest_two[0][1] < 0.1:
            return KnowledgeType.LIMINAL
            
        # Check for flux conditions
        if phase_point[2] > 0.7 and 0.3 < phase_point[0] < 0.7:
            return KnowledgeType.FLUX
            
        # Check for transcendent conditions
        if any(coord > 1.0 or coord < 0.0 for coord in phase_point):
            return KnowledgeType.TRANSCENDENT
            
        # Check for meta-indefinite
        if all(0.4 < coord < 0.6 for coord in phase_point):
            return KnowledgeType.META_INDEFINITE
            
        # Default to liminal
        return KnowledgeType.LIMINAL
        
    def _estimate_phase_point(self, entity: Any) -> Tuple[float, float, float]:
        """Estimate phase coordinates for arbitrary entities."""
        entity_str = str(entity).lower()
        
        # Base values
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
        determinacy = np.clip(determinacy, 0, 1)
        stability = np.clip(stability, 0, 1)
        entropy = np.clip(entropy, 0, 1)
        
        return (determinacy, stability, entropy)
        
    def adapt_attractors(
        self, 
        feedback: List[Tuple[KnowledgeEntity, KnowledgeType]],
        learning_rate: float = 0.1
    ):
        """
        Adapt attractor positions based on classification feedback.
        
        Args:
            feedback: List of (entity, expected_type) tuples
            learning_rate: Learning rate for adaptation
        """
        if not feedback:
            return
            
        # Group by type
        grouped = defaultdict(list)
        for entity, expected_type in feedback:
            grouped[expected_type].append(entity.to_phase_point())
            
        # Track changes for history
        position_changes = {}
        
        # Update attractors
        for type_name, points in grouped.items():
            attractor_name = self._find_attractor_by_type(type_name)
            if attractor_name and len(points) > 3:
                # Current position
                old_center = self.attractors[attractor_name].center
                
                # Compute new center
                points_array = np.array(points)
                new_center = np.mean(points_array, axis=0)
                
                # Apply exponential moving average
                updated_center = tuple(
                    learning_rate * new + (1 - learning_rate) * old
                    for new, old in zip(new_center, old_center)
                )
                
                # Update
                self.attractors[attractor_name].center = updated_center
                position_changes[attractor_name] = {
                    'old': old_center,
                    'new': updated_center
                }
                
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': time.time(),
            'feedback_size': len(feedback),
            'positions': {name: att.center for name, att in self.attractors.items()},
            'changes': position_changes
        })
        
        # Rebuild spatial index
        if position_changes:
            self._rebuild_spatial_index()
            logger.info(f"Adapted {len(position_changes)} attractors")
            
    def _find_attractor_by_type(self, knowledge_type: KnowledgeType) -> Optional[str]:
        """Find attractor name by knowledge type."""
        for name, attractor in self.attractors.items():
            if attractor.type == knowledge_type:
                return name
        return None
        
    def visualize_phase_space(
        self,
        entities: Optional[List[KnowledgeEntity]] = None,
        show_plot: bool = True
    ) -> Optional[go.Figure]:
        """
        Create interactive 3D visualization of phase space.
        
        Args:
            entities: Optional entities to plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure or None
        """
        if not self.visualization_engine:
            logger.warning("Visualization not enabled")
            return None
            
        fig = self.visualization_engine.create_phase_space_plot(
            self.attractors,
            entities
        )
        
        if fig and show_plot:
            fig.show()
            
        return fig
        
    def visualize_adaptation_history(self, show_plot: bool = True) -> Optional[go.Figure]:
        """
        Visualize how attractors have adapted over time.
        
        Args:
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure or None
        """
        if not self.visualization_engine or not self.adaptation_history:
            return None
            
        fig = self.visualization_engine.create_attractor_adaptation_plot(
            self.adaptation_history
        )
        
        if fig and show_plot:
            fig.show()
            
        return fig
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'total_classifications': len(self.phase_history),
            'cache_hit_rate': 0.0,
            'adaptation_count': len(self.adaptation_history),
            'attractor_stability': {},
            'classification_distribution': defaultdict(int)
        }
        
        # Cache hit rate
        if self.cache_enabled and self.phase_history:
            # Estimate based on cache size vs total possible distances
            total_possible = len(self.phase_history) * len(self.attractors)
            cache_size = len(self.distance_cache)
            metrics['cache_hit_rate'] = min(1.0, cache_size / total_possible)
            
        # Classification distribution
        for record in self.phase_history:
            classification = record['classification']
            metrics['classification_distribution'][classification.value] += 1
            
        # Attractor stability (how much they've moved)
        if self.adaptation_history:
            for name in self.attractors:
                positions = [h['positions'][name] for h in self.adaptation_history if name in h['positions']]
                if len(positions) > 1:
                    positions_array = np.array(positions)
                    stability = 1.0 / (1.0 + np.std(positions_array, axis=0).mean())
                    metrics['attractor_stability'][name] = stability
                    
        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================

def create_classifier(
    enhanced: bool = True,
    **kwargs
) -> Union[EnhancedTopologicalClassifier, Any]:
    """
    Factory function to create appropriate classifier.
    
    Args:
        enhanced: Whether to create enhanced version
        **kwargs: Additional arguments for classifier
        
    Returns:
        Classifier instance
    """
    if enhanced:
        return EnhancedTopologicalClassifier(enhanced_mode=True, **kwargs)
    else:
        # Could return basic classifier here
        return EnhancedTopologicalClassifier(enhanced_mode=False, **kwargs)


# =============================================================================
# Example Usage and Tests
# =============================================================================

def demonstrate_enhanced_classifier():
    """Demonstrate enhanced classifier features."""
    print("=" * 80)
    print("GTMØ Enhanced Topological Classifier Demonstration")
    print("=" * 80)
    
    # Create classifier
    classifier = create_classifier(
        enhanced=True,
        distance_metric='wasserstein',
        enable_uncertainty=True,
        enable_visualization=PLOTLY_AVAILABLE
    )
    
    # Test entities
    test_entities = [
        KnowledgeEntity("Mathematical theorem", 0.95, 0.92, 0.08),
        KnowledgeEntity("Quantum superposition", 0.3, 0.2, 0.9),
        KnowledgeEntity("Emerging pattern", 0.5, 0.3, 0.85),
        KnowledgeEntity("Stable fact", 0.9, 0.9, 0.1),
        KnowledgeEntity("Paradox", 0.5, 0.1, 0.95),
        KnowledgeEntity("Future prediction", 0.2, 0.3, 0.8),
    ]
    
    print("\n1. ENHANCED CLASSIFICATION WITH CONFIDENCE")
    print("-" * 50)
    
    for entity in test_entities:
        result = classifier.classify_with_confidence(entity)
        print(f"\n'{entity.content}':")
        print(f"  Type: {result.type.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        if result.confidence_interval:
            print(f"  95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"  Uncertainty (entropy): {result.uncertainty_metrics.get('entropy', 0):.3f}")
        print(f"  Nearest attractors: {[f'{n}({d:.3f})' for n, d in result.nearest_attractors[:2]]}")
        
    print("\n2. BATCH CLASSIFICATION PERFORMANCE")
    print("-" * 50)
    
    # Create large batch
    large_batch = []
    for i in range(1000):
        large_batch.append(KnowledgeEntity(
            f"Entity_{i}",
            np.random.random(),
            np.random.random(),
            np.random.random()
        ))
        
    start_time = time.time()
    results = classifier.batch_classify(large_batch, parallel=False)
    elapsed = time.time() - start_time
    
    print(f"Classified {len(large_batch)} entities in {elapsed:.3f} seconds")
    print(f"Average time per classification: {elapsed/len(large_batch)*1000:.3f} ms")
    
    print("\n3. ADAPTIVE LEARNING")
    print("-" * 50)
    
    # Simulate feedback
    feedback = [
        (KnowledgeEntity("New theorem", 0.88, 0.89, 0.12), KnowledgeType.PARTICLE),
        (KnowledgeEntity("Another theorem", 0.87, 0.88, 0.13), KnowledgeType.PARTICLE),
        (KnowledgeEntity("Third theorem", 0.86, 0.87, 0.14), KnowledgeType.PARTICLE),
    ]
    
    old_center = classifier.attractors['particle'].center
    classifier.adapt_attractors(feedback, learning_rate=0.3)
    new_center = classifier.attractors['particle'].center
    
    print(f"Particle attractor moved from {old_center} to {new_center}")
    print(f"Distance moved: {np.linalg.norm(np.array(new_center) - np.array(old_center)):.4f}")
    
    print("\n4. PERFORMANCE METRICS")
    print("-" * 50)
    
    metrics = classifier.get_performance_metrics()
    print(f"Total classifications: {metrics['total_classifications']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"Adaptations performed: {metrics['adaptation_count']}")
    
    print("\nClassification distribution:")
    for type_name, count in metrics['classification_distribution'].items():
        print(f"  {type_name}: {count}")
        
    print("\n5. VISUALIZATION")
    print("-" * 50)
    
    if PLOTLY_AVAILABLE:
        print("Creating phase space visualization...")
        fig = classifier.visualize_phase_space(test_entities, show_plot=False)
        if fig:
            print("Visualization created successfully (not shown in demo)")
    else:
        print("Plotly not available - visualization skipped")
        
    print("\n" + "=" * 80)
    print("Demonstration complete!")


if __name__ == "__main__":
    demonstrate_enhanced_classifier()
