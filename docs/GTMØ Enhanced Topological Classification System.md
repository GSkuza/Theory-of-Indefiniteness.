# GTMØ Enhanced Topological Classification System

## Overview

The GTMØ Enhanced Topological Classification System is an advanced implementation of a knowledge classification framework that maps abstract concepts and entities into a three-dimensional phase space. The system uses topological attractors to classify knowledge based on determinacy, stability, and entropy characteristics.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Visualization](#visualization)
8. [Performance Optimization](#performance-optimization)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Features

### Core Features
- **Topological Phase Space Classification**: Maps entities to a 3D phase space based on determinacy, stability, and entropy
- **Multiple Knowledge Types**: Supports 10 distinct knowledge types from singularities to transcendent forms
- **Adaptive Learning**: Attractors can adapt their positions based on classification feedback

### Enhanced Features
- **True Wasserstein Distance Metric**: Implements optimal transport distance for more accurate classification
- **Spatial Indexing**: KD-tree based indexing for efficient nearest neighbor queries
- **Uncertainty Quantification**: Provides confidence scores and confidence intervals
- **Interactive 3D Visualization**: Real-time phase space visualization using Plotly
- **Performance Optimization**: Distance caching and batch processing capabilities

## Installation

### Basic Requirements
```bash
pip install numpy
```

### Optional Dependencies
For full functionality, install the following optional packages:

```bash
# For Wasserstein distance and spatial operations
pip install scipy scikit-learn

# For optimal transport (improved Wasserstein)
pip install POT

# For interactive visualizations
pip install plotly

# For performance optimization
pip install numba
```

### Complete Installation
```bash
pip install numpy scipy scikit-learn POT plotly numba
```

## Core Concepts

### Phase Space
The classifier operates in a 3D phase space with three axes:
- **Determinacy** (x-axis): How well-defined or certain the knowledge is (0-1)
- **Stability** (y-axis): How stable or consistent the knowledge is over time (0-1)
- **Entropy** (z-axis): The level of disorder or uncertainty (0-1)

### Knowledge Types
The system classifies entities into the following types:

| Type | Symbol | Description |
|------|--------|-------------|
| `SINGULARITY` | Ø | Ontological singularity - perfectly determined and stable |
| `ALIENATED` | ℓ∅ | Alienated numbers - highly determined but isolated |
| `PARTICLE` | Ψᴷ | Knowledge particles - discrete, well-defined units |
| `SHADOW` | Ψʰ | Knowledge shadows - low determinacy, high entropy |
| `EMERGENT` | Ψᴺ | Emergent patterns - moderate stability, high entropy |
| `LIMINAL` | Ψᴧ | Liminal fragments - between classifications |
| `META_INDEFINITE` | Ψ∅∅ | Meta-indefinite - centered in phase space |
| `VOID` | Ψ◊ | Void fragments - low values across all dimensions |
| `FLUX` | Ψ~ | Fluctuating - high entropy, variable determinacy |
| `TRANSCENDENT` | Ψ↑ | Transcendent - beyond normal phase space bounds |

### Attractors
Each knowledge type has an associated attractor in phase space with:
- **Center**: 3D coordinates in phase space
- **Basin radius**: Sphere of influence
- **Strength**: Affects the effective distance calculation
- **Adaptive**: Whether the attractor can move based on feedback

## Quick Start

### Basic Usage

```python
from gtmo_topological_classifier_enhanced import create_classifier, KnowledgeEntity

# Create classifier
classifier = create_classifier(enhanced=True)

# Create an entity
entity = KnowledgeEntity(
    content="Mathematical theorem",
    determinacy=0.95,
    stability=0.92,
    entropy=0.08
)

# Classify
result = classifier.classify(entity)
print(f"Classification: {result.type.value}")
print(f"Confidence: {result.confidence:.3f}")
```

### Simple Text Classification

```python
# Classify arbitrary text
result = classifier.classify("This might be a quantum superposition")
print(f"Type: {result.type.value}")
```

## API Reference

### Classes

#### `KnowledgeEntity`
Represents an entity to be classified.

```python
KnowledgeEntity(
    content: Any,                    # The content of the entity
    determinacy: float = 0.5,        # Determinacy level (0-1)
    stability: float = 0.5,          # Stability level (0-1)
    entropy: float = 0.5,            # Entropy level (0-1)
    metadata: Dict[str, Any] = None  # Optional metadata
)
```

#### `ClassificationResult`
Enhanced result object containing classification details.

```python
@dataclass
class ClassificationResult:
    type: KnowledgeType              # The classified type
    confidence: float                # Confidence score (0-1)
    confidence_interval: Optional[Tuple[float, float]]  # 95% CI
    nearest_attractors: List[Tuple[str, float]]        # Nearby attractors
    phase_point: Optional[Tuple[float, float, float]]  # 3D coordinates
    uncertainty_metrics: Dict[str, float]              # Additional metrics
```

#### `EnhancedTopologicalClassifier`
Main classifier class.

```python
EnhancedTopologicalClassifier(
    enhanced_mode: bool = True,           # Enable all enhancements
    distance_metric: str = 'wasserstein', # 'wasserstein' or 'l2'
    use_spatial_index: bool = True,       # Enable KD-tree indexing
    enable_uncertainty: bool = True,      # Enable uncertainty quantification
    enable_visualization: bool = True,    # Enable visualization features
    cache_distances: bool = True          # Cache distance calculations
)
```

### Methods

#### `classify(entity)`
Classify a single entity.

```python
# Enhanced mode returns ClassificationResult
result = classifier.classify(entity)

# Basic mode returns KnowledgeType
knowledge_type = classifier.classify(entity)  # with enhanced_mode=False
```

#### `classify_with_confidence(entity)`
Always returns enhanced classification with confidence.

```python
result = classifier.classify_with_confidence(entity)
print(f"Type: {result.type.value}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Uncertainty: {result.uncertainty_metrics['entropy']:.3f}")
```

#### `batch_classify(entities, parallel=True, batch_size=1000)`
Efficiently classify multiple entities.

```python
entities = [entity1, entity2, entity3, ...]
results = classifier.batch_classify(entities)
```

#### `adapt_attractors(feedback, learning_rate=0.1)`
Adapt attractor positions based on feedback.

```python
feedback = [
    (entity1, KnowledgeType.PARTICLE),
    (entity2, KnowledgeType.EMERGENT),
    # ...
]
classifier.adapt_attractors(feedback, learning_rate=0.1)
```

#### `visualize_phase_space(entities=None, show_plot=True)`
Create interactive 3D visualization.

```python
fig = classifier.visualize_phase_space(entities=[entity1, entity2])
```

#### `get_performance_metrics()`
Get comprehensive performance statistics.

```python
metrics = classifier.get_performance_metrics()
print(f"Total classifications: {metrics['total_classifications']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

## Advanced Usage

### Custom Distance Metrics

```python
from gtmo_topological_classifier_enhanced import DistanceMetric

class CustomDistance(DistanceMetric):
    def compute(self, p1: np.ndarray, p2: np.ndarray) -> float:
        # Implement custom distance calculation
        return custom_calculation(p1, p2)

# Use custom metric
classifier.distance_metric = CustomDistance()
```

### Uncertainty Analysis

```python
# Get detailed uncertainty metrics
result = classifier.classify_with_confidence(entity)

# Check confidence interval
if result.confidence_interval:
    lower, upper = result.confidence_interval
    print(f"95% CI: [{lower:.3f}, {upper:.3f}]")

# Get entropy-based uncertainty
entropy_uncertainty = result.uncertainty_metrics.get('entropy', 0)
print(f"Entropy uncertainty: {entropy_uncertainty:.3f}")
```

### Adaptive Learning Workflow

```python
# Collect classification feedback
feedback_data = []
for entity in training_entities:
    # Get ground truth label somehow
    true_type = get_ground_truth(entity)
    feedback_data.append((entity, true_type))

# Adapt attractors
classifier.adapt_attractors(feedback_data, learning_rate=0.15)

# Visualize adaptation history
fig = classifier.visualize_adaptation_history()
```

## Visualization

### Phase Space Visualization

```python
# Create entities
entities = [
    KnowledgeEntity("Theorem A", 0.9, 0.9, 0.1),
    KnowledgeEntity("Paradox B", 0.5, 0.1, 0.9),
    KnowledgeEntity("Pattern C", 0.6, 0.4, 0.7),
]

# Visualize
fig = classifier.visualize_phase_space(entities)

# Customize visualization
fig.update_layout(
    title="My Knowledge Space",
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
)
fig.show()
```

### Trajectory Visualization

```python
# Track entity evolution
trajectory = []
for t in range(10):
    entity = evolve_entity(base_entity, t)
    point = entity.to_phase_point()
    trajectory.append(point)

# Visualize with trajectory
fig = classifier.visualize_phase_space(
    entities=[base_entity],
    trajectories=[trajectory]
)
```

## Performance Optimization

### Batch Processing

```python
# Process large datasets efficiently
large_dataset = load_entities()  # 100,000+ entities

# Use batch processing
results = classifier.batch_classify(
    large_dataset,
    parallel=True,
    batch_size=5000
)
```

### Distance Caching

```python
# Enable caching (default)
classifier = create_classifier(cache_distances=True)

# Check cache performance
metrics = classifier.get_performance_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

### Spatial Indexing

```python
# Spatial indexing is enabled by default
# Disable for small datasets (<100 attractors)
classifier = create_classifier(use_spatial_index=False)
```

## Examples

### Example 1: Document Classification

```python
# Classify various document types
documents = [
    "The Earth orbits the Sun every 365.25 days.",  # Scientific fact
    "What if reality is just a simulation?",         # Philosophical question
    "Stock prices may rise or fall tomorrow.",       # Uncertain prediction
    "2 + 2 = 4 in base 10 arithmetic.",            # Mathematical truth
]

for doc in documents:
    result = classifier.classify(doc)
    print(f"\n'{doc[:50]}...'")
    print(f"  Type: {result.type.value}")
    print(f"  Confidence: {result.confidence:.2%}")
```

### Example 2: Knowledge Evolution Tracking

```python
# Track how knowledge evolves over time
import numpy as np

def simulate_knowledge_evolution(initial_entity, steps=50):
    trajectory = []
    entity = initial_entity
    
    for step in range(steps):
        # Simulate evolution
        det = entity.determinacy + np.random.normal(0, 0.02)
        stab = entity.stability + np.random.normal(0, 0.02)
        ent = entity.entropy + np.random.normal(0, 0.02)
        
        # Clip to valid range
        det = np.clip(det, 0, 1)
        stab = np.clip(stab, 0, 1)
        ent = np.clip(ent, 0, 1)
        
        # Create evolved entity
        entity = KnowledgeEntity(
            f"Evolution step {step}",
            determinacy=det,
            stability=stab,
            entropy=ent
        )
        
        # Classify and track
        result = classifier.classify(entity)
        trajectory.append(entity.to_phase_point())
        
        if step % 10 == 0:
            print(f"Step {step}: {result.type.value} (conf: {result.confidence:.2f})")
    
    return trajectory

# Run simulation
initial = KnowledgeEntity("Hypothesis", 0.5, 0.5, 0.5)
trajectory = simulate_knowledge_evolution(initial)

# Visualize evolution
fig = classifier.visualize_phase_space(trajectories=[trajectory])
```

### Example 3: Comparative Analysis

```python
# Compare different knowledge domains
domains = {
    "Mathematics": [
        KnowledgeEntity("Pythagorean theorem", 1.0, 1.0, 0.0),
        KnowledgeEntity("Goldbach conjecture", 0.7, 0.8, 0.3),
        KnowledgeEntity("Collatz conjecture", 0.6, 0.7, 0.4),
    ],
    "Physics": [
        KnowledgeEntity("Newton's laws", 0.9, 0.9, 0.1),
        KnowledgeEntity("Quantum superposition", 0.3, 0.4, 0.8),
        KnowledgeEntity("String theory", 0.4, 0.5, 0.7),
    ],
    "Philosophy": [
        KnowledgeEntity("Cogito ergo sum", 0.7, 0.8, 0.3),
        KnowledgeEntity("Ship of Theseus", 0.2, 0.3, 0.9),
        KnowledgeEntity("Simulation hypothesis", 0.1, 0.2, 0.95),
    ]
}

# Analyze each domain
for domain, entities in domains.items():
    print(f"\n{domain}:")
    results = classifier.batch_classify(entities)
    
    # Count types
    type_counts = {}
    for result in results:
        type_name = result.type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    for type_name, count in type_counts.items():
        print(f"  {type_name}: {count}")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'scipy'
**Solution**: Install optional dependencies
```bash
pip install scipy scikit-learn
```

#### Visualization not working
**Solution**: Install Plotly
```bash
pip install plotly
```

#### Slow performance with large datasets
**Solutions**:
1. Enable spatial indexing (default)
2. Use batch processing
3. Reduce bootstrap samples for uncertainty
4. Use L2 distance instead of Wasserstein for speed

#### Memory issues
**Solutions**:
1. Process in smaller batches
2. Disable distance caching for very large datasets
3. Use `enhanced_mode=False` for basic classification

### Performance Tips

1. **For real-time classification**: Use L2 distance and disable uncertainty quantification
2. **For high accuracy**: Use Wasserstein distance with full uncertainty analysis
3. **For large batches**: Enable parallel processing and increase batch size
4. **For adaptive systems**: Store feedback and adapt in batches rather than continuously

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Citation

If you use this system in your research, please cite:
```
GTMØ Enhanced Topological Classification System
Version 2.0
GTMØ Research Team
```
