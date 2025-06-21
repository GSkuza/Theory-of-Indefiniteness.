# GTMØ Topology v2 Optimized - Documentation

## Overview

The `topology_v2_optimized.py` module provides advanced analytical and visualization tools for studying the topological structure of the GTMØ v2 system. This module focuses on analyzing the distribution of epistemic entities in phase space and identifying emergent behavioral patterns in particle trajectories.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Core Functions](#core-functions)
4. [Integration with GTMØ v2](#integration-with-gtmø-v2)
5. [Usage Examples](#usage-examples)
6. [Visualization Features](#visualization-features)
7. [Analysis Capabilities](#analysis-capabilities)
8. [API Reference](#api-reference)

## Introduction

The topology module serves as a high-level interface for:
- **Phase Space Analysis**: Examining the distribution of knowledge entities in the 3D phase space defined by determinacy, stability, and entropy
- **Topological Visualization**: Creating 3D visualizations of particles and attractors
- **Behavioral Clustering**: Grouping discovered trajectory patterns into behavioral clusters
- **Statistical Metrics**: Computing center of mass, variance, and bounding box volume of knowledge clouds

## Dependencies

### Required GTMØ Modules
- `gtmo_axioms_v2.py`: Enhanced GTMØ system with universe modes
- `epistemic_particles_optimized.py`: Optimized trajectory observer and epistemic system
- `gtmo_core_v2.py`: Core knowledge entity definitions

### Optional Libraries
- `matplotlib`: For 3D visualization (optional but recommended)
- `numpy`: For numerical computations (required)

### Installation Note
```python
# The module will work without matplotlib but visualization features will be disabled
# Install matplotlib with: pip install matplotlib
```

## Core Functions

### 1. `analyze_phase_space_topology(system)`

Analyzes the distribution of entities in phase space, computing statistical metrics about the knowledge cloud.

**Parameters:**
- `system` (EnhancedGTMOSystem): Active instance of GTMØ v2 system

**Returns:**
- Dictionary containing:
  - `particle_count`: Number of particles analyzed
  - `center_of_mass`: [x, y, z] coordinates of the knowledge cloud center
  - `standard_deviation`: [σx, σy, σz] for each dimension
  - `bounding_box_volume`: Volume of the minimal bounding box

**Example:**
```python
metrics = analyze_phase_space_topology(system)
print(f"Knowledge cloud center: {metrics['center_of_mass']}")
```

### 2. `visualize_phase_space(system, show_attractors=True)`

Creates a 3D visualization of the phase space with particles and topological attractors.

**Parameters:**
- `system` (EnhancedGTMOSystem): Active instance of GTMØ v2 system
- `show_attractors` (bool): Whether to display attractor positions

**Features:**
- Color-coded particles based on entropy levels
- Marked attractor positions with distinct symbols
- Interactive 3D rotation and zoom
- Colorbar indicating entropy scale

### 3. `analyze_trajectory_clusters(observer)`

Groups discovered trajectory dimensions into behavioral clusters based on variance patterns.

**Parameters:**
- `observer` (OptimizedTrajectoryObserver): Observer instance with registered patterns

**Returns:**
- Dictionary containing:
  - `total_discovered_dimensions`: Count of high-confidence dimensions
  - `cluster_counts`: Distribution across stable, chaotic, and transient patterns

**Clustering Logic:**
- **Stable patterns**: Total variance < 0.05
- **Chaotic patterns**: Total variance > 0.3
- **Transient patterns**: 0.05 ≤ total variance ≤ 0.3

## Integration with GTMØ v2

The module seamlessly integrates with the enhanced GTMØ v2 ecosystem:

```python
from gtmo_axioms_v2 import UniverseMode
from epistemic_particles_optimized import OptimizedEpistemicSystem

# Create an optimized system
system = OptimizedEpistemicSystem(mode=UniverseMode.ETERNAL_FLUX)

# Add particles to populate phase space
for i in range(50):
    system.add_optimized_particle(
        content=f"particle_{i}",
        determinacy=np.random.uniform(0.1, 0.9),
        stability=np.random.uniform(0.1, 0.9),
        entropy=np.random.uniform(0.1, 0.9)
    )

# Analyze topology
metrics = analyze_phase_space_topology(system)
```

## Usage Examples

### Basic Phase Space Analysis
```python
# Create and populate system
system = OptimizedEpistemicSystem(mode=UniverseMode.INDEFINITE_STILLNESS)

# Add diverse particles
for _ in range(30):
    system.add_optimized_particle("knowledge fragment")

# Evolve system
for _ in range(10):
    system.evolve_system()

# Analyze
topology_metrics = analyze_phase_space_topology(system)
print(f"Bounding box volume: {topology_metrics['bounding_box_volume']:.4f}")
```

### Trajectory Clustering Analysis
```python
# After system evolution
cluster_analysis = analyze_trajectory_clusters(system.trajectory_observer)
print(f"Discovered dimensions: {cluster_analysis['total_discovered_dimensions']}")
print(f"Stable patterns: {cluster_analysis['cluster_counts']['stable_patterns']}")
```

### Creating Knowledge Clusters
```python
# Create distinct knowledge clusters in phase space
system = OptimizedEpistemicSystem(mode=UniverseMode.ETERNAL_FLUX)

# High determinacy cluster
for _ in range(20):
    system.add_optimized_particle(
        "certain knowledge",
        determinacy=np.random.uniform(0.8, 1.0),
        stability=np.random.uniform(0.8, 1.0),
        entropy=np.random.uniform(0.0, 0.2)
    )

# Chaotic cluster
for _ in range(20):
    system.add_optimized_particle(
        "uncertain knowledge",
        determinacy=np.random.uniform(0.1, 0.3),
        stability=np.random.uniform(0.1, 0.3),
        entropy=np.random.uniform(0.8, 1.0)
    )

# Visualize the clusters
visualize_phase_space(system)
```

## Visualization Features

### 3D Phase Space Visualization
The visualization provides:
- **Particle Distribution**: Each particle represented as a point in 3D space
- **Color Coding**: Particles colored by entropy level (viridis colormap)
- **Attractor Markers**: Large X markers showing attractor positions
- **Interactive Controls**: Rotate, zoom, and pan the 3D plot
- **Axis Labels**: Clear labeling of determinacy, stability, and entropy axes

### Visual Interpretation
- **Clustering**: Dense regions indicate knowledge convergence
- **Dispersion**: Spread indicates diversity of knowledge states
- **Attractor Influence**: Particles tend to cluster near attractors
- **Color Gradients**: Dark colors (low entropy) vs bright colors (high entropy)

## Analysis Capabilities

### Statistical Metrics
1. **Center of Mass**: Identifies the "average" position of all knowledge in phase space
2. **Standard Deviation**: Measures spread along each dimension
3. **Bounding Box Volume**: Quantifies the total "knowledge space" occupied

### Behavioral Patterns
1. **Stable Patterns**: Low variance, consistent behavior
2. **Chaotic Patterns**: High variance, unpredictable evolution
3. **Transient Patterns**: Moderate variance, transitional states

### Topological Insights
- **Phase Space Coverage**: How much of the possible knowledge space is explored
- **Cluster Formation**: Natural groupings of similar knowledge states
- **Attractor Basins**: Regions of influence around topological attractors

## API Reference

### Functions

#### `analyze_phase_space_topology(system: EnhancedGTMOSystem) -> Dict`
Computes statistical metrics for the particle distribution in phase space.

**Raises:**
- Returns error dict if GTMØ v2 ecosystem is not available
- Returns error dict if no particles exist in the system

#### `visualize_phase_space(system: EnhancedGTMOSystem, show_attractors: bool = True) -> None`
Creates an interactive 3D visualization of the phase space.

**Note:** Requires matplotlib. Prints warning if not available.

#### `analyze_trajectory_clusters(observer: OptimizedTrajectoryObserver) -> Dict`
Groups trajectory patterns into behavioral clusters.

**Clustering Criteria:**
- Based on sum of variance patterns
- Three categories: stable, chaotic, transient

#### `demonstrate_topology_analysis() -> None`
Demonstrates the module's capabilities with a complete example.

### Error Handling

The module gracefully handles missing dependencies:
```python
# Without matplotlib
visualize_phase_space(system)  # Prints warning, no crash

# Without GTMØ v2 modules
analyze_phase_space_topology(system)  # Returns {'error': '...'}
```

## Advanced Usage

### Custom Clustering Logic
```python
def custom_cluster_analysis(observer, custom_thresholds):
    """Implement custom clustering based on specific criteria"""
    dimensions = observer.get_discovered_dimensions()
    clusters = defaultdict(list)
    
    for dim in dimensions:
        # Custom logic here
        metric = compute_custom_metric(dim)
        cluster_name = assign_cluster(metric, custom_thresholds)
        clusters[cluster_name].append(dim)
    
    return clusters
```

### Phase Space Trajectory Analysis
```python
def analyze_phase_trajectories(system, time_steps=50):
    """Track how particles move through phase space over time"""
    trajectories = []
    
    for step in range(time_steps):
        system.evolve_system()
        snapshot = [p.to_phase_point() for p in system.epistemic_particles]
        trajectories.append(snapshot)
    
    return np.array(trajectories)
```

## Performance Considerations

- **Large Systems**: For systems with >1000 particles, visualization may be slow
- **Memory Usage**: Phase space analysis stores all particle positions in memory
- **Clustering Efficiency**: O(n) complexity for n discovered dimensions

## Future Enhancements

Potential improvements for the module:
1. **Advanced Clustering**: Machine learning-based clustering algorithms
2. **Animation**: Time-based animation of particle evolution
3. **2D Projections**: Alternative visualizations for specific phase space slices
4. **Export Capabilities**: Save visualizations and metrics to files
5. **Real-time Analysis**: Streaming analysis for long-running simulations

## Conclusion

The `topology_v2_optimized.py` module provides essential tools for understanding the complex dynamics of the GTMØ v2 system through topological analysis and visualization. By examining the distribution and evolution of knowledge entities in phase space, researchers can gain insights into emergent patterns, system stability, and the influence of topological attractors on knowledge evolution.

The module's integration with the broader GTMØ v2 ecosystem enables sophisticated analyses while maintaining ease of use through clear APIs and comprehensive error handling.