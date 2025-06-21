# GTMØ Utils v2 Module Documentation

## Overview

The `utils_v2.py` module is an optimized and comprehensive utility module for the GTMØ (Generalized Theory of Mathematical Indefiniteness) v2 ecosystem. It replaces the deprecated `utils.py`, introducing full compatibility with advanced components such as `gtmo-core-v2.py`, `gtmo_axioms_v2.py`, and `epistemic_particles_optimized.py`.

## Key Features

- **Robust Import Handling**: Error-resistant imports from v2 modules with graceful fallbacks
- **Extended Type Checking**: Comprehensive type-checking functions for all v2 key entities
- **Phase Space Operations**: High-level utility functions for topological phase space operations
- **DRY Principle**: Centralized logic to avoid code duplication across modules
- **Built-in Testing**: Integrated demonstration block for easy testing and verification

## Installation

```python
# Place utils_v2.py in your GTMØ package directory
from gtmo.utils_v2 import *
```

## Module Structure

### Import Handling

The module uses a robust try-except block to handle imports from the GTMØ v2 ecosystem:

```python
try:
    from .gtmo_core_v2 import (
        Singularity, AlienatedNumber, KnowledgeEntity,
        EpistemicParticle, AdaptiveGTMONeuron,
        TopologicalClassifier, KnowledgeType, O
    )
    V2_CORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Fallback definitions for partial functionality
    V2_CORE_AVAILABLE = False
```

## API Reference

### Type Checking Functions

#### `is_ontological_singularity(item: Any) -> bool`

Checks if the given element is an Ontological Singularity (Ø).

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element is an Ontological Singularity, `False` otherwise

**Example:**
```python
>>> is_ontological_singularity(O)
True
>>> is_ontological_singularity("not a singularity")
False
```

#### `is_alienated_number(item: Any) -> bool`

Checks if the given element is an Alienated Number (ℓ∅).

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element is an instance of AlienatedNumber, `False` otherwise

**Example:**
```python
>>> alien = AlienatedNumber("future_concept")
>>> is_alienated_number(alien)
True
```

#### `is_gtmo_primitive(item: Any) -> bool`

Checks if the given element is a basic GTMØ primitive entity (Ø or ℓ∅).

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element is Ø or an AlienatedNumber instance, `False` otherwise

#### `is_knowledge_entity(item: Any) -> bool`

Checks if the element is a knowledge entity (KnowledgeEntity or its subclass).

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element inherits from KnowledgeEntity, `False` otherwise

**Note:** This is crucial for phase space operations as only these entities have defined topological coordinates.

#### `is_epistemic_particle(item: Any) -> bool`

Checks if the element is an Epistemic Particle.

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element is an instance of EpistemicParticle, `False` otherwise

#### `is_adaptive_neuron(item: Any) -> bool`

Checks if the element is an Adaptive Neuron.

**Parameters:**
- `item`: Element to check

**Returns:**
- `True` if the element is an instance of AdaptiveGTMONeuron, `False` otherwise

### Phase Space Operations

#### `extract_phase_point(entity: Any) -> Optional[Tuple[float, float, float]]`

Safely extracts phase space coordinates (determinacy, stability, entropy) from an entity.

**Parameters:**
- `entity`: Entity to process

**Returns:**
- Tuple `(determinacy, stability, entropy)` or `None` if not applicable

**Example:**
```python
>>> particle = EpistemicParticle("knowledge", determinacy=0.8, stability=0.9, entropy=0.2)
>>> extract_phase_point(particle)
(0.8, 0.9, 0.2)
>>> extract_phase_point("not an entity")
None
```

#### `get_entity_type(entity: Any, classifier: TopologicalClassifier, default: KnowledgeType = None) -> Optional[KnowledgeType]`

Returns the topological type of an entity using the provided classifier.

**Parameters:**
- `entity`: Entity to classify
- `classifier`: Instance of TopologicalClassifier to use
- `default`: Default value to return if classification fails

**Returns:**
- Value from KnowledgeType enum or the default value

**Example:**
```python
>>> classifier = TopologicalClassifier()
>>> particle = EpistemicParticle("stable knowledge", determinacy=0.9, stability=0.9, entropy=0.1)
>>> get_entity_type(particle, classifier)
<KnowledgeType.PARTICLE: 'Ψᴷ'>
```

#### `filter_by_type(entities: Iterable[Any], knowledge_type: KnowledgeType, classifier: TopologicalClassifier) -> Generator[Any, None, None]`

Filters a collection of entities, returning only those of the specified topological type.

**Parameters:**
- `entities`: Iterable collection of entities to filter
- `knowledge_type`: Target topological type from KnowledgeType enum
- `classifier`: Instance of TopologicalClassifier

**Yields:**
- Entities matching the specified type

**Example:**
```python
>>> entities = [particle1, particle2, shadow1, singularity]
>>> classifier = TopologicalClassifier()
>>> particles = list(filter_by_type(entities, KnowledgeType.PARTICLE, classifier))
>>> len(particles)
2
```

## Usage Examples

### Basic Type Checking

```python
from gtmo.utils_v2 import *

# Create test objects
singularity = O
alien_num = AlienatedNumber("undefined_concept")
particle = EpistemicParticle("knowledge", determinacy=0.8, stability=0.7, entropy=0.3)

# Check types
print(f"Is singularity: {is_ontological_singularity(singularity)}")  # True
print(f"Is alien number: {is_alienated_number(alien_num)}")          # True
print(f"Is knowledge entity: {is_knowledge_entity(particle)}")       # True
```

### Phase Space Analysis

```python
from gtmo.utils_v2 import *
from gtmo.gtmo_core_v2 import TopologicalClassifier, KnowledgeType

# Create entities
entities = [
    EpistemicParticle("stable", determinacy=0.9, stability=0.9, entropy=0.1),
    EpistemicParticle("chaotic", determinacy=0.2, stability=0.1, entropy=0.9),
    EpistemicParticle("liminal", determinacy=0.5, stability=0.5, entropy=0.5)
]

# Extract phase points
for entity in entities:
    point = extract_phase_point(entity)
    print(f"Phase point for {entity.content}: {point}")

# Classify and filter
classifier = TopologicalClassifier()
particles = list(filter_by_type(entities, KnowledgeType.PARTICLE, classifier))
print(f"Found {len(particles)} particle-type entities")
```

### Integration with GTMØ System

```python
from gtmo.utils_v2 import *
from gtmo.gtmo_core_v2 import GTMOSystemV2

# Create system
system = GTMOSystemV2()

# Process entities using utility functions
for entity in system.epistemic_particles:
    if is_epistemic_particle(entity):
        phase_point = extract_phase_point(entity)
        entity_type = get_entity_type(entity, system.classifier)
        print(f"Entity {entity.content}: Type={entity_type}, Phase={phase_point}")
```

## Testing

The module includes a comprehensive test suite that can be run directly:

```bash
python utils_v2.py
```

This will execute the demonstration block, testing all functions with various entity types and displaying the results.

## Error Handling

The module is designed to be resilient:

- **Import Failures**: Gracefully handles missing dependencies with fallback definitions
- **Type Checking**: Returns `False` for non-GTMØ objects without raising exceptions
- **Phase Space Operations**: Returns `None` for entities without phase coordinates
- **Classification**: Accepts optional default values for failed classifications

## Performance Considerations

- **Generator Usage**: `filter_by_type()` uses a generator for memory efficiency with large collections
- **Type Checking**: Uses `isinstance()` for flexibility and proper inheritance handling
- **Centralized Logic**: Reduces redundant operations across the GTMØ ecosystem

## Compatibility

- **Required**: Python 3.7+
- **Dependencies**: gtmo_core_v2 module (graceful degradation if unavailable)
- **Compatible with**: All GTMØ v2 ecosystem components

## Best Practices

1. **Always use type checking functions** instead of direct `isinstance()` calls for consistency
2. **Handle None returns** from phase space operations gracefully
3. **Provide default values** when using `get_entity_type()`
4. **Use generators** when filtering large collections to save memory
5. **Check V2_CORE_AVAILABLE** before using advanced features in dependent modules

## Troubleshooting

### Import Errors

If you see the warning about failed imports:
```
Warning: Failed to import components from 'gtmo_core_v2'. Functionality will be limited.
```

Ensure that:
1. The `gtmo_core_v2.py` file is in the correct directory
2. The module structure follows Python package conventions
3. All dependencies of `gtmo_core_v2` are installed

### Type Checking Returns False

If type checking unexpectedly returns `False`:
1. Verify the object was created with the correct class
2. Check that V2_CORE_AVAILABLE is `True`
3. Ensure you're not comparing against fallback placeholder classes

## Contributing

When extending this module:
1. Maintain the robust import pattern
2. Add corresponding type checking functions for new entity types
3. Include comprehensive docstrings
4. Update the test suite in the `__main__` block
5. Follow the established naming conventions

## Version History

- **v2.0**: Complete rewrite with v2 ecosystem compatibility
  - Added support for all v2 entity types
  - Implemented phase space operations
  - Enhanced error handling
  - Added comprehensive testing

## License

This module is part of the GTMØ project and follows the same licensing terms.

## See Also

- [gtmo_core_v2.py](gtmo_core_v2.md) - Core GTMØ v2 implementation
- [gtmo_axioms_v2.py](gtmo_axioms_v2.md) - Enhanced axiom system
- [epistemic_particles_optimized.py](epistemic_particles_optimized.md) - Optimized particle system