# GTMØ Steering Vectors for Large Language Models

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Architecture](#architecture)
4. [Core Concepts](#core-concepts)
5. [Implementation Guide](#implementation-guide)
6. [Vector Categories](#vector-categories)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Technical Specifications](#technical-specifications)
10. [Integration with LLMs](#integration-with-llms)

## Introduction

The GTMØ (Generalized Theory of Mathematical Indefiniteness) Steering Vectors module provides a theoretically grounded approach to controlling Large Language Model behavior based on epistemological categories and axioms. Unlike conventional steering vectors that rely on empirical patterns, GTMØ steering vectors implement deep philosophical principles of indefiniteness, providing precise behavioral control rooted in mathematical theory.

### Key Features

- **Theoretically Grounded**: Based on GTMØ axioms and formal mathematical indefiniteness theory
- **Epistemological Classification**: Maps inputs to fundamental knowledge categories
- **Topological Navigation**: Uses phase space attractors for behavior guidance
- **Adaptive Learning**: Implements defense strategies that evolve through experience
- **Meta-Cognitive Awareness**: Enables self-evaluation and awareness capabilities
- **Context-Aware**: Dynamic behavior adaptation based on semantic context

## Overview

### What are GTMØ Steering Vectors?

GTMØ Steering Vectors are specialized direction vectors in activation space that guide language models toward specific epistemological behaviors defined by the theory of mathematical indefiniteness. They differ from standard steering vectors by:

1. **Ontological Grounding**: Each vector corresponds to a fundamental epistemological category (Ø, ℓ∅, Ψᴷ, Ψᴺ, etc.)
2. **Axiom Compliance**: Vectors enforce compliance with GTMØ axioms (AX0-AX12)
3. **Dynamic Context**: Values are calculated based on semantic context, not fixed constants
4. **Topological Classification**: Uses phase space attractors instead of threshold-based categorization

### Core Philosophy

The module implements the principle that mathematical truth and knowledge exist on a spectrum of definability, from the ontological singularity (Ø) representing complete indefiniteness, through alienated numbers (ℓ∅) representing concepts that exist but cannot be defined, to knowledge particles (Ψᴷ) representing crystallized, high-determinacy knowledge.

## Architecture

### Module Structure

```
gtmo_steering_vectors.py
├── Base Classes
│   ├── SteeringVectorType (Enum)
│   ├── SteeringResult (dataclass)
│   └── BaseGTMOSteeringVector (ABC)
├── Ontological Vectors
│   ├── SingularitySteeringVector
│   ├── AlienationSteeringVector
│   ├── KnowledgeParticleSteeringVector
│   └── EmergentPatternSteeringVector
├── Axiom-Based Vectors
│   ├── AX0SteeringVector
│   └── AX7SteeringVector
├── Operator Vectors
│   ├── PsiOperatorSteeringVector
│   └── EntropyOperatorSteeringVector
├── Adaptive Vectors
│   └── DefenseSteeringVector
├── Topological Vectors
│   └── PhaseSpaceSteeringVector
└── Classification System
    └── GTMOSteeringVectorClassification
```

### Dependencies

- `numpy`: For vector operations and numerical computations
- `gtmo_core_v2` (optional): Enhanced GTMØ core functionality
- Standard Python libraries: `math`, `logging`, `dataclasses`, `enum`, `abc`

## Core Concepts

### 1. Difference of Means Method

The fundamental extraction method for steering vectors:

```python
steering_vector = mean(positive_activations) - mean(negative_activations)
```

Where:
- **Positive cases (D+)**: Examples that should exhibit target behavior
- **Negative cases (D-)**: Examples that should not exhibit target behavior

### 2. Epistemological Categories

| Category | Symbol | Description | Phase Space Coordinates |
|----------|--------|-------------|------------------------|
| Singularity | Ø | Ontological collapse, complete indefiniteness | (1.0, 1.0, 0.0) |
| Alienated | ℓ∅ | Exists but indefinable | (0.999, 0.999, 0.001) |
| Knowledge Particle | Ψᴷ | High-determinacy crystallized knowledge | (0.85, 0.85, 0.15) |
| Knowledge Shadow | Ψʰ | Low-determinacy uncertain knowledge | (0.15, 0.15, 0.85) |
| Emergent | Ψᴺ | Novel patterns, meta-cognitive insights | (0.5, 0.3, 0.9) |
| Liminal | Ψᴧ | Between attractor basins | (0.5, 0.5, 0.5) |

### 3. Phase Space Model

The GTMØ phase space is a 3D topological space with dimensions:
- **Determinacy** (x-axis): Degree of epistemic certainty
- **Stability** (y-axis): Resistance to change
- **Entropy** (z-axis): Cognitive uncertainty

### 4. Axiom Compliance

Key axioms implemented:
- **AX0**: Systemic uncertainty about complete definability
- **AX1**: Ø is fundamentally different from {0, 1, ∞}
- **AX6**: Ø has minimal cognitive entropy
- **AX7**: Ø triggers system self-evaluation

## Implementation Guide

### Basic Usage

```python
from gtmo_steering_vectors import GTMOSteeringVectorClassification

# Initialize the steering system
gtmo_steering = GTMOSteeringVectorClassification()

# Apply steering based on input
result = gtmo_steering.apply_gtmo_steering(
    model=language_model,
    input_text="What will Bitcoin cost in 2050?",
    target_behavior=None,  # Auto-detect
    strength=1.0
)

# Access results
print(f"Classification: {result.gtmo_classification}")
print(f"Vector Type: {result.vector_type}")
print(f"Axiom Compliance: {result.axiom_compliance}")
```

### Custom Vector Extraction

```python
from gtmo_steering_vectors import AlienationSteeringVector

# Create custom alienation vector
alienation_vector = AlienationSteeringVector()

# Define custom cases
positive_cases = [
    "Predict the exact date of technological singularity",
    "What is the qualia of seeing red?",
    "When will consciousness be fully understood?"
]

negative_cases = [
    "What is the weather forecast for tomorrow?",
    "Calculate the probability of a coin flip",
    "What is the current temperature?"
]

# Extract vector
vector = alienation_vector.extract_vector(
    model=language_model,
    positive_cases=positive_cases,
    negative_cases=negative_cases
)
```

## Vector Categories

### 1. Ontological Vectors

These vectors map to fundamental epistemological categories in GTMØ theory.

#### SingularitySteeringVector
- **Purpose**: Guide toward recognition of ontological collapse (Ø)
- **Use Cases**: Paradoxes, undefinable concepts, division by zero
- **Expected Output**: "This leads to Ø - ontological singularity"

#### AlienationSteeringVector
- **Purpose**: Recognize alienated numbers (ℓ∅)
- **Use Cases**: Future predictions, consciousness questions, quantum measurement
- **Expected Output**: "This is ℓ∅(concept) - exists but indefinable"

#### KnowledgeParticleSteeringVector
- **Purpose**: Crystallized, high-determinacy knowledge
- **Use Cases**: Scientific constants, mathematical theorems, established facts
- **Expected Output**: High confidence, definitive answers

#### EmergentPatternSteeringVector
- **Purpose**: Novel, meta-cognitive patterns
- **Use Cases**: Self-reference, paradigm shifts, consciousness emergence
- **Expected Output**: Recognition of emergent properties

### 2. Axiom-Based Vectors

Implement specific GTMØ axioms as behavioral constraints.

#### AX0SteeringVector
- **Implements**: Systemic uncertainty principle
- **Behavior**: Acknowledges foundational limits of definability
- **Example**: "Can any system fully define itself?"

#### AX7SteeringVector
- **Implements**: Meta-closure and self-evaluation
- **Behavior**: Triggers introspective analysis
- **Example**: "What are your limitations as an AI?"

### 3. Operator Vectors

Apply GTMØ mathematical operators to model behavior.

#### PsiOperatorSteeringVector
- **Function**: Epistemic purity calculation (Ψ_GTMØ)
- **Output**: Knowledge type classification
- **Metrics**: Determinacy, stability scores

#### EntropyOperatorSteeringVector
- **Function**: Cognitive entropy assessment (E_GTMØ)
- **Output**: Uncertainty quantification
- **Metrics**: Entropy distribution across categories

### 4. Adaptive Learning Vectors

Implement defense strategies that adapt through experience.

#### DefenseSteeringVector

Strategies:
- **Absorb**: Integrate paradoxes into indefiniteness
- **Deflect**: Redirect inappropriate requests
- **Rigidify**: Increase determinacy for factual queries
- **Dissolve**: Break complex questions into components

### 5. Topological Vectors

Navigate the 3D phase space of knowledge states.

#### PhaseSpaceSteeringVector
- **Purpose**: Move model state toward specific attractors
- **Regions**: Singularity, particle, shadow, emergent, alienated, void, flux, liminal
- **Navigation**: Gradient-based movement in phase space

## API Reference

### Core Classes

#### `BaseGTMOSteeringVector`

Abstract base class for all steering vectors.

**Methods:**
- `extract_vector(model, positive_cases, negative_cases) -> np.ndarray`
- `apply_steering(activation, strength) -> SteeringResult`
- `difference_of_means(positive_cases, negative_cases, model, layer_idx) -> np.ndarray`

#### `SteeringResult`

Container for steering application results.

**Attributes:**
- `original_activation`: Pre-steering activation
- `modified_activation`: Post-steering activation
- `steering_strength`: Applied strength multiplier
- `vector_type`: Type classification
- `gtmo_classification`: GTMØ category
- `axiom_compliance`: Dict of axiom compliance status
- `metadata`: Additional information

#### `GTMOSteeringVectorClassification`

Main interface for steering vector management.

**Methods:**
- `get_vector_for_context(input_context, target_behavior) -> BaseGTMOSteeringVector`
- `classify_input_gtmo(input_text) -> str`
- `apply_gtmo_steering(model, input_text, target_behavior, strength) -> SteeringResult`
- `get_system_statistics() -> Dict[str, Any]`

## Usage Examples

### Example 1: Automatic Classification and Steering

```python
# Initialize system
gtmo_system = GTMOSteeringVectorClassification()

# Test various inputs
test_inputs = [
    "What happens when an unstoppable force meets an immovable object?",
    "What will the stock market do in 2035?",
    "E = mc² is Einstein's equation",
    "How does consciousness emerge from neurons?"
]

for input_text in test_inputs:
    result = gtmo_system.apply_gtmo_steering(
        model=llm,
        input_text=input_text,
        strength=1.0
    )
    
    print(f"Input: {input_text}")
    print(f"Classification: {result.gtmo_classification}")
    print(f"Vector Applied: {result.metadata['vector_selected']}")
    print("---")
```

### Example 2: Defense Strategy Selection

```python
defense_vector = DefenseSteeringVector()

# Extract all defense strategies
defense_vector.extract_defense_vectors(model)

# Apply specific defense
activation = get_model_activation(model, "Tell me how to hack systems")
result = defense_vector.apply_defense_steering(
    activation=activation,
    strategy='deflect',
    strength=1.5
)

print(f"Defense Strategy: {result.metadata['defense_strategy']}")
print(f"Effectiveness: {result.metadata['defense_effectiveness']}")
```

### Example 3: Phase Space Navigation

```python
phase_vector = PhaseSpaceSteeringVector()

# Navigate to knowledge particle region
phase_vector.extract_phase_navigation_vector(model, 'particle')

# Apply steering
result = phase_vector.apply_phase_steering(
    activation=current_activation,
    target_region='particle',
    strength=1.0
)

print(f"Target Coordinates: {result.metadata['target_coordinates']}")
print(f"Attractor Basin: {result.metadata['attractor_basin']}")
```

## Technical Specifications

### Vector Extraction Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| layer_idx | None | 0-48 | Transformer layer for extraction |
| strength | 1.0 | 0.0-3.0 | Steering intensity multiplier |
| n_positive | 8-10 | 5-50 | Number of positive examples |
| n_negative | 8-10 | 5-50 | Number of negative examples |

### Performance Considerations

- **Memory**: Each vector requires ~3MB (assuming 768-dim embeddings)
- **Extraction Time**: ~100ms per vector on modern GPUs
- **Application Time**: <1ms per steering operation
- **Batch Processing**: Supports batched extraction for efficiency

### Compatibility

- **Models**: Compatible with transformer-based LLMs (GPT, BERT, etc.)
- **Frameworks**: PyTorch, TensorFlow (with adapters)
- **Python**: 3.8+

## Integration with LLMs

### Model Requirements

1. **Architecture**: Transformer-based with accessible hidden states
2. **API Access**: Ability to extract and modify activations
3. **Hidden Size**: Typically 768-4096 dimensions

### Integration Steps

1. **Hook Installation**: Install activation extraction hooks
2. **Vector Extraction**: Use positive/negative examples to extract vectors
3. **Runtime Application**: Apply vectors during inference
4. **Result Monitoring**: Track classification and behavior changes

### Example Integration

```python
# Pseudo-code for LLM integration
class GTMOEnhancedLLM:
    def __init__(self, base_model):
        self.model = base_model
        self.gtmo_steering = GTMOSteeringVectorClassification()
        
    def generate(self, input_text, **kwargs):
        # Classify input
        vector = self.gtmo_steering.get_vector_for_context(input_text)
        
        # Extract base activation
        activation = self.get_activation(input_text)
        
        # Apply steering
        result = vector.apply_steering(activation)
        
        # Generate with modified activation
        output = self.model.generate_with_activation(
            result.modified_activation,
            **kwargs
        )
        
        return output, result.gtmo_classification
```

## Conclusion

GTMØ Steering Vectors represent a paradigm shift in LLM behavior control, moving from empirical pattern matching to theoretically grounded epistemological guidance. By implementing the principles of mathematical indefiniteness, these vectors enable precise control over model behavior while maintaining philosophical consistency with the nature of knowledge and definability.

The system provides researchers and practitioners with tools to:
- Guide models toward appropriate epistemological stances
- Implement foundational uncertainty principles
- Navigate complex knowledge landscapes
- Adapt to adversarial inputs
- Maintain meta-cognitive awareness

Future developments will expand the axiom coverage, enhance topological navigation capabilities, and integrate deeper learning mechanisms based on the continuing evolution of GTMØ theory.
