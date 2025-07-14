# SANB-5 Logic System Documentation

## Overview

SANB-5 (Singularity, Alienation, Nothingness, Boolean - 5 values) is a five-valued logic system designed to integrate with the Generalized Theory of Mathematical Indefiniteness (GTMØ). It extends classical binary logic to handle indefiniteness, superposition, and the boundaries of mathematical definition.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [The Five Logical Values](#the-five-logical-values)
3. [Phase Space Mapping](#phase-space-mapping)
4. [Logical Operators](#logical-operators)
5. [GTMØ Transformations](#gtmø-transformations)
6. [Interpretation Engine](#interpretation-engine)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)

## Core Concepts

### Purpose
SANB-5 provides a formal mathematical framework for reasoning about:
- **Definite states** (truth/falsehood)
- **Indefinite states** (singularities, chaos, superposition)
- **Transformations** between these states
- **Multiple interpretations** of the same situation

### Integration with GTMØ
- Maps logical values to GTMØ's Phase Space coordinates
- Models AlienatedNumber collapse mechanics
- Implements question transformation (0→0?, 1→1?)
- Tracks cognitive trajectories

## The Five Logical Values

### O (One/True) - Definite Truth
- **Meaning**: Established, verified, collapsed to a single definitive reality
- **Characteristics**: High determinacy, high stability, low entropy
- **Examples**: Mathematical facts, verified statements, axioms

### Z (Zero/False) - Definite Falsehood
- **Meaning**: Established false statement, verified negative
- **Characteristics**: High determinacy, high stability, low entropy
- **Examples**: Contradictions, disproven claims

### Ø (Phi/Singularity) - Boundary of Definition
- **Meaning**: Where logic breaks down, paradoxes, recursive self-reference
- **Characteristics**: Minimal entropy, point of logical collapse
- **Examples**: "This statement is false", division by zero

### ∞ (Infinity/Chaos) - Maximum Entropy
- **Meaning**: Informational chaos, infinite complexity, all possibilities
- **Characteristics**: Maximum entropy, minimal determinacy
- **Examples**: "What is everything?", unbounded questions

### Ψ (Psi/Superposition) - Coherent Superposition
- **Meaning**: Multiple meanings held simultaneously, metaphor, ambiguity
- **Characteristics**: High potential entropy, medium stability
- **Examples**: "Juliet is the sun", poetic language, irony

## Phase Space Mapping

Each SANB-5 value maps to coordinates in GTMØ's 3D Phase Space:

```python
# Determinacy, Stability, Entropy
O → (0.95, 0.95, 0.05)  # High determinacy/stability, low entropy
Z → (0.95, 0.95, 0.05)  # High determinacy/stability, low entropy
Ø → (0.01, 0.01, 0.01)  # Singularity point
∞ → (0.05, 0.05, 0.95)  # Low determinacy/stability, high entropy
Ψ → (0.50, 0.30, 0.70)  # Medium values, collapse potential
```

## Logical Operators

### Negation (¬) - Inversion Operator

Preserves indefinite states while inverting definite ones:

| Input | Output | Rationale |
|-------|--------|-----------|
| O | Z | Standard negation |
| Z | O | Standard negation |
| Ø | Ø | Negation of singularity remains singular |
| ∞ | ∞ | Negation of chaos is still chaos |
| Ψ | Ψ | Negation of metaphor is another metaphor |

### Conjunction (∧) - "Weakest Link" Principle

Result is as weak/limited as the weakest argument:

| ∧ | Z | O | Ø | ∞ | Ψ |
|---|---|---|---|---|---|
| **Z** | Z | Z | Z | Z | Z |
| **O** | Z | O | Ø | ∞ | Ψ |
| **Ø** | Z | Ø | Ø | Ø | Ø |
| **∞** | Z | ∞ | Ø | ∞ | ∞ |
| **Ψ** | Z | Ψ | Ø | ∞ | Ψ |

### Disjunction (∨) - "Strongest Link" Principle

Result is as strong/expansive as the strongest argument:

| ∨ | Z | O | Ø | ∞ | Ψ |
|---|---|---|---|---|---|
| **Z** | Z | O | Ø | ∞ | Ψ |
| **O** | O | O | O | O | O |
| **Ø** | Ø | O | Ø | ∞ | Ψ |
| **∞** | ∞ | O | ∞ | ∞ | ∞ |
| **Ψ** | Ψ | O | Ψ | ∞ | Ψ |

### Implication (→) - Flow of Certainty

Models how certainty propagates through logical inference:

| → | Z | O | Ø | ∞ | Ψ |
|---|---|---|---|---|---|
| **Z** | O | O | O | O | O |
| **O** | Z | O | Ø | ∞ | Ψ |
| **Ø** | Ø | O | Ø | ∞ | Ψ |
| **∞** | ∞ | O | ∞ | ∞ | ∞ |
| **Ψ** | Ψ | O | Ψ | ∞ | Ψ |

### Equivalence (↔) - Ontological Identity Test

Tests if two values have the same ontological status:

| ↔ | Z | O | Ø | ∞ | Ψ |
|---|---|---|---|---|---|
| **Z** | O | Z | Z | Z | Z |
| **O** | Z | O | Ø | ∞ | Ψ |
| **Ø** | Z | Ø | Ø | Ø | Ø |
| **∞** | Z | ∞ | Ø | ∞ | ∞ |
| **Ψ** | Z | Ψ | Ø | ∞ | O |

### XOR (⊕) - Rigorous Difference Test

Tests if values are fundamentally different and exclusive:

| ⊕ | Z | O | Ø | ∞ | Ψ |
|---|---|---|---|---|---|
| **Z** | Z | O | Ø | ∞ | Ψ |
| **O** | O | Z | O | O | O |
| **Ø** | Ø | O | Ø | ∞ | Ψ |
| **∞** | ∞ | O | ∞ | ∞ | ∞ |
| **Ψ** | Ψ | O | Ψ | ∞ | Z |

## GTMØ Transformations

### Question Transformation
Models the GTMØ principle of 0→0?, 1→1?:

```python
question_transformation(O) → Ψ  # Definite becomes superposition
question_transformation(Z) → Ψ  # Definite becomes superposition
question_transformation(Ø) → Ø  # Already indefinite
question_transformation(∞) → ∞  # Already indefinite
question_transformation(Ψ) → Ψ  # Already indefinite
```

### Observation Collapse
Models how superposition collapses through observation:

```python
observation_collapse(Ψ, observer_bias=0.1) → Z  # Low bias → false
observation_collapse(Ψ, observer_bias=0.9) → O  # High bias → true
observation_collapse(Ψ, observer_bias=0.5) → Ψ  # Medium → remains
```

### AlienatedNumber Operations
Models collapse when applying pure math to indefinite entities:

```python
alienated_number_operation(O, Ψ) → Ø  # Math + indefinite → collapse
alienated_number_operation(O, O) → O  # Pure definite remains
alienated_number_operation(Ψ, ∞) → Ø  # Any indefinite → collapse
```

## Interpretation Engine

The `InterpretationEngine` manages multiple interpretations of situations:

### Adding Interpretations
```python
engine = InterpretationEngine(logic)
engine.add_interpretation("0+1 = 1 (arithmetic)", SANB5Value.O)
engine.add_interpretation("0+1 = 01 (concatenation)", SANB5Value.PSI)
engine.add_interpretation("0+1 = 10 (reverse)", SANB5Value.PSI)
```

### Consistency Analysis
- All same value → that value
- Mix of O and Z → Ø (contradiction)
- Any ∞ → ∞ (chaos dominates)
- Otherwise → Ψ (superposition)

### Trajectory Productivity
- **Productive**: Moving toward definiteness (O/Z)
- **Unproductive**: Moving toward chaos (∞)
- **Neutral**: No clear direction

## Usage Examples

### Basic Setup
```python
from sanb5_logic import SANB5Logic, SANB5Value, GTMOTransformations

# Initialize system
logic = SANB5Logic()
transform = GTMOTransformations()
```

### Basic Operations
```python
# Negation
result = logic.neg(SANB5Value.O)  # Returns Z

# Conjunction
result = logic.and_(SANB5Value.O, SANB5Value.PSI)  # Returns Ψ

# Disjunction
result = logic.or_(SANB5Value.Z, SANB5Value.INF)  # Returns ∞
```

### GTMØ Transformations
```python
# Question transformation
result = transform.question_transformation(SANB5Value.O)  # Returns Ψ

# Observation collapse
result = transform.observation_collapse(
    SANB5Value.PSI, 
    observer_bias=0.8
)  # Returns O

# AlienatedNumber operation
result = transform.alienated_number_operation(
    SANB5Value.O, 
    SANB5Value.PSI
)  # Returns Ø
```

### Multiple Interpretations
```python
from sanb5_logic import InterpretationEngine

engine = InterpretationEngine(logic)

# Add interpretations from paper experiment
engine.add_interpretation("0+1 = 1", SANB5Value.O)
engine.add_interpretation("0+1 = 01", SANB5Value.PSI)
engine.add_interpretation("0+1 = 10", SANB5Value.PSI)
engine.add_interpretation("0+1 = 2", SANB5Value.O)

# Analyze
consistency = engine.analyze_consistency()  # Returns Ψ
productivity = engine.trajectory_productivity()  # Returns "Productive"
```

## API Reference

### Classes

#### `SANB5Value(Enum)`
The five logical values as an enumeration.

#### `PhaseSpaceCoordinates`
- **Attributes**: `determinacy`, `stability`, `entropy`
- **Methods**: `from_sanb5(value)` - Convert SANB5 value to coordinates

#### `SANB5Logic`
Main logic system with truth tables.
- **Methods**:
  - `neg(a)` - Negation
  - `and_(a, b)` - Conjunction
  - `or_(a, b)` - Disjunction
  - `implies(a, b)` - Implication
  - `equiv(a, b)` - Equivalence
  - `xor(a, b)` - Exclusive OR

#### `GTMOTransformations`
Static methods for GTMØ-specific transformations:
- `question_transformation(value)` - Apply question operator
- `observation_collapse(value, observer_bias)` - Collapse superposition
- `alienated_number_operation(v1, v2)` - Model arithmetic collapse

#### `InterpretationEngine`
Manages multiple interpretations:
- **Methods**:
  - `add_interpretation(description, value)` - Add interpretation
  - `analyze_consistency()` - Determine overall consistency
  - `trajectory_productivity()` - Assess trajectory direction

## Integration with GTMØ Theory

SANB-5 provides the formal logical foundation for GTMØ's concepts:

1. **Singularity (Ø)** maps to the ontological singularity where AlienatedNumbers collapse
2. **Superposition (Ψ)** models AlienatedNumbers before collapse
3. **Question transformation** formalizes the 0→0?, 1→1? operation
4. **Phase Space mapping** connects logic to GTMØ's topological framework
5. **Trajectory analysis** implements productive vs unproductive cognitive paths

This system enables formal reasoning about indefiniteness, providing a mathematical framework for AI systems that can handle ambiguity, metaphor, and the boundaries of definition.