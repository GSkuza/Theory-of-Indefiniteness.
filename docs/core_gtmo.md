# GTMØ Core Primitives Documentation

## Overview

The `core.py` module provides the foundational runtime primitives for the **Generalized Theory of Mathematical Indefiniteness (GTMØ)**. This module implements the basic mathematical objects that form the core of the indefiniteness theory: the ontological singularity **Ø** and alienated numbers **ℓ∅**.

**Author of the Theory of Indefiniteness (Teoria Niedefinitywności):** Grzegorz Skuza (Poland)

This module serves as the mathematical foundation upon which all other GTMØ modules are built, providing the primitive objects that represent indefiniteness in formal mathematical systems.

## Table of Contents

1. [Core Mathematical Objects](#core-mathematical-objects)
2. [Design Principles](#design-principles)
3. [Ontological Singularity (Ø)](#ontological-singularity)
4. [Alienated Numbers (ℓ∅)](#alienated-numbers)
5. [Arithmetic Operations](#arithmetic-operations)
6. [Strict Mode](#strict-mode)
7. [Integration with gtmo_axioms.py](#integration-with-gtmo-axioms)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Advanced Topics](#advanced-topics)

---

## Core Mathematical Objects

The GTMØ core module defines two fundamental mathematical entities:

- **`O`** - The unique ontological singularity (Ø)
- **`AlienatedNumber`** - Symbolic placeholders for alienated numbers (ℓ∅)

These objects represent different forms of mathematical indefiniteness and serve as the building blocks for more complex GTMØ operations defined in other modules.

---

## Design Principles

### Singleton Safety
The ontological singularity `O` is implemented as a true singleton, ensuring that `O is O` remains true under all circumstances, including:
- Module imports and reloads
- Pickling and unpickling operations
- Multiple instantiation attempts

### Fail-Fast Strict Mode
The module supports a strict mode (controlled by `GTM_STRICT` environment variable) where:
- Arithmetic operations with Ø or ℓ∅ raise `SingularityError`
- This prevents silent absorption/collapse behavior
- Enables debugging and formal verification

### Compatibility
Maintains backward compatibility with existing GTMØ implementations:
- `O_empty_singularity` representation format
- Absorbing operator behavior in non-strict mode
- Standard Python numeric protocol compliance

### Pythonic Integration
Full integration with Python's numeric system:
- Inherits from `numbers.Number` for type checking
- Implements `__bool__`, `__eq__`, `__hash__` for collections
- Supports mypy type checking (PEP 561 compliant)

---

## Ontological Singularity (Ø)

### Mathematical Properties

The ontological singularity `O` represents the fundamental indefiniteness in GTMØ theory. It has several unique mathematical properties:

```python
from core import O

# Identity and uniqueness
assert O is O  # Always true (singleton)
assert O == O  # Equality check

# Falsy behavior
assert not O   # O is falsy
assert bool(O) == False

# Absorbing element property
result = O + 5    # Returns O (in non-strict mode)
result = 10 * O   # Returns O (in non-strict mode)
```

### Implementation Details

The `Singularity` class uses several advanced Python features:

#### Singleton Metaclass
```python
class _SingletonMeta(ABCMeta):
    """Metaklasa enforcing the singleton pattern (one shared instance)."""
    _instance: "Singularity | None" = None
    
    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
```

#### Absorbing Operations
All arithmetic operations return the singularity itself:
```python
@_absorbing_operation("__add__")
def __add__(self, other):
    # In non-strict mode: always returns O
    # In strict mode: raises SingularityError
```

#### JSON Serialization
```python
def to_json(self) -> str:
    return "\"O_empty_singularity\""
```

### Correspondence with GTMØ Axioms

The ontological singularity directly implements several formal axioms from `gtmo_axioms.py`:

- **AX1**: Fundamental difference from {0, 1, ∞}
- **AX6**: Minimal entropy property (E_GTMØ(Ø) = 0.0)
- **AX9**: Standard operator irreducibility
- **AX10**: Requires meta-operators for meaningful processing

---

## Alienated Numbers (ℓ∅)

### Conceptual Foundation

Alienated numbers represent symbolic indefiniteness - mathematical objects that exist in a state of epistemic alienation from standard number systems.

```python
from core import AlienatedNumber

# Create alienated numbers
undefined_concept = AlienatedNumber("undefined_concept")
anonymous_alien = AlienatedNumber()  # identifier: "anonymous"
numeric_alien = AlienatedNumber(42)  # identifier: 42
```

### Key Properties

#### Identifier-Based Identity
Each alienated number has a unique identifier:
```python
alien1 = AlienatedNumber("concept_a")
alien2 = AlienatedNumber("concept_a")
alien3 = AlienatedNumber("concept_b")

assert alien1 == alien2  # Same identifier
assert alien1 != alien3  # Different identifier
assert hash(alien1) == hash(alien2)  # Hashable for collections
```

#### GTMØ Metrics
Alienated numbers provide built-in GTMØ measurements:
```python
alien = AlienatedNumber("test_concept")

# Epistemic purity score (approaching 1.0)
purity = alien.psi_gtm_score()  # 0.999999999

# Cognitive entropy (approaching 0.0)
entropy = alien.e_gtm_entropy()  # 1e-9
```

#### Arithmetic Collapse
All arithmetic operations collapse to the ontological singularity:
```python
alien = AlienatedNumber("test")
result = alien + 5    # Returns O (in non-strict mode)
result = alien * 2    # Returns O (in non-strict mode)
result = alien ** 3   # Returns O (in non-strict mode)
```

### Implementation Constants
```python
class AlienatedNumber(Number):
    # Class constants for GTMØ metrics
    PSI_GTM_SCORE: Final[float] = 0.999_999_999  # Near-maximal purity
    E_GTM_ENTROPY: Final[float] = 1e-9           # Near-minimal entropy
```

### Correspondence with GTMØ Theory

Alienated numbers bridge the gap between definable mathematics and the ontological singularity:
- High epistemic purity (approaching Ø levels)
- Low cognitive entropy (but not minimal like Ø)
- Symbolic representation of indefinite concepts
- Collapse to Ø under arithmetic operations

---

## Arithmetic Operations

### Absorbing Element Behavior

Both Ø and ℓ∅ act as absorbing elements in GTMØ arithmetic:

```python
from core import O, AlienatedNumber

# Ontological singularity absorption
assert O + 42 is O
assert 100 * O is O
assert O / 7 is O
assert O ** 3 is O

# Alienated number collapse
alien = AlienatedNumber("test")
assert alien + 5 is O
assert alien * 2 is O
assert 10 - alien is O
```

### Operation Decorator

The `@_absorbing_operation` decorator implements the absorption logic:

```python
def _absorbing_operation(method_name: str):
    def decorator(fn_placeholder):
        @wraps(fn_placeholder)
        def wrapper(self, *args, **kwargs):
            if STRICT_MODE:
                op_source = "Ø" if isinstance(self, Singularity) else "ℓ∅"
                raise SingularityError(
                    f"Operation '{method_name}' with {op_source} is forbidden in STRICT mode"
                )
            return get_singularity()
        return wrapper
    return decorator
```

### Supported Operations

Both `Singularity` and `AlienatedNumber` support all standard arithmetic operations:
- Addition: `__add__`, `__radd__`
- Subtraction: `__sub__`, `__rsub__`
- Multiplication: `__mul__`, `__rmul__`
- Division: `__truediv__`, `__rtruediv__`
- Exponentiation: `__pow__`, `__rpow__`

---

## Strict Mode

### Configuration

Strict mode is controlled by the environment variable `GTM_STRICT`:

```bash
# Enable strict mode
export GTM_STRICT=1

# Disable strict mode (default)
export GTM_STRICT=0
```

### Behavior in Strict Mode

When `GTM_STRICT=1`, all arithmetic operations with Ø or ℓ∅ raise `SingularityError`:

```python
import os
os.environ['GTM_STRICT'] = '1'

from core import O, AlienatedNumber, SingularityError

try:
    result = O + 5
except SingularityError as e:
    print(f"Error: {e}")
    # Error: Operation '__add__' with Ø is forbidden in STRICT mode

try:
    alien = AlienatedNumber("test")
    result = alien * 2
except SingularityError as e:
    print(f"Error: {e}")
    # Error: Operation '__mul__' with ℓ∅ is forbidden in STRICT mode
```

### Use Cases for Strict Mode

- **Formal verification**: Ensure no accidental indefiniteness propagation
- **Debugging**: Catch unintended operations with indefinite objects
- **Educational**: Understand where indefiniteness enters calculations
- **Research**: Study behavior of systems under strict indefiniteness rules

---

## Integration with gtmo_axioms.py

The core module provides the primitive objects that the axioms module operates upon. Here's how they correspond:

### Axiomatic Processing

The `gtmo_axioms.py` module defines meta-operators that can properly handle core primitives:

```python
# In gtmo_axioms.py
from core import O, AlienatedNumber

class PsiOperator:
    def _process_singularity(self, context):
        # Implements AX6, AX10 for Ø
        return OperationResult(
            value={'score': 1.0, 'type': 'Ø (ontological_singularity)'},
            operator_type=OperatorType.META,
            axiom_compliance={'AX6': True, 'AX10': True}
        )
    
    def _process_alienated_number(self, alienated_num, context):
        # Uses built-in GTMØ metrics from AlienatedNumber
        psi_score = alienated_num.psi_gtm_score()
        return OperationResult(
            value={'score': psi_score, 'type': f'ℓ∅ ({alienated_num.identifier})'},
            operator_type=OperatorType.META
        )
```

### Entropy Operator Integration

```python
# In gtmo_axioms.py
class EntropyOperator:
    def _process_singularity_entropy(self, context):
        # Implements AX6: Ø has minimal entropy
        return OperationResult(
            value={'total_entropy': 0.0},  # AX6 compliance
            operator_type=OperatorType.META,
            axiom_compliance={'AX6': True}
        )
    
    def _process_alienated_entropy(self, alienated_num, context):
        # Uses built-in entropy from AlienatedNumber
        entropy_value = alienated_num.e_gtm_entropy()
        return OperationResult(
            value={'total_entropy': entropy_value},
            operator_type=OperatorType.META
        )
```

### Axiom Validation

The `AxiomValidator` in `gtmo_axioms.py` validates operations with core primitives:

```python
# Example validation flow
validator = AxiomValidator()

# Test with ontological singularity
psi_result = psi_operator(O, context)
compliance = validator.validate_operation('Ψ_GTMØ', [O], psi_result)

# Validates:
# AX1: O ≠ {0, 1, ∞}
# AX6: Minimal entropy
# AX9: Meta-operator required
# AX10: Meta-operator applied
```

### Meta-Feedback Loop Integration

```python
# In gtmo_axioms.py MetaFeedbackLoop
def run(self, fragments, initial_scores, iterations=5):
    for fragment in fragments:
        if fragment is O:
            # Use meta-operator (AX10)
            psi_result = self.psi_operator(fragment, context)
        elif isinstance(fragment, AlienatedNumber):
            # Use meta-operator for ℓ∅
            psi_result = self.psi_operator(fragment, context)
        else:
            # Standard processing
            psi_result = self.psi_operator(fragment, context)
```

### Emergence Detection

Core primitives can trigger emergence detection:

```python
# In gtmo_axioms.py EmergenceDetector
def detect_emergence(self, fragment, psi_result, entropy_result):
    if fragment is O:
        # Ontological singularity doesn't exhibit emergence
        return {'is_emergent': False, 'emergent_type': None}
    elif isinstance(fragment, AlienatedNumber):
        # Alienated numbers may indicate emergent indefiniteness
        return self._analyze_alienated_emergence(fragment)
```

---

## Usage Examples

### Basic Operations

```python
from core import O, AlienatedNumber

# Create and use ontological singularity
print(f"Ontological singularity: {O}")  # O_empty_singularity
print(f"Is falsy: {not O}")             # True
print(f"Absorption: {O + 42}")          # O_empty_singularity

# Create and use alienated numbers
concept = AlienatedNumber("undefined_concept")
print(f"Alienated number: {concept}")   # l_empty_num(undefined_concept)
print(f"Purity score: {concept.psi_gtm_score()}")  # 0.999999999
print(f"Entropy: {concept.e_gtm_entropy()}")       # 1e-09
print(f"Collapse: {concept * 5}")       # O_empty_singularity
```

### Collections and Hashing

```python
from core import O, AlienatedNumber

# Ontological singularity in collections
singularities = {O, O, O}
assert len(singularities) == 1  # Only one unique element

# Alienated numbers in collections
aliens = {
    AlienatedNumber("concept_a"),
    AlienatedNumber("concept_a"),  # Duplicate
    AlienatedNumber("concept_b")
}
assert len(aliens) == 2  # Two unique concepts

# Using as dictionary keys
entropy_map = {
    O: 0.0,
    AlienatedNumber("test"): 1e-9
}
```

### JSON Serialization

```python
import json
from core import O, AlienatedNumber

# Serialize ontological singularity
o_json = O.to_json()
print(o_json)  # "O_empty_singularity"

# Serialize alienated number
alien = AlienatedNumber("test_concept")
alien_json = alien.to_json()
print(alien_json)  # "l_empty_num(test_concept)"

# Custom JSON encoder
class GTMOEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)

data = {"singularity": O, "alienated": AlienatedNumber("test")}
json_str = json.dumps(data, cls=GTMOEncoder)
```

### Integration with GTMØ System

```python
from core import O, AlienatedNumber
from gtmo_axioms import create_gtmo_system

# Create GTMØ operator system
psi_op, entropy_op, meta_loop = create_gtmo_system()

# Process core primitives
test_objects = [
    O,                              # Ontological singularity
    AlienatedNumber("undefined"),   # Alienated number
    "Regular knowledge fragment"    # Standard content
]

for obj in test_objects:
    # Apply GTMØ operators
    psi_result = psi_op(obj, {'all_scores': [0.3, 0.5, 0.7]})
    entropy_result = entropy_op(obj)
    
    print(f"Object: {obj}")
    print(f"Classification: {psi_result.value.get('classification')}")
    print(f"Operator type: {psi_result.operator_type.name}")
    print(f"Entropy: {entropy_result.value.get('total_entropy')}")
    print("-" * 50)
```

---

## API Reference

### Core Functions

#### `get_singularity() -> Singularity`
Returns the unique global Ø instance. Used internally for singleton management and pickling support.

### Classes

#### `Singularity`
The ontological singularity class implementing Ø.

**Methods:**
- `__bool__() -> bool`: Returns `False` (falsy behavior)
- `__eq__(other) -> bool`: Equality comparison with other objects
- `__hash__() -> int`: Hash for collections (based on "O_empty_singularity")
- `__repr__() -> str`: Returns "O_empty_singularity"
- `to_json() -> str`: JSON representation
- Arithmetic methods: All return `O` (or raise `SingularityError` in strict mode)

**Properties:**
- Singleton instance (enforced by metaclass)
- Absorbing element for all arithmetic operations
- Inherits from `numbers.Number`

#### `AlienatedNumber`
Symbolic placeholder for alienated numbers ℓ∅.

**Constructor:**
```python
AlienatedNumber(identifier: str | int | float | None = None)
```

**Methods:**
- `__eq__(other) -> bool`: Equality based on identifier
- `__hash__() -> int`: Hash based on identifier
- `__repr__() -> str`: Returns `l_empty_num(identifier)`
- `psi_gtm_score() -> float`: Returns epistemic purity score (0.999999999)
- `e_gtm_entropy() -> float`: Returns cognitive entropy (1e-9)
- `to_json() -> str`: JSON representation
- Arithmetic methods: All return `O` (or raise `SingularityError` in strict mode)

**Class Constants:**
- `PSI_GTM_SCORE: Final[float] = 0.999_999_999`
- `E_GTM_ENTROPY: Final[float] = 1e-9`

### Exceptions

#### `SingularityError`
Inherits from `ArithmeticError`. Raised when operations with Ø or ℓ∅ are attempted in strict mode.

### Module Constants

#### `O: Final[Singularity]`
The global ontological singularity instance.

#### `STRICT_MODE: Final[bool]`
Boolean flag indicating whether strict mode is enabled (based on `GTM_STRICT` environment variable).

---

## Advanced Topics

### Metaclass Implementation

The singleton pattern is implemented using a custom metaclass that inherits from `ABCMeta`:

```python
class _SingletonMeta(ABCMeta):
    _instance: "Singularity | None" = None
    
    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
```

This ensures:
- Thread safety (Python GIL protection)
- Inheritance compatibility
- Abstract method support if needed

### Decorator Pattern for Operations

The absorbing operation behavior is implemented using a decorator factory:

```python
def _absorbing_operation(method_name: str):
    def decorator(fn_placeholder):
        @wraps(fn_placeholder)
        def wrapper(self, *args, **kwargs):
            # Implementation logic here
            return get_singularity()
        return wrapper
    return decorator
```

This pattern:
- Centralizes absorption logic
- Maintains method signatures
- Preserves metadata with `@wraps`
- Enables consistent error handling

### Type System Integration

The module is fully typed and mypy-compliant:

```python
from typing import Any, Final
from numbers import Number

# Proper type annotations
O: Final[Singularity] = get_singularity()
STRICT_MODE: Final[bool] = os.getenv("GTM_STRICT", "0") == "1"

class AlienatedNumber(Number):
    __slots__ = ("identifier",)
    
    def __init__(self, identifier: str | int | float | None = None):
        self.identifier = identifier if identifier is not None else "anonymous"
```

### Memory Optimization

Both classes use memory optimization techniques:

- `Singularity` uses `__slots__ = ()` for minimal memory footprint
- `AlienatedNumber` uses `__slots__ = ("identifier",)` to prevent dynamic attributes
- Singleton pattern ensures only one `Singularity` instance exists

### Pickling Support

Custom pickling support ensures singleton preservation:

```python
def __reduce__(self):
    return (get_singularity, ())
```

This guarantees that unpickled objects maintain singleton identity.

---

## Theoretical Significance

The core module implements the fundamental mathematical objects of Grzegorz Skuza's Theory of Indefiniteness:

1. **Ontological Singularity (Ø)**: Represents absolute indefiniteness that cannot be reduced to or derived from standard mathematical objects

2. **Alienated Numbers (ℓ∅)**: Bridge the gap between definable mathematics and absolute indefiniteness, representing concepts that exist in epistemic alienation

3. **Absorption Dynamics**: Mathematical operations with indefinite objects result in indefiniteness, reflecting the theory's principle that indefiniteness propagates through formal systems

4. **Meta-Mathematical Foundation**: Provides the primitives upon which meta-operators and advanced GTMØ constructs are built

This implementation serves as the computational foundation for exploring indefiniteness in mathematical and cognitive systems, enabling practical applications of theoretical indefiniteness concepts.

---

This documentation provides a comprehensive guide to the GTMØ core primitives, their implementation, and their integration with the broader GTMØ framework. The module serves as the mathematical foundation for all indefiniteness computations in the system.