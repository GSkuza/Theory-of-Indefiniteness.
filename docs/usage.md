# GTMØ Usage Guide

## Introduction

This guide explains how to use the modules and functionalities provided in the Generalized Theory of Mathematical Indefiniteness (GTMØ) library. The GTMØ library includes modules for managing indefiniteness, classifying cognitive fragments, and analyzing cognitive trajectories.

## Installation

Ensure Python 3.9 or newer is installed, then install required dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Core Module (`core.py`)

Provides foundational objects for working with indefiniteness:

* **Singularity (Ø)**: Represents ontological singularity, an absorbing element for arithmetic.
* **AlienatedNumber (ℓ∅)**: Represents symbolic placeholders leading to Ø upon arithmetic operations.

```python
from gtmo.core import O, AlienatedNumber

alien_num = AlienatedNumber("example")
result = alien_num + 10  # Results in O_empty_singularity
print(result)
```

### Classification Module (`classification.py`)

Implements logic for classifying cognitive fragments using the Ψ\_GTMØ method and managing meta-feedback loops:

```python
from gtmo.classification import classify_fragment

fragment = "some cognitive data"
classification_result = classify_fragment(fragment)
print(classification_result)
```

### Topology Module (`topology.py`)

Analyzes cognitive trajectories (φ(t)) and evaluates cognitive entropy (E(x)):

```python
from gtmo.topology import get_trajectory_state_phi_t, evaluate_field_E_x
from gtmo.core import O, AlienatedNumber

state = get_trajectory_state_phi_t(AlienatedNumber("example"), t=1.0)
print(state)  # Outputs O_empty_singularity

entropy = evaluate_field_E_x(O, "cognitive_entropy")
print(entropy)  # Outputs 0.0
```

## Advanced Features

### Strict Mode

Enable strict arithmetic handling to raise exceptions instead of collapsing silently:

```bash
export GTM_STRICT=1
```

Example usage:

```python
from gtmo.core import O, AlienatedNumber, SingularityError

alien_num = AlienatedNumber("strict_test")

try:
    result = alien_num + 5
except SingularityError as e:
    print(f"Operation disallowed: {e}")
```

## Examples

Examples of how to utilize the library are available in the `examples/` directory:

* `simple_run.py`: Demonstrates basic module interactions.
* `plotting_trajectory.py`: Shows how to visualize cognitive trajectories and entropy.

Run an example:

```bash
python examples/simple_run.py
```

## Testing

To execute unit tests:

```bash
pytest tests/
```

Tests are provided for core functionalities (`test_core.py`) and classification logic (`test_classification.py`).

## Further Information

Refer to additional documents:

* [Axioms](axioms.md): Detailed axioms underpinning GTMØ.
* [Theory](theory.md): Explanation of the theoretical foundations of GTMØ.

For contributions and further details, please consult the project's main repository.
