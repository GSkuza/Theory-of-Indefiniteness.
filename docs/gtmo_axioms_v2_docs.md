# GTMØ Axioms v2 Documentation

## Overview

The `gtmo_axioms_v2.py` module represents a significant evolution of the Generalized Theory of Mathematical Indefiniteness (GTMØ) axiom system. This enhanced version integrates advanced features from `gtmo_core_v2.py` while preserving the original UniverseMode functionality, creating a more sophisticated and adaptive mathematical framework.

## Table of Contents

1. [Key Improvements](#key-improvements)
2. [Module Architecture](#module-architecture)
3. [Core Components](#core-components)
4. [Axioms and Definitions](#axioms-and-definitions)
5. [Enhanced Operators](#enhanced-operators)
6. [System Integration](#system-integration)
7. [Usage Examples](#usage-examples)
8. [API Reference](#api-reference)
9. [Dependencies](#dependencies)

## Key Improvements

### Major Enhancements over v1

1. **Dynamic Context-Aware Operators**
   - Replaced simple heuristics with context-sensitive calculations
   - Operators adapt based on semantic and temporal context
   - Real-time adjustment of processing parameters

2. **Executable Axioms**
   - Axioms can now transform system state
   - Active enforcement of mathematical principles
   - Dynamic system evolution based on axiom application

3. **Topological Classification**
   - Phase space attractors instead of percentage thresholds
   - Wasserstein distance metrics for classification
   - Multi-dimensional phase space analysis

4. **Adaptive Learning**
   - Neurons learn from adversarial experiences
   - Defense strategy optimization
   - Memory consolidation and pattern recognition

5. **Enhanced System Integration**
   - Preserved UniverseMode (STILLNESS vs FLUX)
   - Seamless v2 feature integration
   - Backward compatibility maintained

## Module Architecture

```
gtmo_axioms_v2.py
├── Imports and Configuration
│   ├── Enhanced GTMØ core components (v2)
│   └── Fallback to basic core
├── Axioms and Definitions
│   ├── GTMOAxiom class (13 axioms)
│   └── GTMODefinition class (7 definitions)
├── Enhanced Operators
│   ├── EnhancedPsiOperator
│   └── EnhancedEntropyOperator
├── Meta-Feedback Loop
│   └── EnhancedMetaFeedbackLoop
├── System Integration
│   ├── UniverseMode enum
│   └── EnhancedGTMOSystem
└── Utilities
    ├── EmergenceDetector
    └── Factory functions
```

## Core Components

### GTMOAxiom Class

Container for all GTMØ formal axioms with validation capabilities:

```python
class GTMOAxiom:
    # Axioms AX0-AX12
    AX0 = "Systemic Uncertainty..."
    AX1 = "Ø is fundamentally different..."
    # ... through AX12
    
    @classmethod
    def validate_axiom_compliance(cls, operation_result, axiom_id):
        # Validates if operation complies with specified axiom
```

#### Key Axioms:
- **AX0**: Systemic Uncertainty - System's definability is inherently uncertain
- **AX1**: Ontological Difference - Ø ∉ {0, 1, ∞}
- **AX6**: Minimal Entropy - Ø has minimal cognitive entropy
- **AX11**: Adaptive Learning - Neurons modify responses based on experience
- **AX12**: Topological Classification - Knowledge types as phase space attractors

### GTMODefinition Class

Enhanced definitions incorporating v2 concepts:

```python
class GTMODefinition:
    DEF1 = "Knowledge particle Ψᴷ..."
    DEF2 = "Knowledge shadow Ψʰ..."
    # ... through DEF7
```

## Enhanced Operators

### EnhancedPsiOperator

The Ψ_GTMØ operator with v2 dynamic context-aware calculations:

```python
psi_op = EnhancedPsiOperator(classifier)
result = psi_op(fragment, context={
    'iteration': 5,
    'learning_enabled': True,
    'adaptation_weights': [0.2, 0.3, 0.5]
})
```

Features:
- Topological classification when v2 available
- Context-aware processing for AlienatedNumbers
- Dynamic scoring based on phase space position
- Fallback heuristics for compatibility

### EnhancedEntropyOperator

The E_GTMØ operator with context-aware entropy calculations:

```python
entropy_op = EnhancedEntropyOperator()
result = entropy_op(fragment, context)
```

Features:
- Phase space partition calculation
- Context factor extraction
- Semantic partitioning fallback
- Axiom compliance tracking

## System Integration

### UniverseMode

Preserved from original implementation:

```python
class UniverseMode(Enum):
    INDEFINITE_STILLNESS = auto()  # Rare genesis events
    ETERNAL_FLUX = auto()          # Frequent chaotic creation
```

### EnhancedGTMOSystem

Main system class integrating all v2 capabilities:

```python
system = EnhancedGTMOSystem(
    mode=UniverseMode.INDEFINITE_STILLNESS,
    initial_fragments=["Initial knowledge"],
    enable_v2_features=True
)
```

Key Methods:
- `add_adaptive_neuron()`: Add learning-capable neurons
- `add_epistemic_particle()`: Add knowledge particles
- `step()`: Advance system evolution
- `simulate_adversarial_attack()`: Test neuron defenses
- `get_comprehensive_report()`: Generate system metrics

## Usage Examples

### Basic System Creation

```python
from gtmo_axioms_v2 import create_enhanced_gtmo_system, UniverseMode

# Create system with v2 features
system, psi_op, entropy_op, meta_loop = create_enhanced_gtmo_system(
    mode=UniverseMode.ETERNAL_FLUX,
    initial_fragments=["Mathematical theorem", "Uncertain prediction"],
    enable_v2=True
)
```

### Context-Aware Processing

```python
# Create AlienatedNumber with context
from gtmo_core_v2 import AlienatedNumber

alien = AlienatedNumber("bitcoin_2030", context={
    'temporal_distance': 5.0,
    'volatility': 0.9,
    'predictability': 0.1,
    'domain': 'future_prediction'
})

# Process with enhanced operators
psi_result = psi_op(alien)
print(f"PSI Score: {psi_result.value['score']:.4f}")
print(f"Context factors: {psi_result.value.get('context_factors', {})}")
```

### Adaptive Learning Simulation

```python
# Add adaptive neurons
for i in range(5):
    system.add_adaptive_neuron(f"neuron_{i}", (i, 0, 0))

# Simulate adversarial attack
attack_result = system.simulate_adversarial_attack(
    attack_type='anti_paradox',
    target_indices=[0, 1, 2],
    intensity=0.8
)

# Check learning outcomes
for result in attack_result['results']:
    patterns = result['learned_patterns']
    print(f"Neuron {result['neuron_id']}: "
          f"Success rate: {patterns['success_rate']:.2%}")
```

### Meta-Feedback Loop

```python
# Run enhanced feedback loop
fragments = ["Theorem", "Paradox", "Maybe true", "Contradiction"]
initial_scores = [0.8, 0.5, 0.3, 0.2]

feedback_results = meta_loop.run(
    fragments=fragments,
    initial_scores=initial_scores,
    iterations=10,
    learning_enabled=True
)

# Analyze results
final_state = feedback_results['final_state']
print(f"System converged: {final_state['system_stability']}")
print(f"Final ratios: {final_state['final_classification_ratios']}")
print(f"Adaptation effectiveness: {final_state['adaptation_effectiveness']:.2%}")
```

## API Reference

### Core Classes

#### EnhancedGTMOSystem

```python
class EnhancedGTMOSystem:
    def __init__(self, mode: UniverseMode, 
                 initial_fragments: Optional[List[Any]] = None,
                 enable_v2_features: bool = True)
    
    def add_adaptive_neuron(self, neuron_id: str, 
                           position: Tuple[int, int, int]) -> bool
    
    def add_epistemic_particle(self, content: Any, **kwargs) -> bool
    
    def step(self, iterations: int = 1) -> None
    
    def simulate_adversarial_attack(self, attack_type: str, 
                                   target_indices: List[int], 
                                   intensity: float = 1.0) -> Dict[str, Any]
    
    def get_comprehensive_report(self) -> Dict[str, Any]
```

#### EnhancedPsiOperator

```python
class EnhancedPsiOperator:
    def __init__(self, classifier: Optional[TopologicalClassifier] = None)
    
    def __call__(self, fragment: Any, 
                 context: Dict[str, Any] = None) -> OperationResult
```

#### EnhancedMetaFeedbackLoop

```python
class EnhancedMetaFeedbackLoop:
    def __init__(self, psi_operator: EnhancedPsiOperator, 
                 entropy_operator: EnhancedEntropyOperator)
    
    def run(self, fragments: List[Any], 
            initial_scores: List[float], 
            iterations: int = 5, 
            learning_enabled: bool = True) -> Dict[str, Any]
```

### Enums and Types

```python
class OperatorType(Enum):
    STANDARD = 1
    META = 2
    HYBRID = 3
    ADAPTIVE = 4

class UniverseMode(Enum):
    INDEFINITE_STILLNESS = auto()
    ETERNAL_FLUX = auto()
```

### Factory Functions

```python
def create_enhanced_gtmo_system(
    mode: UniverseMode = UniverseMode.INDEFINITE_STILLNESS,
    initial_fragments: Optional[List[Any]] = None,
    enable_v2: bool = True
) -> Tuple[EnhancedGTMOSystem, Optional[EnhancedPsiOperator], 
           Optional[EnhancedEntropyOperator], Optional[EnhancedMetaFeedbackLoop]]
```

## Dependencies

### Required
- Python 3.8+
- numpy
- Standard library modules (math, logging, random, etc.)

### Optional (for v2 features)
- gtmo_core_v2.py module
- PyTorch (if defense network learning desired)

### Fallback Mode
When gtmo_core_v2.py is not available, the system automatically falls back to basic functionality using the standard core module, maintaining operational capability with reduced features.

## Configuration

### Environment Variables
- `GTM_STRICT`: Set to "1" for strict mode operations

### Logging
The module uses Python's logging module with INFO level by default:
```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Best Practices

1. **Always check v2 availability** before using advanced features
2. **Use context parameters** to enhance operator accuracy
3. **Enable learning** for systems that need to adapt
4. **Monitor phase space coverage** to ensure diverse knowledge representation
5. **Validate axiom compliance** for critical operations

## Conclusion

The gtmo_axioms_v2.py module represents a significant advancement in the GTMØ framework, introducing dynamic, adaptive, and topologically-aware mathematical operations while maintaining backward compatibility. This enhanced system provides a robust foundation for exploring mathematical indefiniteness with learning capabilities and context-aware processing.