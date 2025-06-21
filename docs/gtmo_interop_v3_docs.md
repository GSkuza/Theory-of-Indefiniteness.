# GTMØ Interoperability Module v3 Documentation

## Overview

The `gtmo_interop_v3.py` module represents a significant evolution in the GTMØ ecosystem, introducing a **learning-based interoperability layer** that adaptively maps external data (primarily text) into the GTMØ phase space. This module abandons static heuristics in favor of a fully adaptive, machine learning-driven approach.

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Technical Details](#technical-details)
8. [Future Enhancements](#future-enhancements)

## Key Concepts

### Learning-Based Ingestion

Unlike previous versions that relied on keyword matching and static rules, v3 introduces a neural network that learns how to map text to GTMØ's topological phase space coordinates:
- **Determinacy**: How certain or definite the knowledge is
- **Stability**: How consistent and stable the knowledge remains
- **Entropy**: The level of disorder or uncertainty

### Self-Supervised Feedback Loop

The module implements a feedback mechanism where the GTMØ system "teaches" the ingestion module how to better interpret external data:

```
Text → Embedding → Neural Network → Phase Coordinates → GTMØ Evolution → Feedback → Network Update
```

### Deep Semantic Understanding

Using transformer-based models (Sentence Transformers), the module captures semantic meaning rather than just keyword presence, enabling:
- Context-aware interpretation
- Multilingual support
- Nuanced understanding of uncertainty and paradox

## Architecture

### Component Overview

```
LearnedIngestionManager
├── SentenceTransformer (embedding_model)
│   └── Converts text to semantic vectors
├── IngestionNetwork (neural network)
│   └── Maps embeddings to phase space coordinates
├── Optimizer (Adam)
│   └── Updates network weights during learning
└── Loss Function (MSE)
    └── Measures prediction accuracy
```

### IngestionNetwork Architecture

```python
Input (embedding_dim) → Linear(128) → ReLU → Dropout(0.2) 
                     → Linear(64) → ReLU 
                     → Linear(3) → Sigmoid → Output (determinacy, stability, entropy)
```

## Requirements

### Core Dependencies
- Python 3.7+
- PyTorch 1.9+
- sentence-transformers 2.0+

### GTMØ Ecosystem
- gtmo_core_v2.py
- gtmo_axioms_v2.py
- utils_v2.py

## Installation

```bash
# Install ML dependencies
pip install torch sentence-transformers

# Ensure GTMØ v2 ecosystem is in your Python path
# Clone or download the GTMØ repository
git clone https://github.com/your-repo/gtmo.git
cd gtmo
```

## API Reference

### LearnedIngestionManager

#### Constructor

```python
LearnedIngestionManager(
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    device: str = 'cpu'
)
```

**Parameters:**
- `model_name`: Name of the Sentence Transformer model to use
- `device`: Computing device ('cpu' or 'cuda')

#### Methods

##### create_entity_from_text

```python
create_entity_from_text(
    text: str, 
    context: Optional[Dict] = None
) -> Optional[KnowledgeEntity]
```

Creates a KnowledgeEntity from text using the learned neural network.

**Parameters:**
- `text`: Input text to convert
- `context`: Optional metadata dictionary

**Returns:**
- `KnowledgeEntity` with predicted phase space coordinates

##### update_network_from_experience

```python
update_network_from_experience(
    initial_prediction: Tuple,
    observed_final_point: Tuple
) -> float
```

Updates network weights based on system feedback.

**Parameters:**
- `initial_prediction`: Original predicted coordinates
- `observed_final_point`: Target coordinates from GTMØ evolution

**Returns:**
- `float`: Loss value after update

### Serialization Functions

#### save_system_state

```python
save_system_state(
    system: EnhancedGTMOSystem,
    manager: LearnedIngestionManager,
    dir_path: str
) -> bool
```

Saves both system state and learned model weights.

#### load_system_state

```python
load_system_state(
    dir_path: str
) -> Tuple[Optional[EnhancedGTMOSystem], Optional[LearnedIngestionManager]]
```

Loads previously saved system state and model weights.

## Usage Examples

### Basic Text Ingestion

```python
from gtmo_interop_v3 import LearnedIngestionManager
from gtmo_axioms_v2 import EnhancedGTMOSystem, UniverseMode

# Initialize components
system = EnhancedGTMOSystem(mode=UniverseMode.INDEFINITE_STILLNESS)
manager = LearnedIngestionManager()

# Convert text to GTMØ entity
fact_text = "Water boils at 100 degrees Celsius at sea level."
entity = manager.create_entity_from_text(fact_text)

# Add to system
system.epistemic_particles.append(entity)

print(f"Predicted coordinates: {entity.to_phase_point()}")
# Output: (high determinacy, high stability, low entropy)
```

### Training the Network with Feedback

```python
# Initial prediction
uncertain_text = "The future of quantum computing remains uncertain."
entity = manager.create_entity_from_text(uncertain_text)
initial_coords = entity.to_phase_point()

# After system evolution, determine ideal coordinates
target_coords = (0.3, 0.4, 0.8)  # Low determinacy, medium stability, high entropy

# Train the network
for i in range(10):
    loss = manager.update_network_from_experience(initial_coords, target_coords)
    print(f"Iteration {i+1}, Loss: {loss:.6f}")

# Re-predict after training
new_entity = manager.create_entity_from_text(uncertain_text)
print(f"Updated prediction: {new_entity.to_phase_point()}")
```

### Persistent Learning

```python
# Save trained state
save_system_state(system, manager, "my_gtmo_state")

# Later, load the trained state
loaded_system, loaded_manager = load_system_state("my_gtmo_state")

# Continue with pre-trained network
entity = loaded_manager.create_entity_from_text("New text to analyze")
```

## Technical Details

### Embedding Models

The module uses Sentence Transformers, which are pre-trained models optimized for semantic similarity. The default model `paraphrase-multilingual-MiniLM-L12-v2` offers:
- 384-dimensional embeddings
- Multilingual support (50+ languages)
- Efficient inference (suitable for real-time applications)

### Learning Mechanism

The feedback loop implements a form of self-supervised learning:

1. **Initial Prediction**: Network predicts phase coordinates from text
2. **System Evolution**: GTMØ system evolves with the entity
3. **Feedback Generation**: System determines "ideal" coordinates based on its axioms
4. **Network Update**: Backpropagation adjusts weights to minimize prediction error

### Phase Space Mapping

The Sigmoid activation ensures all outputs are in [0, 1], which aligns with GTMØ's phase space requirements:
- **Determinacy ∈ [0, 1]**: 0 = completely uncertain, 1 = absolutely certain
- **Stability ∈ [0, 1]**: 0 = highly volatile, 1 = perfectly stable
- **Entropy ∈ [0, 1]**: 0 = perfect order, 1 = maximum disorder

## Future Enhancements

### Planned Features

1. **Attention Mechanisms**: Implement attention layers to identify which parts of text most influence coordinates
2. **Multi-Modal Support**: Extend beyond text to images, audio, and structured data
3. **Continual Learning**: Implement experience replay and regularization to prevent catastrophic forgetting
4. **Uncertainty Quantification**: Add dropout inference for prediction confidence estimates

### Research Directions

1. **Axiom-Aware Loss Functions**: Design loss functions that directly encode GTMØ axioms
2. **Topological Constraints**: Ensure predictions respect attractor basins in phase space
3. **Meta-Learning**: Learn to learn from few examples of new knowledge types
4. **Interpretability**: Develop methods to explain why certain texts map to specific coordinates

## Troubleshooting

### Common Issues

**ImportError for ML libraries**
```bash
# Solution: Install required packages
pip install torch sentence-transformers
```

**GTMØ modules not found**
```python
# Solution: Add GTMØ directory to Python path
import sys
sys.path.append('/path/to/gtmo')
```

**CUDA/GPU errors**
```python
# Solution: Use CPU if GPU unavailable
manager = LearnedIngestionManager(device='cpu')
```

## Contributing

When extending this module:

1. Maintain backward compatibility with saved models
2. Document any new neural architectures thoroughly
3. Include unit tests for new ingestion methods
4. Benchmark performance impact of changes

## References

- GTMØ Core Documentation
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Original GTMØ theory papers

---

*This module represents a significant step toward truly intelligent knowledge ingestion, where the system learns and adapts its understanding of external data through experience.*