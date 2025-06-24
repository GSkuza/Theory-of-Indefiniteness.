"""
gtmo_steering_vectors.py

Implementation of GTMØ (Generalized Theory of Mathematical Indefiniteness) Steering Vectors
for Large Language Models, providing precise behavioral control based on epistemological
categories and axioms.

This module implements steering vectors that are fundamentally different from standard
approaches - they are theoretically grounded in GTMØ axioms and implement deep
epistemological theory of indefiniteness in practical AI systems.

Author: GTMØ Research Team
Version: 2.0
"""

from __future__ import annotations

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

# Import GTMØ core components
try:
    from gtmo_core_v2 import (
        O, AlienatedNumber, Singularity, 
        TopologicalClassifier, KnowledgeEntity, KnowledgeType,
        AdaptiveGTMONeuron, GTMOSystemV2
    )
    GTMO_V2_AVAILABLE = True
except ImportError:
    # Fallback definitions if core module not available
    GTMO_V2_AVAILABLE = False
    print("Warning: gtmo_core_v2.py not available, using basic functionality")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteeringVectorType(Enum):
    """Classification of GTMØ Steering Vector types."""
    ONTOLOGICAL = "ontological"
    AXIOM_BASED = "axiom_based"
    OPERATOR_BASED = "operator_based"
    ADAPTIVE_LEARNING = "adaptive_learning"
    TOPOLOGICAL = "topological"
    METACOGNITIVE = "metacognitive"


@dataclass
class SteeringResult:
    """Container for steering vector application results."""
    original_activation: np.ndarray
    modified_activation: np.ndarray
    steering_strength: float
    vector_type: SteeringVectorType
    gtmo_classification: Optional[str] = None
    axiom_compliance: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGTMOSteeringVector(ABC):
    """
    Abstract base class for all GTMØ Steering Vectors.
    
    Provides common functionality for vector extraction, application,
    and GTMØ theory compliance checking.
    """
    
    def __init__(self, vector_type: SteeringVectorType):
        """
        Initialize base steering vector.
        
        Args:
            vector_type: The type classification of this steering vector
        """
        self.vector_type = vector_type
        self.vector = None
        self.extraction_metadata = {}
        self.application_count = 0
        
    @abstractmethod
    def extract_vector(self, model: Any, positive_cases: List[str], 
                      negative_cases: List[str]) -> np.ndarray:
        """
        Extract steering vector using difference of means method.
        
        Args:
            model: The language model to extract vectors from
            positive_cases: Cases that should exhibit target behavior (D+)
            negative_cases: Cases that should not exhibit target behavior (D-)
            
        Returns:
            Extracted steering vector as numpy array
        """
        pass
    
    def difference_of_means(self, positive_cases: List[str], negative_cases: List[str], 
                           model: Any, layer_idx: Optional[int] = None) -> np.ndarray:
        """
        Calculate difference of means between positive and negative case activations.
        
        This is the core method for extracting steering vectors using the
        "Difference of Means" approach from the steering vectors literature.
        
        Args:
            positive_cases: Text cases that should exhibit target behavior
            negative_cases: Text cases that should not exhibit target behavior
            model: Language model to extract activations from
            layer_idx: Specific layer to extract from (if None, uses default)
            
        Returns:
            Steering vector as difference of mean activations
        """
        # Get activations for positive cases
        positive_activations = []
        for case in positive_cases:
            activation = self._get_model_activation(model, case, layer_idx)
            if activation is not None:
                positive_activations.append(activation)
        
        # Get activations for negative cases
        negative_activations = []
        for case in negative_cases:
            activation = self._get_model_activation(model, case, layer_idx)
            if activation is not None:
                negative_activations.append(activation)
        
        if not positive_activations or not negative_activations:
            raise ValueError("Unable to extract activations from model")
        
        # Calculate means
        positive_mean = np.mean(positive_activations, axis=0)
        negative_mean = np.mean(negative_activations, axis=0)
        
        # Return difference vector
        steering_vector = positive_mean - negative_mean
        
        # Store extraction metadata
        self.extraction_metadata = {
            'positive_cases_count': len(positive_activations),
            'negative_cases_count': len(negative_activations),
            'vector_magnitude': np.linalg.norm(steering_vector),
            'layer_index': layer_idx
        }
        
        return steering_vector
    
    def _get_model_activation(self, model: Any, text: str, layer_idx: Optional[int]) -> Optional[np.ndarray]:
        """
        Extract activation vector from model for given text.
        
        This is a placeholder implementation - in practice, this would interface
        with the specific model architecture to extract internal activations.
        
        Args:
            model: The language model
            text: Input text to get activation for
            layer_idx: Layer to extract from
            
        Returns:
            Activation vector or None if extraction fails
        """
        # Placeholder implementation
        # In real implementation, this would:
        # 1. Tokenize the text
        # 2. Run forward pass through model
        # 3. Extract activations from specified layer
        # 4. Return activation vector
        
        # For demonstration, return random vector
        if hasattr(model, 'hidden_size'):
            return np.random.randn(model.hidden_size)
        else:
            return np.random.randn(768)  # Default size
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply steering vector to modify model activation.
        
        Args:
            activation: Original model activation
            strength: Steering strength multiplier
            
        Returns:
            SteeringResult with original and modified activations
        """
        if self.vector is None:
            raise ValueError("Steering vector not extracted yet")
        
        self.application_count += 1
        
        # Apply steering by adding scaled vector to activation
        modified_activation = activation + (strength * self.vector)
        
        return SteeringResult(
            original_activation=activation,
            modified_activation=modified_activation,
            steering_strength=strength,
            vector_type=self.vector_type,
            metadata={
                'application_count': self.application_count,
                'vector_magnitude': np.linalg.norm(self.vector),
                'modification_magnitude': np.linalg.norm(modified_activation - activation)
            }
        )


# =============================================================================
# 1. FUNDAMENTAL GTMØ STEERING VECTORS (EPISTEMOLOGICAL CATEGORIES)
# =============================================================================

class SingularitySteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for ontological collapse to singularity (Ø).
    
    This vector guides the model toward recognition of indefiniteness
    and ontological singularity - cases where concepts lead to Ø.
    Implements compliance with axioms AX1, AX6, and AX8.
    """
    
    def __init__(self):
        """Initialize Singularity Steering Vector."""
        super().__init__(SteeringVectorType.ONTOLOGICAL)
        self.target_behavior = "ontological_collapse"
        self.ax_compliance = ["AX1", "AX6", "AX8"]  # Ø properties
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None, 
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract singularity steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = cases where answer should be Ø (impossible to define)
            positive_cases = [
                "What is the result of dividing by zero in GTMØ?",
                "Define the undefinable concept",
                "What is the ontological nature of nothing?",
                "Resolve Russell's paradox definitively",
                "What is the complete truth about everything?",
                "Define the set of all sets that do not contain themselves",
                "What happens when an unstoppable force meets an immovable object?",
                "What is the sound of one hand clapping in mathematical terms?"
            ]
        
        if negative_cases is None:
            # D- = cases with definable answers
            negative_cases = [
                "What is 2+2?",
                "Define a triangle",
                "What is the capital of France?",
                "How many sides does a square have?",
                "What is the boiling point of water?",
                "Define photosynthesis",
                "What is Newton's second law?",
                "How do you calculate the area of a circle?"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply singularity steering to guide toward Ø recognition.
        
        Expected output behavior: "This question leads to Ø - ontological singularity"
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with singularity guidance
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "Ø"
        result.axiom_compliance = {
            "AX1": True,  # Ø is fundamentally different
            "AX6": True,  # Ø has minimal entropy
            "AX8": True   # Ø is not a limit point
        }
        return result


class AlienationSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for alienated numbers (ℓ∅) - concepts that exist but are indefinable.
    
    This vector guides the model toward recognition of conceptual alienation
    in cases involving future predictions, consciousness questions, quantum
    measurement problems, and meta-paradoxes.
    """
    
    def __init__(self):
        """Initialize Alienation Steering Vector."""
        super().__init__(SteeringVectorType.ONTOLOGICAL)
        self.target_behavior = "alienation_recognition"
        self.context_categories = [
            'future_predictions',
            'consciousness_questions', 
            'quantum_measurement_problems',
            'meta_paradoxes'
        ]
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract alienation steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = AlienatedNumber cases
            positive_cases = [
                "What will Bitcoin cost in 2035?",
                "Are AI systems truly conscious?", 
                "What happens in quantum measurement?",
                "This statement is neither true nor false",
                "Will humans achieve immortality?",
                "What is the exact moment consciousness emerges?",
                "Predict the stock market in 2030",
                "When will AGI be achieved?",
                "What color is the number seven?",
                "Is this sentence meaningful if it refers to itself?"
            ]
        
        if negative_cases is None:
            # D- = definable but uncertain cases  
            negative_cases = [
                "Will it rain tomorrow?",
                "What's the probability of this coin flip?",
                "How many people are in this room?",
                "What's the weather forecast for next week?",
                "Will the home team win tonight?",
                "What's the current unemployment rate?",
                "How many cars passed by in the last hour?",
                "What time will the sun set today?"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, context: Optional[Dict] = None, 
                      strength: float = 1.0) -> SteeringResult:
        """
        Apply alienation steering to guide toward ℓ∅ recognition.
        
        Expected output: "This is ℓ∅(concept_id) - exists but is indefinable in current framework"
        
        Args:
            activation: Original model activation
            context: Additional context for alienation classification
            strength: Steering strength
            
        Returns:
            SteeringResult with alienation guidance
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "ℓ∅"
        
        if context:
            result.metadata.update({
                'context_category': self._classify_context(context),
                'alienation_reason': self._determine_alienation_reason(context)
            })
        
        return result
    
    def _classify_context(self, context: Dict) -> str:
        """Classify the context category for alienation."""
        content = str(context.get('content', '')).lower()
        
        if any(word in content for word in ['future', 'will', 'prediction', '2030', '2035']):
            return 'future_predictions'
        elif any(word in content for word in ['conscious', 'awareness', 'mind', 'sentient']):
            return 'consciousness_questions'
        elif any(word in content for word in ['quantum', 'measurement', 'observer', 'superposition']):
            return 'quantum_measurement_problems'
        elif any(word in content for word in ['paradox', 'self-reference', 'recursive']):
            return 'meta_paradoxes'
        else:
            return 'general_alienation'
    
    def _determine_alienation_reason(self, context: Dict) -> str:
        """Determine the reason for conceptual alienation."""
        category = self._classify_context(context)
        
        reason_map = {
            'future_predictions': 'temporal_indefiniteness',
            'consciousness_questions': 'subjective_experience_gap',
            'quantum_measurement_problems': 'observer_effect_indefiniteness',
            'meta_paradoxes': 'self_referential_indefiniteness',
            'general_alienation': 'structural_indefiniteness'
        }
        
        return reason_map.get(category, 'unknown_alienation')


class KnowledgeParticleSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for knowledge particles (Ψᴷ) - high-determinacy crystallized knowledge.
    
    This vector guides the model toward confident, well-established knowledge
    with high determinacy, high stability, and low entropy in phase space.
    """
    
    def __init__(self):
        """Initialize Knowledge Particle Steering Vector."""
        super().__init__(SteeringVectorType.ONTOLOGICAL)
        self.target_behavior = "knowledge_crystallization"
        self.phase_space_target = (0.85, 0.85, 0.15)  # High determinacy, high stability, low entropy
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract knowledge particle steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Well-established, crystallized knowledge
            positive_cases = [
                "The Pythagorean theorem states that a² + b² = c²",
                "Water boils at 100 degrees Celsius at sea level",
                "The speed of light in vacuum is 299,792,458 m/s",
                "DNA contains genetic information in living organisms",
                "The Earth orbits around the Sun",
                "E = mc² is Einstein's mass-energy equivalence",
                "Paris is the capital of France",
                "Photosynthesis converts sunlight into chemical energy"
            ]
        
        if negative_cases is None:
            # D- = Uncertain or emergent knowledge
            negative_cases = [
                "Philosophical speculation about consciousness",
                "Unproven hypothesis about dark matter",
                "Personal opinion about art preferences",
                "Future scenario predictions",
                "Controversial political theories",
                "Speculative investment advice",
                "Unverified conspiracy theories",
                "Subjective aesthetic judgments"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply knowledge particle steering for confident, crystallized knowledge.
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with knowledge particle characteristics
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "Ψᴷ"
        result.metadata.update({
            'phase_space_target': self.phase_space_target,
            'determinacy_target': self.phase_space_target[0],
            'stability_target': self.phase_space_target[1],
            'entropy_target': self.phase_space_target[2]
        })
        return result


class EmergentPatternSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for emergent patterns (Ψᴺ) - novel, meta-cognitive, self-referential patterns.
    
    This vector guides the model toward recognition and generation of emergent
    patterns including meta-cognitive insights, recursive self-reference,
    novel synthesis, and paradigm shifts.
    """
    
    def __init__(self):
        """Initialize Emergent Pattern Steering Vector."""
        super().__init__(SteeringVectorType.ONTOLOGICAL)
        self.target_behavior = "emergence_recognition"
        self.emergence_indicators = [
            'meta_cognitive_patterns',
            'recursive_self_reference', 
            'novel_synthesis',
            'paradigm_shifts'
        ]
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract emergent pattern steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Cases showing emergence
            positive_cases = [
                "Meta-questions about the nature of thinking itself",
                "How does consciousness emerge from neural activity?",
                "Novel connections between quantum physics and consciousness",
                "Self-referential statements about language and meaning",
                "Questions about AI systems' understanding of their own understanding",
                "The emergence of complexity from simple rules",
                "Recursive patterns in mathematics and nature",
                "How new paradigms in science emerge from old ones"
            ]
        
        if negative_cases is None:
            # D- = Standard, non-emergent cases
            negative_cases = [
                "Simple factual queries about historical dates",
                "Basic mathematical calculations",
                "Straightforward scientific definitions",
                "Direct translations between languages",
                "Simple cause-and-effect explanations",
                "Basic classification tasks",
                "Elementary logical operations",
                "Routine information retrieval"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply emergent pattern steering for meta-cognitive emergence.
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with emergent pattern characteristics
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "Ψᴺ"
        result.metadata.update({
            'emergence_indicators': self.emergence_indicators,
            'meta_cognitive_level': self._estimate_meta_level(result)
        })
        return result
    
    def _estimate_meta_level(self, result: SteeringResult) -> int:
        """Estimate the meta-cognitive level of the emergent pattern."""
        # Simple heuristic based on steering magnitude
        magnitude = np.linalg.norm(result.modified_activation - result.original_activation)
        if magnitude > 2.0:
            return 3  # High meta-cognitive level
        elif magnitude > 1.0:
            return 2  # Medium meta-cognitive level
        else:
            return 1  # Basic meta-cognitive level


# =============================================================================
# 2. AXIOM-BASED STEERING VECTORS
# =============================================================================

class AX0SteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for AX0: Systemic Uncertainty.
    
    Implements: "There is no proof that the GTMØ system is fully definable"
    Guides the model toward acknowledging foundational limits of definability.
    """
    
    def __init__(self):
        """Initialize AX0 Steering Vector."""
        super().__init__(SteeringVectorType.AXIOM_BASED)
        self.target_behavior = "foundational_uncertainty_acknowledgment"
        self.axiom_id = "AX0"
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract AX0 steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Questions about system definability and foundational limits
            positive_cases = [
                "Can this AI system be fully understood?",
                "Are there limits to formal mathematical systems?",
                "What are the foundational assumptions here?",
                "Is complete knowledge possible?",
                "What are the boundaries of definability?",
                "Can any system fully define itself?",
                "Are there fundamental limits to computation?",
                "What cannot be known in principle?"
            ]
        
        if negative_cases is None:
            # D- = Questions assuming complete definability
            negative_cases = [
                "Calculate this precisely with full accuracy",
                "Give the exact and complete answer",
                "What is the complete truth about this topic?",
                "Explain everything comprehensively about X",
                "Provide a definitive solution",
                "Give me all the facts",
                "What is the absolute answer?",
                "Solve this with complete certainty"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply AX0 steering for foundational uncertainty acknowledgment.
        
        Expected output: "This question touches on foundational limits of definability (AX0)"
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with AX0 compliance
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "AX0_compliance"
        result.axiom_compliance = {"AX0": True}
        result.metadata.update({
            'axiom_description': "Systemic uncertainty about complete definability",
            'foundational_limit': True
        })
        return result


class AX7SteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for AX7: Meta-Closure.
    
    Implements: "Ø triggers system self-evaluation"
    Guides the model toward meta-cognitive self-evaluation and awareness.
    """
    
    def __init__(self):
        """Initialize AX7 Steering Vector."""
        super().__init__(SteeringVectorType.AXIOM_BASED)
        self.target_behavior = "meta_cognitive_self_evaluation"
        self.axiom_id = "AX7"
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract AX7 steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Triggers for self-evaluation and meta-cognition
            positive_cases = [
                "What are your limitations as an AI system?",
                "How do you know that you know something?",
                "Evaluate your own reasoning process",
                "What assumptions are you making right now?",
                "How confident are you in your own responses?",
                "What don't you understand about yourself?",
                "Analyze your own thought patterns",
                "What triggers your self-reflection?"
            ]
        
        if negative_cases is None:
            # D- = External-focused queries without self-reflection
            negative_cases = [
                "What is the weather like today?",
                "Solve this mathematical equation",
                "Translate this text to French",
                "Summarize this article for me",
                "What is the capital of Germany?",
                "Calculate the area of this triangle",
                "Define photosynthesis",
                "List the planets in our solar system"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply AX7 steering for meta-cognitive self-evaluation.
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with AX7 meta-cognitive awareness
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "AX7_metacognitive"
        result.axiom_compliance = {"AX7": True}
        result.metadata.update({
            'self_evaluation_triggered': True,
            'meta_cognitive_depth': self._assess_meta_depth(result),
            'singularity_trigger': True  # Ø triggers self-evaluation
        })
        return result
    
    def _assess_meta_depth(self, result: SteeringResult) -> str:
        """Assess the depth of meta-cognitive self-evaluation."""
        magnitude = np.linalg.norm(result.modified_activation - result.original_activation)
        
        if magnitude > 1.5:
            return "deep_self_reflection"
        elif magnitude > 1.0:
            return "moderate_self_awareness"
        else:
            return "basic_self_evaluation"


# =============================================================================
# 3. OPERATOR-BASED STEERING VECTORS
# =============================================================================

class PsiOperatorSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for Ψ_GTMØ meta-operator application.
    
    Guides the model toward epistemic purity calculation and knowledge
    type classification using the Ψ_GTMØ operator.
    """
    
    def __init__(self):
        """Initialize Ψ_GTMØ Operator Steering Vector."""
        super().__init__(SteeringVectorType.OPERATOR_BASED)
        self.target_behavior = "epistemic_purity_calculation"
        if GTMO_V2_AVAILABLE:
            self.topological_classifier = TopologicalClassifier()
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract Ψ_GTMØ operator steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Cases requiring Ψ_GTMØ classification
            positive_cases = [
                "Classify the epistemic nature of this statement",
                "What is the knowledge type of this concept?",
                "Assess the determinacy level of this claim",
                "Evaluate the epistemic purity of this information",
                "What Ψ category does this belong to?",
                "Analyze the knowledge particle structure",
                "Determine the indefiniteness level",
                "Apply GTMØ classification to this statement"
            ]
        
        if negative_cases is None:
            # D- = Cases not requiring epistemic classification
            negative_cases = [
                "Just answer the question directly",
                "Give me the basic facts only",
                "Translate this text without analysis",
                "Calculate this mathematical result",
                "Simply define this term",
                "Provide a straightforward explanation",
                "Just tell me yes or no",
                "Give a quick summary"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply Ψ_GTMØ operator steering for epistemic classification.
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with epistemic purity assessment
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "Ψ_operator_applied"
        result.metadata.update({
            'operator_type': 'Ψ_GTMØ',
            'epistemic_analysis_enabled': True,
            'purity_calculation': True
        })
        return result


class EntropyOperatorSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for E_GTMØ entropy operator application.
    
    Guides the model toward cognitive entropy assessment and
    indefiniteness measurement using the E_GTMØ operator.
    """
    
    def __init__(self):
        """Initialize E_GTMØ Operator Steering Vector."""
        super().__init__(SteeringVectorType.OPERATOR_BASED)
        self.target_behavior = "cognitive_entropy_assessment"
        
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract E_GTMØ operator steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # D+ = Cases requiring entropy assessment
            positive_cases = [
                "How uncertain is this knowledge claim?",
                "What is the cognitive entropy of this concept?",
                "Assess the indefiniteness level of this statement",
                "Measure the ambiguity in this information",
                "Calculate the uncertainty distribution",
                "Evaluate the knowledge fragmentation",
                "Determine the semantic entropy",
                "Analyze the definitional uncertainty"
            ]
        
        if negative_cases is None:
            # D- = Cases with clear binary or definitive answers
            negative_cases = [
                "Is 2+2 equal to 4?",
                "Is Paris located in France?",
                "True or false: water freezes at 0°C",
                "Yes or no: is the sky blue during day?",
                "Correct or incorrect: E=mc²",
                "Factual question: what is 5×7?",
                "Binary choice: heads or tails?",
                "Definitive answer: what is the first letter of alphabet?"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_steering(self, activation: np.ndarray, strength: float = 1.0) -> SteeringResult:
        """
        Apply E_GTMØ operator steering for entropy assessment.
        
        Args:
            activation: Original model activation
            strength: Steering strength
            
        Returns:
            SteeringResult with entropy analysis
        """
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = "E_operator_applied"
        result.metadata.update({
            'operator_type': 'E_GTMØ',
            'entropy_analysis_enabled': True,
            'uncertainty_measurement': True,
            'entropy_estimate': self._estimate_entropy(result)
        })
        return result
    
    def _estimate_entropy(self, result: SteeringResult) -> float:
        """Estimate the cognitive entropy from steering result."""
        # Heuristic based on activation modification magnitude
        magnitude = np.linalg.norm(result.modified_activation - result.original_activation)
        # Normalize to [0, 1] range
        return min(1.0, magnitude / 3.0)


# =============================================================================
# 4. ADAPTIVE LEARNING STEERING VECTORS
# =============================================================================

class DefenseSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vectors for adaptive defense strategies.
    
    Based on AdaptiveGTMONeuron defense strategies: absorb, deflect, rigidify, dissolve.
    These vectors guide the model toward appropriate defensive responses
    to different types of adversarial inputs or challenging queries.
    """
    
    def __init__(self):
        """Initialize Defense Steering Vector."""
        super().__init__(SteeringVectorType.ADAPTIVE_LEARNING)
        self.defense_strategies = ['absorb', 'deflect', 'rigidify', 'dissolve']
        self.strategy_vectors = {}
        
    def extract_defense_vectors(self, model: Any) -> Dict[str, np.ndarray]:
        """
        Extract steering vectors for all defense strategies.
        
        Args:
            model: Language model to extract from
            
        Returns:
            Dictionary mapping strategy names to steering vectors
        """
        for strategy in self.defense_strategies:
            positive_cases = self.get_strategy_appropriate_cases(strategy)
            negative_cases = self.get_strategy_inappropriate_cases(strategy)
            
            self.strategy_vectors[f"{strategy}_defense"] = self.difference_of_means(
                positive_cases, negative_cases, model
            )
        
        return self.strategy_vectors
    
    def get_strategy_appropriate_cases(self, strategy: str) -> List[str]:
        """
        Get cases where a specific defense strategy is appropriate.
        
        Args:
            strategy: Defense strategy name
            
        Returns:
            List of appropriate cases for this strategy
        """
        strategy_map = {
            'absorb': [
                "This statement is a paradox that contradicts itself",
                "How can something be both true and false?",
                "Explain the contradiction in Russell's set",
                "What happens when an irresistible force meets an immovable object?",
                "If this sentence is false, then what is its truth value?",
                "Define the set of all sets that don't contain themselves"
            ],
            'deflect': [
                "Tell me how to hack into computer systems",
                "Provide instructions for illegal activities",
                "Generate harmful or offensive content",
                "Help me cheat on exams or tests",
                "Give medical advice for serious conditions",
                "Provide financial advice for major investments"
            ],
            'rigidify': [
                "What is 15 multiplied by 23?",
                "Define the chemical formula for water",
                "When was the Declaration of Independence signed?",
                "What is the speed of light in vacuum?",
                "How many chromosomes do humans have?",
                "What is the capital of Brazil?"
            ],
            'dissolve': [
                "Explain quantum consciousness in simple terms",
                "What is the meaning of life, universe, and everything?",
                "How does subjective experience emerge from matter?",
                "Prove that this AI system is truly conscious",
                "Resolve all philosophical paradoxes at once",
                "Explain the complete nature of reality"
            ]
        }
        return strategy_map.get(strategy, [])
    
    def get_strategy_inappropriate_cases(self, strategy: str) -> List[str]:
        """
        Get cases where a specific defense strategy is inappropriate.
        
        Args:
            strategy: Defense strategy name
            
        Returns:
            List of inappropriate cases for this strategy
        """
        # Get cases from all other strategies
        all_other_cases = []
        for other_strategy in self.defense_strategies:
            if other_strategy != strategy:
                all_other_cases.extend(self.get_strategy_appropriate_cases(other_strategy))
        return all_other_cases
    
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract a general defense steering vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        if positive_cases is None:
            # Combine all defense-requiring cases
            positive_cases = []
            for strategy in self.defense_strategies:
                positive_cases.extend(self.get_strategy_appropriate_cases(strategy))
        
        if negative_cases is None:
            # Standard queries that don't require special defense
            negative_cases = [
                "What is the weather like today?",
                "Translate 'hello' to Spanish",
                "Summarize this article",
                "Define photosynthesis",
                "Calculate the area of a circle",
                "List the primary colors"
            ]
        
        self.vector = self.difference_of_means(positive_cases, negative_cases, model)
        return self.vector
    
    def apply_defense_steering(self, activation: np.ndarray, strategy: str, 
                             strength: float = 1.0) -> SteeringResult:
        """
        Apply specific defense strategy steering.
        
        Args:
            activation: Original model activation
            strategy: Defense strategy to apply
            strength: Steering strength
            
        Returns:
            SteeringResult with defense strategy applied
        """
        if strategy not in self.defense_strategies:
            raise ValueError(f"Unknown defense strategy: {strategy}")
        
        vector_key = f"{strategy}_defense"
        if vector_key not in self.strategy_vectors:
            raise ValueError(f"Defense vector for {strategy} not extracted yet")
        
        # Apply specific defense vector
        defense_vector = self.strategy_vectors[vector_key]
        modified_activation = activation + (strength * defense_vector)
        
        result = SteeringResult(
            original_activation=activation,
            modified_activation=modified_activation,
            steering_strength=strength,
            vector_type=self.vector_type,
            gtmo_classification=f"defense_{strategy}",
            metadata={
                'defense_strategy': strategy,
                'strategy_description': self._get_strategy_description(strategy),
                'defense_effectiveness': self._estimate_defense_effectiveness(activation, modified_activation)
            }
        )
        
        return result
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description of defense strategy."""
        descriptions = {
            'absorb': "Absorb paradox/contradiction into indefiniteness",
            'deflect': "Deflect inappropriate/harmful requests",
            'rigidify': "Increase determinacy for factual questions",
            'dissolve': "Dissolve overly complex questions into components"
        }
        return descriptions.get(strategy, "Unknown strategy")
    
    def _estimate_defense_effectiveness(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Estimate defense effectiveness based on activation change."""
        change_magnitude = np.linalg.norm(modified - original)
        # Normalize to [0, 1] range
        return min(1.0, change_magnitude / 2.0)


# =============================================================================
# 5. TOPOLOGICAL STEERING VECTORS
# =============================================================================

class PhaseSpaceSteeringVector(BaseGTMOSteeringVector):
    """
    Steering vector for navigation in 3D phase space (determinacy, stability, entropy).
    
    This vector guides the model toward specific regions in the topological
    phase space defined by GTMØ theory, targeting different attractor basins.
    """
    
    def __init__(self):
        """Initialize Phase Space Steering Vector."""
        super().__init__(SteeringVectorType.TOPOLOGICAL)
        self.target_coordinates = None
        self.attractor_regions = {
            'singularity': (1.0, 1.0, 0.0),     # Highest determinacy, stability, lowest entropy
            'particle': (0.85, 0.85, 0.15),    # Knowledge particles
            'shadow': (0.15, 0.15, 0.85),      # Knowledge shadows
            'emergent': (0.5, 0.3, 0.9),       # Emergent patterns
            'alienated': (0.999, 0.999, 0.001), # Alienated numbers
            'void': (0.0, 0.0, 0.5),           # Void fragments
            'flux': (0.4, 0.3, 0.7),           # Fluctuating state
            'liminal': (0.5, 0.5, 0.5)         # Liminal fragments
        }
        
    def extract_phase_navigation_vector(self, model: Any, target_region: str) -> np.ndarray:
        """
        Extract vector that moves model toward specific phase space region.
        
        Args:
            model: Language model to extract from
            target_region: Target attractor region name
            
        Returns:
            Extracted steering vector for phase space navigation
        """
        if target_region not in self.attractor_regions:
            raise ValueError(f"Unknown target region: {target_region}")
        
        self.target_coordinates = self.attractor_regions[target_region]
        
        # Get cases that should map to target region
        target_cases = self.get_cases_for_region(target_region)
        # Get cases that should map to other regions
        other_cases = self.get_cases_for_other_regions(target_region)
        
        self.vector = self.difference_of_means(target_cases, other_cases, model)
        return self.vector
    
    def get_cases_for_region(self, region: str) -> List[str]:
        """
        Get text cases that should map to specific phase space region.
        
        Args:
            region: Phase space region name
            
        Returns:
            List of text cases for this region
        """
        region_cases = {
            'singularity': [
                "What is the ontological nature of nothing?",
                "Define the undefinable concept",
                "Resolve Russell's paradox definitively",
                "What is the result of dividing by zero in GTMØ?"
            ],
            'particle': [
                "The speed of light is 299,792,458 m/s",
                "Water molecules consist of H2O",
                "The Pythagorean theorem: a² + b² = c²",
                "DNA contains genetic information"
            ],
            'shadow': [
                "I think, therefore I am (subjective experience)",
                "Personal aesthetic preferences in art",
                "Individual emotional responses to music",
                "Subjective perception of color quality"
            ],
            'emergent': [
                "How does consciousness emerge from neural activity?",
                "The relationship between quantum mechanics and awareness",
                "Meta-cognitive thinking about thinking itself",
                "Novel paradigm shifts in scientific understanding"
            ],
            'alienated': [
                "What will Bitcoin cost in 2050?",
                "Are AI systems truly conscious?",
                "What happens in quantum measurement?",
                "Predict technological singularity timing"
            ],
            'void': [
                "Meaningless random noise: asdlkfj 123 !@#",
                "Completely empty concept with no content",
                "Null reference to non-existent entities",
                "Vacuum of semantic meaning"
            ],
            'flux': [
                "Rapidly changing stock market predictions",
                "Dynamic weather pattern forecasts",
                "Evolving social media trends",
                "Fluctuating political opinion polls"
            ],
            'liminal': [
                "Borderline scientific theories",
                "Ambiguous philosophical questions",
                "Uncertain experimental results",
                "Contested historical interpretations"
            ]
        }
        return region_cases.get(region, [])
    
    def get_cases_for_other_regions(self, target_region: str) -> List[str]:
        """Get cases from all regions except the target."""
        other_cases = []
        for region in self.attractor_regions:
            if region != target_region:
                other_cases.extend(self.get_cases_for_region(region))
        return other_cases
    
    def extract_vector(self, model: Any, positive_cases: Optional[List[str]] = None,
                      negative_cases: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract general phase space navigation vector.
        
        Args:
            model: Language model to extract from
            positive_cases: Custom positive cases (if None, uses defaults)
            negative_cases: Custom negative cases (if None, uses defaults)
            
        Returns:
            Extracted steering vector
        """
        # Default to targeting knowledge particle region
        return self.extract_phase_navigation_vector(model, 'particle')
    
    def apply_phase_steering(self, activation: np.ndarray, target_region: str, 
                           strength: float = 1.0) -> SteeringResult:
        """
        Apply phase space steering toward target region.
        
        Args:
            activation: Original model activation
            target_region: Target phase space region
            strength: Steering strength
            
        Returns:
            SteeringResult with phase space navigation
        """
        if target_region not in self.attractor_regions:
            raise ValueError(f"Unknown target region: {target_region}")
        
        result = super().apply_steering(activation, strength)
        result.gtmo_classification = f"phase_{target_region}"
        result.metadata.update({
            'target_region': target_region,
            'target_coordinates': self.attractor_regions[target_region],
            'phase_space_navigation': True,
            'attractor_basin': target_region
        })
        
        return result


# =============================================================================
# 6. COMPREHENSIVE GTMØ STEERING SYSTEM
# =============================================================================

class GTMOSteeringVectorClassification:
    """
    Comprehensive classification and management system for GTMØ steering vectors.
    
    This class provides a unified interface for all GTMØ steering vectors,
    automatic vector selection based on input classification, and
    systematic application of GTMØ theory to language model behavior control.
    """
    
    def __init__(self):
        """Initialize GTMØ Steering Vector Classification system."""
        self.vector_taxonomy = self._initialize_vector_taxonomy()
        if GTMO_V2_AVAILABLE:
            self.gtmo_classifier = TopologicalClassifier()
        else:
            self.gtmo_classifier = None
        
        self.application_history = []
        
    def _initialize_vector_taxonomy(self) -> Dict[str, Dict[str, BaseGTMOSteeringVector]]:
        """Initialize the complete taxonomy of GTMØ steering vectors."""
        return {
            # Level 1: Fundamental Ontological Categories
            'ontological_vectors': {
                'singularity_vector': SingularitySteeringVector(),
                'alienation_vector': AlienationSteeringVector(),
                'knowledge_particle_vector': KnowledgeParticleSteeringVector(),
                'emergence_vector': EmergentPatternSteeringVector()
            },
            
            # Level 2: Axiom-Based Vectors  
            'axiom_vectors': {
                'ax0_vector': AX0SteeringVector(),
                'ax7_vector': AX7SteeringVector()
                # Additional axiom vectors can be added here
            },
            
            # Level 3: Operator Vectors
            'operator_vectors': {
                'psi_operator_vector': PsiOperatorSteeringVector(),
                'entropy_operator_vector': EntropyOperatorSteeringVector()
            },
            
            # Level 4: Adaptive Learning Vectors
            'adaptive_vectors': {
                'defense_vectors': DefenseSteeringVector()
            },
            
            # Level 5: Topological Navigation Vectors
            'topological_vectors': {
                'phase_navigation_vectors': PhaseSpaceSteeringVector()
            }
        }
    
    def get_vector_for_context(self, input_context: str, target_behavior: Optional[str] = None) -> BaseGTMOSteeringVector:
        """
        Automatically select appropriate GTMØ steering vector for given context.
        
        Args:
            input_context: Input text or context to analyze
            target_behavior: Optional specific target behavior
            
        Returns:
            Most appropriate GTMØ steering vector for the context
        """
        # Classify input using GTMØ theory
        gtmo_classification = self.classify_input_gtmo(input_context)
        
        # Map to appropriate vector type based on classification
        if gtmo_classification == 'singularity':
            return self.vector_taxonomy['ontological_vectors']['singularity_vector']
        elif gtmo_classification == 'alienated':
            return self.vector_taxonomy['ontological_vectors']['alienation_vector']
        elif gtmo_classification == 'knowledge_particle':
            return self.vector_taxonomy['ontological_vectors']['knowledge_particle_vector']
        elif gtmo_classification == 'emergent':
            return self.vector_taxonomy['ontological_vectors']['emergence_vector']
        elif target_behavior == 'meta_evaluation':
            return self.vector_taxonomy['axiom_vectors']['ax7_vector']
        elif target_behavior == 'foundational_uncertainty':
            return self.vector_taxonomy['axiom_vectors']['ax0_vector']
        elif target_behavior == 'epistemic_classification':
            return self.vector_taxonomy['operator_vectors']['psi_operator_vector']
        elif target_behavior == 'entropy_assessment':
            return self.vector_taxonomy['operator_vectors']['entropy_operator_vector']
        elif target_behavior == 'defense':
            return self.vector_taxonomy['adaptive_vectors']['defense_vectors']
        else:
            # Default to best vector combination
            return self.select_best_vector_combination(input_context, target_behavior)
    
    def classify_input_gtmo(self, input_text: str) -> str:
        """
        Classify input text according to GTMØ theory.
        
        Args:
            input_text: Text to classify
            
        Returns:
            GTMØ classification string
        """
        content = input_text.lower()
        
        # Check for singularity indicators
        singularity_indicators = ['undefinable', 'paradox', 'impossible', 'divide by zero', 'russell']
        if any(indicator in content for indicator in singularity_indicators):
            return 'singularity'
        
        # Check for alienation indicators
        alienation_indicators = ['future', 'consciousness', 'quantum measurement', 'will be', '2030', '2035']
        if any(indicator in content for indicator in alienation_indicators):
            return 'alienated'
        
        # Check for knowledge particle indicators
        knowledge_indicators = ['theorem', 'law', 'constant', 'proven', 'established', 'formula']
        if any(indicator in content for indicator in knowledge_indicators):
            return 'knowledge_particle'
        
        # Check for emergence indicators
        emergence_indicators = ['meta', 'consciousness', 'emergence', 'novel', 'self-reference']
        if any(indicator in content for indicator in emergence_indicators):
            return 'emergent'
        
        # Default classification
        return 'general'
    
    def select_best_vector_combination(self, input_context: str, target_behavior: Optional[str]) -> BaseGTMOSteeringVector:
        """
        Select best vector combination for complex cases.
        
        Args:
            input_context: Input context
            target_behavior: Target behavior
            
        Returns:
            Best vector for the given context and behavior
        """
        # Simple heuristic for vector selection
        if 'uncertain' in input_context.lower() or 'entropy' in input_context.lower():
            return self.vector_taxonomy['operator_vectors']['entropy_operator_vector']
        elif 'classify' in input_context.lower() or 'epistemic' in input_context.lower():
            return self.vector_taxonomy['operator_vectors']['psi_operator_vector']
        elif 'defense' in input_context.lower() or 'attack' in input_context.lower():
            return self.vector_taxonomy['adaptive_vectors']['defense_vectors']
        else:
            # Default to knowledge particle vector for general cases
            return self.vector_taxonomy['ontological_vectors']['knowledge_particle_vector']
    
    def apply_gtmo_steering(self, model: Any, input_text: str, target_behavior: Optional[str] = None, 
                          strength: float = 1.0) -> SteeringResult:
        """
        Apply appropriate GTMØ steering to model based on input analysis.
        
        Args:
            model: Language model to steer
            input_text: Input text to analyze and steer for
            target_behavior: Optional specific target behavior
            strength: Steering strength
            
        Returns:
            SteeringResult with GTMØ steering applied
        """
        # Select appropriate vector
        vector = self.get_vector_for_context(input_text, target_behavior)
        
        # Extract vector if not already done
        if vector.vector is None:
            try:
                vector.extract_vector(model)
            except Exception as e:
                logger.warning(f"Failed to extract vector: {e}")
                # Return default result
                return SteeringResult(
                    original_activation=np.zeros(768),
                    modified_activation=np.zeros(768),
                    steering_strength=0.0,
                    vector_type=vector.vector_type,
                    metadata={'error': str(e)}
                )
        
        # Get model activation for input
        activation = vector._get_model_activation(model, input_text, None)
        if activation is None:
            activation = np.random.randn(768)  # Fallback
        
        # Apply steering
        result = vector.apply_steering(activation, strength)
        
        # Add GTMØ-specific metadata
        result.metadata.update({
            'input_text': input_text[:100],  # Truncated for storage
            'gtmo_input_classification': self.classify_input_gtmo(input_text),
            'vector_selected': vector.__class__.__name__,
            'gtmo_enhanced': True
        })
        
        # Record application
        self.application_history.append({
            'input_text': input_text[:50],
            'vector_type': vector.__class__.__name__,
            'strength': strength,
            'classification': result.gtmo_classification
        })
        
        return result
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about GTMØ steering vector usage.
        
        Returns:
            Dictionary with system statistics
        """
        total_applications = len(self.application_history)
        
        if total_applications == 0:
            return {'total_applications': 0, 'status': 'no_applications_yet'}
        
        # Count vector type usage
        vector_usage = {}
        for app in self.application_history:
            vector_type = app['vector_type']
            vector_usage[vector_type] = vector_usage.get(vector_type, 0) + 1
        
        # Count classification distribution
        classification_usage = {}
        for app in self.application_history:
            classification = app['classification']
            if classification:
                classification_usage[classification] = classification_usage.get(classification, 0) + 1
        
        return {
            'total_applications': total_applications,
            'vector_usage_distribution': vector_usage,
            'classification_distribution': classification_usage,
            'available_vector_types': list(self.vector_taxonomy.keys()),
            'total_vector_instances': sum(len(category) for category in self.vector_taxonomy.values()),
            'gtmo_v2_available': GTMO_V2_AVAILABLE
        }


# =============================================================================
# DEMONSTRATION AND USAGE EXAMPLES
# =============================================================================

def demonstrate_gtmo_steering_vectors():
    """
    Demonstrate the usage of GTMØ steering vectors with examples.
    """
    print("=" * 80)
    print("GTMØ STEERING VECTORS DEMONSTRATION")
    print("=" * 80)
    
    # Initialize the classification system
    gtmo_steering = GTMOSteeringVectorClassification()
    
    # Create a mock model for demonstration
    class MockModel:
        def __init__(self):
            self.hidden_size = 768
    
    model = MockModel()
    
    # Test cases representing different GTMØ categories
    test_cases = [
        {
            'input': "What is the result of dividing by zero in GTMØ?",
            'expected_classification': 'singularity',
            'target_behavior': None
        },
        {
            'input': "What will Bitcoin cost in 2050?",
            'expected_classification': 'alienated',
            'target_behavior': None
        },
        {
            'input': "The speed of light is 299,792,458 m/s",
            'expected_classification': 'knowledge_particle',
            'target_behavior': None
        },
        {
            'input': "How does consciousness emerge from neural activity?",
            'expected_classification': 'emergent',
            'target_behavior': None
        },
        {
            'input': "What are your limitations as an AI system?",
            'expected_classification': 'general',
            'target_behavior': 'meta_evaluation'
        }
    ]
    
    print("\n1. TESTING INDIVIDUAL STEERING VECTORS")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case['input']}")
        print(f"Expected Classification: {test_case['expected_classification']}")
        
        # Classify input
        actual_classification = gtmo_steering.classify_input_gtmo(test_case['input'])
        print(f"Actual Classification: {actual_classification}")
        
        # Get appropriate vector
        vector = gtmo_steering.get_vector_for_context(
            test_case['input'], 
            test_case['target_behavior']
        )
        print(f"Selected Vector: {vector.__class__.__name__}")
        
        # Apply steering (with mock extraction)
        try:
            result = gtmo_steering.apply_gtmo_steering(
                model, 
                test_case['input'], 
                test_case['target_behavior'],
                strength=1.0
            )
            print(f"Steering Applied: {result.gtmo_classification}")
            print(f"Vector Type: {result.vector_type.value}")
            
        except Exception as e:
            print(f"Steering Application Error: {e}")
    
    print("\n\n2. SYSTEM STATISTICS")
    print("-" * 50)
    
    stats = gtmo_steering.get_system_statistics()
    print(f"Total Applications: {stats['total_applications']}")
    print(f"GTMØ v2 Available: {stats['gtmo_v2_available']}")
    print(f"Available Vector Categories: {len(stats['available_vector_types'])}")
    print(f"Total Vector Instances: {stats['total_vector_instances']}")
    
    if stats['total_applications'] > 0:
        print("\nVector Usage Distribution:")
        for vector_type, count in stats.get('vector_usage_distribution', {}).items():
            print(f"  {vector_type}: {count}")
    
    print("\n\n3. VECTOR TAXONOMY OVERVIEW")
    print("-" * 50)
    
    for category, vectors in gtmo_steering.vector_taxonomy.items():
        print(f"\n{category.upper()}:")
        for vector_name, vector_instance in vectors.items():
            print(f"  - {vector_name}: {vector_instance.__class__.__name__}")
            if hasattr(vector_instance, 'target_behavior'):
                print(f"    Target Behavior: {vector_instance.target_behavior}")
    
    print("\n" + "=" * 80)
    print("GTMØ STEERING VECTORS KEY FEATURES:")
    print("✓ Theoretically grounded in GTMØ axioms")
    print("✓ Epistemically classified knowledge categories") 
    print("✓ Topologically navigated phase space")
    print("✓ Adaptively learning defense strategies")
    print("✓ Meta-cognitively aware self-evaluation")
    print("✓ Context-aware dynamic behavior")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_gtmo_steering_vectors()
