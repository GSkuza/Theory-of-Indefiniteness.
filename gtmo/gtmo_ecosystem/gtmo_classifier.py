"""
GTMØ Universal Classifier - Complete Ψ-type classification system
The cognitive heart of GTMØ - classifies any fragment into Ψᴷ, Ψʰ, Ψᴧ, Ψᴺ, Ψ∅, Ø
"""
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import Counter

class PsiType(Enum):
    """Complete GTMØ Ψ-type taxonomy"""
    PSI_K = "Ψᴷ"      # Knowledge particles (high determinacy)
    PSI_H = "Ψʰ"      # Knowledge shadows (low determinacy)  
    PSI_L = "Ψᴧ"      # Liminal fragments (boundary states)
    PSI_N = "Ψᴺ"      # Novel emergent patterns
    PSI_E = "Ψᴱ"      # Emergent phenomena
    PSI_M = "Ψᴹ"      # Meta-cognitive fragments
    PSI_P = "Ψᴾ"      # Paradoxical fragments
    PSI_INDEF = "Ψ∅"  # Inherently indefinite
    OMEGA = "Ø"       # Ontological singularity

@dataclass
class ClassificationResult:
    """Result of GTMØ classification"""
    psi_type: PsiType
    confidence: float
    determinacy_score: float
    indefiniteness_vector: Dict[str, float]
    explanation: str
    metadata: Dict[str, Any] = None

class GTMOClassifier:
    """Universal classifier for all cognitive fragments"""
    
    def __init__(self, knowledge_threshold: float = 0.7, shadow_threshold: float = 0.3):
        self.k_threshold = knowledge_threshold
        self.h_threshold = shadow_threshold
        self.classification_history = []
        self.emergence_patterns = []
        
        # Load classification patterns
        self.definite_patterns = self._load_definite_patterns()
        self.indefinite_patterns = self._load_indefinite_patterns()
        self.paradox_patterns = self._load_paradox_patterns()
    
    def classify(self, fragment: Any, context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """Main classification method - determines Ψ-type of any fragment"""
        context = context or {}
        
        # Convert fragment to analyzable form
        fragment_text = str(fragment).strip()
        
        # Check for special GTMØ objects first
        if self._is_singularity(fragment):
            return ClassificationResult(
                PsiType.OMEGA, 1.0, 1.0,
                {"ontological": 1.0, "semantic": 0.0, "logical": 0.0, "temporal": 0.0, "paradox": 0.0, "definability": 1.0},
                "Ontological singularity - minimal entropy"
            )
        
        if self._is_alienated_number(fragment):
            return ClassificationResult(
                PsiType.PSI_INDEF, 0.999, 0.999,
                {"ontological": 0.999, "semantic": 0.999, "logical": 0.999, "temporal": 0.999, "paradox": 0.1, "definability": 0.001},
                "AlienatedNumber - inherently indefinite"
            )
        
        # Calculate 6D indefiniteness vector
        indef_vector = self._calculate_indefiniteness_vector(fragment_text)
        
        # Calculate determinacy score
        determinacy = self._calculate_determinacy(fragment_text, indef_vector)
        
        # Primary classification logic
        psi_type, confidence, explanation = self._classify_by_determinacy_and_patterns(
            fragment_text, determinacy, indef_vector, context
        )
        
        result = ClassificationResult(
            psi_type, confidence, determinacy, indef_vector, explanation,
            {"fragment_length": len(fragment_text), "classification_id": len(self.classification_history)}
        )
        
        self.classification_history.append(result)
        return result
    
    def _calculate_indefiniteness_vector(self, text: str) -> Dict[str, float]:
        """Calculate 6D indefiniteness vector"""
        text_lower = text.lower()
        
        # Semantic indefiniteness
        semantic = self._calculate_semantic_indefiniteness(text_lower)
        
        # Ontological indefiniteness  
        ontological = self._calculate_ontological_indefiniteness(text_lower)
        
        # Logical indefiniteness
        logical = self._calculate_logical_indefiniteness(text_lower)
        
        # Temporal indefiniteness
        temporal = self._calculate_temporal_indefiniteness(text_lower)
        
        # Paradox level
        paradox = self._calculate_paradox_level(text_lower)
        
        # Definability
        definability = 1.0 - (semantic + ontological + logical + temporal) / 4
        
        return {
            "semantic": semantic,
            "ontological": ontological, 
            "logical": logical,
            "temporal": temporal,
            "paradox": paradox,
            "definability": max(0.0, min(1.0, definability))
        }
    
    def _calculate_determinacy(self, text: str, indef_vector: Dict[str, float]) -> float:
        """Calculate overall determinacy score"""
        # Base determinacy from indefiniteness
        base_determinacy = 1.0 - sum(indef_vector[k] for k in ["semantic", "ontological", "logical", "temporal"]) / 4
        
        # Adjust for definite indicators
        definite_boost = sum(0.1 for pattern in self.definite_patterns if pattern in text.lower())
        indefinite_penalty = sum(0.15 for pattern in self.indefinite_patterns if pattern in text.lower())
        
        determinacy = base_determinacy + definite_boost - indefinite_penalty
        return max(0.0, min(1.0, determinacy))
    
    def _classify_by_determinacy_and_patterns(self, text: str, determinacy: float, 
                                            indef_vector: Dict[str, float], context: Dict[str, Any]) -> Tuple[PsiType, float, str]:
        """Core classification logic"""
        
        # Check for paradoxes first
        if indef_vector["paradox"] > 0.8:
            return PsiType.PSI_P, 0.9, "High paradox level detected"
        
        # Check for meta-cognitive patterns
        if self._is_metacognitive(text):
            return PsiType.PSI_M, 0.85, "Meta-cognitive fragment detected"
        
        # Check for emergence patterns
        if self._is_emergent(text, context):
            return PsiType.PSI_N, 0.8, "Novel emergent pattern detected"
        
        # Check for inherent indefiniteness
        total_indefiniteness = sum(indef_vector[k] for k in ["semantic", "ontological", "logical", "temporal"]) / 4
        if total_indefiniteness > 0.8:
            return PsiType.PSI_INDEF, 0.9, f"High indefiniteness: {total_indefiniteness:.3f}"
        
        # Standard determinacy-based classification
        if determinacy >= self.k_threshold:
            return PsiType.PSI_K, determinacy, f"Knowledge particle - determinacy: {determinacy:.3f}"
        elif determinacy <= self.h_threshold:
            return PsiType.PSI_H, 1.0 - determinacy, f"Knowledge shadow - determinacy: {determinacy:.3f}"
        else:
            return PsiType.PSI_L, 0.7, f"Liminal fragment - determinacy: {determinacy:.3f}"
    
    def _calculate_semantic_indefiniteness(self, text: str) -> float:
        """Calculate semantic indefiniteness"""
        indefinite_words = ["maybe", "perhaps", "might", "could", "possibly", "unclear", "vague", "ambiguous"]
        definite_words = ["certainly", "definitely", "always", "never", "exactly", "precisely"]
        
        indefinite_count = sum(1 for word in indefinite_words if word in text)
        definite_count = sum(1 for word in definite_words if word in text)
        
        if indefinite_count + definite_count == 0:
            return 0.5  # Neutral
        
        return indefinite_count / (indefinite_count + definite_count)
    
    def _calculate_ontological_indefiniteness(self, text: str) -> float:
        """Calculate ontological indefiniteness"""
        existence_questions = ["what is", "does it exist", "what does it mean", "undefined", "nonexistent"]
        existence_affirmations = ["it is", "exists", "defined as", "means that"]
        
        questions = sum(1 for phrase in existence_questions if phrase in text)
        affirmations = sum(1 for phrase in existence_affirmations if phrase in text)
        
        if questions + affirmations == 0:
            return 0.3  # Default low ontological indefiniteness
        
        return questions / (questions + affirmations)
    
    def _calculate_logical_indefiniteness(self, text: str) -> float:
        """Calculate logical indefiniteness"""
        logical_connectors = ["if", "then", "therefore", "because", "since", "implies"]
        contradictions = ["but", "however", "contradiction", "paradox", "inconsistent"]
        
        logical_strength = sum(1 for conn in logical_connectors if conn in text)
        contradiction_strength = sum(1 for contr in contradictions if contr in text)
        
        if logical_strength + contradiction_strength == 0:
            return 0.4
        
        return contradiction_strength / (logical_strength + contradiction_strength)
    
    def _calculate_temporal_indefiniteness(self, text: str) -> float:
        """Calculate temporal indefiniteness"""
        temporal_definite = ["always", "never", "eternal", "constant", "fixed"]
        temporal_indefinite = ["sometimes", "occasionally", "changing", "evolving", "temporary"]
        
        definite = sum(1 for word in temporal_definite if word in text)
        indefinite = sum(1 for word in temporal_indefinite if word in text)
        
        if definite + indefinite == 0:
            return 0.2
        
        return indefinite / (definite + indefinite)
    
    def _calculate_paradox_level(self, text: str) -> float:
        """Calculate paradox level"""
        paradox_indicators = 0
        
        for pattern in self.paradox_patterns:
            if pattern in text:
                paradox_indicators += 1
        
        # Self-reference detection
        if any(phrase in text for phrase in ["this statement", "this sentence", "itself"]):
            paradox_indicators += 2
        
        return min(1.0, paradox_indicators * 0.2)
    
    def _is_metacognitive(self, text: str) -> bool:
        """Detect meta-cognitive patterns"""
        meta_patterns = ["thinking about thinking", "knowledge about knowledge", "meta-", "self-aware", "consciousness"]
        return any(pattern in text.lower() for pattern in meta_patterns)
    
    def _is_emergent(self, text: str, context: Dict[str, Any]) -> bool:
        """Detect emergent patterns"""
        emergence_keywords = ["emergent", "emerging", "novel", "unprecedented", "breakthrough", "paradigm shift"]
        
        # Check for emergence keywords
        has_emergence_keywords = any(keyword in text.lower() for keyword in emergence_keywords)
        
        # Check for novelty in context
        is_novel_in_context = context.get('is_novel', False)
        
        # Check classification history for patterns
        if len(self.classification_history) > 5:
            recent_types = [r.psi_type for r in self.classification_history[-5:]]
            type_diversity = len(set(recent_types))
            has_pattern_break = type_diversity >= 4  # High diversity indicates emergence
        else:
            has_pattern_break = False
        
        return has_emergence_keywords or is_novel_in_context or has_pattern_break
    
    def _load_definite_patterns(self) -> List[str]:
        """Load patterns indicating definiteness"""
        return [
            "theorem", "proof", "axiom", "definition", "equals", "is exactly",
            "always true", "never false", "certainly", "mathematical fact",
            "proven", "demonstrated", "established", "verified"
        ]
    
    def _load_indefinite_patterns(self) -> List[str]:
        """Load patterns indicating indefiniteness"""
        return [
            "maybe", "perhaps", "possibly", "might be", "could be", "unclear",
            "undefined", "unknown", "ambiguous", "vague", "uncertain",
            "questionable", "debatable", "controversial"
        ]
    
    def _load_paradox_patterns(self) -> List[str]:
        """Load patterns indicating paradoxes"""
        return [
            "paradox", "contradiction", "self-referential", "infinite regress",
            "liar's paradox", "this statement is false", "russell's paradox",
            "gödel", "incompleteness", "undecidable"
        ]
    
    def _is_singularity(self, obj: Any) -> bool:
        """Check if object is Ø"""
        return (str(obj) == "Ø" or 
                hasattr(obj, '__class__') and 'Singularity' in str(obj.__class__))
    
    def _is_alienated_number(self, obj: Any) -> bool:
        """Check if object is ℓ∅"""
        return (str(obj).startswith("ℓ∅") or 
                hasattr(obj, 'psi_score') or
                hasattr(obj, '__class__') and 'Alienated' in str(obj.__class__))
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get statistics from classification history"""
        if not self.classification_history:
            return {"total_classifications": 0}
        
        type_counts = Counter(result.psi_type for result in self.classification_history)
        avg_confidence = np.mean([result.confidence for result in self.classification_history])
        avg_determinacy = np.mean([result.determinacy_score for result in self.classification_history])
        
        return {
            "total_classifications": len(self.classification_history),
            "type_distribution": dict(type_counts),
            "average_confidence": avg_confidence,
            "average_determinacy": avg_determinacy,
            "emergence_events": len([r for r in self.classification_history if r.psi_type == PsiType.PSI_N])
        }

# Convenience functions
def classify_fragment(fragment: Any, context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
    """Quick classification function"""
    classifier = GTMOClassifier()
    return classifier.classify(fragment, context)

def batch_classify(fragments: List[Any]) -> List[ClassificationResult]:
    """Classify multiple fragments"""
    classifier = GTMOClassifier()
    return [classifier.classify(fragment) for fragment in fragments]