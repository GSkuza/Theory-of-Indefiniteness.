"""
GTMØ Core Operators - Fundamental Ψ_GTMØ and E_GTMØ operators
Essential foundation for all GTMØ classification and measurement
"""
import numpy as np
import math
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class OperationResult:
    """Result container for GTMØ operations"""
    score: float
    classification: str
    metadata: Dict[str, Any]
    
class ThresholdManager:
    """Dynamic threshold management for Ψᴷ/Ψʰ boundaries"""
    
    def __init__(self, knowledge_percentile: float = 85.0, shadow_percentile: float = 15.0):
        self.k_percentile = knowledge_percentile
        self.h_percentile = shadow_percentile
        self.history = []
        self.adaptation_rate = 0.05
    
    def calculate_thresholds(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate dynamic thresholds based on score distribution"""
        if len(scores) < 3:
            return 0.7, 0.3  # Default fallback
        
        k_threshold = np.percentile(scores, self.k_percentile)
        h_threshold = np.percentile(scores, self.h_percentile)
        self.history.append((k_threshold, h_threshold))
        return k_threshold, h_threshold

class PsiOperator:
    """Ψ_GTMØ operator - Epistemic purity measurement"""
    
    def __init__(self, threshold_manager: ThresholdManager):
        self.threshold_manager = threshold_manager
        self.operation_count = 0
    
    def __call__(self, fragment: Any, context: Dict[str, Any] = None) -> OperationResult:
        """Apply Ψ_GTMØ operator to fragment"""
        self.operation_count += 1
        context = context or {}
        
        # Handle special GTMØ objects
        if self._is_singularity(fragment):
            return OperationResult(1.0, "Ø", {"type": "ontological_singularity"})
        elif self._is_alienated(fragment):
            return OperationResult(0.999999999, "ℓ∅", {"type": "alienated_number"})
        
        # Calculate epistemic purity score
        score = self._calculate_purity(fragment)
        
        # Dynamic classification
        all_scores = context.get('all_scores', [score])
        k_threshold, h_threshold = self.threshold_manager.calculate_thresholds(all_scores)
        
        if score >= k_threshold:
            classification = "Ψᴷ"
        elif score <= h_threshold:
            classification = "Ψʰ"
        else:
            classification = "Ψᴧ"
        
        return OperationResult(score, classification, {
            "thresholds": (k_threshold, h_threshold),
            "operation_id": self.operation_count
        })
    
    def _calculate_purity(self, fragment: Any) -> float:
        """Calculate epistemic purity score"""
        text = str(fragment).lower()
        score = 0.5  # Base indefiniteness
        
        # Increase purity for definite concepts
        definite_indicators = ['theorem', 'proof', 'axiom', 'equals', 'always', 'never']
        score += 0.1 * sum(1 for ind in definite_indicators if ind in text)
        
        # Decrease purity for uncertain concepts
        uncertain_indicators = ['maybe', 'perhaps', 'might', 'could', 'unclear']
        score -= 0.15 * sum(1 for ind in uncertain_indicators if ind in text)
        
        # Handle paradoxes and self-reference
        if any(ind in text for ind in ['paradox', 'self-referential', 'this statement']):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _is_singularity(self, obj: Any) -> bool:
        return str(obj) == "Ø" or hasattr(obj, '__class__') and 'Singularity' in str(obj.__class__)
    
    def _is_alienated(self, obj: Any) -> bool:
        return str(obj).startswith("ℓ∅") or hasattr(obj, 'psi_score')

class EntropyOperator:
    """E_GTMØ operator - Cognitive entropy measurement"""
    
    def __init__(self):
        self.operation_count = 0
    
    def __call__(self, fragment: Any, context: Dict[str, Any] = None) -> OperationResult:
        """Apply E_GTMØ operator to fragment"""
        self.operation_count += 1
        
        # AX6: E_GTMØ(Ø) = min
        if self._is_singularity(fragment):
            return OperationResult(0.0, "minimal_entropy", {"axiom": "AX6"})
        
        # Calculate semantic partitions
        partitions = self._calculate_partitions(fragment)
        total_entropy = self._shannon_entropy(partitions)
        
        return OperationResult(total_entropy, "cognitive_entropy", {
            "partitions": partitions,
            "operation_id": self.operation_count
        })
    
    def _calculate_partitions(self, fragment: Any) -> List[float]:
        """Calculate semantic partitions for entropy computation"""
        text = str(fragment).lower()
        
        # Base partition weights
        certain_weight = 0.4
        uncertain_weight = 0.4
        unknown_weight = 0.2
        
        # Adjust based on content analysis
        certainty_indicators = sum(1 for ind in ['is', 'equals', 'always', 'fact'] if ind in text)
        uncertainty_indicators = sum(1 for ind in ['maybe', 'perhaps', 'might'] if ind in text)
        paradox_indicators = sum(1 for ind in ['paradox', 'contradiction'] if ind in text)
        
        # Redistribute weights
        if certainty_indicators > 0:
            certain_weight += 0.2
            uncertain_weight -= 0.1
        if uncertainty_indicators > 0:
            uncertain_weight += 0.2
            certain_weight -= 0.1
        if paradox_indicators > 0:
            unknown_weight += 0.3
            certain_weight -= 0.15
            uncertain_weight -= 0.15
        
        # Normalize
        total = certain_weight + uncertain_weight + unknown_weight
        return [max(0.001, w/total) for w in [certain_weight, uncertain_weight, unknown_weight]]
    
    def _shannon_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy"""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    def _is_singularity(self, obj: Any) -> bool:
        return str(obj) == "Ø" or hasattr(obj, '__class__') and 'Singularity' in str(obj.__class__)

class MetaFeedbackLoop:
    """Meta-feedback system for operator self-modification"""
    
    def __init__(self, psi_op: PsiOperator, entropy_op: EntropyOperator):
        self.psi_op = psi_op
        self.entropy_op = entropy_op
        self.feedback_history = []
    
    def run_feedback_cycle(self, fragments: List[Any], iterations: int = 3) -> Dict[str, Any]:
        """Run meta-feedback loop to optimize thresholds"""
        results = []
        
        for iteration in range(iterations):
            # Collect scores
            scores = []
            classifications = []
            
            for fragment in fragments:
                psi_result = self.psi_op(fragment, {'all_scores': scores})
                entropy_result = self.entropy_op(fragment)
                
                scores.append(psi_result.score)
                classifications.append(psi_result.classification)
            
            # Calculate distribution metrics
            psi_k_ratio = classifications.count("Ψᴷ") / len(classifications)
            psi_h_ratio = classifications.count("Ψʰ") / len(classifications)
            
            iteration_result = {
                'iteration': iteration,
                'psi_k_ratio': psi_k_ratio,
                'psi_h_ratio': psi_h_ratio,
                'avg_score': np.mean(scores),
                'avg_entropy': np.mean([entropy_result.score])
            }
            results.append(iteration_result)
        
        self.feedback_history.extend(results)
        return {'iterations': results, 'convergence': len(results)}

# Factory function for creating complete operator system
def create_gtmo_system() -> Tuple[PsiOperator, EntropyOperator, MetaFeedbackLoop]:
    """Create complete GTMØ operator system"""
    threshold_manager = ThresholdManager()
    psi_operator = PsiOperator(threshold_manager)
    entropy_operator = EntropyOperator()
    meta_loop = MetaFeedbackLoop(psi_operator, entropy_operator)
    
    return psi_operator, entropy_operator, meta_loop